from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.chat_engine import get_bot
from src.recommendation_engine import get_engine

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: warm up singletons before first request ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚡ Warming up models...")
    get_engine()   # loads recommendation_model.pkl
    get_bot()      # loads chatbot_model.pkl
    logger.info("✅ Models ready.")
    yield
    logger.info("🛑 Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Blyss AI API",
    description=(
        "REST API powering the Blyss wellness platform:\n\n"
        "- **POST /recommend** — personalised service recommendations\n"
        "- **POST /chatbot**   — multi-turn NLP chatbot\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    customer_id: str = Field(
        ...,
        examples=["CUST_001"],
        description="Unique customer identifier",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return (1–20)",
    )

    @field_validator("customer_id")
    @classmethod
    def customer_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_id must not be empty")
        return v.strip()


class ServiceRecommendation(BaseModel):
    service: str
    score: float
    reason: str


class RecommendResponse(BaseModel):
    customer_id: str
    recommendations: list[ServiceRecommendation]
    model_used: str
    is_known_customer: bool


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["I'd like to book a deep tissue massage"],
        description="User's message to the chatbot",
    )
    session_id: str | None = Field(
        default=None,
        description="Pass the session_id from a previous response to continue the conversation",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    confidence: float
    response: str
    history_length: int
    model_used: str


class SessionClearRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to delete")


class HealthResponse(BaseModel):
    status: str
    recommendation_engine: str
    chatbot: str
    uptime_seconds: float


# ── Request timing middleware ─────────────────────────────────────────────────
_start_time = time.time()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - t0) * 1000:.1f}ms"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root() -> dict[str, Any]:
    return {
        "service": "Blyss AI API",
        "version": "1.0.0",
        "endpoints": {
            "POST /recommend": "Get personalised service recommendations",
            "POST /chatbot": "Chat with the Blyss wellness assistant",
            "DELETE /chatbot/session": "Clear a conversation session",
            "GET /health": "Service health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health() -> HealthResponse:
    engine = get_engine()
    bot    = get_bot()
    return HealthResponse(
        status="ok",
        recommendation_engine="loaded" if engine.is_ready else "fallback_mode",
        chatbot="loaded" if bot.is_ready else "rule_based_mode",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    tags=["Recommendation"],
    summary="Get service recommendations for a customer",
)
async def recommend(body: RecommendRequest) -> RecommendResponse:
    """
    Return top-N personalised service recommendations for a given `customer_id`.

    - If the customer is **known** (exists in training data), the trained model is used.
    - If the customer is **new/unknown**, a popularity-based fallback is returned.
    """
    try:
        engine = get_engine()
        result = engine.recommend(customer_id=body.customer_id, top_n=body.top_n)
    except Exception as exc:
        logger.exception("Recommendation error for customer %s", body.customer_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine error: {exc}",
        )

    return RecommendResponse(
        customer_id=result["customer_id"],
        recommendations=[ServiceRecommendation(**r) for r in result["recommendations"]],
        model_used=result["model_used"],
        is_known_customer=result["is_known_customer"],
    )


@app.post(
    "/chatbot",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chatbot"],
    summary="Send a message to the Blyss chatbot",
)
async def chatbot(body: ChatRequest) -> ChatResponse:
    """
    Send a user message and receive a contextual AI response.

    - Pass `session_id` from a previous response to **continue** the conversation.
    - Omit `session_id` (or pass `null`) to **start a new session**.
    - The chatbot supports intents: greeting, book_service, service_inquiry,
      pricing, cancellation, recommendation, complaint, feedback, farewell.
    """
    try:
        bot    = get_bot()
        result = bot.chat(user_message=body.message, session_id=body.session_id)
    except Exception as exc:
        logger.exception("Chatbot error for session %s", body.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot error: {exc}",
        )

    return ChatResponse(**result)


@app.delete(
    "/chatbot/session",
    status_code=status.HTTP_200_OK,
    tags=["Chatbot"],
    summary="Clear a conversation session",
)
async def clear_session(body: SessionClearRequest) -> dict[str, str]:
    """
    Delete all conversation history for a given `session_id`.
    Useful for implementing a "Start over" button on the frontend.
    """
    bot     = get_bot()
    cleared = bot.clear_session(body.session_id)
    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found.",
        )
    return {"message": f"Session '{body.session_id}' cleared."}


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )