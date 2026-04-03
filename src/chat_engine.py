from __future__ import annotations

import logging
import pickle
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).parent
_MODEL_PATH = _BASE / "outputs" / "models" / "chatbot_model.pkl"

# ── Intent → response templates ──────────────────────────────────────────────
_RESPONSES: dict[str, list[str]] = {
    "greeting": [
        "Hi there! Welcome to Blyss 🌿 How can I help you today?",
        "Hello! I'm your Blyss wellness assistant. What can I do for you?",
    ],
    "book_service": [
        "I'd love to help you book a session! Which service are you interested in — "
        "massage, aromatherapy, reflexology, or something else?",
        "Great choice! To confirm your booking I'll need: your preferred service, date, and time. "
        "What service would you like?",
    ],
    "service_inquiry": [
        "We offer a wide range of wellness services including Swedish Massage, Deep Tissue, "
        "Hot Stone, Aromatherapy, Reflexology, Sports Massage, Prenatal Massage, and Couples Massage. "
        "Which one would you like to know more about?",
    ],
    "pricing": [
        "Our pricing varies by service and duration:\n"
        "  • 60-min massage: from AUD 120\n"
        "  • 90-min massage: from AUD 160\n"
        "  • Aromatherapy (60 min): AUD 130\n"
        "  • Couples Massage (60 min): AUD 230\n\n"
        "Would you like to book a session?",
    ],
    "cancellation": [
        "You can cancel or reschedule up to 4 hours before your appointment at no charge. "
        "Cancellations within 4 hours incur a 50% fee. "
        "Would you like help rescheduling instead?",
    ],
    "recommendation": [
        "Based on popular choices, I'd suggest our Swedish Massage for relaxation or "
        "Deep Tissue if you have muscle tension. "
        "Would you like me to pull personalised recommendations using your customer ID?",
    ],
    "complaint": [
        "I'm really sorry to hear that. Your experience matters to us. "
        "Could you share more details so I can escalate this to our team?",
        "Thank you for letting us know. I'll flag this for our quality team right away. "
        "Can I get your booking reference number?",
    ],
    "feedback": [
        "Thank you so much for your feedback — we genuinely appreciate it! "
        "Is there anything else I can help you with?",
    ],
    "farewell": [
        "Take care! We look forward to seeing you again at Blyss 🌿",
        "Goodbye! Don't forget to treat yourself — you deserve it. 😊",
    ],
    "fallback": [
        "I didn't quite catch that. Could you rephrase? I can help with bookings, "
        "services, pricing, cancellations, and recommendations.",
        "Sorry, I'm not sure I understand. Try asking about our services, pricing, or bookings!",
    ],
}

# Round-robin index per intent so responses vary across turns
_response_idx: dict[str, int] = {k: 0 for k in _RESPONSES}


def _pick_response(intent: str) -> str:
    options = _RESPONSES.get(intent, _RESPONSES["fallback"])
    idx = _response_idx.get(intent, 0)
    response = options[idx % len(options)]
    _response_idx[intent] = idx + 1
    return response


# ── Rule-based intent detection (fallback when model absent) ─────────────────
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(hi|hello|hey|howdy|good\s*(morning|afternoon|evening))\b", re.I), "greeting"),
    (re.compile(r"\b(bye|goodbye|see\s*you|later|farewell|thanks?\s*bye)\b", re.I), "farewell"),
    (re.compile(r"\b(book|schedule|reserve|appointment|session)\b", re.I), "book_service"),
    (re.compile(r"\b(price|cost|fee|how\s*much|rate|charges?)\b", re.I), "pricing"),
    (re.compile(r"\b(cancel|reschedule|refund|postpone)\b", re.I), "cancellation"),
    (re.compile(r"\b(recommend|suggest|best|top|popular|what.*should)\b", re.I), "recommendation"),
    (re.compile(r"\b(services?|offer|menu|what.*you\s*do|massage|aromatherapy|reflexology)\b", re.I), "service_inquiry"),
    (re.compile(r"\b(complain|issue|problem|unhappy|bad|worst|terrible|awful)\b", re.I), "complaint"),
    (re.compile(r"\b(feedback|review|experience|loved|great|awesome|excellent)\b", re.I), "feedback"),
]


def _rule_based_intent(text: str) -> str:
    for pattern, intent in _RULES:
        if pattern.search(text):
            return intent
    return "fallback"


# ── Conversation state ────────────────────────────────────────────────────────
@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[dict[str, str]] = field(default_factory=list)
    last_intent: str = ""
    context: dict[str, Any] = field(default_factory=dict)  # e.g. pending service, booking slot

    def add_turn(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text})

    def recent_context(self, n: int = 4) -> str:
        """Last n turns as a plain string for prompt context."""
        return "\n".join(
            f"{t['role'].capitalize()}: {t['text']}" for t in self.history[-n:]
        )


# ── ChatBot ───────────────────────────────────────────────────────────────────
class ChatBot:
    """Load once at startup; call .chat() per request."""

    def __init__(self) -> None:
        self._vectorizer: Any = None
        self._classifier: Any = None
        self._label_encoder: Any = None
        self._intents: list[str] = list(_RESPONSES.keys())
        self._ready = False

        # In-memory session store  {session_id: ConversationState}
        self._sessions: dict[str, ConversationState] = {}

        self._load()

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _MODEL_PATH.exists():
            logger.warning(
                "chatbot_model.pkl not found at %s — running in rule-based mode.",
                _MODEL_PATH,
            )
            return

        try:
            with open(_MODEL_PATH, "rb") as fh:
                bundle = pickle.load(fh)

            if isinstance(bundle, dict):
                self._vectorizer    = bundle.get("vectorizer")
                self._classifier    = bundle.get("classifier")
                self._label_encoder = bundle.get("label_encoder")
                if "intents" in bundle:
                    self._intents = bundle["intents"]
            else:
                logger.warning("Unexpected pkl format — falling back to rule-based mode.")
                return

            self._ready = True
            logger.info("Chatbot model loaded (intents: %s)", self._intents)
        except Exception as exc:
            logger.error("Failed to load chatbot model: %s", exc)

    def _classify_intent(self, text: str) -> tuple[str, float]:
        """Return (intent, confidence). Falls back to rules if model not ready."""
        if not self._ready or self._vectorizer is None or self._classifier is None:
            return _rule_based_intent(text), 0.0

        try:
            vec = self._vectorizer.transform([text])
            pred = self._classifier.predict(vec)[0]

            # Decode label
            if self._label_encoder is not None:
                intent = self._label_encoder.inverse_transform([pred])[0]
            elif isinstance(pred, (int, np.integer)):
                intent = self._intents[pred] if pred < len(self._intents) else "fallback"
            else:
                intent = str(pred)

            # Confidence (if classifier supports predict_proba)
            confidence = 0.0
            if hasattr(self._classifier, "predict_proba"):
                proba = self._classifier.predict_proba(vec)[0]
                confidence = float(proba.max())

            # Fall back to rules if model is not confident
            if confidence < 0.35:
                return _rule_based_intent(text), confidence

            return intent, confidence
        except Exception as exc:
            logger.warning("Intent classification failed: %s", exc)
            return _rule_based_intent(text), 0.0

    def _contextual_response(
        self,
        intent: str,
        state: ConversationState,
        user_text: str,
    ) -> str:
        """
        Augment the base template response with conversation context.
        Handles simple multi-turn flows (e.g. collect service → date → confirm).
        """
        base = _pick_response(intent)

        # ── Multi-turn: booking flow ─────────────────────────────────────────
        if intent == "book_service":
            if "service" not in state.context:
                # Try to extract service name from user message
                services_pattern = re.compile(
                    r"\b(swedish|deep tissue|hot stone|aromatherapy|reflexology|"
                    r"sports|prenatal|couples)\b",
                    re.I,
                )
                m = services_pattern.search(user_text)
                if m:
                    state.context["service"] = m.group(0).title()
                    return (
                        f"Great choice! {state.context['service']} is wonderful. "
                        "What date and time work best for you?"
                    )
                return base
            elif "date" not in state.context:
                state.context["date"] = user_text  # store raw date string
                return (
                    f"Perfect! I'll note {user_text} for your "
                    f"{state.context.get('service', 'session')}. "
                    "Can I confirm your name and contact number to complete the booking?"
                )
            else:
                state.context.clear()
                return (
                    "Your booking request has been received! Our team will confirm "
                    "within 30 minutes. Is there anything else I can help with?"
                )

        # ── Contextual follow-up after recommendation ────────────────────────
        if intent == "recommendation" and state.last_intent == "service_inquiry":
            return (
                "Based on what you were asking about, I'd especially recommend "
                "our Deep Tissue Massage or Aromatherapy. "
                "Would you like to book one of those?"
            )

        return base

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_or_create_session(self, session_id: str | None = None) -> ConversationState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        state = ConversationState(session_id=session_id or str(uuid.uuid4()))
        self._sessions[state.session_id] = state
        return state

    def clear_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def chat(
        self,
        user_message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message and return a response dict.

        Parameters
        ----------
        user_message : str
        session_id   : str | None  — pass None to start a new session

        Returns
        -------
        {
            "session_id": str,
            "intent": str,
            "confidence": float,
            "response": str,
            "history_length": int,
            "model_used": str,
        }
        """
        state = self.get_or_create_session(session_id)

        # Classify intent
        intent, confidence = self._classify_intent(user_message)

        # Build contextual response
        response = self._contextual_response(intent, state, user_message)

        # Update state
        state.add_turn("user", user_message)
        state.add_turn("assistant", response)
        state.last_intent = intent

        model_used = "ml_classifier" if self._ready and confidence >= 0.35 else "rule_based"

        return {
            "session_id": state.session_id,
            "intent": intent,
            "confidence": round(confidence, 4),
            "response": response,
            "history_length": len(state.history) // 2,  # turns (user+assistant = 1 turn)
            "model_used": model_used,
        }


# ── numpy import guard (only needed for confidence float cast) ────────────────
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# ── Singleton ─────────────────────────────────────────────────────────────────
_bot: ChatBot | None = None


def get_bot() -> ChatBot:
    global _bot
    if _bot is None:
        _bot = ChatBot()
    return _bot