from __future__ import annotations

import logging
import pickle
import joblib
import re
import uuid
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)

#  Paths 
_BASE       = Path(__file__).parent
_MODEL_PATH = _BASE.parent / "models" / "chatbot_model.pkl"

#  Dialogue States (mirrors notebook ConversationState enum) 
class DialogueState(Enum):
    IDLE               = auto()
    AWAIT_RESCHEDULE   = auto()
    AWAIT_NEW_DATETIME = auto()
    AWAIT_BOOK_SERVICE = auto()
    AWAIT_BOOK_DATETIME= auto()
    AWAIT_CANCEL_REF   = auto()
    AWAIT_PRICE_SVC    = auto()
    AWAIT_REC_PREF     = auto()


#   Fallback rule-based intent detection 
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(hi|hello|hey|howdy|good\s*(morning|afternoon|evening)|greetings|hiya)\b", re.I), "greet"),
    (re.compile(r"\b(bye|goodbye|see\s*you|later|farewell|cheers|take\s*care|all\s*set|thank\s*you\s*so\s*much|that'?s?\s*all|i'?m?\s*done)\b", re.I), "farewell"),
    (re.compile(r"\b(yes|yeah|yep|sure|absolutely|of\s*course|go\s*ahead|correct|affirmative|ok\s*yes|sounds\s*good|confirmed|proceed|alright|i\s*agree)\b", re.I), "confirm_yes"),
    (re.compile(r"\b(no\b|nope|no\s*thanks|i'?m?\s*good|not\s*really|don'?t\s*bother|i'?ll\s*pass|nah|forget\s*it|never\s*mind|not\s*interested)\b", re.I), "confirm_no"),
    (re.compile(r"\b(reschedule|change.*appointment|move.*booking|shift.*booking|postpone|different.*date|different.*time)\b", re.I), "change_booking"),
    (re.compile(r"\b(cancel|cancellation|remove.*appointment|call\s*off)\b", re.I), "cancel_booking"),
    (re.compile(r"\b(book|schedule|reserve|appointment|session|make.*booking)\b", re.I), "make_booking"),
    (re.compile(r"\b(price|cost|fee|how\s*much|rate|charges?|pricing|quote)\b", re.I), "check_price"),
    (re.compile(r"\b(recommend|suggest|best|top|popular|what.*should|guide\s*me|help\s*me\s*choose)\b", re.I), "get_recommendation"),
]


def _rule_based_intent(text: str) -> str:
    for pattern, intent in _RULES:
        if pattern.search(text):
            return intent
    return "fallback"


#   Entity extraction helpers  
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    from dateutil import parser as date_parser
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False
    logger.warning("spaCy / dateutil not installed — entity extraction uses regex only.")


def _build_phrase_matcher(svc_names: list[str]):
    """Build a spaCy PhraseMatcher for the given service names, or None."""
    if not _NLP_AVAILABLE or not svc_names:
        return None, None
    try:
        nlp = spacy.load("en_core_web_sm")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(n.lower()) for n in svc_names]
        matcher.add("SERVICE", patterns)
        return nlp, matcher
    except Exception as exc:
        logger.warning("spaCy model load failed: %s", exc)
        return None, None


def extract_service_regex(text: str, svc_names: list[str]) -> str | None:
    """Fallback regex service extractor when spaCy is unavailable."""
    lower = text.lower()
    # Sort longest-first to prefer more specific matches
    for name in sorted(svc_names, key=len, reverse=True):
        if name.lower() in lower:
            return name
    return None


def extract_datetime_from_text(text: str) -> datetime | None:
    """
    Parse a datetime from user text.
    Tries regex patterns first, then dateutil fuzzy parse.
    Returns None if nothing parseable found.
    """
    if _NLP_AVAILABLE:
        from dateutil import parser as date_parser
        patterns = [
            r"\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}(?::\d{2})?\s*[aApP][mM]",
            r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*[aApP][mM]",
            r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}(?::\d{2})?\s*[aApP][mM]",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return date_parser.parse(m.group(), dayfirst=True)
                except Exception:
                    pass
        try:
            return date_parser.parse(text, dayfirst=True, fuzzy=True)
        except Exception:
            return None
    else:
        # Simple regex fallback without dateutil
        m = re.search(
            r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}|"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
            r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\b",
            text, re.IGNORECASE
        )
        return m.group() if m else None  # Return string when dateutil is absent


def _format_dt(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%d %b %Y at %I:%M %p")
    return str(dt)  # fallback for plain string


#   Price tier data (mirrors notebook defaults, overridden from pkl)  
_DEFAULT_PRICE_GUIDE = {
    "budget":  "$50 – $95",
    "mid":     "$110 – $190",
    "premium": "$220 – $380",
}

_DEFAULT_SVC_NAMES = [
    "Swedish Massage", "Deep Tissue Massage", "Hot Stone Massage",
    "Aromatherapy Massage", "Sports Massage", "Prenatal Massage",
    "Couples Massage", "Classic Facial", "Anti-Aging Facial",
    "Hydrating Facial", "Acne Treatment Facial", "Reflexology",
    "Body Scrub & Wrap", "Lymphatic Drainage", "Wellness Package",
    "Corporate Wellness Session", "Meditation & Breathwork",
    "Infrared Sauna Session", "Cryotherapy", "Cupping Therapy",
    "Reiki Healing", "Sound Bath Therapy", "Ayurvedic Treatment",
    "Float Tank Session", "Stretching & Mobility Session",
]


#   Conversation state  
@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[dict[str, str]] = field(default_factory=list)
    dialogue_state: DialogueState = field(default=DialogueState.IDLE)
    booked_service: str | None = None
    booked_at: Any = None         # datetime or raw string
    last_intent: str = ""

    def add_turn(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text})


#   ChatBot  
class ChatBot:
    """Load once at startup; call .chat() per request."""

    def __init__(self) -> None:
        # ML components
        self._intent_pipeline: Any = None   # sklearn Pipeline (vectoriser + classifier)
        self._ready = False

        # Catalog (loaded from pkl or defaults)
        self._svc_names: list[str] = list(_DEFAULT_SVC_NAMES)
        self._svc_tier: dict[str, str] = {}
        self._price_guide: dict[str, str] = dict(_DEFAULT_PRICE_GUIDE)
        self._by_cat: dict[str, list[str]] = {}
        self._by_tier: dict[str, list[str]] = {}

        # spaCy entity extractors (built after catalog is loaded)
        self._nlp = None
        self._phrase_matcher = None

        # Session store {session_id: ConversationState}
        self._sessions: dict[str, ConversationState] = {}

        self._load()
        self._nlp, self._phrase_matcher = _build_phrase_matcher(self._svc_names)

    #   loading  

    def _load(self) -> None:
        if not _MODEL_PATH.exists():
            logger.warning(
                "chatbot_model.pkl not found at %s — running in rule-based mode.",
                _MODEL_PATH,
            )
            return

        try:
            try:
                bundle = joblib.load(_MODEL_PATH)
            except Exception:
                with open(_MODEL_PATH, "rb") as fh:
                    bundle = pickle.load(fh)

            if not isinstance(bundle, dict):
                logger.warning("Unexpected pkl format — falling back to rule-based mode.")
                return

            #   FIX: the notebook saves the pipeline under "intent_pipeline" ──
            pipeline = bundle.get("intent_pipeline")
            if pipeline is None:
                logger.warning(
                    "Key 'intent_pipeline' not found in bundle (found: %s). "
                    "Falling back to rule-based mode.",
                    list(bundle.keys()),
                )
                return

            self._intent_pipeline = pipeline

            # Load catalog data if present
            if "service_names"    in bundle: self._svc_names   = bundle["service_names"]
            if "service_to_tier"  in bundle: self._svc_tier    = bundle["service_to_tier"]
            if "price_guide"      in bundle: self._price_guide = bundle["price_guide"]
            if "services_by_tier" in bundle: self._by_tier     = bundle["services_by_tier"]
            if "services_by_category" in bundle: self._by_cat  = bundle["services_by_category"]

            self._ready = True
            labels = bundle.get("intent_labels", [])
            logger.info("Chatbot model loaded (intents: %s)", labels)

        except Exception as exc:
            logger.error("Failed to load chatbot model: %s", exc)

    #   entity extraction  

    def _extract_service(self, text: str) -> str | None:
        if self._nlp and self._phrase_matcher:
            try:
                doc = self._nlp(text.lower())
                matches = self._phrase_matcher(doc)
                if matches:
                    _, start, end = matches[0]
                    matched_lower = doc[start:end].text
                    for svc in self._svc_names:
                        if svc.lower() == matched_lower:
                            return svc
            except Exception:
                pass
        return extract_service_regex(text, self._svc_names)

    def _extract_datetime(self, text: str):
        return extract_datetime_from_text(text)

    #   intent classification  

    def _classify_intent(self, text: str) -> tuple[str, float]:
        """Return (intent, confidence). Falls back to rule-based when model not ready."""
        if self._ready and self._intent_pipeline is not None:
            try:
                pred = self._intent_pipeline.predict([text])[0]
                confidence = 0.0
                clf = self._intent_pipeline.named_steps.get("classifier")
                if clf is not None and hasattr(clf, "predict_proba"):
                    vec = self._intent_pipeline.named_steps["vectoriser"]
                    X = vec.transform([text])
                    confidence = float(clf.predict_proba(X)[0].max())
                elif clf is not None and hasattr(clf, "decision_function"):
                    vec = self._intent_pipeline.named_steps["vectoriser"]
                    X = vec.transform([text])
                    scores = clf.decision_function(X)[0]
                    # Normalise decision scores to a rough confidence proxy
                    scores_shifted = scores - scores.min()
                    total = scores_shifted.sum()
                    if total > 0:
                        confidence = float(scores_shifted.max() / total)

                # Fall back to rules if not confident enough
                if confidence < 0.30:
                    return _rule_based_intent(text), confidence
                return str(pred), confidence
            except Exception as exc:
                logger.warning("ML classification failed: %s", exc)

        return _rule_based_intent(text), 0.0

    #   recommendation helper  

    def _pick_recommendation(self, pref_text: str) -> tuple[str, str]:
        lower = pref_text.lower()
        if any(kw in lower for kw in ["stress", "relax", "calm", "peaceful", "unwind", "sooth"]):
            pool = (self._by_cat.get("massage", []) +
                    ["Meditation & Breathwork", "Sound Bath Therapy", "Float Tank Session"])
        elif any(kw in lower for kw in ["pain", "muscle", "sport", "athletic", "tension", "stiff"]):
            pool = ["Deep Tissue Massage", "Sports Massage", "Stretching & Mobility Session", "Cupping Therapy"]
        elif any(kw in lower for kw in ["skin", "face", "facial", "glow", "anti-aging", "acne"]):
            pool = self._by_cat.get("facial", ["Classic Facial", "Hydrating Facial"])
        elif any(kw in lower for kw in ["budget", "cheap", "affordable", "value"]):
            pool = self._by_tier.get("budget", [])
        elif any(kw in lower for kw in ["luxury", "premium", "indulge", "special"]):
            pool = self._by_tier.get("premium", [])
        elif any(kw in lower for kw in ["couple", "partner", "date", "together"]):
            pool = ["Couples Massage", "Wellness Package"]
        else:
            pool = self._by_cat.get("massage", self._svc_names[:5])

        if not pool:
            pool = self._svc_names[:5]

        pick = random.choice(pool)
        tier = self._svc_tier.get(pick, "mid")
        price = self._price_guide.get(tier, "see website for pricing")
        return pick, price

    #   state machine response  

    def _state_machine_response(
        self,
        intent: str,
        state: ConversationState,
        user_text: str,
        svc_ent: str | None,
        dt_ent: Any,
    ) -> str:

        ds = state.dialogue_state

        #   AWAIT_RESCHEDULE: waiting for yes/no  
        if ds == DialogueState.AWAIT_RESCHEDULE:
            if intent == "confirm_yes":
                state.dialogue_state = DialogueState.AWAIT_NEW_DATETIME
                return "Please provide the new date and time you'd like to reschedule to."
            else:
                state.dialogue_state = DialogueState.IDLE
                return ("No problem! Your booking remains unchanged. "
                        "Let me know if there's anything else I can help you with.")

        #   AWAIT_NEW_DATETIME: waiting for a date  
        if ds == DialogueState.AWAIT_NEW_DATETIME:
            if dt_ent:
                state.booked_at = dt_ent
                state.dialogue_state = DialogueState.IDLE
                return (f"Done! Reschedule request sent — your new date: {_format_dt(dt_ent)}. "
                        "You'll be notified once it's confirmed.")
            return ('I couldn\'t read that date. Please try a format like "30 Mar 2026 10 am" or "April 15, 2026 at 2 PM".')

        #   AWAIT_BOOK_SERVICE: waiting for service name  
        if ds == DialogueState.AWAIT_BOOK_SERVICE:
            if svc_ent:
                state.booked_service = svc_ent
                state.dialogue_state = DialogueState.AWAIT_BOOK_DATETIME
                return (f"Great choice — {svc_ent}! "
                        "What date and time would you like to book it?")
            svc_sample = ", ".join(random.sample(self._svc_names, min(3, len(self._svc_names))))
            return (f"I didn't catch a service name. We offer services like {svc_sample}, and more. "
                    "Which would you like?")

        #   AWAIT_BOOK_DATETIME: waiting for date/time  
        if ds == DialogueState.AWAIT_BOOK_DATETIME:
            if dt_ent:
                state.booked_at = dt_ent
                svc = state.booked_service
                state.dialogue_state = DialogueState.IDLE
                return (f"Your {svc} has been booked for {_format_dt(dt_ent)}. "
                        "You'll receive a confirmation shortly. Is there anything else?")
            return ('Please share the date and time, e.g. "April 15, 2026 at 2 PM". '
                    "When would you like to come in?")

        #   AWAIT_CANCEL_REF: waiting for a booking reference  
        if ds == DialogueState.AWAIT_CANCEL_REF:
            ref_match = re.search(r"\b(BK[\-\s]?\d{4,}[\-\d]*|\d{6,})\b", user_text, re.IGNORECASE)
            if ref_match:
                ref_no = ref_match.group().upper()
                state.dialogue_state = DialogueState.IDLE
                return (f"Booking {ref_no} has been successfully cancelled. "
                        "A confirmation will be sent to your registered email. "
                        "Is there anything else I can help with?")
            return ("I need your booking reference number to proceed. "
                    "It looks like BK-YYYYMMDD-XXX and can be found in your confirmation email.")

        #   AWAIT_PRICE_SVC: waiting for which service to price  
        if ds == DialogueState.AWAIT_PRICE_SVC:
            if svc_ent:
                tier  = self._svc_tier.get(svc_ent, "mid")
                price = self._price_guide.get(tier, "see website")
                state.dialogue_state = DialogueState.IDLE
                return (f"{svc_ent} is a {tier}-tier service, priced at {price}. "
                        "Would you like to book it?")
            return ("Which service are you asking about? "
                    "For example: 'Swedish Massage', 'Hot Stone Massage', 'Classic Facial'.")

        #   AWAIT_REC_PREF: waiting for preference text  
        if ds == DialogueState.AWAIT_REC_PREF:
            suggested, price = self._pick_recommendation(user_text)
            state.dialogue_state = DialogueState.IDLE
            return (f"Based on your preference, I'd recommend {suggested} ({price}). "
                    "Would you like to book it?")

        #   IDLE: route by intent  
        if intent == "greet":
            return ("Hello! Welcome to Blyss Wellness 🌿 "
                    "I can help you book, cancel, reschedule, get pricing, "
                    "or recommend a service. What would you like to do?")

        if intent == "farewell":
            return "You're welcome! Have a wonderful day. We look forward to seeing you soon. 🙏"

        if intent == "change_booking":
            if dt_ent:
                state.booked_at = dt_ent
                return (f"Reschedule request sent — new date: {_format_dt(dt_ent)}. "
                        "You'll be notified once confirmed.")
            state.dialogue_state = DialogueState.AWAIT_RESCHEDULE
            return ("Yes, you can reschedule your booking. "
                    "Would you like me to assist you with that?")

        if intent == "make_booking":
            if svc_ent and dt_ent:
                state.booked_service = svc_ent
                state.booked_at = dt_ent
                return (f"Your {svc_ent} has been booked for {_format_dt(dt_ent)}. "
                        "You'll receive a confirmation shortly!")
            if svc_ent:
                state.booked_service = svc_ent
                state.dialogue_state = DialogueState.AWAIT_BOOK_DATETIME
                return (f"Great choice — {svc_ent}! "
                        "What date and time would you like to book it?")
            state.dialogue_state = DialogueState.AWAIT_BOOK_SERVICE
            svc_sample = ", ".join(random.sample(self._svc_names, min(3, len(self._svc_names))))
            return (f"I'd love to help you make a booking! "
                    f"Which service are you interested in? E.g. {svc_sample}.")

        if intent == "cancel_booking":
            state.dialogue_state = DialogueState.AWAIT_CANCEL_REF
            return ("I can help you cancel your booking. "
                    "Please provide your booking reference number "
                    "(found in your confirmation email, e.g. BK-20260315-001).")

        if intent == "check_price":
            if svc_ent:
                tier  = self._svc_tier.get(svc_ent, "mid")
                price = self._price_guide.get(tier, "see website")
                return (f"{svc_ent} is a {tier}-tier service, priced at {price}. "
                        "Would you like to book it?")
            state.dialogue_state = DialogueState.AWAIT_PRICE_SVC
            return ("Happy to share pricing information! "
                    "Which service are you interested in?")

        if intent == "get_recommendation":
            state.dialogue_state = DialogueState.AWAIT_REC_PREF
            return ("I'd love to help you find the perfect treatment! "
                    "Could you tell me a bit about what you're looking for? "
                    "(e.g. relaxation, muscle relief, skincare, something luxurious)")

        if intent in ("confirm_yes", "confirm_no"):
            return "Got it! Is there anything else I can help you with?"

        # Fallback
        return ("I'm not quite sure I understood that. "
                "I can help with booking, cancellation, rescheduling, pricing, "
                "or service recommendations. What would you like?")

    #   public API  

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

        Returns
        -------
        {
            "session_id": str,
            "intent": str,
            "confidence": float,
            "response": str,
            "history_length": int,
            "model_used": str,
            "dialogue_state": str,
        }
        """
        state = self.get_or_create_session(session_id)

        intent, confidence = self._classify_intent(user_message)
        svc_ent = self._extract_service(user_message)
        dt_ent  = self._extract_datetime(user_message)

        response = self._state_machine_response(intent, state, user_message, svc_ent, dt_ent)

        state.add_turn("user", user_message)
        state.add_turn("assistant", response)
        state.last_intent = intent

        model_used = "ml_classifier" if self._ready and confidence >= 0.30 else "rule_based"

        return {
            "session_id":    state.session_id,
            "intent":        intent,
            "confidence":    round(confidence, 4),
            "response":      response,
            "history_length": len(state.history) // 2,
            "model_used":    model_used,
            "dialogue_state": state.dialogue_state.name,
        }


#   Singleton  
_bot: ChatBot | None = None


def get_bot() -> ChatBot:
    global _bot
    if _bot is None:
        _bot = ChatBot()
    return _bot