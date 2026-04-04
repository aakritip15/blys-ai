import time
import uuid
import requests
import streamlit as st

#   Page Config  
st.set_page_config(
    page_title="Blyss AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
import os
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "customer_behaviour_report.pdf")  # adjust as needed

#   Styling  
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #FAFAF8;
    color: #1a1a18;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8E6DF;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Brand header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
}
.brand-logo {
    width: 36px;
    height: 36px;
    background: #1a1a18;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1a1a18;
    letter-spacing: -0.3px;
}
.brand-tagline {
    font-size: 11px;
    color: #888780;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: -2px;
}

/* ── Nav pills ── */
.nav-section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #B4B2A9;
    margin: 1.5rem 0 0.5rem;
    padding-left: 2px;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}
.status-ok {
    background: #EAF3DE;
    color: #3B6D11;
}
.status-error {
    background: #FCEBEB;
    color: #A32D2D;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
}
.status-dot-ok    { background: #639922; }
.status-dot-error { background: #E24B4A; }

/* ── Page heading ── */
.page-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    color: #1a1a18;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
    line-height: 1.2;
}
.page-subheading {
    font-size: 15px;
    color: #888780;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* ── Metric cards ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 2rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8E6DF;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: #888780;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #1a1a18;
    line-height: 1;
}
.metric-detail {
    font-size: 12px;
    color: #B4B2A9;
    margin-top: 4px;
}

/* ── Rec card ── */
.rec-card {
    background: #FFFFFF;
    border: 1px solid #E8E6DF;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.15s;
}
.rec-rank {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #D3D1C7;
    min-width: 28px;
}
.rec-body { flex: 1; }
.rec-service {
    font-weight: 500;
    font-size: 15px;
    color: #1a1a18;
    margin-bottom: 3px;
}
.rec-reason {
    font-size: 13px;
    color: #888780;
}
.rec-score {
    font-size: 13px;
    font-weight: 500;
    color: #3B6D11;
    background: #EAF3DE;
    padding: 3px 10px;
    border-radius: 20px;
}
.top-rec-card {
    border: 1.5px solid #C0DD97;
    background: #F7FBF0;
}

/* ── Chat window ── */
.chat-container {
    background: #FFFFFF;
    border: 1px solid #E8E6DF;
    border-radius: 16px;
    overflow: hidden;
}
.chat-header {
    background: #1a1a18;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.chat-header-icon {
    width: 32px;
    height: 32px;
    background: #2C2C2A;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.chat-header-title {
    font-size: 15px;
    font-weight: 500;
    color: #FFFFFF;
}
.chat-header-sub {
    font-size: 12px;
    color: #888780;
}
.chat-messages {
    padding: 1.25rem;
    min-height: 320px;
    max-height: 480px;
    overflow-y: auto;
}

/* ── Chat bubbles ── */
.msg-row {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
    align-items: flex-start;
}
.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 500;
    flex-shrink: 0;
}
.avatar-bot  { background: #1a1a18; color: #FAFAF8; }
.avatar-user { background: #E8E6DF; color: #444441; }

.msg-bubble {
    display: inline-block;  
    width: fit-content;
    max-width: 72%;
    padding: 10px 14px;
    border-radius: 14px;
    font-size: 14px;
    line-height: 1.55;
}
.bubble-bot {
    background: #F4F3EF;
    color: #1a1a18;
    border-bottom-left-radius: 4px;
}
.bubble-user {
    background: #1a1a18;
    color: #FAFAF8;
    border-bottom-right-radius: 4px;
}
.msg-meta {
    font-size: 11px;
    color: #B4B2A9;
    margin-top: 4px;
}
.msg-intent {
    display: inline-block;
    background: #F4F3EF;
    color: #5F5E5A;
    font-size: 10px;
    letter-spacing: 0.4px;
    padding: 2px 8px;
    border-radius: 10px;
    margin-top: 4px;
}

/* ── Input row ── */
.chat-input-row {
    border-top: 1px solid #E8E6DF;
    padding: 12px 1.25rem;
    background: #FAFAF8;
}

/* ── Report card ── */
.report-card {
    background: #FFFFFF;
    border: 1px solid #E8E6DF;
    border-radius: 14px;
    padding: 1.75rem 2rem;
    display: flex;
    align-items: center;
    gap: 20px;
}
.report-icon {
    width: 52px;
    height: 52px;
    background: #F4F3EF;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}
.report-title {
    font-weight: 500;
    font-size: 16px;
    color: #1a1a18;
    margin-bottom: 3px;
}
.report-desc {
    font-size: 13px;
    color: #888780;
}

/* ── Divider ── */
.thin-divider {
    border: none;
    border-top: 1px solid #E8E6DF;
    margin: 1.5rem 0;
}

/* ── Quick chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-bottom: 12px;
}
.chip {
    background: #F4F3EF;
    border: 1px solid #E8E6DF;
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 12px;
    color: #5F5E5A;
    cursor: pointer;
    transition: all 0.12s;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: #1a1a18 !important;
    color: #FAFAF8 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #E8E6DF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    background: #FFFFFF !important;
}
.stSelectbox > div > div {
    border-radius: 8px !important;
    border: 1px solid #E8E6DF !important;
}
div[data-testid="stRadio"] > div {
    gap: 8px;
}
.stAlert {
    border-radius: 10px !important;
}
/* Hide hamburger menu default header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


#   Helpers  
def api_get(path: str, timeout: int = 6):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Cannot connect to API. Make sure the server is running on port 8000."
    except Exception as e:
        return None, str(e)


def api_post(path: str, payload: dict, timeout: int = 10):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Cannot connect to API. Make sure the server is running on port 8000."
    except requests.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def api_delete(path: str, payload: dict, timeout: int = 6):
    try:
        r = requests.delete(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def check_health():
    data, err = api_get("/health")
    return data, err


#   Session State  
if "chat_session_id"  not in st.session_state:
    st.session_state.chat_session_id  = None
if "chat_history"     not in st.session_state:
    st.session_state.chat_history     = []   # list of {"role", "text", "intent", "confidence"}
if "health_cache"     not in st.session_state:
    st.session_state.health_cache     = None
if "health_ts"        not in st.session_state:
    st.session_state.health_ts        = 0
if "active_page"      not in st.session_state:
    st.session_state.active_page      = "Chatbot"


#   Sidebar  
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🌿</div>
        <div>
            <div class="brand-name">Blyss</div>
            <div class="brand-tagline">Wellness AI Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    #   Health check (cached 30s)  
    now = time.time()
    if now - st.session_state.health_ts > 30:
        hdata, herr = check_health()
        st.session_state.health_cache = (hdata, herr)
        st.session_state.health_ts    = now
    else:
        hdata, herr = st.session_state.health_cache or (None, None)

    if hdata and hdata.get("status") == "ok":
        st.markdown("""
        <div class="status-pill status-ok">
            <div class="status-dot status-dot-ok"></div>
            API connected
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pill status-error">
            <div class="status-dot status-dot-error"></div>
            API offline
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "💬  Chatbot":        "Chatbot",
        "✦   Recommendations": "Recommendations",
        "📊  Analysis Report": "Report",
    }
    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#B4B2A9; padding: 0 2px;">
        <div style="margin-bottom:6px;"><strong style="color:#888780;">API Base URL</strong></div>
        <code style="font-size:11px; background:#F4F3EF; padding:3px 7px; border-radius:5px;">
            localhost:8000
        </code>
    </div>
    """, unsafe_allow_html=True)

    if hdata:
        st.markdown(f"""
        <div style="font-size:12px; color:#B4B2A9; margin-top:10px; padding: 0 2px;">
            <span>Uptime: <strong style="color:#5F5E5A;">{int(hdata.get('uptime_seconds', 0))}s</strong></span>
        </div>
        """, unsafe_allow_html=True)


#   Page: Chatbot  
if st.session_state.active_page == "Chatbot":

    st.markdown('<div class="page-heading">AI Wellness Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Book, reschedule, cancel, and get personalised recommendations — all in one conversation.</div>', unsafe_allow_html=True)

    col_chat, col_info = st.columns([2, 1], gap="large")

    with col_chat:
        #   Chat window  
        st.markdown("""
        <div class="chat-container">
          <div class="chat-header">
            <div class="chat-header-icon">🌿</div>
            <div>
              <div class="chat-header-title">Blyss Assistant</div>
              <div class="chat-header-sub">Powered by NLP intent classification</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Messages area
        chat_area = st.container()
        with chat_area:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center; padding: 2.5rem 1rem;">
                    <div style="font-size:36px; margin-bottom:12px;">🌿</div>
                    <div style="font-size:15px; color:#5F5E5A; font-weight:500;">Hello! I'm your Blyss wellness assistant.</div>
                    <div style="font-size:13px; color:#B4B2A9; margin-top:6px;">I can help you book sessions, reschedule appointments,<br>check pricing, and recommend services.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    text = msg["text"]
                    if role == "user":
                        st.markdown(f"""
                        <div class="msg-row user">
                            <div class="msg-avatar avatar-user">Y</div>
                            <div>
                                <div class="msg-bubble bubble-user">{text}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        intent_html = ""
                        if msg.get("intent"):
                            conf_pct = int(msg.get("confidence", 0) * 100)
                            intent_html = f'<div><span class="msg-intent">{msg["intent"]}  ·  {conf_pct}%</span></div>'
                        st.markdown(f"""
                        <div class="msg-row">
                            <div class="msg-avatar avatar-bot">B</div>
                            <div>
                                <div class="msg-bubble bubble-bot">{text}</div>
                                {intent_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        #   Quick suggestions  
        QUICK_MESSAGES = [
            "Can I reschedule my booking?",
            "I'd like to book a Swedish Massage",
            "How much does a Hot Stone Massage cost?",
            "What do you recommend?",
            "Cancel my appointment",
        ]
        st.markdown('<div class="chip-row">', unsafe_allow_html=True)
        qcols = st.columns(len(QUICK_MESSAGES))
        for i, (col, qmsg) in enumerate(zip(qcols, QUICK_MESSAGES)):
            with col:
                if st.button(qmsg[:22] + ("…" if len(qmsg) > 22 else ""), key=f"q_{i}"):
                    st.session_state._pending_message = qmsg
        st.markdown('</div>', unsafe_allow_html=True)

        #  Input row 
        with st.form("chat_form", clear_on_submit=True):
            inp_col, btn_col = st.columns([5, 1])
            with inp_col:
                user_input = st.text_input(
                    "Message",
                    label_visibility="collapsed",
                    placeholder="Type your message…",
                    key="chat_input",
                )
            with btn_col:
                submitted = st.form_submit_button("Send", use_container_width=True)

        # Handle quick-chip press
        if hasattr(st.session_state, "_pending_message"):
            user_input = st.session_state._pending_message
            submitted  = True
            del st.session_state._pending_message

        if submitted and user_input and user_input.strip():
            msg_text = user_input.strip()

            # Add user message
            st.session_state.chat_history.append({"role": "user", "text": msg_text})

            payload = {"message": msg_text}
            if st.session_state.chat_session_id:
                payload["session_id"] = st.session_state.chat_session_id

            with st.spinner(""):
                data, err = api_post("/chatbot", payload)

            if err:
                st.session_state.chat_history.append({
                    "role": "bot",
                    "text": f"⚠ Error: {err}",
                    "intent": None,
                    "confidence": 0,
                })
            else:
                st.session_state.chat_session_id = data.get("session_id")
                st.session_state.chat_history.append({
                    "role":       "bot",
                    "text":       data.get("response", "—"),
                    "intent":     data.get("intent"),
                    "confidence": data.get("confidence", 0),
                })
            st.rerun()

        #  Session controls 
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("🗑  Clear conversation", use_container_width=True):
                if st.session_state.chat_session_id:
                    api_delete("/chatbot/session",
                               {"session_id": st.session_state.chat_session_id})
                st.session_state.chat_history    = []
                st.session_state.chat_session_id = None
                st.rerun()

        with ctrl_col2:
            if st.session_state.chat_session_id:
                st.markdown(f"""
                <div style="font-size:12px; color:#B4B2A9; text-align:right; padding-top:8px;">
                    Session: <code style="font-size:11px;">{st.session_state.chat_session_id[:12]}…</code>
                </div>
                """, unsafe_allow_html=True)

    #  Right info panel 
    with col_info:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E8E6DF; border-radius:14px; padding:1.25rem;">
            <div style="font-size:13px; font-weight:500; color:#1a1a18; margin-bottom:10px;">What I can do</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
        """, unsafe_allow_html=True)

        capabilities = [
            ("📅", "Book a service", "Schedule any of our 25+ wellness treatments"),
            ("🔄", "Reschedule", "Move your appointment to a new date & time"),
            ("✕", "Cancel booking", "Cancel using your booking reference"),
            ("💰", "Check pricing", "Get price info for any service or tier"),
            ("✦", "Recommendations", "Get personalised service suggestions"),
        ]
        for icon, title, desc in capabilities:
            st.markdown(f"""
            <div style="display:flex; gap:10px; align-items:flex-start; padding:8px 10px; background:#FAFAF8; border-radius:9px;">
                <span style="font-size:16px; flex-shrink:0;">{icon}</span>
                <div>
                    <div style="font-size:13px; font-weight:500; color:#1a1a18;">{title}</div>
                    <div style="font-size:12px; color:#888780; margin-top:1px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Example reschedule flow card
        st.markdown("""
        <div style="background:#F7FBF0; border:1px solid #C0DD97; border-radius:14px; padding:1.25rem;">
            <div style="font-size:12px; font-weight:500; letter-spacing:0.5px; text-transform:uppercase; color:#3B6D11; margin-bottom:10px;">Example: Reschedule flow</div>
            <div style="font-size:13px; color:#444441; display:flex; flex-direction:column; gap:7px;">
                <div><span style="color:#B4B2A9; font-size:11px;">You</span><br>"Can I reschedule my booking?"</div>
                <div><span style="color:#3B6D11; font-size:11px;">Blyss</span><br>"Yes, you can reschedule through the Blyss app. Would you like me to assist?"</div>
                <div><span style="color:#B4B2A9; font-size:11px;">You</span><br>"Yes"</div>
                <div><span style="color:#3B6D11; font-size:11px;">Blyss</span><br>"Please provide the new date and time."</div>
                <div><span style="color:#B4B2A9; font-size:11px;">You</span><br>"30 Mar 2025 10 am"</div>
                <div><span style="color:#3B6D11; font-size:11px;">Blyss</span><br>"Sent to pro — you'll be notified once confirmed."</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


#  Page: Recommendations 
elif st.session_state.active_page == "Recommendations":

    st.markdown('<div class="page-heading">Service Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Personalised treatment suggestions powered by the hybrid SVD recommendation model.</div>', unsafe_allow_html=True)

    # Input row
    in_col1, in_col2, in_col3 = st.columns([2, 1, 1])
    with in_col1:
        customer_id = st.text_input(
            "Customer ID",
            value="1253",
            placeholder="e.g. 1096",
            help="Enter an existing customer ID from the training dataset.",
        )
    with in_col2:
        top_n = st.selectbox("Top N results", options=[1,2,3], index=2)
    with in_col3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("Get recommendations", use_container_width=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    if run_btn:
        if not customer_id.strip():
            st.warning("Please enter a customer ID.")
        else:
            with st.spinner("Running recommendation engine…"):
                data, err = api_post("/recommend", {
                    "customer_id": customer_id.strip(),
                    "top_n": top_n,
                })

            if err:
                st.error(f"API error: {err}")
            else:
                recs            = data.get("recommendations", [])
                model_used      = data.get("model_used", "—")
                is_known        = data.get("is_known_customer", False)

                #  Metrics strip  
                st.markdown(f"""
                <div class="metric-strip">
                    <div class="metric-card">
                        <div class="metric-label">Customer</div>
                        <div class="metric-value" style="font-size:20px;">{data.get('customer_id','—')}</div>
                        <div class="metric-detail">{"Known customer" if is_known else "New customer — popularity fallback"}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Results returned</div>
                        <div class="metric-value">{len(recs)}</div>
                        <div class="metric-detail">of {top_n} requested</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Model</div>
                        <div class="metric-value" style="font-size:15px; padding-top:6px;">{model_used.replace('_',' ').title()}</div>
                        <div class="metric-detail">{"Personalised" if is_known else "Popularity-based"}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if not is_known:
                    st.info("Customer not found in training data — showing popularity-based recommendations.")

                #  Recommendation cards 
                for i, rec in enumerate(recs):
                    svc    = rec.get("service", "—")
                    score  = rec.get("score", 0.0)
                    reason = rec.get("reason", "")
                    top    = "top-rec-card" if i == 0 else ""
                    st.markdown(f"""
                    <div class="rec-card {top}">
                        <div class="rec-rank">{"01" if i == 0 else f"0{i+1}" if i < 9 else str(i+1)}</div>
                        <div class="rec-body">
                            <div class="rec-service">{svc}</div>
                            <div class="rec-reason">{reason}</div>
                        </div>
                        <div class="rec-score">{score:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size:40px; margin-bottom:14px;">✦</div>
            <div style="font-size:16px; color:#5F5E5A; font-weight:500;">Enter a customer ID and click "Get recommendations"</div>
            <div style="font-size:13px; color:#B4B2A9; margin-top:6px;">The hybrid SVD + content-based model will personalise results.</div>
        </div>
        """, unsafe_allow_html=True)


#  Page: Analysis Report 
elif st.session_state.active_page == "Report":

    st.markdown('<div class="page-heading">Customer Analysis Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Insights from customer booking behaviour, sentiment analysis, and service trends.</div>', unsafe_allow_html=True)

    #  Report card 
    st.markdown("""
    <div class="report-card" style="margin-bottom:24px;">
        <div class="report-icon">📊</div>
        <div style="flex:1;">
            <div class="report-title">Blyss Customer Intelligence Report</div>
            <div class="report-desc">Comprehensive analysis including booking patterns, revenue segmentation, NPS scores, sentiment trends, and recommendation model performance.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    #  Embedded report preview 
    st.markdown("#### Preview")
    try:
        import base64
        with open(REPORT_PATH, "rb") as f:
            pdf_bytes = f.read()

        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="700" 
            type="application/pdf">
        </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        # Placeholder cards when report isn't present
        st.markdown("""
        <div style="background:#FFF8E7; border:1px solid #FAC775; border-radius:12px; padding:1.25rem; margin-bottom:1rem;">
            <div style="font-size:13px; color:#854F0B; font-weight:500;">Report file not found</div>
            <div style="font-size:12px; color:#BA7517; margin-top:4px;">
                Update the <code>REPORT_PATH</code> constant at the top of <code>app.py</code> to point to your PDF report file.
                <br>Current path: <code style="font-size:11px;">{REPORT_PATH}</code>
            </div>
        </div>
        """.format(REPORT_PATH=REPORT_PATH), unsafe_allow_html=True)

        # Show placeholder stats instead
        st.markdown("""
        <div class="metric-strip">
            <div class="metric-card"><div class="metric-label">Total customers</div><div class="metric-value">1,247</div><div class="metric-detail">Analysed in Section 1</div></div>
            <div class="metric-card"><div class="metric-label">Services covered</div><div class="metric-value">25</div><div class="metric-detail">Across 5 categories</div></div>
            <div class="metric-card"><div class="metric-label">Avg. sentiment</div><div class="metric-value">+0.42</div><div class="metric-detail">Positive customer reviews</div></div>
        </div>
        """, unsafe_allow_html=True)

        report_sections = [
            ("📈", "Booking frequency analysis", "Distribution of booking rates by customer segment and service tier"),
            ("💬", "Sentiment & NPS trends", "TextBlob polarity scores, review analysis, and Net Promoter Score breakdown"),
            ("🛍", "Revenue segmentation", "Avg. spend per customer, tier migration patterns, and LTV estimates"),
            ("✦", "Recommendation model performance", "Precision, Recall, Hit-Rate@N, and confusion matrices"),
            ("📅", "Seasonal & temporal patterns", "Booking peaks, recency decay analysis, and churn indicators"),
        ]

        for icon, title, desc in report_sections:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:14px; padding:14px; background:#FFFFFF; border:1px solid #E8E6DF; border-radius:10px; margin-bottom:8px;">
                <span style="font-size:20px;">{icon}</span>
                <div>
                    <div style="font-size:14px; font-weight:500; color:#1a1a18;">{title}</div>
                    <div style="font-size:12px; color:#888780; margin-top:2px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    #  Download report button 
    r_col1, r_col2 = st.columns([1, 2])
    with r_col1:

        if st.button("📥  Download report", use_container_width=True):
            try:
                with open(REPORT_PATH, "rb") as f:
                    report_pdf = f.read()

                st.download_button(
                    label="Click to download",
                    data=report_pdf,
                    file_name="blyss_customer_analysis_report.pdf",
                    mime="application/pdf",
                )
            except FileNotFoundError:
                st.error(f"Report file not found at `{REPORT_PATH}`. Update the `REPORT_PATH` constant in app.py.")

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)


