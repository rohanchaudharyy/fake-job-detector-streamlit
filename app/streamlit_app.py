import streamlit as st
from dotenv import load_dotenv
from typing import Iterator
from pathlib import Path
import sys
import os
import re
import time

import ollama

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.predict_and_explain import HybridDetector

load_dotenv()
st.set_page_config(page_title="JobGuard · Is This Job Real?", page_icon="🛡️", layout="centered")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 720px; padding-top: 1.5rem; }

/* ── Brand ── */
.brand { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.brand-name { font-family:'Syne',sans-serif; font-weight:800; font-size:1.55rem; color:#0f172a; letter-spacing:-0.5px; }
.brand-tag  { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; border-radius:99px; font-size:0.65rem; font-weight:600; padding:3px 9px; letter-spacing:.4px; text-transform:uppercase; }

/* ── Verdict card ── */
.verdict-card { border-radius:16px; padding:1.4rem 1.5rem 1.1rem; margin:0.8rem 0 1rem; }
.verdict-fake { background:#fff1f2; border:2px solid #fca5a5; }
.verdict-real { background:#f0fdf4; border:2px solid #86efac; }
.verdict-uncertain { background:#fffbeb; border:2px solid #fcd34d; }

.verdict-emoji { font-size:2.2rem; line-height:1; }
.verdict-headline { font-family:'Syne',sans-serif; font-weight:800; font-size:1.3rem; margin:4px 0 2px; }
.fake-color { color:#be123c; }
.real-color { color:#15803d; }
.warn-color { color:#92400e; }
.verdict-sub  { font-size:0.875rem; color:#4b5563; margin-bottom:0.9rem; line-height:1.6; }

/* ── Confidence bar ── */
.conf-label { font-size:0.75rem; font-weight:500; color:#6b7280; margin-bottom:4px; }
.conf-track { background:#e5e7eb; border-radius:99px; height:10px; overflow:hidden; width:100%; }
.conf-fill-fake { background:linear-gradient(90deg,#f87171,#dc2626); height:100%; border-radius:99px; }
.conf-fill-real { background:linear-gradient(90deg,#4ade80,#16a34a); height:100%; border-radius:99px; }
.conf-fill-warn { background:linear-gradient(90deg,#fbbf24,#d97706); height:100%; border-radius:99px; }

/* ── Sections inside card ── */
.section-title { font-size:0.7rem; font-weight:600; letter-spacing:.6px; text-transform:uppercase; color:#9ca3af; margin:1rem 0 6px; }
.pill-list { display:flex; flex-wrap:wrap; gap:6px; margin:0; padding:0; }
.pill-red  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; font-size:0.78rem; border-radius:99px; padding:3px 10px; }
.pill-green{ background:#dcfce7; color:#166534; border:1px solid #86efac; font-size:0.78rem; border-radius:99px; padding:3px 10px; }
.pill-gray { background:#f3f4f6; color:#374151; border:1px solid #d1d5db; font-size:0.78rem; border-radius:99px; padding:3px 10px; }

/* ── Plain-English box ── */
.plain-box { background:white; border:1px solid #e5e7eb; border-radius:10px; padding:0.9rem 1rem; font-size:0.875rem; color:#1e293b; line-height:1.7; margin-top:0.5rem; }

/* ── Tips box ── */
.tips-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:0.85rem 1rem; font-size:0.82rem; color:#1e40af; line-height:1.65; margin-top:0.7rem; }
.tips-title { font-weight:600; margin-bottom:4px; }

/* ── Setup error ── */
.setup-box { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:1rem; font-size:0.85rem; color:#9a3412; }

/* ── Typing dots (ChatGPT style) ── */
.typing-dots { display:inline-flex; align-items:center; gap:5px; padding:4px 2px; }
.typing-dots span {
    width:8px; height:8px; border-radius:50%; background:#94a3b8;
    display:inline-block;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay:0.2s; }
.typing-dots span:nth-child(3) { animation-delay:0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity:0.4; }
    40%           { transform: scale(1.1); opacity:1;   }
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_detector() -> HybridDetector:
    return HybridDetector()


@st.cache_resource
def get_chat_client() -> tuple[ollama.Client, str]:
    client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return client, model


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_analysis_request(text: str) -> bool:
    lowered = text.lower().strip()
    if len(lowered) >= 280:
        return True
    triggers = ["job description","job posting","is this fake","real or fake",
                "analyze this","analyse this","check this","fraudulent","scam",
                "company profile","salary range","hiring","vacancy","position","role"]
    return any(p in lowered for p in triggers)


def generate_smalltalk_reply(user_text: str) -> str:
    import random
    lowered = user_text.strip().lower()

    # ── Greetings ──
    if re.search(r"^(hi+|hey+|hello|hiya|howdy|sup|what'?s up)[!?.]*$", lowered):
        options = [
            "Hey hey! 👋 So glad you're here. Got a job posting you're not sure about? Drop it in and I'll give you the honest truth!",
            "Hiii! 😄 Welcome to JobGuard — your personal scam detector. Paste any job listing and I'll break it down for you in plain language.",
            "Hey there! 👋 I'm JobGuard, and I'm basically that one friend who can spot a dodgy job listing from a mile away. Go ahead — paste one and let's take a look! 🔍",
        ]
        return random.choice(options)

    # ── How are you ──
    if re.search(r"how are you|how('?re| are) (you|u)|you good", lowered):
        options = [
            "I'm doing awesome, thanks for asking! 🙌 Ready and caffeinated (metaphorically) to scan job postings for you. What have you got?",
            "Feeling great! 😊 Honestly I love what I do — helping people dodge job scams is pretty rewarding. Got one for me to look at?",
            "Living my best life scanning job postings! 😄 How about you? If you've got a listing you're unsure about, I'm all yours.",
        ]
        return random.choice(options)

    # ── Bored / testing ──
    if re.search(r"bored|just testing|test|trying (you|this) out", lowered):
        return "Ha, fair enough — gotta kick the tyres! 😄 Go ahead and paste a job posting, real or made-up, and I'll show you what I can do. I promise I won't disappoint!"

    # ── Compliments ──
    if re.search(r"you('?re| are) (great|amazing|cool|awesome|good|helpful|nice)", lowered):
        return "Aww, you're too kind! 🥹 That genuinely made my day. Now let's put these skills to work — paste a job posting and I'll analyse it for you! 💪"

    # ── Confused / don't know what to do ──
    if re.search(r"(what|how) (do i|should i|can i)|not sure|confused|help|what now", lowered):
        return (
            "No worries at all — it's super simple! 😊\n\n"
            "Just **copy and paste the full job posting** (the one you found on LinkedIn, Indeed, WhatsApp, wherever) into this chat, and hit enter. "
            "I'll instantly tell you if it looks real or fake, and explain exactly why in plain language. Give it a go!"
        )

    # ── Worry / anxiety about job hunting ──
    if re.search(r"scared|worried|nervous|anxious|stress|overwhelm|job hunt(ing)?|applying", lowered):
        return (
            "Ugh, job hunting can be SO stressful — I totally get it. 😔 And scam postings make it even harder. "
            "That's exactly why I'm here! Paste any posting you're unsure about and I'll check it for you. "
            "You focus on the real opportunities — I'll filter out the rubbish. 💪"
        )

    # ── Gratitude ──
    if re.search(r"thank(s| you)|ty|cheers|appreciate", lowered):
        options = [
            "Aw, you're so welcome! 🥰 That's what I'm here for. Stay sharp out there — and if you ever spot another sketchy listing, you know where to find me!",
            "Anytime! 🙌 Seriously, protecting job seekers from scams is my whole thing. Good luck with your job search — you've got this! 💪",
            "Happy to help! 😊 Share another posting any time — I'm always here. And good luck out there, you deserve a great job! 🌟",
        ]
        return random.choice(options)

    # ── Goodbye ──
    if re.search(r"bye|goodbye|see ya|later|gotta go|talk (to you |)later", lowered):
        return "Take care! 👋 Good luck with the job hunt — and remember, if a posting ever feels off, trust your gut and come back here. You've got this! 🌟"

    # ── What can you do ──
    if re.search(r"what (can|do) you (do|help)|your (purpose|job|function)|capabilities", lowered):
        return (
            "Great question! 🔍 Here's what I do:\n\n"
            "- You paste a job posting\n"
            "- I scan it for red flags (vague company info, suspicious salary claims, requests for upfront payment, etc.)\n"
            "- I give you a **plain-English verdict**: real, fake, or suspicious\n"
            "- And I tell you *exactly* what to watch out for\n\n"
            "Think of me as your savvy friend who's seen every job scam trick in the book. 😄 Go ahead, paste one!"
        )

    # ── LLM fallback with a warmer system prompt ──
    try:
        client, model = get_chat_client()
        resp = client.chat(model=model, messages=[
            {"role": "system", "content": (
                "You are JobGuard — a warm, witty, encouraging assistant inside a fake-job detection app. "
                "Your personality: friendly, casual, like a supportive older sibling who works in tech. "
                "Use contractions, light humour, and the occasional emoji. "
                "Keep replies to 2-3 sentences. "
                "Always end by gently inviting the user to paste a job posting for analysis."
            )},
            {"role": "user", "content": user_text},
        ])
        text = resp["message"]["content"].strip()
        if text:
            return text
    except Exception:
        pass

    # ── Final fallback ──
    fallbacks = [
        "I'm all ears! 😊 If you've got a job posting you want me to check, just paste it and I'm on it.",
        "Love the chat! 😄 Whenever you're ready, drop in a job listing and I'll give you the full scoop on whether it's legit.",
        "You know what's fun? Catching scam job postings before they waste your time. 😤 Paste one and let's go!",
    ]
    return random.choice(fallbacks)


def _stream_chunks(text: str) -> Iterator[str]:
    for i, word in enumerate(text.split(" ")):
        yield word + (" " if i < len(text.split()) - 1 else "")


# ── Friendly result renderer ──────────────────────────────────────────────────
def render_friendly_result(result: dict) -> None:
    """
    Renders a warm, plain-English verdict card — no raw probabilities,
    no ML jargon. Designed for fresh grads and general public.
    """
    is_fake   = result["prediction"] == "fake"
    fake_prob = result["confidence_fake"]          # 0–1 float from your model
    real_prob = 1.0 - fake_prob

    # ── Decide tone ──
    if is_fake and fake_prob >= 0.75:
        tier, card_cls, color_cls, fill_cls = "high", "verdict-fake", "fake-color", "conf-fill-fake"
        emoji, headline = "🚨", "This looks like a SCAM"
        subtitle = (
            "Our analysis found several classic warning signs used by fake job postings. "
            "We strongly recommend <strong>not applying</strong> and not sharing any personal details."
        )
        conf_pct = int(fake_prob * 100)
        conf_label = f"We're {conf_pct}% confident this is fake"
    elif is_fake:
        tier, card_cls, color_cls, fill_cls = "med", "verdict-fake", "fake-color", "conf-fill-fake"
        emoji, headline = "⚠️", "Something feels off here"
        subtitle = (
            "This posting has some red flags. It might be real, but we'd be cautious. "
            "Do a bit of extra research before you invest your time applying."
        )
        conf_pct = int(fake_prob * 100)
        conf_label = f"We're {conf_pct}% confident this is suspicious"
    elif real_prob >= 0.75:
        tier, card_cls, color_cls, fill_cls = "real", "verdict-real", "real-color", "conf-fill-real"
        emoji, headline = "✅", "This looks LEGIT"
        subtitle = (
            "Good news! We didn't find major red flags. This posting appears to be from a real employer. "
            "Still, always verify the company on LinkedIn or their official website before sharing your ID or bank details."
        )
        conf_pct = int(real_prob * 100)
        conf_label = f"We're {conf_pct}% confident this is real"
    else:
        tier, card_cls, color_cls, fill_cls = "unc", "verdict-uncertain", "warn-color", "conf-fill-warn"
        emoji, headline = "🔍", "We're not sure — tread carefully"
        subtitle = (
            "The signals are mixed — some things look okay, others raised a small flag. "
            "We'd suggest doing extra research on the company before applying."
        )
        conf_pct = 50
        conf_label = "Mixed signals — can't call it confidently"

    # ── Build red / green pill lists ──
    red_flags    = result.get("rule_findings") or []
    good_signals = []  # your model may expose these — add if available

    # Human-readable rewrites of common raw findings
    FRIENDLY_FLAGS = {
        "no_company_profile":      "No clear info about the company",
        "vague_location":          "Location is vague or missing",
        "upfront_payment":         "Asks you to pay to get the job",
        "personal_info_requested": "Asks for sensitive personal info too early",
        "generic_email":           "Uses a personal email (Gmail/Yahoo) not a company one",
        "unrealistic_salary":      "Promises unusually high pay",
        "no_experience_required":  "Suspiciously no skills or experience needed",
        "urgency_language":        "Creates artificial urgency ('apply now!')",
        "typos_poor_grammar":      "Poor grammar and lots of typos",
        "missing_requirements":    "No real job requirements listed",
    }

    def humanise(flag: str) -> str:
        for key, label in FRIENDLY_FLAGS.items():
            if key in flag.lower() or flag.lower() in key:
                return label
        # Fall back: clean up snake_case / technical text
        cleaned = re.sub(r"[_\-]", " ", flag)
        cleaned = re.sub(r"\b(score|index|prob|val|std|avg|mean)\b", "", cleaned, flags=re.I)
        return cleaned.strip().capitalize()

    friendly_flags = [humanise(f) for f in red_flags if f]

    # ── Render the card ──
    pills_red   = "".join(f'<span class="pill-red">⚠ {f}</span>' for f in friendly_flags) or '<span class="pill-gray">No obvious red flags found</span>'
    pills_green = "".join(f'<span class="pill-green">✓ {s}</span>' for s in good_signals) or '<span class="pill-gray">Not enough positive signals detected</span>'

    st.markdown(f"""
    <div class="verdict-card {card_cls}">
      <div class="verdict-emoji">{emoji}</div>
      <div class="verdict-headline {color_cls}">{headline}</div>
      <div class="verdict-sub">{subtitle}</div>

      <div class="conf-label">{conf_label}</div>
      <div class="conf-track">
        <div class="conf-fill-{fill_cls.split('-')[-1]}" style="width:{conf_pct}%"></div>
      </div>

      <div class="section-title" style="margin-top:1.1rem">What raised flags 🚩</div>
      <div class="pill-list">{pills_red}</div>

    </div>
    """, unsafe_allow_html=True)

    # ── Plain-English explanation (generated, never raw model text) ──
    flags = result.get("rule_findings") or []
    missing_company  = any("company" in f.lower() for f in flags)
    missing_salary   = any("salary" in f.lower() for f in flags)

    if is_fake and fake_prob >= 0.75:
        plain = (
            f"We're quite confident this is a fake posting ({int(fake_prob*100)}% probability). "
            "It shows multiple patterns that scam jobs commonly use. "
            "We strongly recommend not applying and not sharing any personal details."
        )
    elif is_fake:
        parts = []
        if missing_company: parts.append("the company isn't clearly identified")
        if missing_salary:  parts.append("salary details are vague or missing")
        issues = " and ".join(parts) if parts else "some suspicious patterns were detected"
        plain = (
            f"This posting has a {int(fake_prob*100)}% chance of being fake. "
            f"Specifically, {issues}. It might still be real, but we'd do some extra research before applying."
        )
    elif real_prob >= 0.75:
        parts = []
        if missing_company: parts.append("double-check the company on LinkedIn")
        if missing_salary:  parts.append("confirm the salary range before applying")
        tip = " and ".join(parts) if parts else "always verify the company on LinkedIn before sharing personal info"
        plain = (
            f"This looks like a genuine job posting ({int(real_prob*100)}% confidence). "
            f"We didn't find major red flags — just remember to {tip}. Good luck! 🌟"
        )
    else:
        plain = (
            "The signals are mixed on this one. Some things look fine, but a couple of details "
            "couldn't be fully verified. Take a few minutes to look up the company independently before applying."
        )

    st.markdown(f"""
    <div class="plain-box">{plain}</div>
    """, unsafe_allow_html=True)

    # ── Safety tips ──
    if is_fake or tier == "unc":
        st.markdown("""
        <div class="tips-box">
          <div class="tips-title">🔒 What to do next</div>
          <ul style="margin:4px 0 0; padding-left:18px;">
            <li>Search the company name on LinkedIn or Google before applying.</li>
            <li>Never pay any fee to get a job — real employers don't ask for this.</li>
            <li>Don't share your ID, bank details, or tax number until you're hired officially.</li>
            <li>If it feels too good to be true, it usually is.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tips-box">
          <div class="tips-title">✅ A few healthy habits before you apply</div>
          <ul style="margin:4px 0 0; padding-left:18px;">
            <li>Verify the company on their official website and LinkedIn.</li>
            <li>Look up reviews on Glassdoor or Indeed.</li>
            <li>Never share sensitive personal info until you have a written offer.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Collapsible technical details (for those who want them) ──
    with st.expander("🔬 Show technical details (for the curious)"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model verdict", result["prediction"].upper())
            st.metric("Fake probability", f"{result['confidence_fake']:.1%}")
            st.metric("Real probability", f"{1 - result['confidence_fake']:.1%}")
            st.caption("Decision rule: fake if probability ≥ 50%")
        with col2:
            st.write("**Raw rule-based findings**")
            for f in (result.get("rule_findings") or ["None triggered"]):
                st.write(f"- {f}")

        st.write("**Similar cases from training data**")
        examples = result.get("similar_examples") or []
        if not examples:
            st.caption("No similar examples retrieved.")
        else:
            for i, ex in enumerate(examples, 1):
                label = "✅ Real" if ex["label"] == 0 else "🚨 Fake"
                match_pct = int(ex["similarity"] * 100)
                # Show only first 120 chars of text, cleanly
                preview = ex["text"].strip().replace("\n", " ")[:120] + "…"
                st.write(f"**{label}** — {match_pct}% similar")
                st.caption(f"_{preview}_")


def render_setup_error(exc: Exception) -> None:
    st.markdown(f"""
    <div class="setup-box">
      <strong>⚙️ Setup needed before analysis can run</strong><br><br>
      The model files haven't been built yet. Run these commands once in your terminal, then restart the app:
    </div>
    """, unsafe_allow_html=True)
    st.code("\n".join([
        "python -m src.data.preprocess",
        "python -m src.models.train_baseline",
        "python -m src.retrieval.build_faiss_index",
    ]))
    st.caption(f"Error detail: {exc}")


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    # Brand header
    st.markdown("""
    <div class="brand">
      <span style="font-size:1.6rem">🛡️</span>
      <span class="brand-name">JobGuard</span>
      <span class="brand-tag">AI-powered</span>
    </div>
    <p style="color:#64748b; font-size:0.875rem; margin-bottom:1rem;">
      Paste any job posting below — I'll tell you in plain English whether it looks real or fake, and exactly why.
    </p>
    <hr style="border:none; border-top:1px solid #e2e8f0; margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hey! 👋 I'm **JobGuard** — your personal fake job detector.\n\n"
                "Job scams are everywhere right now, and they can be really hard to spot — "
                "that's where I come in! 🔍\n\n"
                "**Just paste any job posting** you've found (LinkedIn, WhatsApp, Indeed, anywhere) "
                "and I'll tell you in plain, simple English whether it looks real or fake, and *why*. "
                "No tech jargon, I promise!\n\n"
                "Go ahead, drop one in — I'm ready. 💪"
            ),
        }]
    if "stream_next_assistant" not in st.session_state:
        st.session_state.stream_next_assistant = False

    # Render chat history
    last_idx = len(st.session_state.messages) - 1
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            should_stream = (
                st.session_state.stream_next_assistant
                and idx == last_idx
                and message["role"] == "assistant"
            )
            if should_stream:
                st.write_stream(_stream_chunks(message["content"]))
                st.session_state.stream_next_assistant = False
            else:
                st.write(message["content"])

            # Render result card if this message has analysis
            if "result" in message:
                render_friendly_result(message["result"])
            if "error" in message:
                render_setup_error(Exception(message["error"]))

    # Chat input
    user_input = st.chat_input("Type a message or paste a job posting here…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if not is_analysis_request(user_input):
            reply = generate_smalltalk_reply(user_input)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                # Show dots word by word so Streamlit renders it
                for dots in ["●", "● ●", "● ● ●"]:
                    placeholder.markdown(dots)
                    time.sleep(0.4)
                placeholder.empty()
                st.write_stream(_stream_chunks(reply))
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.stream_next_assistant = False
        else:
            try:
                detector = get_detector()
            except Exception as exc:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I can't run the analysis yet — the model files need to be set up first.",
                    "error": str(exc),
                })
                st.session_state.stream_next_assistant = True
            else:
                with st.chat_message("assistant"):
                    typing2 = st.empty()
                    typing2.markdown("⬤  ⬤  ⬤")
                    result = detector.predict(job_text=user_input)
                    typing2.empty()
                import random
                if result["prediction"] == "fake":
                    intros = [
                        "Okay, I've scanned this one — and yeah, we need to talk. 😬 Here's what I found:",
                        "Hmm, this one's raising some alarms for me. 🚨 Let me break it down:",
                        "Glad you checked this before applying! Here's the honest scoop 👇",
                    ]
                else:
                    intros = [
                        "Good news — this one's looking pretty solid! Here's the full picture 👇",
                        "Alright, I've had a good look at this. Here's what I think 🔍",
                        "Not bad! Here's my read on this posting 👇",
                    ]
                intro = random.choice(intros)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": intro,
                    "result": result,
                })
                st.session_state.stream_next_assistant = True

        st.rerun()


if __name__ == "__main__":
    main()
