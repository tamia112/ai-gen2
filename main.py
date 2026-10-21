import html
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_helper import LLMError, get_secret
from post_generator import (POST_TYPES, TONES, LENGTHS, generate_posts,
                            refine_post, examples_from_posts)

LIMIT = 3000
HOOK_CHARS = 210  # roughly what shows before "...see more"

st.set_page_config(page_title="LinkedIn Post Generator", page_icon="💼", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)


def step(n, title):
    st.markdown(f'<div class="step"><span class="num">{n}</span>{title}</div>', unsafe_allow_html=True)


def preview(text, name, headline):
    hook, rest = html.escape(text[:HOOK_CHARS]), html.escape(text[HOOK_CHARS:])
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "?"
    return (f'<div class="li-card"><div class="li-head"><div class="li-avatar">{html.escape(initials)}</div>'
            f'<div><div class="li-name">{html.escape(name)}</div><div class="li-sub">{html.escape(headline)} · now · 🌐</div></div></div>'
            f'<div class="li-body">{hook}<span class="rest">{rest}</span></div></div>')


# ---------- Server-side config and usage limits (keys never appear in the UI) ----------
llm = dict(temperature=0.8)   # backend and model come from LLM_BACKEND / LLM_MODEL or the available key
SESSION_MAX = int(get_secret("SESSION_LIMIT", 15))
DAILY_MAX = int(get_secret("DAILY_LIMIT", 200))
LIMIT_MSG = "You've reached the usage limit for now. Please come back later."


@st.cache_resource
def _daily():
    return {"day": None, "n": 0}


def allow():
    """Protects the free tier / your credit from being used up."""
    d, today = _daily(), date.today().isoformat()
    if d["day"] != today:
        d.update(day=today, n=0)
    if st.session_state.get("used", 0) >= SESSION_MAX or d["n"] >= DAILY_MAX:
        return False
    st.session_state["used"] = st.session_state.get("used", 0) + 1
    d["n"] += 1
    return True


# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI WRITING ASSISTANT</div>
  <h1>LinkedIn posts that sound like you.</h1>
  <p>Drop in a rough idea. Get three polished posts with strong hooks, ready to edit and publish.</p>
  <div class="chips"><span class="chip">✍️ 7 post types</span><span class="chip">🎯 Hook preview</span>
  <span class="chip">⚡ One-click rewrites</span><span class="chip">⚡ Ready in seconds</span></div>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
with st.container(key="card_idea"):
    step(1, "Your idea")
    topic = st.text_area("What do you want to post about?", height=120, label_visibility="collapsed",
                         placeholder="e.g. I built my first full-stack project and nearly gave up on the database part")
    audience = st.text_input("Who is it for? (optional)", placeholder="e.g. recruiters, junior developers")

with st.container(key="card_style"):
    step(2, "Style")
    c1, c2, c3 = st.columns(3)
    post_type = c1.selectbox("Post type", list(POST_TYPES))
    tone = c2.selectbox("Tone", TONES)
    length = c3.selectbox("Length", list(LENGTHS), index=1)
    o1, o2, o3 = st.columns(3)
    n = o1.slider("Variations", 1, 3, 3)
    hashtags = o2.toggle("Add hashtags", value=True)
    emojis = o3.toggle("Use emojis", value=False)
    with st.expander("More options"):
        past = st.file_uploader("Match my writing style: upload past posts (CSV with a 'text' column)", type="csv")
        p1, p2 = st.columns(2)
        name = p1.text_input("Name shown in preview", value="Tamia Mwandu")
        headline = p2.text_input("Headline shown in preview", value="Software developer")

go = st.button("🚀 Generate posts", type="primary", disabled=not topic.strip(), use_container_width=True)

if go and not allow():
    st.warning(LIMIT_MSG)
elif go:
    examples = ""
    if past is not None:
        try:
            examples = examples_from_posts(pd.read_csv(past))
        except Exception as e:
            st.warning(f"Couldn't read your CSV, continuing without it: {e}")
    try:
        with st.spinner("Writing your posts..."):
            posts = generate_posts(topic, post_type, tone, audience, length, n, hashtags, emojis, examples, **llm)
        for key in [k for k in st.session_state if k.startswith("post_")]:
            del st.session_state[key]
        for i, p in enumerate(posts):
            st.session_state[f"post_{i}"] = p
        st.session_state.update(count=len(posts))
        st.session_state.pop("error", None)
    except LLMError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")


def refine(i, instruction):
    if not allow():
        st.session_state["error"] = LIMIT_MSG
        return
    try:
        st.session_state[f"post_{i}"] = refine_post(st.session_state[f"post_{i}"], instruction, **llm)
        st.session_state.pop("error", None)
    except LLMError as e:
        st.session_state["error"] = str(e)


# ---------- Results ----------
if st.session_state.get("error"):
    st.error(st.session_state["error"])

count = st.session_state.get("count", 0)
st.markdown('<div class="section-title">Your posts</div>', unsafe_allow_html=True)
if not count:
    st.markdown('<div class="empty"><b>Your posts will appear here</b><br>'
                'Write your idea above and press Generate.</div>', unsafe_allow_html=True)
else:
    tabs = st.tabs([f"Option {i + 1}" for i in range(count)])
    for i, tab in enumerate(tabs):
        key = f"post_{i}"
        with tab:
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown('<div class="label">EDIT</div>', unsafe_allow_html=True)
                text = st.text_area("Post", key=key, height=430, label_visibility="collapsed")
                chars = len(text)
                pct = min(100, chars * 100 // LIMIT)
                warn = " ⚠️ too long" if chars > LIMIT else ""
                st.markdown(f'<div class="meter"><div style="width:{pct}%"></div></div>'
                            f'<div class="meter-text">{chars:,} / {LIMIT:,} characters{warn}</div>', unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                b1.button("✂️ Shorter", key=f"s{i}", on_click=refine, args=(i, "make it about 30% shorter"), use_container_width=True)
                b2.button("🎣 Better hook", key=f"h{i}", on_click=refine, args=(i, "write a stronger, more specific first line"), use_container_width=True)
                b3.button("💬 Personal", key=f"p{i}", on_click=refine, args=(i, "make it more personal and conversational"), use_container_width=True)
            with right:
                st.markdown('<div class="label">PREVIEW</div>', unsafe_allow_html=True)
                st.markdown(preview(text, name, headline), unsafe_allow_html=True)
                st.markdown('<div class="hint">Bright text is what readers see before “see more”.</div>', unsafe_allow_html=True)
                with st.expander("📋 Copy text"):
                    st.code(text, language=None, wrap_lines=True)
                st.download_button("⬇️ Download .txt", text, file_name=f"linkedin_post_{i + 1}.txt",
                                   key=f"d{i}", use_container_width=True)

st.markdown('<div class="hint" style="text-align:center;margin-top:3rem">Built by Tamia Mwandu · Streamlit + AI</div>',
            unsafe_allow_html=True)