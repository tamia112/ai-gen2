"""LLM backends. API keys live on the server (env / Streamlit secrets), never in the UI.

Backends: groq (free tier), claude (paid), ollama (free, local).
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",   # free tier; see console.groq.com/docs/models if this name changes
    "claude": "claude-haiku-5-5",        # paid, fast and cheap
    "ollama": "llama3.2",                # free, local
}
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


class LLMError(Exception):
    """Readable error that the UI can show directly."""


def get_secret(name, default=None):
    """Read from Streamlit secrets (when deployed) or environment / .env (locally)."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def default_backend():
    """LLM_BACKEND wins; otherwise use whichever key is configured (free Groq first)."""
    chosen = get_secret("LLM_BACKEND")
    if chosen:
        return chosen.lower()
    if get_secret("GROQ_API_KEY"):
        return "groq"
    if get_secret("ANTHROPIC_API_KEY"):
        return "claude"
    return "none"


GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]


def _groq_error(r):
    try:
        return r.json()["error"]["message"]
    except Exception:
        return r.text[:200]


def _groq_pick_model(key):
    """Ask Groq which models are active right now and pick the best text model."""
    r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    ids = [m["id"] for m in r.json().get("data", [])]
    for m in GROQ_PREFERRED:
        if m in ids:
            return m
    skip = ("whisper", "tts", "guard", "orpheus", "embed", "compound")
    chat = [i for i in ids if not any(x in i for x in skip)]
    return chat[0] if chat else None


def call_groq(prompt, system, model, max_tokens=1200, temperature=0.8):
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set on the server.")

    def post(m):
        return requests.post(
            f"{GROQ_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": m, "max_tokens": max_tokens, "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": prompt}]},
            timeout=90)

    try:
        r = post(model)
        # Model retired or renamed? Find a working one automatically and retry once.
        if r.status_code in (400, 404) and "model" in _groq_error(r).lower():
            alt = _groq_pick_model(key)
            if alt and alt != model:
                r = post(alt)
    except requests.RequestException as e:
        raise LLMError(f"Couldn't reach the AI service: {e}")
    if r.status_code == 429:
        raise LLMError("The free AI service is busy right now. Please try again in a minute.")
    if r.status_code in (401, 403):
        raise LLMError("The server's Groq API key was rejected. Create a new key at console.groq.com/keys.")
    if not r.ok:
        raise LLMError(f"Groq error {r.status_code}: {_groq_error(r)}")
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def call_claude(prompt, system, model, max_tokens=1200, temperature=0.8):
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set on the server.")
    try:
        import anthropic
    except ImportError:
        raise LLMError("The 'anthropic' package is missing. Run: pip install anthropic")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                     system=system, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    except anthropic.AuthenticationError:
        raise LLMError("Claude rejected the server's API key. Check it and its credit.")
    except anthropic.NotFoundError:
        raise LLMError(f"Model '{model}' was not found. Set LLM_MODEL to a valid name.")
    except anthropic.APIError as e:
        raise LLMError(f"Claude API error: {e}")


def call_ollama(prompt, system, model, max_tokens=1200, temperature=0.8):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "stream": False,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                  "options": {"temperature": temperature, "num_predict": max_tokens}},
            timeout=300)
    except requests.ConnectionError:
        raise LLMError("Can't reach Ollama. Start it with 'ollama serve' and run 'ollama pull llama3.2'.")
    if r.status_code == 404:
        raise LLMError(f"Ollama doesn't have '{model}'. Run: ollama pull {model}")
    if not r.ok:
        raise LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
    return r.json()["message"]["content"].strip()


def call_llm(prompt, system, backend=None, model=None, **kw):
    backend = backend or default_backend()
    if backend == "none":
        raise LLMError("The AI isn't set up yet. The app owner needs to add GROQ_API_KEY (free) or ANTHROPIC_API_KEY.")
    if backend not in DEFAULT_MODELS:
        raise LLMError(f"Unknown LLM_BACKEND '{backend}'. Use groq, claude or ollama.")
    model = model or get_secret("LLM_MODEL") or DEFAULT_MODELS[backend]
    fn = {"groq": call_groq, "claude": call_claude, "ollama": call_ollama}[backend]
    return fn(prompt, system, model, **kw)