"""Prompt building, generation and refinement of LinkedIn posts."""
import re
from llm_helper import call_llm

POST_TYPES = {
    "Story / personal lesson": "Open with a specific moment, then the lesson. Make it personal and honest.",
    "Motivational": "Be grounded, not preachy. One clear idea the reader can act on today.",
    "Technical insight": "Explain one concrete thing you built or learned. Include a specific detail or number.",
    "Project announcement": "Say what you built, why it matters, and what you learned. Invite feedback.",
    "Tips / how-to list": "Give 3 to 5 practical tips, each on its own short line.",
    "Career update": "Share the news simply, thank people specifically, and look forward.",
    "Opinion / hot take": "Take a clear stance in the first line, back it with one reason, invite disagreement.",
}
TONES = ["Professional", "Friendly", "Inspiring", "Bold", "Humble", "Witty"]
LENGTHS = {"Short (60-100 words)": "60 to 100 words",
           "Medium (120-180 words)": "120 to 180 words",
           "Long (200-280 words)": "200 to 280 words"}

SYSTEM = """You are an expert LinkedIn ghostwriter. Write posts that sound like a real person, not an ad.
Rules:
- The first line is a hook that earns the click on "see more": specific, curious or surprising. Never start with "I'm thrilled", "Excited to announce" or "In today's fast-paced world".
- Short paragraphs of 1 to 2 lines, with blank lines between them. One idea per post.
- Plain text only: no markdown, no bold, no headings, no asterisks.
- Be concrete. Never invent facts, numbers, employers or achievements: use only what the user gave you, and use [brackets] for details they should fill in.
- End with a genuine question or a simple call to action.
- Never exceed 2,800 characters."""


def _build_prompt(topic, post_type, tone, audience, length, n, hashtags, emojis, examples):
    parts = [
        f"Write {n} different LinkedIn post(s) about:\n{topic.strip()}\n",
        f"Post type: {post_type}. {POST_TYPES[post_type]}",
        f"Tone: {tone}.",
        f"Length: {LENGTHS[length]} each.",
    ]
    if audience.strip():
        parts.append(f"Target audience: {audience.strip()}.")
    parts.append("Emojis: " + ("use 1 to 3, sparingly." if emojis else "do not use any."))
    parts.append("Hashtags: " + ("end with 3 to 5 relevant hashtags." if hashtags else "do not add hashtags."))
    if examples:
        parts.append("Match the voice of the author's past posts:\n" + examples)
    parts.append(
        "Make each post take a clearly different angle and a different hook.\n"
        "Output format: start each post with a line exactly like === POST 1 === (then 2, 3...). "
        "Output nothing else: no intro, no commentary."
    )
    return "\n".join(parts)


def parse_posts(raw):
    chunks = re.split(r"=+\s*POST\s*\d+\s*=+", raw, flags=re.I)
    posts = [c.strip().strip('"').strip() for c in chunks if c.strip()]
    return posts or [raw.strip()]


def generate_posts(topic, post_type, tone, audience="", length="Medium (120-180 words)",
                   n=3, hashtags=True, emojis=False, examples="", **llm):
    prompt = _build_prompt(topic, post_type, tone, audience, length, n, hashtags, emojis, examples)
    return parse_posts(call_llm(prompt, SYSTEM, **llm))[:n]


def refine_post(post, instruction, **llm):
    prompt = (f"Here is a LinkedIn post:\n\n{post}\n\nRewrite it with this change: {instruction}.\n"
              "Keep the same facts and voice. Output only the rewritten post.")
    return call_llm(prompt, SYSTEM, **llm).strip().strip('"')


def examples_from_posts(df, k=3):
    """Use the user's own past posts (a 'text' column) as style examples."""
    if df is None or "text" not in df.columns:
        return ""
    texts = df["text"].dropna().astype(str).tail(k).tolist()
    return "\n\n".join(f"Example {i}:\n{t}" for i, t in enumerate(texts, 1))