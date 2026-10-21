# 💼 LinkedIn Post Generator

Turn a rough idea into scroll-stopping LinkedIn posts in seconds. Runs free on Groq or Ollama, or on Claude for top quality.

Built with **Python**, **Streamlit**, and the **Groq / Anthropic / Ollama** APIs.

## Features

- **7 post types**: story, motivational, technical insight, project announcement, tips list, career update, hot take
- **Tone, audience and length controls**
- **Up to 3 variations** per request, each with a different angle and hook
- **Hook preview**: see the first lines readers see before "see more"
- **One-click rewrites**: shorter, better hook, more personal
- **Character counter** against LinkedIn's 3,000 limit
- **Learns your voice**: upload your past posts (CSV with a `text` column)
- **Free to run**: Groq free tier, Ollama locally, or Claude if you want top quality
- **Editable results** and `.txt` download

## Quick start

```bash
git clone https://github.com/tamia112/ai-gen2.git
cd ai-gen2
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your key (see below)
streamlit run main.py
```

### AI engine (the key stays on the server, visitors never see it)
Visitors just type their idea. You choose the engine once, in `.env` locally or in Streamlit **Secrets** when deployed:

| Engine | Cost | Setup |
|--------|------|-------|
| **Groq** (default if `GROQ_API_KEY` is set) | Free tier, rate-limited | Key from https://console.groq.com/keys, no card needed |
| **Claude** (if `ANTHROPIC_API_KEY` is set) | Pay-as-you-go, best quality | Key from https://console.anthropic.com |
| **Ollama** (`LLM_BACKEND=ollama`) | Free, local only | `ollama pull llama3.2` and `ollama serve` |

Built-in limits (`SESSION_LIMIT`, `DAILY_LIMIT`) stop visitors from using up your free quota or credit.

## Project structure

```
├── main.py              Streamlit interface
├── post_generator.py    Prompts, post types, parsing, rewrites
├── llm_helper.py        Claude and Ollama backends
├── sample_posts.csv     Example format for "your past posts"
├── requirements.txt
└── .env.example
```

## How it works

1. Your idea, post type, tone, audience and length become a structured prompt.
2. A system prompt carries LinkedIn writing rules: a strong hook, short paragraphs, plain text, no invented facts, a closing question.
3. The model returns several posts separated by markers, which the app splits into editable cards.
4. Rewrite buttons send a single post back with one focused instruction.

## What this project demonstrates

- Prompt engineering with a system prompt, constraints and few-shot style examples
- Multiple LLM backends behind one interface, with readable error handling
- Streamlit state management (editable results, in-place rewrites)

## Roadmap

- Post scheduling and a LinkedIn-style preview
- Carousel and poll ideas
- Saved post history

## Author

**Tamia Mwandu** · [@tamia112](https://github.com/tamia112)