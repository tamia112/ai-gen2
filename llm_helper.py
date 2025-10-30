import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

# Load your OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def call_llm_with_openai(prompt, max_tokens=250, temperature=0.7):
    """
    Sends a text prompt to OpenAI's ChatCompletion API and returns the generated response.
    """
    try:
        import openai
        openai.api_key = OPENAI_API_KEY

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # You can change to gpt-3.5-turbo if preferred
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Warning] OpenAI API call failed: {e}")
        return None


def call_llm_with_ollama(prompt, model="llama3.2"):
    """
    Runs a local Ollama model to generate text.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"[Error] Ollama call failed: {e}")
        return None


def call_llm(prompt, backend="openai", **kwargs):
    """
    Wrapper function to call LLM.
    Supports both 'openai' and 'ollama' backends.
    """
    if backend == "openai":
        result = call_llm_with_openai(prompt, **kwargs)
        if result is None:  # fallback to Ollama if OpenAI fails
            print("[Fallback] Using Ollama backend instead.")
            result = call_llm_with_ollama(prompt)
        return result

    elif backend == "ollama":
        return call_llm_with_ollama(prompt)

    else:
        raise ValueError("Unsupported backend. Use 'openai' or 'ollama'.")

