
# post_generator.py
from llm_helper import call_llm
from few_shot import build_few_shot_examples
import textwrap

PROMPT_TEMPLATE = """
You are a helpful assistant that writes LinkedIn posts in the same style as the examples.
Instructions:
- Use the chosen topic: {topic}
- Target language: {language}
- Target length: {length}
- Include a clear opening, a short body with at least 2 sentences, and a CTA (call-to-action).

Here are example posts to set the style:
{few_shot}

Now, generate 3 variations of new LinkedIn posts (numbered), each matching the length and language.
Be concise and engaging.
"""

def generate_posts(posts_df, topic, language="en", length="medium", backend="ollama"):
    few_shot = build_few_shot_examples(posts_df, chosen_topic=topic, language=language, length_category=length, n_examples=3)
    prompt = PROMPT_TEMPLATE.format(topic=topic, language=language, length=length, few_shot=few_shot)
    result = call_llm(prompt, backend=backend, max_tokens=400, temperature=0.75)
    # simple post-splitting — keep raw string
    return result

