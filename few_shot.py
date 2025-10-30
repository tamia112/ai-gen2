
# few_shot.py
from typing import List

def build_few_shot_examples(posts_df, chosen_topic=None, language=None, length_category=None, n_examples=3):
    df = posts_df
    if chosen_topic:
        # naive filter: check keywords column contains chosen topic token
        df = df[df["keywords"].str.contains(chosen_topic, na=False, case=False)]
    if language:
        df = df[df["language"] == language]
    if length_category:
        df = df[df["length_category"] == length_category]
    examples = df["text"].head(n_examples).tolist()
    # build prompt examples in a consistent format
    few_shot = ""
    for i, ex in enumerate(examples, 1):
        few_shot += f"Example {i}:\n{ex}\n\n"
    return few_shot

