import pandas as pd
import re
from collections import Counter
from langdetect import detect

def load_posts(path="data/posts.csv"):
    df = pd.read_csv(path)
    
    # Handle missing 'text' column gracefully
    if "text" not in df.columns:
        print("⚠️ No 'text' column found. Creating placeholder text for testing.")
        df["text"] = "Sample AI-generated text for testing."
    
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    return df

def estimate_length_category(text):
    tokens = len(text.split())
    if tokens < 25:
        return "short"
    elif tokens < 80:
        return "medium"
    else:
        return "long"

def detect_language_safe(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"

def simple_keywords(text, top_n=5):
    # naive keyword extraction: split and choose frequent words excluding stopwords
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in text.split() if len(w) > 3]
    counts = Counter(words)
    common = [w for w, _ in counts.most_common(top_n)]
    return ", ".join(common)

def extract_metadata(df):
    df = df.copy()
    if "text" not in df.columns:
        print("⚠️ Adding placeholder 'text' column for metadata extraction.")
        df["text"] = "This is a sample text about AI."
    df["length_category"] = df["text"].apply(estimate_length_category)
    df["language"] = df["text"].apply(detect_language_safe)
    df["keywords"] = df["text"].apply(simple_keywords)
    return df
