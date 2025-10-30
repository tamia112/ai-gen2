import streamlit as st
import pandas as pd
from preprocess import load_posts, extract_metadata
from post_generator import generate_posts
import os
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(page_title="AI LinkedIn Post Generator", layout="wide")


st.title("💼 LinkedIn Post Generator — GenAI (Llama 3.2 Powered)")
st.markdown("""
Welcome to your **AI-powered LinkedIn Post Generator**!  
Upload your dataset of LinkedIn posts or use the default one, then generate engaging new posts using **Ollama (offline)** or **OpenAI (online)**.
""")

# -------------------------------
# Sidebar Settings
# -------------------------------
st.sidebar.header("⚙️ Model Settings")

# Backend: choose between Ollama or OpenAI
backend = st.sidebar.selectbox("LLM Backend", ["ollama", "openai"], index=0)
model_name = st.sidebar.text_input("Model Name", value="llama3.2")

# Default topic input
default_topic = st.sidebar.text_input("Default Topic", value="startup innovation")

# -------------------------------
# CSV Upload / Fallback Dataset
# -------------------------------
st.header("📂 Upload or Use Default Dataset")

uploaded = st.file_uploader("Upload your LinkedIn posts CSV", type=["csv"])
default_path = "Advertising_2.csv"  # your uploaded CSV file

# Load the data
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded CSV loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to read uploaded file: {e}")
        st.stop()
else:
    try:
        df = load_posts(default_path)
        st.info("ℹ️ Using local dataset: Advertising_2.csv")
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        st.stop()

# -------------------------------
# Process Metadata
# -------------------------------
try:
    df_meta = extract_metadata(df)
    st.subheader("📊 Sample Processed Data")
    st.dataframe(df_meta.head(5))
except Exception as e:
    st.error(f"⚠️ Metadata extraction failed: {e}")
    st.stop()


languages = sorted(df_meta["language"].dropna().unique().tolist())
lengths = sorted(df_meta["length_category"].dropna().unique().tolist())

language = st.selectbox("Select Language", options=(["en"] + languages), index=0)
length = st.selectbox("Select Post Length", options=(["short", "medium", "long"] + lengths), index=1)


topic = st.text_input("🧠 Topic to generate posts for", value=default_topic)

if st.button("🚀 Generate Posts"):
    with st.spinner("✨ Crafting LinkedIn posts using AI..."):
        try:
            result = generate_posts(df_meta, topic=topic, language=language, length=length, backend=backend)
            st.subheader("📝 Generated Posts")
            st.code(result)
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")


st.markdown("---")
st.markdown("💡 *Developed by Tamia Mwandu | Powered by Llama 3.2 & Streamlit*")

