import os
import json
import secrets
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set page title and styling
st.set_page_config(page_title="Legal Guidance Platform", page_icon="⚖️")
st.title("⚖️ Legal Guidance AI Platform")

# 1) Load Knowledge Base
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_DIR, "legal_kb.json")

if not os.path.exists(KB_PATH):
    st.error("Missing 'legal_kb.json' file in the directory.")
    st.stop()

@st.cache_resource
def initialize_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    
    # Initialize FAISS and Embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    scenarios = [item["scenario"] for item in kb_data]
    embeddings = embed_model.encode(scenarios, convert_to_numpy=True).astype("float32")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return kb_data, embed_model, index

kb, embed_model, index = initialize_knowledge_base()

# 2) Setup Groq Client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.warning("Please configure your GROQ_API_KEY secret in the settings.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# 3) Main UI Experience
user_query = st.text_area("Describe your legal legal situation/dispute in detail:", 
                          placeholder="e.g., Someone withdrew money from my bank account via a fake UPI link...")

if st.button("Analyze Assessment Case"):
    if user_query.strip():
        with st.spinner("Processing context matching and generating AI legal profile..."):
            # Compute similarity match
            query_vector = embed_model.encode([user_query], convert_to_numpy=True).astype("float32")
            D, I = index.search(query_vector, 1)
            matched_scenario = kb[I[0][0]]
            
            # Request LLM Guidance Evaluation
            prompt = f"""
            You are an expert legal assistant. Analyze this user issue: "{user_query}"
            The closely matching legal template is: {json.dumps(matched_scenario)}
            Provide immediate actionable next steps, applicable laws, and relevant handling authorities.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            
            ai_response = completion.choices[0].message.content
            
            # Render evaluation findings
            st.subheader(f"Matched Category: {matched_scenario['title']}")
            st.markdown(ai_response)
    else:
        st.error("Please insert a valid case scenario explanation before submitting.")
