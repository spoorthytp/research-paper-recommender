import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Research Paper Finder", page_icon="🔍")

@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    papers = pd.read_parquet("papers.parquet")
    index = faiss.read_index("papers.index")
    return model, papers, index

model, papers, index = load_all()

st.title("🔍 Research Paper Recommendation App (% Match)")
query = st.text_input("Enter paper title or keywords:")
top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        q_emb = model.encode(query, convert_to_numpy=True).astype("float32").reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
        for score, idx in zip(D[0], I[0]):
            row = papers.iloc[int(idx)]
            percent = round(float(max(0.0, min(1.0, score)) * 100.0), 2)
            st.subheader(f"{row['title']} — {percent}% match")
            st.write(row['abstract'][:500])
            st.markdown("---")
