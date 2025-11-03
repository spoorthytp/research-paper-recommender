import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

df = pd.DataFrame({
    "title": [
        "Deep Learning for Image Classification",
        "Quantum Computing Basics",
        "Natural Language Processing Advances",
        "Graph Neural Networks for Recommendation"
    ],
    "abstract": [
        "A paper on CNNs for image classification tasks.",
        "An introduction to qubits and quantum gates.",
        "Latest transformer models for NLP applications.",
        "Using GNNs to improve recommendation accuracy."
    ]
})

df.to_parquet("papers.parquet", index=False)

model = SentenceTransformer("all-MiniLM-L6-v2")
text = (df["title"] + " " + df["abstract"]).tolist()
emb = model.encode(text, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, "papers.index")

print("✅ Created small demo files: papers.parquet and papers.index")
