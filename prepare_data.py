import pandas as pd

arxiv = pd.read_csv("arxiv_data.csv")[['title', 'abstract']]

try:
    ieee = pd.read_csv("ieee_data.csv")
    ieee = ieee.rename(columns={"Title": "title", "Abstract": "abstract"})
    ieee = ieee[['title', 'abstract']]
except FileNotFoundError:
    print("⚠️ IEEE dataset not found — continuing with only arXiv.")
    ieee = pd.DataFrame(columns=["title", "abstract"])

combined = pd.concat([arxiv, ieee], ignore_index=True)
combined.reset_index(inplace=True)
combined.rename(columns={'index': 'paper_id'}, inplace=True)
combined.to_csv("papers.csv", index=False)

print(f"✅ Combined dataset saved with {len(combined)} papers.")
