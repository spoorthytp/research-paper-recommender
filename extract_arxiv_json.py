import json
import pandas as pd

input_file = "arxiv-metadata-oai-snapshot.json"
output_file = "arxiv_data.csv"

titles = []
abstracts = []

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            titles.append(data.get("title", "").replace("\n", " ").strip())
            abstracts.append(data.get("abstract", "").replace("\n", " ").strip())
        except json.JSONDecodeError:
            continue

        if (i + 1) % 50000 == 0:
            print(f"Processed {i+1} lines...")

df = pd.DataFrame({"title": titles, "abstract": abstracts})
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"✅ Saved {len(df)} rows to {output_file}")
