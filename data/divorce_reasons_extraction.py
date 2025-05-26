# %%
"""
Clustering Divorce Reasons from Indonesian Court Decisions

This notebook-style script demonstrates two approaches for clustering 'alasan perceraian' (divorce reasons):
- Option A: UMAP + HDBSCAN
- Option B: BERTopic

It follows the implementation plan for unsupervised grouping of similar divorce reasons, using Indonesian legal texts.
"""
# %%
# Install dependencies (uncomment if running in a new environment)
# !pip install pandas scikit-learn sentence-transformers umap-learn hdbscan bertopic Sastrawi tqdm

# %%
import json
import pandas as pd
from tqdm import tqdm
import re

# ---
# 1. Load a sample of court decision texts
# ---
# For demo, we use a small sample. Adjust 'max_docs' as needed.
sample_path = 'court_cases_output_with_text.jsonl'
reasons = []
max_docs = 100  # adjust for demo
reason_keywords = [
    'karena', 'sebab', 'alasan', 'tidak cocok', 'kekerasan', 'nafkah', 'perselingkuhan',
    'selingkuh', 'ditinggalkan', 'ekonomi', 'kdrt', 'mabuk', 'judi', 'penganiayaan', 'kdrt',
    'tidak pulang', 'tidak memberi', 'tidak menafkahi', 'tidak bertanggung jawab', 'tidak harmonis'
]
legal_boilerplate = [
    'pasal', 'huruf', 'undang-undang', 'kompilasi', 'putusan', 'pengadilan', 'hukum', 'perdata',
    'putusan', 'putusan pengadilan', 'putusan hakim', 'putusan nomor', 'putusan perkara', 'putusan banding',
    'putusan kasasi', 'putusan mahkamah', 'putusan pengadilan agama', 'putusan pengadilan negeri',
    'putusan pengadilan tinggi', 'putusan pengadilan tata usaha negara', 'putusan pengadilan militer',
    'putusan pengadilan niaga', 'putusan pengadilan hubungan industrial', 'putusan pengadilan tipikor',
    'putusan pengadilan tindak pidana korupsi', 'putusan pengadilan anak', 'putusan pengadilan agama islam',
    'putusan pengadilan agama kristen', 'putusan pengadilan agama katolik', 'putusan pengadilan agama hindu',
    'putusan pengadilan agama budha', 'putusan pengadilan agama konghucu', 'putusan pengadilan agama lainnya'
]
with open(sample_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_docs:
            break
        obj = json.loads(line)
        text = obj.get('text', '')
        for sent in re.split(r'[\n\.!?]', text):
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in reason_keywords):
                if not any(law in sent_lower for law in legal_boilerplate):
                    if 20 < len(sent) < 400:
                        reasons.append(sent.strip())
    # Fallback: if extraction yields too few, use first 1-2 sentences
    if not reasons and text:
        sents = re.split(r'[\n\.!?]', text)
        reasons.extend([s.strip() for s in sents[:2] if len(s.strip()) > 20])
print(f"Loaded {len(reasons)} extracted divorce reason sentences.")

# For demo, if extraction yields too few, use a hardcoded sample
if len(reasons) < 10:
    reasons = [
        "Tergugat tidak memberikan nafkah lahir dan batin selama dua tahun.",
        "Penggugat sering mengalami kekerasan fisik dari tergugat.",
        "Tergugat berselingkuh dengan wanita lain dan tidak pulang ke rumah.",
        "Sering terjadi pertengkaran karena masalah ekonomi.",
        "Tergugat meninggalkan rumah tanpa izin selama berbulan-bulan.",
        "Tidak ada kecocokan lagi antara penggugat dan tergugat.",
        "Tergugat melakukan kekerasan verbal dan fisik.",
        "Penggugat merasa tidak dihargai dan sering diabaikan.",
        "Tergugat berjudi dan tidak memberikan nafkah.",
        "Penggugat dan tergugat sering bertengkar karena perbedaan prinsip."
    ]

# Create DataFrame
import numpy as np
df = pd.DataFrame({
    "id": np.arange(1, len(reasons)+1),
    "reason_text": reasons
})
df = df.drop_duplicates(subset=["reason_text"]).reset_index(drop=True)
print(df.head())

# %%
# ---
# 2. Preprocessing: Use straycat for Indonesian text cleaning and tokenization
# ---
from straycat.text_preprocessing import TextPreprocessing

prep = TextPreprocessing()
# auto_text_prep returns a list of tokens per document
processed_tokens = prep.auto_text_prep(df['reason_text'].tolist())
# For embedding, join tokens back to string
cleaned_strs = [' '.join(tokens) for tokens in processed_tokens]
df['cleaned'] = cleaned_strs

# Show a sample of the tokenized output
print("Sample tokenized output:")
for i, tokens in enumerate(processed_tokens[:5]):
    print(f"{df['reason_text'][i]}\n-> {tokens}\n")

# %%
# ---
# 3. Embedding: Use IndoBERT or multilingual model
# ---
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
except Exception:
    print("Falling back to multilingual model...")
    model = SentenceTransformer("distiluse-base-multilingual-cased")

embeddings = model.encode(df['cleaned'].tolist(), show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# %%
# ---
# 4A. Option A: UMAP + HDBSCAN
# ---
import umap
import hdbscan

reducer = umap.UMAP(n_neighbors=10, n_components=5, metric='cosine', random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
labels = clusterer.fit_predict(umap_embeddings)
df['umap_hdbscan_cluster'] = labels
print(df[['reason_text', 'umap_hdbscan_cluster']].head())

# %%
# ---
# 5. Cluster Interpretation: Show top keywords per cluster (UMAP+HDBSCAN)
# ---
print("\nExample reasons per UMAP+HDBSCAN cluster:")
for label in sorted(df['umap_hdbscan_cluster'].unique()):
    print(f"\nCluster {label}:")
    print(df[df['umap_hdbscan_cluster'] == label]['reason_text'].head(5).to_string(index=False))

# %%
# ---
# 6. (Optional) Trend Analysis & Evaluation
# ---
# If you have year/region columns, you can do:
# df.groupby(["year", "bertopic_topic"]).size().unstack().plot()

# Internal evaluation (Silhouette Score)
from sklearn.metrics import silhouette_score
if len(set(labels)) > 1 and len(df) > len(set(labels)):
    sil = silhouette_score(umap_embeddings, labels)
    print(f"Silhouette Score (UMAP+HDBSCAN): {sil:.3f}")
else:
    print("Silhouette Score not available (only one cluster or too few samples).")

# --- End of notebook --- 