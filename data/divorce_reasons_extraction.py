# Divorce Reason Extraction & Clustering Pipeline
# Workflow:
# 1. Run this script to extract, cluster, and export divorce reason sentences.
# 2. Annotate/review the exported CSV (data/annotated_divorce_reason_sample_for_labelstudio.csv).
# 3. Run data/divorce_reason_classifier.py to train and evaluate the classifier.
# Note: Classifier logic is now in a separate script (data/divorce_reason_classifier.py).

# %%
# -*- coding: utf-8 -*-
"""
Divorce Reason Extraction & Clustering Pipeline (Standalone, Portable Version)
- All-in-one script for Colab/Kaggle/local use
- Context-aware, hybrid detection, cluster validation, reporting, export, and evaluation
- No external project/module dependencies
"""

# %%
# =============================
# SECTION 1: Imports & Device Info
# =============================
import os
import re
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import spacy
from sentence_transformers import SentenceTransformer, util
import hdbscan
import umap
from keybert import KeyBERT
import nltk
import importlib.util
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt', quiet=True)

try:
    import torch
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU detected)")
except ImportError:
    print("Torch not installed, device info unavailable.")

# %%
# =============================
# SECTION 2: Config
# =============================
class Config:
    SAMPLE_PATH = 'court_cases_output_with_text.jsonl'  # Path to input JSONL
    MAX_DOCS = 200  # Set higher for full runs
    MIN_REASON_LENGTH = 20
    MAX_REASON_LENGTH = 400
    EMBEDDING_MODEL_PRIMARY = "firqaaa/indo-sentence-bert-base"
    EMBEDDING_MODEL_FALLBACK = "distiluse-base-multilingual-cased"
    UMAP_PARAMS = {
        'n_neighbors': 15,
        'n_components': 10,
        'metric': 'cosine',
        'random_state': 42,
        'min_dist': 0.1
    }
    HDBSCAN_PARAMS = {
        'min_cluster_size': 5,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'min_samples': 2
    }
    CSV_EXPORT = 'clustered_reasons_for_review_organized.csv'
    HTML_EXPORT = 'divorce_reason_clusters.html'

# %%
# =============================
# SECTION 3: Logging Setup
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("divorce_reason_pipeline")

# %%
# =============================
# SECTION 4: Legal & Linguistic Resources
# =============================
class LanguageResources:
    LEGAL_BOILERPLATE = [
        'pasal', 'huruf', 'undang-undang', 'kompilasi', 'putusan', 'pengadilan', 
        'hukum', 'perdata', 'nomor', 'tahun', 'perkara', 'mahkamah', 'agama',
        'negeri', 'tinggi', 'banding', 'kasasi', 'mengadili', 'memutus',
        'menetapkan', 'dalil', 'pokok', 'pertimbangan', 'mengingat', 'demi'
    ]
    INDONESIAN_STOPWORDS = [
        'dan', 'atau', 'yang', 'untuk', 'dengan', 'dari', 'pada', 'oleh', 
        'dalam', 'adalah', 'ke', 'di', 'sebagai', 'tidak', 'sudah', 'akan', 
        'karena', 'juga', 'lebih', 'agar', 'bagi', 'dapat', 'tersebut', 
        'setelah', 'telah', 'bahwa', 'oleh', 'sehingga', 'maka', 'setiap',
        'diri', 'masih', 'harus', 'bukan', 'saat', 'sampai', 'sejak', 'antara',
        'namun', 'tetapi', 'hanya', 'saja', 'jika', 'bila', 'pun', 'per'
    ]
    LEGAL_ACTION_WORDS = [
        'mengajukan', 'mengabulkan', 'menolak', 'menyatakan', 'memutuskan',
        'menghukum', 'menetapkan', 'mempertimbangkan', 'menerima', 'menyampaikan',
        'memeriksa', 'mendengar', 'mempertimbangkan', 'menimbang', 'menguatkan'
    ]
    @classmethod
    def get_stopwords(cls) -> List[str]:
        return list(set(cls.LEGAL_BOILERPLATE + cls.INDONESIAN_STOPWORDS + cls.LEGAL_ACTION_WORDS))
    REASON_PATTERNS = {
        'financial': ['nafkah', 'ekonomi', 'biaya hidup', 'tidak memberi nafkah', 'penghasilan'],
        'violence': ['kekerasan', 'memukul', 'menyakiti', 'kekerasan fisik', 'kekerasan verbal'],
        'infidelity': ['selingkuh', 'perselingkuhan', 'wanita lain', 'pria lain', 'hubungan terlarang'],
        'abandonment': ['meninggalkan', 'tidak pulang', 'pergi tanpa kabar', 'tinggal rumah', 'pergi begitu saja'],
        'disharmony': ['tidak cocok', 'sering bertengkar', 'tidak harmonis', 'konflik terus', 'pertikaian'],
        'addiction': ['kecanduan', 'narkoba', 'minuman keras', 'judi', 'obat terlarang'],
        'neglect': ['mengabaikan', 'tidak memperhatikan', 'tidak peduli', 'acuh tak acuh'],
        'incompatibility': ['beda prinsip', 'tidak sejalan', 'tidak sepaham', 'perbedaan mendasar'],
        'mental_health': ['sakit jiwa', 'gangguan mental', 'depresi', 'psikologis']
    }

# %%
# =============================
# SECTION 5: Text Extraction & Sentence Splitting
# =============================
try:
    nlp_id = spacy.load("xx_sent_ud_sm")
except Exception as e:
    log.warning(f"spaCy model load failed: {e}. Falling back to regex.")
    nlp_id = None

def split_sentences(text: str) -> List[str]:
    """Use spaCy for robust Indonesian sentence segmentation, fallback to regex."""
    if nlp_id:
        doc = nlp_id(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return [s.strip() for s in re.split(r'[\n\.!?]', text) if s.strip()]

class TextExtractor:
    @staticmethod
    def extract_duduk_perkara(text: str) -> str:
        """Extract the 'duduk perkara' section with multiple pattern fallbacks."""
        if not isinstance(text, str) or not text.strip():
            return ""
        patterns = [
            r'duduk[\s\-]perkara(.*?)(?:pertimbangan hukum|menimbang|putusan|demikianlah|mengingat)',
            r'(alasan[\s\-]pokok|alasan[\s\-]perceraian|sebab[\s\-]perceraian)(.*?)(?:dalam[\s\-]perkara|pertimbangan)',
            r'(latar[\s\-]belakang|sebab[\s\-]musabab)(.*?)(?:dalam[\s\-]perkara|pertimbangan)',
            r'(menyatakan bahwa|mengajukan bahwa)(.*?)(?:oleh karena itu|sehubungan dengan)',
            r'alasan-alasan yang dikemukakan oleh penggugat(.*?)(?:pertimbangan|putusan|demikianlah|mengingat)',
            r'sebab-sebab perceraian menurut penggugat(.*?)(?:pertimbangan|putusan|demikianlah|mengingat)',
            r'faktor-faktor yang menyebabkan perceraian(.*?)(?:pertimbangan|putusan|demikianlah|mengingat)'
        ]
        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    extracted = re.sub(
                        r'(pasal|ayat|uu|peraturan|nomor|tahun)[\s\d\-]+[^\s]*',
                        '',
                        extracted,
                        flags=re.IGNORECASE
                    )
                    return extracted
            except Exception as e:
                log.warning(f"Error processing pattern {pattern[:20]}...: {str(e)}")
                continue
        return ""
    @staticmethod
    def is_reason_sentence(sent: str) -> bool:
        if not isinstance(sent, str) or not sent.strip():
            return False
        if len(sent) < Config.MIN_REASON_LENGTH:
            return False
        sent_lower = sent.lower()
        exclusion_patterns = [
            r'berdasarkan (pasal|ayat|uu|peraturan)',
            r'dalil[\s\-]pokok',
            r'pengadilan (agama|negeri)',
            r'perkara nomor',
            r'(memutus|menetapkan|mengadili)',
            r'(dengan ini|demi keadilan)',
            r'^[^\w]*(yang|dengan|untuk|pada)'
        ]
        inclusion_patterns = [
            r'(karena|sebab|disebabkan|akibat|oleh)',
            r'(tidak|pernah|jarang|sering|selalu) (memberi|melakukan|pulang|menghargai)',
            r'(kekerasan|selingkuh|pertengkaran|ekonomi|nafkah)',
            r'(meninggalkan|pergi|tidak pulang)',
            r'(tidak (ada|terjadi) (kecocokan|harmoni))',
            r'(perbuatan (tercela|asusila|melanggar))'
        ]
        if any(re.search(p, sent_lower) for p in exclusion_patterns):
            return False
        return any(re.search(p, sent_lower) for p in inclusion_patterns)
    @staticmethod
    def extract_reasons_from_text(text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        duduk_perkara = TextExtractor.extract_duduk_perkara(text)
        section = duduk_perkara if duduk_perkara else text
        sentences = split_sentences(section)
        filtered = [s for s in sentences if (Config.MIN_REASON_LENGTH <= len(s) <= Config.MAX_REASON_LENGTH and TextExtractor.is_reason_sentence_hybrid(s))]
        if len(filtered) < 3 and text:
            sents = split_sentences(text)
            filtered.extend([
                s for s in sents[:5]
                if (isinstance(s, str) and s.strip() and Config.MIN_REASON_LENGTH <= len(s.strip()) <= Config.MAX_REASON_LENGTH)
            ])
        return filtered
    @staticmethod
    def is_reason_sentence_hybrid(sent: str) -> bool:
        return TextExtractor.is_reason_sentence(sent) or is_reason_sentence_ml(sent)

# %%
# =============================
# SECTION 1A: Stubs for ML-based Reason Detection (replace with real logic if available)
def is_reason_sentence_ml(sent: str) -> bool:
    """Stub for ML-based reason sentence detection. Returns False by default."""
    return False

def reason_sentence_confidence(sent: str) -> float:
    """Stub for ML-based confidence score for reason sentence. Returns 0.5 by default."""
    return 0.5

# %%
# =============================
# SECTION 7: Data Loading & Preparation
# =============================
def load_and_prepare_data():
    """
    Load court decisions and extract potential reason sentences.
    Returns: DataFrame with extracted reason sentences.
    """
    reasons = []
    docs_processed = 0
    if not os.path.exists(Config.SAMPLE_PATH):
        raise FileNotFoundError(f"Input file not found: {Config.SAMPLE_PATH}")
    with open(Config.SAMPLE_PATH, encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), desc="Processing documents"):
            if i >= Config.MAX_DOCS:
                break
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                if not isinstance(text, str):
                    continue
                extracted = TextExtractor.extract_reasons_from_text(text)
                if extracted:
                    reasons.extend(extracted)
                    docs_processed += 1
            except (json.JSONDecodeError, AttributeError) as e:
                log.warning(f"Error processing line {i}: {str(e)}")
                continue
    log.info(f"Processed {docs_processed} documents, extracted {len(reasons)} reason sentences")
    if len(reasons) < 10:
        log.warning("Few reasons extracted, using sample data")
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
    df = pd.DataFrame({
        "id": np.arange(1, len(reasons)+1),
        "reason_text": reasons
    })
    df = df.drop_duplicates(subset=["reason_text"]).reset_index(drop=True)
    log.info(f"Loaded {len(df)} unique reason sentences.")
    return df

# %%
# =============================
# SECTION 8: Text Preprocessing
# =============================
class DummyTextPreprocessing:
    """Replace with straycat.TextPreprocessing if available."""
    def auto_text_prep(self, texts):
        # Simple tokenization and lowercasing as fallback
        return [[w.lower() for w in re.findall(r'\w+', t)] for t in texts]
try:
    from straycat.text_preprocessing import TextPreprocessing
    prep = TextPreprocessing()
except ImportError:
    log.warning("straycat not found, using dummy preprocessing.")
    prep = DummyTextPreprocessing()
def preprocess_text(df):
    processed_tokens = prep.auto_text_prep(df['reason_text'].tolist())
    cleaned_strs = [' '.join(tokens) for tokens in processed_tokens]
    df['cleaned'] = cleaned_strs
    log.info("\nSample preprocessing results:")
    for i, tokens in enumerate(processed_tokens[:3]):
        log.info(f"Original: {df['reason_text'].iloc[i]}")
        log.info(f"Processed: {tokens}\n")
    return df

# %%
# =============================
# SECTION 9: Contextual Embedding Extraction
# =============================
def extract_with_context_weighted(text, max_context=3, weight=3, model=None):
    sentences = split_sentences(text)
    reasons = []
    for i, sent in enumerate(sentences):
        if TextExtractor.is_reason_sentence_hybrid(sent):
            start = max(0, i - max_context)
            end = min(len(sentences), i + max_context + 1)
            context = sentences[start:end]
            reason_idx = i - start
            if model:
                weighted_context = context[:reason_idx] + [context[reason_idx]]*weight + context[reason_idx+1:]
                emb = model.encode(' '.join(weighted_context))
                reasons.append((context, reason_idx, emb))
            else:
                reasons.append((context, reason_idx))
    return reasons

def generate_embeddings(df):
    try:
        model = SentenceTransformer(Config.EMBEDDING_MODEL_PRIMARY)
        log.info(f"Using primary model: {Config.EMBEDDING_MODEL_PRIMARY}")
    except Exception as e:
        log.warning(f"Failed to load primary model: {str(e)}")
        model = SentenceTransformer(Config.EMBEDDING_MODEL_FALLBACK)
        log.info(f"Using fallback model: {Config.EMBEDDING_MODEL_FALLBACK}")
    weighted_embs = []
    for text in tqdm(df['reason_text'], desc="Embedding reasons"):
        reasons = extract_with_context_weighted(text, model=model)
        if reasons:
            _, _, emb = reasons[0]
            weighted_embs.append(emb)
        else:
            weighted_embs.append(model.encode(text))
    embeddings = np.vstack(weighted_embs)
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings

# %%
# =============================
# SECTION 10: Clustering & Validation
# =============================
def perform_clustering(embeddings):
    reducer = umap.UMAP(**Config.UMAP_PARAMS)
    umap_embeddings = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(**Config.HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(umap_embeddings)
    log.info(f"Cluster distribution: {pd.Series(labels).value_counts().to_dict()}")
    return umap_embeddings, labels
def cluster_coherence(embeddings, labels, cluster_id):
    idxs = [i for i, l in enumerate(labels) if l == cluster_id]
    if len(idxs) < 2:
        return 0
    cluster_embs = embeddings[idxs]
    sim_matrix = cosine_similarity(cluster_embs)
    n = len(cluster_embs)
    return (sim_matrix.sum() - n) / (n*(n-1))
def validate_clusters(df, labels, embeddings, thresholds=None):
    df = df.copy()
    df['cluster'] = labels
    valid_clusters = set()
    thresholds = thresholds or {'pattern': 0.5, 'clf': 0.5, 'coherence': 0.2}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_df = df[df['cluster'] == cluster_id]
        if len(cluster_df) < 2:
            continue
        pattern_score = cluster_df['reason_text'].apply(TextExtractor.is_reason_sentence).mean()
        clf_scores = cluster_df['reason_text'].apply(reason_sentence_confidence).mean()
        coherence = cluster_coherence(embeddings, labels, cluster_id)
        if (pattern_score > thresholds['pattern'] and clf_scores > thresholds['clf'] and coherence > thresholds['coherence']):
            valid_clusters.add(cluster_id)
    df['valid_reason_cluster'] = df['cluster'].apply(lambda x: x if x in valid_clusters else -1)
    log.info(f"Cluster validation: {len(valid_clusters)} valid clusters identified (pattern+clf+coherence)")
    return df
def merge_similar_clusters(df, embeddings, similarity_threshold=0.85):
    if len(df) != len(embeddings):
        log.error(f"Dimension mismatch: DataFrame has {len(df)} rows, embeddings has {len(embeddings)}")
        return df
    if 'valid_reason_cluster' not in df.columns:
        log.warning("No valid clusters to merge")
        return df
    valid_indices = df[df['valid_reason_cluster'] != -1].index
    valid_embeddings = embeddings[valid_indices]
    valid_clusters = df.loc[valid_indices]
    if len(valid_clusters) == 0:
        return df
    cluster_centroids = {}
    for cluster_id in valid_clusters['valid_reason_cluster'].unique():
        cluster_mask = valid_clusters['valid_reason_cluster'] == cluster_id
        cluster_embeddings = valid_embeddings[cluster_mask]
        cluster_centroids[cluster_id] = cluster_embeddings.mean(axis=0)
    cluster_ids = list(cluster_centroids.keys())
    similarity_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))
    for i, id1 in enumerate(cluster_ids):
        for j, id2 in enumerate(cluster_ids):
            if i < j:
                sim = util.cos_sim(cluster_centroids[id1], cluster_centroids[id2]).item()
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    merge_map = {}
    for i, id1 in enumerate(cluster_ids):
        if id1 in merge_map:
            continue
        similar_clusters = [cluster_ids[j] for j in range(len(cluster_ids)) if similarity_matrix[i, j] > similarity_threshold]
        if len(similar_clusters) > 1:
            for cid in similar_clusters:
                merge_map[cid] = min(similar_clusters)
    if merge_map:
        df['merged_cluster'] = df['valid_reason_cluster'].apply(lambda x: merge_map.get(x, x))
        log.info(f"Merged {len(merge_map)} clusters into {len(set(merge_map.values()))} groups")
    else:
        df['merged_cluster'] = df['valid_reason_cluster']
        log.info("No similar clusters found to merge")
    return df

# %%
# =============================
# SECTION 11: Keyword Extraction
# =============================
class KeywordExtractor:
    def __init__(self):
        try:
            self.model = KeyBERT(Config.EMBEDDING_MODEL_PRIMARY)
        except Exception:
            self.model = KeyBERT(Config.EMBEDDING_MODEL_FALLBACK)
        self.stopwords = LanguageResources.get_stopwords() + [
            'pengadilan', 'negeri', 'agama', 'perkara', 'nomor', 'tahun',
            'memutus', 'menetapkan', 'mengadili', 'dengan ini', 'demi keadilan'
        ]
    def get_keybert_keywords(self, texts: List[str], n: int = 5) -> str:
        filtered_texts = [
            text for text in texts 
            if not any(re.search(rf'\b{re.escape(term)}\b', text.lower()) 
               for term in self.stopwords[:20])
        ]
        if not filtered_texts:
            filtered_texts = texts
        joined = ' '.join(filtered_texts)
        keywords = self.model.extract_keywords(
            joined,
            keyphrase_ngram_range=(1, 3),
            stop_words=self.stopwords,
            top_n=n*2,
            use_mmr=True,
            diversity=0.7
        )
        final_keywords = []
        for kw, score in keywords:
            if not any(term in kw.lower() for term in self.stopwords):
                if len(kw) > 2 and not kw.isnumeric():
                    final_keywords.append(kw)
                    if len(final_keywords) >= n:
                        break
        return ', '.join(final_keywords) if final_keywords else "No keywords extracted"
    def get_enhanced_keywords(self, texts: List[str], n: int = 5) -> str:
        keybert_kws = self.get_keybert_keywords(texts, n=n)
        keybert_kw_list = [kw.strip() for kw in keybert_kws.split(',') if kw.strip() and kw != 'No keywords extracted']
        pattern_matches = set()
        for category, patterns in LanguageResources.REASON_PATTERNS.items():
            for text in texts:
                for p in patterns:
                    if p in text.lower():
                        pattern_matches.add(p)
        combined = list(dict.fromkeys(keybert_kw_list + list(pattern_matches)))
        return ', '.join(combined[:n]) if combined else "No keywords extracted"

# %%
# =============================
# SECTION 12: Reporting, Export, Visualization, Evaluation
# =============================
def cluster_summary(df: pd.DataFrame, n_keywords: int = 5, n_examples: int = 3) -> None:
    extractor = KeywordExtractor()
    cluster_col = 'merged_cluster' if 'merged_cluster' in df.columns else 'valid_reason_cluster'
    log.info("\nCluster Summary:")
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            continue
        cluster_df = df[df[cluster_col] == cluster_id]
        log.info(f"\nCluster {cluster_id} (Count: {len(cluster_df)})")
        keywords = extractor.get_enhanced_keywords(cluster_df['reason_text'].tolist(), n=n_keywords)
        log.info(f"Top Enhanced keywords: {keywords}")
        log.info("Examples:")
        for ex in cluster_df['reason_text'].head(n_examples):
            log.info(f"  - {ex}")
def export_clustered_reasons(df: pd.DataFrame, filename=Config.CSV_EXPORT):
    cluster_col = 'merged_cluster' if 'merged_cluster' in df.columns else 'valid_reason_cluster'
    extractor = KeywordExtractor()
    df_to_export = df[df[cluster_col] != -1][[cluster_col, 'reason_text']].copy()
    df_to_export['manual_label'] = ''
    df_to_export['review_notes'] = ''
    df_to_export = df_to_export.sort_values(cluster_col)
    summary_rows = []
    for cluster_id in sorted(df_to_export[cluster_col].unique()):
        group = df_to_export[df_to_export[cluster_col] == cluster_id]
        keywords = extractor.get_enhanced_keywords(group['reason_text'].tolist(), n=5)
        summary_row = {
            cluster_col: cluster_id,
            'reason_text': f'--- CLUSTER SUMMARY: Cluster {cluster_id} | Count: {len(group)} | Top keywords: {keywords} ---',
            'manual_label': '',
            'review_notes': ''
        }
        summary_rows.append((group.index.min(), summary_row))
    for idx, row in sorted(summary_rows, reverse=True):
        upper = df_to_export.iloc[:idx]
        lower = df_to_export.iloc[idx:]
        df_to_export = pd.concat([upper, pd.DataFrame([row]), lower], ignore_index=True)
    df_to_export.to_csv(filename, index=False)
    log.info(f"Exported clustered reasons to '{filename}' for manual review.")
def visualize_clusters(df: pd.DataFrame, umap_embeddings: np.ndarray, filename=Config.HTML_EXPORT):
    try:
        import altair as alt
    except ImportError:
        log.warning("Altair not available, skipping visualization")
        return None
    extractor = KeywordExtractor()
    cluster_col = 'merged_cluster' if 'merged_cluster' in df.columns else 'valid_reason_cluster'
    umap_2d = umap.UMAP(
        n_neighbors=Config.UMAP_PARAMS['n_neighbors'],
        n_components=2,
        metric=Config.UMAP_PARAMS['metric'],
        random_state=42
    )
    umap_2d_embeddings = umap_2d.fit_transform(umap_embeddings[:, :Config.UMAP_PARAMS['n_components']])
    df['umap_x'] = umap_2d_embeddings[:, 0]
    df['umap_y'] = umap_2d_embeddings[:, 1]
    df['cluster_keywords'] = ''
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:
            continue
        mask = df[cluster_col] == cluster_id
        df.loc[mask, 'cluster_keywords'] = extractor.get_enhanced_keywords(
            df.loc[mask, 'reason_text'].tolist(), n=3
        )
    df['cluster_size'] = df.groupby(cluster_col)[cluster_col].transform('count')
    valid_clusters = df[df[cluster_col] != -1]
    if len(valid_clusters) == 0:
        log.warning("No valid clusters to visualize")
        return None
    chart = alt.Chart(valid_clusters).mark_circle(size=60).encode(
        x=alt.X('umap_x', title='UMAP-1'),
        y=alt.Y('umap_y', title='UMAP-2'),
        color=alt.Color(f'{cluster_col}:N', title='Cluster', scale=alt.Scale(scheme='category20')),
        tooltip=[
            alt.Tooltip('reason_text', title='Reason'),
            alt.Tooltip(f'{cluster_col}:N', title='Cluster'),
            alt.Tooltip('cluster_keywords', title='Enhanced Keywords'),
            alt.Tooltip('cluster_size', title='Cluster Size')
        ]
    ).properties(
        width=800,
        height=600,
        title='UMAP Projection of Valid Divorce Reason Clusters (Enhanced Keywords)'
    ).interactive()
    chart.save(filename)
    log.info(f"Cluster visualization saved to '{filename}'")

# %%
# =============================
# SECTION 13: Main Pipeline
# =============================
def main() -> pd.DataFrame:
    """
    Run the divorce reason extraction and clustering pipeline.
    Returns the DataFrame of clustered reasons.
    """
    log.info("Starting divorce reason clustering pipeline (standalone)")
    log.info("Loading and preparing data...")
    df = load_and_prepare_data()
    log.info("Preprocessing text...")
    df = preprocess_text(df)
    log.info("Generating embeddings...")
    embeddings = generate_embeddings(df)
    log.info("Performing clustering...")
    umap_embeddings, labels = perform_clustering(embeddings)
    log.info("Validating clusters...")
    df = validate_clusters(df, labels, embeddings)
    log.info("Merging similar clusters...")
    df = merge_similar_clusters(df, embeddings)
    log.info("Generating cluster summary...")
    cluster_summary(df)
    log.info("Visualizing clusters...")
    visualize_clusters(df, umap_embeddings)
    log.info("Exporting clustered reasons...")
    export_clustered_reasons(df)
    log.info("Pipeline complete.")
    return df

if __name__ == "__main__":
    main()
