# -*- coding: utf-8 -*-
"""
Enhanced Divorce Reason Clustering from Indonesian Court Decisions

Key Improvements:
1. Robust reason section extraction with multiple fallback patterns
2. Advanced sentence filtering using inclusion/exclusion patterns
3. Refined stopword lists for legal Indonesian
4. Optimized clustering parameters
5. Post-clustering validation
6. Comprehensive visualization and analysis
"""

# %% Initial Setup
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import altair as alt
from straycat.text_preprocessing import TextPreprocessing

# Disable Altair max rows limit
alt.data_transformers.disable_max_rows()

# %% Constants and Configuration
class Config:
    SAMPLE_PATH = 'court_cases_output_with_text.jsonl'
    MAX_DOCS = 200  # Adjust based on available resources
    MIN_REASON_LENGTH = 20
    MAX_REASON_LENGTH = 400
    OUTPUT_KEYWORDS_FILE = 'auto_reason_keywords.txt'
    
    # Model selection
    EMBEDDING_MODEL_PRIMARY = "firqaaa/indo-sentence-bert-base"
    EMBEDDING_MODEL_FALLBACK = "distiluse-base-multilingual-cased"
    
    # UMAP parameters
    UMAP_PARAMS = {
        'n_neighbors': 15,
        'n_components': 10,
        'metric': 'cosine',
        'random_state': 42,
        'min_dist': 0.1
    }
    
    # HDBSCAN parameters
    HDBSCAN_PARAMS = {
        'min_cluster_size': 3,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'min_samples': 2
    }

# %% Legal and Linguistic Resources
class LanguageResources:
    # Legal boilerplate terms
    LEGAL_BOILERPLATE = [
        'pasal', 'huruf', 'undang-undang', 'kompilasi', 'putusan', 'pengadilan', 
        'hukum', 'perdata', 'nomor', 'tahun', 'perkara', 'mahkamah', 'agama',
        'negeri', 'tinggi', 'banding', 'kasasi', 'mengadili', 'memutus',
        'menetapkan', 'dalil', 'pokok', 'pertimbangan', 'mengingat', 'demi'
    ]
    
    # Indonesian stopwords
    INDONESIAN_STOPWORDS = [
        'dan', 'atau', 'yang', 'untuk', 'dengan', 'dari', 'pada', 'oleh', 
        'dalam', 'adalah', 'ke', 'di', 'sebagai', 'tidak', 'sudah', 'akan', 
        'karena', 'juga', 'lebih', 'agar', 'bagi', 'dapat', 'tersebut', 
        'setelah', 'telah', 'bahwa', 'oleh', 'sehingga', 'maka', 'setiap',
        'diri', 'masih', 'harus', 'bukan', 'saat', 'sampai', 'sejak', 'antara',
        'namun', 'tetapi', 'hanya', 'saja', 'jika', 'bila', 'pun', 'per'
    ]
    
    # Legal action verbs
    LEGAL_ACTION_WORDS = [
        'mengajukan', 'mengabulkan', 'menolak', 'menyatakan', 'memutuskan',
        'menghukum', 'menetapkan', 'mempertimbangkan', 'menerima', 'menyampaikan',
        'memeriksa', 'mendengar', 'mempertimbangkan', 'menimbang', 'menguatkan'
    ]
    
    # Combined stopwords
    @classmethod
    def get_stopwords(cls):
        return list(set(cls.LEGAL_BOILERPLATE + cls.INDONESIAN_STOPWORDS + cls.LEGAL_ACTION_WORDS))
    
    # Common reason keyphrases (for validation)
    REASON_PATTERNS = {
        'financial': ['nafkah', 'ekonomi', 'biaya hidup', 'tidak memberi nafkah', 'penghasilan'],
        'violence': ['kekerasan', 'memukul', 'menyakiti', 'kekerasan fisik', 'kekerasan verbal'],
        'infidelity': ['selingkuh', 'perselingkuhan', 'wanita lain', 'pria lain', 'hubungan terlarang'],
        'abandonment': ['meninggalkan', 'tidak pulang', 'pergi tanpa kabar', 'tinggal rumah', 'pergi begitu saja'],
        'disharmony': ['tidak cocok', 'sering bertengkar', 'tidak harmonis', 'konflik terus', 'pertikaian']
    }

# %% Text Extraction Functions
class TextExtractor:
    @staticmethod
    def extract_duduk_perkara(text):
        """Extract the 'duduk perkara' section with multiple pattern fallbacks"""
        if not isinstance(text, str) or not text.strip():
            return ""
        patterns = [
            r'duduk[\s\-]perkara(.*?)(?:pertimbangan hukum|menimbang|putusan|demikianlah|mengingat)',
            r'(alasan[\s\-]pokok|alasan[\s\-]perceraian|sebab[\s\-]perceraian)(.*?)(?:dalam[\s\-]perkara|pertimbangan)',
            r'(latar[\s\-]belakang|sebab[\s\-]musabab)(.*?)(?:dalam[\s\-]perkara|pertimbangan)',
            r'(menyatakan bahwa|mengajukan bahwa)(.*?)(?:oleh karena itu|sehubungan dengan)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Clean legal references
                extracted = re.sub(
                    r'(pasal|ayat|uu|peraturan|nomor|tahun)[\s\d\-]+[^\s]*',
                    '',
                    extracted,
                    flags=re.IGNORECASE
                )
                return extracted
        return ""  # fallback if no pattern matches
    
    @staticmethod
    def is_reason_sentence(sent):
        """Determine if a sentence contains a divorce reason using multiple criteria"""
        if not isinstance(sent, str) or not sent.strip():
            return False
        if len(sent) < Config.MIN_REASON_LENGTH:
            return False
        sent_lower = sent.lower()
        # Exclusion patterns (legal boilerplate and non-reasons)
        exclusion_patterns = [
            r'berdasarkan (pasal|ayat|uu|peraturan)',
            r'dalil[\s\-]pokok',
            r'pengadilan (agama|negeri)',
            r'perkara nomor',
            r'(memutus|menetapkan|mengadili)',
            r'(dengan ini|demi keadilan)',
            r'^[^\w]*(yang|dengan|untuk|pada)'
        ]
        # Inclusion patterns (common reason indicators)
        inclusion_patterns = [
            r'(karena|sebab|disebabkan|akibat|oleh)',
            r'(tidak|pernah|jarang|sering|selalu) (memberi|melakukan|pulang|menghargai)',
            r'(kekerasan|selingkuh|pertengkaran|ekonomi|nafkah)',
            r'(meninggalkan|pergi|tidak pulang)',
            r'(tidak (ada|terjadi) (kecocokan|harmoni))',
            r'(perbuatan (tercela|asusila|melanggar))'
        ]
        # First check exclusions
        if any(re.search(p, sent_lower) for p in exclusion_patterns):
            return False
        # Then check for positive indicators
        return any(re.search(p, sent_lower) for p in inclusion_patterns)
    
    @staticmethod
    def extract_reasons_from_text(text):
        """Main function to extract potential reason sentences from text"""
        if not isinstance(text, str) or not text.strip():
            return []
        # First try to extract duduk perkara section
        duduk_perkara = TextExtractor.extract_duduk_perkara(text)
        section = duduk_perkara if duduk_perkara else text
        # Split into sentences and filter
        sentences = []
        for sent in re.split(r'[\n\.!?]', section):
            sent = sent.strip()
            if (sent and 
                Config.MIN_REASON_LENGTH <= len(sent) <= Config.MAX_REASON_LENGTH and 
                TextExtractor.is_reason_sentence(sent)):
                sentences.append(sent)
        # Fallback: if extraction yields too few, use first meaningful sentences
        if len(sentences) < 3 and text:
            sents = re.split(r'[\n\.!?]', text)
            sentences.extend([
                s.strip() for s in sents[:5]
                if (isinstance(s, str) and 
                    s.strip() and 
                    Config.MIN_REASON_LENGTH <= len(s.strip()) <= Config.MAX_REASON_LENGTH)
            ])
        return sentences

# %% Data Loading and Preparation
def load_and_prepare_data():
    """Load court decisions and extract potential reason sentences"""
    reasons = []
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
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error processing line {i}: {str(e)}")
                continue
    # Fallback to sample data if extraction yields too few
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
    # Create DataFrame with unique reasons
    df = pd.DataFrame({
        "id": np.arange(1, len(reasons)+1),
        "reason_text": reasons
    })
    df = df.drop_duplicates(subset=["reason_text"]).reset_index(drop=True)
    print(f"\nLoaded {len(df)} unique reason sentences.")
    return df

# %% Text Preprocessing
def preprocess_text(df):
    """Clean and tokenize text using straycat"""
    prep = TextPreprocessing()
    processed_tokens = prep.auto_text_prep(df['reason_text'].tolist())
    cleaned_strs = [' '.join(tokens) for tokens in processed_tokens]
    df['cleaned'] = cleaned_strs
    
    # Show sample preprocessing
    print("\nSample preprocessing results:")
    for i, tokens in enumerate(processed_tokens[:3]):
        print(f"Original: {df['reason_text'][i]}")
        print(f"Processed: {tokens}\n")
    
    return df

# %% Feature Extraction
def extract_features(df):
    """Generate TF-IDF features and embeddings"""
    # TF-IDF for keyword extraction
    vectorizer = TfidfVectorizer(
        max_features=100, 
        stop_words=LanguageResources.get_stopwords()
    )
    tfidf = vectorizer.fit_transform(df['cleaned'].tolist())
    terms = vectorizer.get_feature_names_out()
    means = tfidf.mean(axis=0).A1
    top_idx = means.argsort()[::-1][:20]
    auto_reason_keywords = [terms[i] for i in top_idx]
    
    print("\nTop global TF-IDF keywords:", auto_reason_keywords)
    
    # Save keywords
    with open(Config.OUTPUT_KEYWORDS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(auto_reason_keywords))
    
    # Generate embeddings
    try:
        model = SentenceTransformer(Config.EMBEDDING_MODEL_PRIMARY)
    except Exception:
        print("Primary model not available, falling back to multilingual model...")
        model = SentenceTransformer(Config.EMBEDDING_MODEL_FALLBACK)
    
    embeddings = model.encode(
        df['cleaned'].tolist(), 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    return df, embeddings

# %% Clustering
def perform_clustering(embeddings):
    """Perform UMAP dimensionality reduction and HDBSCAN clustering"""
    # UMAP for dimensionality reduction
    reducer = umap.UMAP(**Config.UMAP_PARAMS)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(**Config.HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(umap_embeddings)
    
    return umap_embeddings, labels

# %% Cluster Validation
def validate_clusters(df, labels):
    """Validate clusters based on reason content"""
    # First calculate cluster validity scores
    df['cluster'] = labels
    valid_clusters = set()
    
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_samples = df[df['cluster'] == cluster_id]['reason_text'].tolist()
        if not cluster_samples:
            continue
            
        # Calculate reason score (percentage of valid reason sentences)
        reason_score = sum(
            1 for text in cluster_samples 
            if TextExtractor.is_reason_sentence(text)
        ) / len(cluster_samples)
        
        # Consider cluster valid if >70% are actual reasons
        if reason_score >= 0.7:
            valid_clusters.add(cluster_id)
    
    # Mark valid clusters
    df['valid_reason_cluster'] = df['cluster'].apply(
        lambda x: x if x in valid_clusters else -1
    )
    
    print(f"\nCluster validation: {len(valid_clusters)} valid clusters identified")
    return df

# %% Cluster Analysis
def analyze_clusters(df):
    """Analyze and visualize clustering results"""
    # Cluster statistics
    cluster_stats = df[df['valid_reason_cluster'] != -1].groupby('valid_reason_cluster').agg({
        'id': 'count',
        'reason_text': lambda x: x.iloc[0][:50] + '...'  # Sample text
    }).rename(columns={'id': 'count'})
    
    print("\nCluster statistics:")
    print(cluster_stats.sort_values('count', ascending=False))
    
    # TF-IDF keywords per cluster
    print("\nTop keywords per valid cluster:")
    for cluster_id in df['valid_reason_cluster'].unique():
        if cluster_id == -1:
            continue
            
        cluster_texts = df[df['valid_reason_cluster'] == cluster_id]['cleaned'].tolist()
        if not cluster_texts:
            continue
            
        vectorizer = TfidfVectorizer(
            max_features=30, 
            stop_words=LanguageResources.get_stopwords()
        )
        tfidf = vectorizer.fit_transform(cluster_texts)
        terms = vectorizer.get_feature_names_out()
        means = tfidf.mean(axis=0).A1
        top_idx = means.argsort()[::-1][:5]
        keywords = [terms[i] for i in top_idx]
        
        print(f"Cluster {cluster_id}: {', '.join(keywords)}")
        print(f"Sample: {df[df['valid_reason_cluster'] == cluster_id]['reason_text'].iloc[0][:80]}...\n")

# %% Cluster Summary and Manual Review Tools

def cluster_summary(df, n_keywords=5, n_examples=3):
    print("\nCluster Summary:")
    for cluster_id in sorted(df['valid_reason_cluster'].unique()):
        if cluster_id == -1:
            continue
        cluster_df = df[df['valid_reason_cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} (Count: {len(cluster_df)})")
        # Top keywords
        vectorizer = TfidfVectorizer(max_features=30, stop_words=LanguageResources.get_stopwords())
        tfidf = vectorizer.fit_transform(cluster_df['cleaned'].tolist())
        terms = vectorizer.get_feature_names_out()
        means = tfidf.mean(axis=0).A1
        top_idx = means.argsort()[::-1][:n_keywords]
        keywords = [terms[i] for i in top_idx]
        print(f"Top keywords: {', '.join(keywords)}")
        # Example sentences
        print("Examples:")
        for ex in cluster_df['reason_text'].head(n_examples):
            print(f"  - {ex}")

# Helper for top keywords per cluster (for visualization)
def get_top_keywords(texts, n=3):
    vectorizer = TfidfVectorizer(max_features=30, stop_words=LanguageResources.get_stopwords())
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    means = tfidf.mean(axis=0).A1
    top_idx = means.argsort()[::-1][:n]
    return ', '.join([terms[i] for i in top_idx])

# %% Visualization
def visualize_clusters(df, umap_embeddings):
    """Create interactive visualization of clusters with top keywords as tooltips"""
    # Prepare 2D UMAP for visualization
    umap_2d = umap.UMAP(
        n_neighbors=Config.UMAP_PARAMS['n_neighbors'],
        n_components=2,
        metric=Config.UMAP_PARAMS['metric'],
        random_state=42
    )
    umap_2d_embeddings = umap_2d.fit_transform(umap_embeddings[:, :Config.UMAP_PARAMS['n_components']])
    df['umap_x'] = umap_2d_embeddings[:, 0]
    df['umap_y'] = umap_2d_embeddings[:, 1]
    # Add a column for top keywords per cluster
    df['cluster_keywords'] = ''
    for cluster_id in df['valid_reason_cluster'].unique():
        if cluster_id == -1:
            continue
        mask = df['valid_reason_cluster'] == cluster_id
        df.loc[mask, 'cluster_keywords'] = get_top_keywords(df.loc[mask, 'cleaned'].tolist())
    # Create Altair chart
    chart = alt.Chart(df[df['valid_reason_cluster'] != -1]).mark_circle(size=60).encode(
        x=alt.X('umap_x', title='UMAP-1'),
        y=alt.Y('umap_y', title='UMAP-2'),
        color=alt.Color('valid_reason_cluster:N', title='Cluster', scale=alt.Scale(scheme='category20')),
        tooltip=[
            alt.Tooltip('reason_text', title='Reason'),
            alt.Tooltip('valid_reason_cluster:N', title='Cluster'),
            alt.Tooltip('cluster_keywords', title='Top Keywords')
        ]
    ).properties(
        width=800,
        height=600,
        title='UMAP Projection of Valid Divorce Reason Clusters'
    ).interactive()
    return chart

# %% Evaluation
def evaluate_clustering(embeddings, labels):
    """Evaluate clustering quality"""
    if len(set(labels)) > 1 and -1 in set(labels):
        # Silhouette score (excluding noise)
        valid_mask = np.array(labels) != -1
        sil_score = silhouette_score(
            embeddings[valid_mask],
            labels[valid_mask]
        )
        print(f"\nSilhouette Score (valid clusters only): {sil_score:.3f}")
    else:
        print("\nNot enough clusters for Silhouette Score calculation")

# %% Main Pipeline
def main():
    # 1. Load and prepare data
    df = load_and_prepare_data()
    
    # 2. Preprocess text
    df = preprocess_text(df)
    
    # 3. Extract features
    df, embeddings = extract_features(df)
    
    # 4. Perform clustering
    umap_embeddings, labels = perform_clustering(embeddings)
    
    # 5. Validate clusters
    df = validate_clusters(df, labels)
    
    # 6. Analyze clusters
    analyze_clusters(df)
    
    # 6a. Cluster summary for manual review
    cluster_summary(df)
    
    # 6b. Export clustered sentences for manual review
    df_to_export = df[df['valid_reason_cluster'] != -1][['valid_reason_cluster', 'reason_text']]
    df_to_export['manual_label'] = ''
    df_to_export['review_notes'] = ''
    # Sort by cluster
    df_to_export = df_to_export.sort_values('valid_reason_cluster')

    # Prepare cluster summaries
    summary_rows = []
    for cluster_id in sorted(df_to_export['valid_reason_cluster'].unique()):
        group = df_to_export[df_to_export['valid_reason_cluster'] == cluster_id]
        # Get top keywords for this cluster
        cluster_texts = group['reason_text'].tolist()
        vectorizer = TfidfVectorizer(max_features=30, stop_words=LanguageResources.get_stopwords())
        tfidf = vectorizer.fit_transform(cluster_texts)
        terms = vectorizer.get_feature_names_out()
        means = tfidf.mean(axis=0).A1
        top_idx = means.argsort()[::-1][:5]
        keywords = ', '.join([terms[i] for i in top_idx])
        summary_row = {
            'valid_reason_cluster': cluster_id,
            'reason_text': f'--- CLUSTER SUMMARY: Cluster {cluster_id} | Count: {len(group)} | Top keywords: {keywords} ---',
            'manual_label': '',
            'review_notes': ''
        }
        summary_rows.append((group.index.min(), summary_row))

    # Insert summary rows at the top of each group
    for idx, row in sorted(summary_rows, reverse=True):
        upper = df_to_export.iloc[:idx]
        lower = df_to_export.iloc[idx:]
        df_to_export = pd.concat([upper, pd.DataFrame([row]), lower], ignore_index=True)

    # Save the organized file
    df_to_export.to_csv('clustered_reasons_for_review_organized.csv', index=False)
    print("\nExported organized clustered reasons to 'clustered_reasons_for_review_organized.csv' for manual review.")
    
    # 7. Visualize results (with top keywords in tooltip)
    chart = visualize_clusters(df, umap_embeddings)
    chart.show()
    
    # 8. Evaluate clustering
    evaluate_clustering(umap_embeddings, df['valid_reason_cluster'].values)
    
    return df

if __name__ == "__main__":
    df = main()
# %%
