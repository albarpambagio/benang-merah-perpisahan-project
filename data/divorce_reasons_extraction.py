# -*- coding: utf-8 -*-
"""
Enhanced Divorce Reason Clustering from Indonesian Court Decisions

Key Improvements in Refined Version:
1. Added all missing imports and functions
2. Fixed error handling and logging
3. Added type hints for better code clarity
4. Optimized model loading and caching
5. Added comprehensive docstrings
6. Improved configuration management
7. Added basic input validation
"""

# %% Initial Setup
import json
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import altair as alt
from straycat.text_preprocessing import TextPreprocessing
from keybert import KeyBERT
from sentence_transformers import util
from sklearn.metrics import silhouette_score
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Disable Altair max rows limit
alt.data_transformers.disable_max_rows()

# %% Constants and Configuration
class Config:
    """Configuration class for all model parameters and file paths"""
    SAMPLE_PATH = 'court_cases_output_with_text.jsonl'
    MAX_DOCS = 200  # Increased from 200 for better clustering
    MIN_REASON_LENGTH = 20
    MAX_REASON_LENGTH = 400
    MIN_CLUSTER_SIZE = 3  # Minimum samples in a valid cluster
    VALID_CLUSTER_THRESHOLD = 0.7  # 70% of sentences must be valid reasons
    
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
        'min_cluster_size': 5,  # Increased from 3 for more robust clusters
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'min_samples': 2
    }

# %% Legal and Linguistic Resources
class LanguageResources:
    """Class containing language-specific resources and patterns"""
    
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
    def get_stopwords(cls) -> List[str]:
        """Combine all stopword lists and return as unique list"""
        return list(set(cls.LEGAL_BOILERPLATE + cls.INDONESIAN_STOPWORDS + cls.LEGAL_ACTION_WORDS))
    
    # Common reason keyphrases (for validation)
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

# %% Text Extraction Functions
class TextExtractor:
    """Class for extracting and processing reason text from court decisions"""
    
    @staticmethod
    def extract_duduk_perkara(text: str) -> str:
        """
        Extract the 'duduk perkara' section with multiple pattern fallbacks
        
        Args:
            text: Raw court decision text
            
        Returns:
            Extracted relevant section or empty string if not found
        """
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
                    # Clean legal references
                    extracted = re.sub(
                        r'(pasal|ayat|uu|peraturan|nomor|tahun)[\s\d\-]+[^\s]*',
                        '',
                        extracted,
                        flags=re.IGNORECASE
                    )
                    return extracted
            except Exception as e:
                logger.warning(f"Error processing pattern {pattern[:20]}...: {str(e)}")
                continue
                
        return ""  # fallback if no pattern matches
    
    @staticmethod
    def is_reason_sentence(sent: str) -> bool:
        """
        Determine if a sentence contains a divorce reason using multiple criteria
        
        Args:
            sent: Input sentence to evaluate
            
        Returns:
            True if sentence appears to contain a divorce reason
        """
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
    def extract_reasons_from_text(text: str) -> List[str]:
        """
        Main function to extract potential reason sentences from text
        
        Args:
            text: Raw court decision text
            
        Returns:
            List of extracted reason sentences
        """
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

def extract_with_context(text: str, max_context: int = 3) -> List[str]:
    """
    Extract reason sentences with surrounding context
    
    Args:
        text: Input text to process
        max_context: Number of sentences to include around each reason
        
    Returns:
        List of reason segments with context
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    reasons = []
    
    for i, sent in enumerate(sentences):
        if TextExtractor.is_reason_sentence(sent):
            # Get surrounding sentences for context
            start = max(0, i - max_context)
            end = min(len(sentences), i + max_context + 1)
            context = " ".join(sentences[start:end])
            reasons.append(context)
            
    return reasons

def normalize_reasons(text: str) -> str:
    """
    Standardize different phrasings of the same reason
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text with standardized phrases
    """
    replacements = [
        (r'tidak (memberi|memberikan|membayar) nafkah', 'tidak memberi nafkah'),
        (r'(kekerasan (fisik|badan|tubuh))', 'kekerasan fisik'),
        (r'(pergi|minggat|kabur) (tanpa (kabar|pemberitahuan)|(tidak pulang))', 'meninggalkan rumah tanpa kabar'),
        (r'(perselingkuhan|hubungan (terlarang|gelap)|(berhubungan|berselingkuh) dengan (pria|wanita) lain)', 'perselingkuhan')
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    return text

def fast_pattern_filter(sentences: List[str]) -> List[str]:
    """
    Filter sentences based on known reason patterns
    
    Args:
        sentences: List of sentences to filter
        
    Returns:
        Filtered list containing only sentences with known reason patterns
    """
    patterns = [p for pats in LanguageResources.REASON_PATTERNS.values() for p in pats]
    filtered = [s for s in sentences if any(p in s.lower() for p in patterns)]
    return filtered

# %% Data Loading and Preparation
def load_and_prepare_data(
    context_aware: bool = True, 
    semantic_filter: bool = True, 
    normalization: bool = True
) -> pd.DataFrame:
    """
    Load court decisions and extract potential reason sentences
    
    Args:
        context_aware: Whether to include context around reason sentences
        semantic_filter: Whether to filter using known reason patterns
        normalization: Whether to normalize reason phrases
        
    Returns:
        DataFrame containing extracted reason sentences
    """
    reasons = []
    docs_processed = 0
    if not os.path.exists(Config.SAMPLE_PATH):
        raise FileNotFoundError(f"Input file not found: {Config.SAMPLE_PATH}")
    try:
        with open(Config.SAMPLE_PATH, encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f), desc="Processing documents"):
                if i >= Config.MAX_DOCS:
                    break
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                    if not isinstance(text, str):
                        continue
                    if context_aware:
                        extracted = extract_with_context(text)
                    else:
                        extracted = TextExtractor.extract_reasons_from_text(text)
                    if normalization:
                        extracted = [normalize_reasons(s) for s in extracted]
                    if semantic_filter:
                        extracted = fast_pattern_filter(extracted)
                    if extracted:
                        reasons.extend(extracted)
                        docs_processed += 1
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Error processing line {i}: {str(e)}")
                    continue
    except FileNotFoundError:
        logger.error(f"Input file not found: {Config.SAMPLE_PATH}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
    logger.info(f"Processed {docs_processed} documents, extracted {len(reasons)} reason sentences")
    if len(reasons) < 10:
        logger.warning("Few reasons extracted, using sample data")
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
    logger.info(f"Loaded {len(df)} unique reason sentences.")
    return df

# %% Text Preprocessing
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and tokenize text using straycat
    
    Args:
        df: DataFrame containing reason_text column
        
    Returns:
        DataFrame with added 'cleaned' column containing processed text
    """
    prep = TextPreprocessing()
    processed_tokens = prep.auto_text_prep(df['reason_text'].tolist())
    cleaned_strs = [' '.join(tokens) for tokens in processed_tokens]
    df['cleaned'] = cleaned_strs
    
    # Show sample preprocessing
    logger.info("\nSample preprocessing results:")
    for i, tokens in enumerate(processed_tokens[:3]):
        logger.info(f"Original: {df['reason_text'].iloc[i]}")
        logger.info(f"Processed: {tokens}\n")
    
    return df

# %% Feature Extraction
def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate sentence embeddings for the cleaned text
    
    Args:
        df: DataFrame containing cleaned text
        
    Returns:
        Tuple of (DataFrame, embeddings numpy array)
    """
    if len(df) > 1000:
        logger.warning(f"Processing {len(df)} documents - this may require significant memory")
    try:
        model = SentenceTransformer(Config.EMBEDDING_MODEL_PRIMARY)
        logger.info(f"Using primary model: {Config.EMBEDDING_MODEL_PRIMARY}")
    except Exception as e:
        logger.warning(f"Failed to load primary model: {str(e)}")
        try:
            model = SentenceTransformer(Config.EMBEDDING_MODEL_FALLBACK)
            logger.info(f"Using fallback model: {Config.EMBEDDING_MODEL_FALLBACK}")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {str(e)}")
            raise
    embeddings = model.encode(
        df['cleaned'].tolist(), 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return df, embeddings

# %% Clustering Functions
def perform_clustering(
    embeddings: np.ndarray, 
    umap_params: Optional[Dict] = None, 
    hdbscan_params: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform UMAP dimensionality reduction and HDBSCAN clustering
    
    Args:
        embeddings: Input sentence embeddings
        umap_params: Optional UMAP parameters
        hdbscan_params: Optional HDBSCAN parameters
        
    Returns:
        Tuple of (UMAP embeddings, cluster labels)
    """
    umap_params = umap_params or Config.UMAP_PARAMS
    hdbscan_params = hdbscan_params or Config.HDBSCAN_PARAMS
    
    # Validate UMAP params
    valid_umap_params = {k: v for k, v in umap_params.items() if k in ['n_neighbors', 'n_components', 'metric', 'random_state', 'min_dist']}
    logger.info("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(**valid_umap_params)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    logger.info("Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    labels = clusterer.fit_predict(umap_embeddings)
    
    logger.info(f"Cluster distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return umap_embeddings, labels

def optimize_cluster_params(embeddings: np.ndarray) -> dict:
    """
    Simple parameter optimization for HDBSCAN clustering
    """
    return {
        'min_cluster_size': max(3, min(10, int(len(embeddings) * 0.02))),
        'min_samples': 1
    }

def merge_similar_clusters(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    similarity_threshold: float = 0.85
) -> pd.DataFrame:
    """
    Merge clusters that are semantically similar
    
    Args:
        df: DataFrame containing cluster assignments
        embeddings: Original sentence embeddings
        similarity_threshold: Cosine similarity threshold for merging
        
    Returns:
        DataFrame with updated 'merged_cluster' column
    """
    # Defensive check for dimensionality mismatch
    if len(df) != len(embeddings):
        logger.error(f"Dimension mismatch: DataFrame has {len(df)} rows, embeddings has {len(embeddings)}")
        return df
    
    if 'valid_reason_cluster' not in df.columns:
        logger.warning("No valid clusters to merge")
        return df
        
    # Get indices of valid clusters in the original DataFrame
    valid_indices = df[df['valid_reason_cluster'] != -1].index
    valid_embeddings = embeddings[valid_indices]
    valid_clusters = df.loc[valid_indices]
    
    if len(valid_clusters) == 0:
        return df
        
    # Calculate cluster centroids using the valid indices
    cluster_centroids = {}
    for cluster_id in valid_clusters['valid_reason_cluster'].unique():
        cluster_mask = valid_clusters['valid_reason_cluster'] == cluster_id
        cluster_embeddings = valid_embeddings[cluster_mask]
        cluster_centroids[cluster_id] = cluster_embeddings.mean(axis=0)
        
    # Compute similarity between centroids
    cluster_ids = list(cluster_centroids.keys())
    similarity_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))
    
    for i, id1 in enumerate(cluster_ids):
        for j, id2 in enumerate(cluster_ids):
            if i < j:
                sim = util.cos_sim(cluster_centroids[id1], cluster_centroids[id2]).item()
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
    # Find clusters to merge
    merge_map = {}
    for i, id1 in enumerate(cluster_ids):
        if id1 in merge_map:
            continue
        similar_clusters = [cluster_ids[j] for j in range(len(cluster_ids)) 
                          if similarity_matrix[i, j] > similarity_threshold]
        if len(similar_clusters) > 1:
            for cid in similar_clusters:
                merge_map[cid] = min(similar_clusters)
                
    # Apply merging
    if merge_map:
        df['merged_cluster'] = df['valid_reason_cluster'].apply(
            lambda x: merge_map.get(x, x)
        )
        logger.info(f"Merged {len(merge_map)} clusters into {len(set(merge_map.values()))} groups")
    else:
        df['merged_cluster'] = df['valid_reason_cluster']
        logger.info("No similar clusters found to merge")
        
    return df

# %% Cluster Validation
def validate_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Validate clusters based on reason content
    
    Args:
        df: DataFrame containing reason texts
        labels: Cluster labels from HDBSCAN
        
    Returns:
        DataFrame with added 'valid_reason_cluster' column
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
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
        
        # Consider cluster valid if > threshold are actual reasons
        if reason_score >= Config.VALID_CLUSTER_THRESHOLD:
            valid_clusters.add(cluster_id)
    
    # Mark valid clusters
    df['valid_reason_cluster'] = df['cluster'].apply(
        lambda x: x if x in valid_clusters else -1
    )
    
    logger.info(f"Cluster validation: {len(valid_clusters)} valid clusters identified")
    return df

# %% Cluster Analysis
def analyze_clusters(df: pd.DataFrame) -> None:
    """
    Analyze and visualize clustering results
    
    Args:
        df: DataFrame containing clustering results
    """
    # Cluster statistics
    valid_clusters = df[df['valid_reason_cluster'] != -1]
    if len(valid_clusters) == 0:
        logger.warning("No valid clusters to analyze")
        return
        
    cluster_stats = valid_clusters.groupby('valid_reason_cluster').agg({
        'id': 'count',
        'reason_text': lambda x: x.iloc[0][:50] + '...'  # Sample text
    }).rename(columns={'id': 'count'})
    
    logger.info("\nCluster statistics:")
    logger.info(cluster_stats.sort_values('count', ascending=False))

# %% Keyword Extraction
class KeywordExtractor:
    """Class for extracting keywords from clusters"""
    
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
        """
        Enhanced KeyBERT keyword extraction with legal boilerplate filtering
        
        Args:
            texts: List of texts to extract keywords from
            n: Number of keywords to return
            
        Returns:
            Comma-separated string of keywords
        """
        # Pre-filter texts to remove sentences dominated by legal boilerplate
        filtered_texts = [
            text for text in texts 
            if not any(re.search(rf'\b{re.escape(term)}\b', text.lower()) 
               for term in self.stopwords[:20])  # Check most common legal terms
        ]
        
        if not filtered_texts:  # Fallback if all texts are filtered
            filtered_texts = texts
            
        joined = ' '.join(filtered_texts)
        
        # Extract keywords with adjusted parameters
        keywords = self.model.extract_keywords(
            joined,
            keyphrase_ngram_range=(1, 3),
            stop_words=self.stopwords,
            top_n=n*2,  # Extract more initially for filtering
            use_mmr=True,
            diversity=0.7
        )
        
        # Post-process keywords
        final_keywords = []
        for kw, score in keywords:
            # Skip if keyword is mostly legal boilerplate
            if not any(term in kw.lower() for term in self.stopwords):
                # Skip single-character keywords
                if len(kw) > 2 and not kw.isnumeric():
                    final_keywords.append(kw)
                    if len(final_keywords) >= n:
                        break
                        
        return ', '.join(final_keywords) if final_keywords else "No keywords extracted"
    
    def get_enhanced_keywords(self, texts: List[str], n: int = 5) -> str:
        """
        Combine KeyBERT keywords and pattern-based matches
        
        Args:
            texts: List of texts to extract keywords from
            n: Number of keywords to return
            
        Returns:
            Comma-separated string of enhanced keywords
        """
        # Get KeyBERT keywords
        keybert_kws = self.get_keybert_keywords(texts, n=n)
        keybert_kw_list = [kw.strip() for kw in keybert_kws.split(',') 
                          if kw.strip() and kw != 'No keywords extracted']
        
        # Get pattern-based matches from REASON_PATTERNS
        pattern_matches = set()
        for category, patterns in LanguageResources.REASON_PATTERNS.items():
            for text in texts:
                for p in patterns:
                    if p in text.lower():
                        pattern_matches.add(p)
        
        # Combine and deduplicate
        combined = list(dict.fromkeys(keybert_kw_list + list(pattern_matches)))
        return ', '.join(combined[:n]) if combined else "No keywords extracted"

# %% Visualization and Reporting
def cluster_summary(df: pd.DataFrame, n_keywords: int = 5, n_examples: int = 3) -> None:
    """
    Generate a summary report of clusters with keywords and examples
    
    Args:
        df: DataFrame containing clustering results
        n_keywords: Number of keywords to show per cluster
        n_examples: Number of examples to show per cluster
    """
    extractor = KeywordExtractor()
    cluster_col = 'merged_cluster' if 'merged_cluster' in df.columns else 'valid_reason_cluster'
    
    logger.info("\nCluster Summary:")
    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            continue
            
        cluster_df = df[df[cluster_col] == cluster_id]
        logger.info(f"\nCluster {cluster_id} (Count: {len(cluster_df)})")
        
        # Enhanced keywords
        keywords = extractor.get_enhanced_keywords(cluster_df['reason_text'].tolist(), n=n_keywords)
        logger.info(f"Top Enhanced keywords: {keywords}")
        
        # Example sentences
        logger.info("Examples:")
        for ex in cluster_df['reason_text'].head(n_examples):
            logger.info(f"  - {ex}")

def visualize_clusters(df: pd.DataFrame, umap_embeddings: np.ndarray) -> alt.Chart:
    try:
        import altair
    except ImportError:
        logger.warning("Altair not available, skipping visualization")
        return None
    extractor = KeywordExtractor()
    cluster_col = 'merged_cluster' if 'merged_cluster' in df.columns else 'valid_reason_cluster'
    
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
    
    # Add enhanced keywords per cluster
    df['cluster_keywords'] = ''
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:
            continue
        mask = df[cluster_col] == cluster_id
        df.loc[mask, 'cluster_keywords'] = extractor.get_enhanced_keywords(
            df.loc[mask, 'reason_text'].tolist(), n=3
        )
    
    # Add cluster size as a quality indicator
    df['cluster_size'] = df.groupby(cluster_col)[cluster_col].transform('count')
    
    # Create Altair chart
    valid_clusters = df[df[cluster_col] != -1]
    if len(valid_clusters) == 0:
        logger.warning("No valid clusters to visualize")
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
    
    return chart

def export_clustered_reasons(df: pd.DataFrame) -> None:
    """
    Export clustered reasons for manual review
    
    Args:
        df: DataFrame containing clustering results
    """
    cluster_col = 'merged_cluster' if 'merged_cluster' in df.columns else 'valid_reason_cluster'
    extractor = KeywordExtractor()
    
    df_to_export = df[df[cluster_col] != -1][[cluster_col, 'reason_text']]
    df_to_export['manual_label'] = ''
    df_to_export['review_notes'] = ''
    df_to_export = df_to_export.sort_values(cluster_col)
    
    # Prepare cluster summaries using enhanced keywords
    summary_rows = []
    for cluster_id in sorted(df_to_export[cluster_col].unique()):
        group = df_to_export[df_to_export[cluster_col] == cluster_id]
        keywords = extractor.get_enhanced_keywords(group['reason_text'].tolist(), n=5)
        summary_row = {
            cluster_col: cluster_id,
            'reason_text': f'--- CLUSTER SUMMARY: Cluster {cluster_id} | Count: {len(group)} | Top Enhanced keywords: {keywords} ---',
            'manual_label': '',
            'review_notes': ''
        }
        summary_rows.append((group.index.min(), summary_row))
    
    # Insert summary rows
    for idx, row in sorted(summary_rows, reverse=True):
        upper = df_to_export.iloc[:idx]
        lower = df_to_export.iloc[idx:]
        df_to_export = pd.concat([upper, pd.DataFrame([row]), lower], ignore_index=True)
    
    try:
        df_to_export.to_csv('clustered_reasons_for_review_organized.csv', index=False)
        logger.info("Exported organized clustered reasons to 'clustered_reasons_for_review_organized.csv' for manual review.")
    except Exception as e:
        logger.error(f"Error exporting clustered reasons: {str(e)}")

# %% Evaluation
def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> None:
    """
    Evaluate clustering quality metrics
    
    Args:
        embeddings: Original sentence embeddings
        labels: Cluster labels
    """
    if len(set(labels)) > 1 and -1 in set(labels):
        # Silhouette score (excluding noise)
        valid_mask = np.array(labels) != -1
        if sum(valid_mask) > 1:  # Need at least 2 samples
            sil_score = silhouette_score(
                embeddings[valid_mask],
                labels[valid_mask]
            )
            logger.info(f"\nSilhouette Score (valid clusters only): {sil_score:.3f}")
        else:
            logger.info("\nNot enough samples for Silhouette Score calculation")
    else:
        logger.info("\nNot enough clusters for evaluation metrics")

# TODO: slice these into smaller chunks of cell
# %% Main Pipeline
def main() -> pd.DataFrame:
    """
    Main execution pipeline for divorce reason clustering
    
    Returns:
        DataFrame containing all extracted reasons with cluster assignments
    """
    logger.info("Starting divorce reason clustering pipeline")
    
    try:
        # 1. Load and prepare data (enable context-aware extraction)
        df = load_and_prepare_data(context_aware=True)
        if len(df) == 0:
            raise ValueError("No valid reasons extracted from input data")
        
        # 2. Preprocess text
        df = preprocess_text(df)
        
        # 3. Extract features
        df, embeddings = extract_features(df)
        
        # 3a. Optimize clustering parameters (for HDBSCAN only)
        best_hdbscan_params = optimize_cluster_params(embeddings)
        logger.info(f"Best HDBSCAN params: {best_hdbscan_params}")
        
        # 4. Perform clustering (UMAP uses Config.UMAP_PARAMS, HDBSCAN uses optimized params)
        umap_embeddings, labels = perform_clustering(
            embeddings,
            umap_params=Config.UMAP_PARAMS,
            hdbscan_params=best_hdbscan_params
        )

        # 5. Validate clusters
        df = validate_clusters(df, labels)

        # 6. Merge similar clusters
        df = merge_similar_clusters(df, embeddings)

        # 7. Analyze clusters
        analyze_clusters(df)

        # 8. Cluster summary
        cluster_summary(df)

        # 9. Visualize clusters (optional, returns Altair chart)
        try:
            chart = visualize_clusters(df, umap_embeddings)
            if chart is not None:
                chart.save('divorce_reason_clusters.html')
                logger.info("Cluster visualization saved to 'divorce_reason_clusters.html'")
        except Exception as e:
            logger.warning(f"Visualization failed: {str(e)}")

        # 10. Export clustered reasons for manual review
        export_clustered_reasons(df)

        # 11. Evaluate clustering
        evaluate_clustering(embeddings, labels)

        logger.info("Pipeline complete.")
        return df
    except MemoryError:
        logger.error("Insufficient memory - try reducing MAX_DOCS")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        return pd.DataFrame()

if __name__ == "__main__":
    main()
    
# %%
