# Shared Utilities for Divorce Reason Extraction & Classification
import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple

# Text Preprocessing Utilities
class DummyTextPreprocessing:
    """Replace with straycat.TextPreprocessing if available."""
    def auto_text_prep(self, texts):
        # Simple tokenization and lowercasing as fallback
        return [[w.lower() for w in re.findall(r'\w+', t)] for t in texts]

def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter (fallback)."""
    return [s.strip() for s in re.split(r'[\n\.!?]', text) if s.strip()]

def preprocess_text(df):
    try:
        from straycat.text_preprocessing import TextPreprocessing
        prep = TextPreprocessing()
    except ImportError:
        prep = DummyTextPreprocessing()
    processed_tokens = prep.auto_text_prep(df['reason_text'].tolist())
    cleaned_strs = [' '.join(tokens) for tokens in processed_tokens]
    df['cleaned'] = cleaned_strs
    return df

# Data Loading

def load_data(data_path: str) -> pd.DataFrame:
    """Load and clean annotated data for classification."""
    df = pd.read_csv(data_path)
    valid_mask = df['label'].astype(str).isin(['0', '1'])
    df = df[valid_mask].copy()
    df['label'] = df['label'].astype(int)
    return df

# TF-IDF Vectorizer Helpers
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(X_train: pd.Series, X_test: pd.Series) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """Fit TF-IDF vectorizer and transform train/test data."""
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec

# Embedding Generation
from sentence_transformers import SentenceTransformer

def get_transformer_embeddings(X_train: pd.Series, X_test: pd.Series = None, model_name: str = "firqaaa/indo-sentence-bert-base"):
    """Get transformer embeddings for train and optionally test data."""
    model = SentenceTransformer(model_name)
    X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
    if X_test is not None:
        X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)
        return X_train_emb, X_test_emb
    else:
        return X_train_emb

def generate_embeddings(df, model_name: str = "firqaaa/indo-sentence-bert-base"):
    model = SentenceTransformer(model_name)
    weighted_embs = []
    for text in df['reason_text']:
        weighted_embs.append(model.encode(text))
    embeddings = np.vstack(weighted_embs)
    return embeddings

# Feature Combination
import numpy as np

def combine_features(X1, X2) -> np.ndarray:
    """Concatenate two feature matrices (sparse or dense)."""
    if hasattr(X1, 'toarray'):
        X1 = X1.toarray()
    return np.hstack([X1, X2])

# Evaluation Plotting
import altair as alt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve

def plot_evaluation(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    """Plot confusion matrix, ROC, and precision-recall curves using Altair."""
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    cm_df = cm_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
    cm_chart = alt.Chart(cm_df).mark_rect().encode(
        x=alt.X('Predicted:N', title='Predicted Label'),
        y=alt.Y('index:N', title='Actual Label'),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['index', 'Predicted', 'Count']
    ).properties(
        title='Confusion Matrix', width=200, height=200
    )
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    roc_chart = alt.Chart(roc_df).mark_line().encode(
        x=alt.X('FPR', scale=alt.Scale(domain=[0,1])),
        y=alt.Y('TPR', scale=alt.Scale(domain=[0,1])),
        tooltip=['FPR', 'TPR']
    ).properties(
        title=f'ROC Curve (AUC={roc_auc:.2f})', width=300, height=200
    )
    roc_diag = alt.Chart(pd.DataFrame({'x':[0,1],'y':[0,1]})).mark_line(strokeDash=[5,5], color='gray').encode(x='x', y='y')
    roc_chart = roc_chart + roc_diag

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_df = pd.DataFrame({'Recall': rec, 'Precision': prec})
    pr_chart = alt.Chart(pr_df).mark_line().encode(
        x=alt.X('Recall', scale=alt.Scale(domain=[0,1])),
        y=alt.Y('Precision', scale=alt.Scale(domain=[0,1])),
        tooltip=['Recall', 'Precision']
    ).properties(
        title='Precision-Recall Curve', width=300, height=200
    )

    # Display charts (side by side)
    chart = alt.hconcat(cm_chart, roc_chart, pr_chart).resolve_scale(color='independent')
    chart.show() 