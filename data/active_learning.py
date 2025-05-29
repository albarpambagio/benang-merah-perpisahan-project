# %%
# Imports
import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from tqdm import tqdm
import json
from divorce_utils import load_data, get_transformer_embeddings, combine_features
from divorce_reason_classifier import main as train_main
from divorce_reasons_extraction import TextExtractor
import os
from datetime import datetime

# Ensure spaCy model is installed
import spacy
import subprocess
import sys

# %%
def ensure_spacy_model(model_name="xx_sent_ud_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        print(f"spaCy model '{model_name}' installed.")

ensure_spacy_model()

# %%
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("active_learning")

# %%
# --- Step 1: Load Unlabeled Documents ---
def load_unlabeled_documents(input_path):
    if input_path.endswith('.jsonl'):
        docs = []
        with open(input_path, encoding='utf-8') as f:
            for line in f:
                docs.append(json.loads(line))
        return docs
    elif input_path.endswith('.csv'):
        return pd.read_csv(input_path).to_dict(orient='records')
    else:
        raise ValueError("Unsupported file format for unlabeled data.")

# %%
# --- Step 2: Extract Candidate Sentences ---
def extract_candidate_sentences(docs, max_sentences=None):
    sentences = []
    for doc in tqdm(docs, desc="Extracting sentences"):
        text = doc.get('text', '')
        doc_id = doc.get('id', None)
        for sent in TextExtractor.extract_reasons_from_text(text):
            sentences.append({'reason_text': sent, 'source_doc_id': doc_id})
            if max_sentences is not None and len(sentences) >= max_sentences:
                return pd.DataFrame(sentences)
    return pd.DataFrame(sentences)

# %%
# --- Step 3: Deduplicate Against Labeled Data ---
def deduplicate_unlabeled(df_unlabeled, labeled_path):
    df_labeled = load_data(labeled_path)
    labeled_set = set(df_labeled['reason_text'])
    before = len(df_unlabeled)
    df_unlabeled = df_unlabeled[~df_unlabeled['reason_text'].isin(labeled_set)]
    after = len(df_unlabeled)
    log.info(f"Deduplicated {before - after} sentences already labeled.")
    return df_unlabeled

# %%
# --- Step 4: Load Model and Vectorizer ---
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    try:
        vocab_size = len(vectorizer.get_feature_names_out())
    except AttributeError:
        vocab_size = len(vectorizer.vocabulary_)
    log.info(f"Vectorizer vocabulary size: {vocab_size}")
    if hasattr(model, 'n_features_in_'):
        log.info(f"Model n_features_in_: {model.n_features_in_}")
    return model, vectorizer

# %%
# --- Step 5: Compute Uncertainty and Model Label ---
def compute_uncertainty_and_label(df_unlabeled, model, vectorizer):
    X_tfidf = vectorizer.transform(df_unlabeled['reason_text'])
    X_emb = get_transformer_embeddings(df_unlabeled['reason_text'], batch_size=32)
    X = combine_features(X_tfidf, X_emb)
    if hasattr(X, "toarray"):
        X = X.toarray()
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    uncertainty = np.abs(probs - 0.5)
    df_unlabeled['uncertainty'] = uncertainty
    df_unlabeled['model_prob'] = probs
    df_unlabeled['model_label'] = preds
    return df_unlabeled

# %%
# --- Step 6: Auto-label and Select for Human Review ---
def split_auto_and_human(df_unlabeled, threshold=0.95):
    auto_mask = (df_unlabeled['model_prob'] >= threshold) | (df_unlabeled['model_prob'] <= (1-threshold))
    df_auto = df_unlabeled[auto_mask].copy()
    df_auto['auto_labeled'] = True
    df_human = df_unlabeled[~auto_mask].copy()
    df_human['auto_labeled'] = False
    return df_auto, df_human

# %%
# --- Step 7: Export for Annotation ---
def export_for_annotation(df, output_path):
    df.to_csv(output_path, index=False)
    log.info(f"Exported {len(df)} samples to {output_path}")

# %%
# --- Step 8: Export Active Learning Report ---
def export_active_learning_report(df_unlabeled, df_auto, df_human, report_dir="report"):
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"active_learning_report_{timestamp}.csv")
    rows = []
    rows.append({"metric": "num_candidates", "value": len(df_unlabeled)})
    rows.append({"metric": "num_auto_labeled", "value": len(df_auto)})
    rows.append({"metric": "num_human_review", "value": len(df_human)})
    if 'uncertainty' in df_unlabeled.columns:
        rows.append({"metric": "uncertainty_min", "value": df_unlabeled['uncertainty'].min()})
        rows.append({"metric": "uncertainty_max", "value": df_unlabeled['uncertainty'].max()})
        rows.append({"metric": "uncertainty_mean", "value": df_unlabeled['uncertainty'].mean()})
        rows.append({"metric": "uncertainty_std", "value": df_unlabeled['uncertainty'].std()})
    if 'model_prob' in df_unlabeled.columns:
        rows.append({"metric": "model_prob_min", "value": df_unlabeled['model_prob'].min()})
        rows.append({"metric": "model_prob_max", "value": df_unlabeled['model_prob'].max()})
        rows.append({"metric": "model_prob_mean", "value": df_unlabeled['model_prob'].mean()})
        rows.append({"metric": "model_prob_std", "value": df_unlabeled['model_prob'].std()})
    pd.DataFrame(rows).to_csv(report_path, index=False)
    log.info(f"Active learning report exported to {report_path}")
    return report_path

# %%
# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Semi-Automated Active Learning for Divorce Reason Classification")
    parser.add_argument('--unlabeled', required=True, help='Path to unlabeled data (jsonl or csv)')
    parser.add_argument('--labeled', required=True, help='Path to labeled data (csv)')
    parser.add_argument('--model', default='model/divorce_reason_rf_classifier.joblib', help='Path to trained model')
    parser.add_argument('--vectorizer', default='model/divorce_reason_rf_vectorizer.joblib', help='Path to vectorizer')
    parser.add_argument('--output_auto', default='active_learning_auto_labeled.csv', help='Output CSV for auto-labeled samples')
    parser.add_argument('--output_human', default='active_learning_human_review.csv', help='Output CSV for human review samples')
    parser.add_argument('--report', default='report', help='Directory for summary report')
    parser.add_argument('-n', type=int, default=20, help='Number of human review samples to select')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for auto-labeling')
    parser.add_argument('--num_sentences', type=int, default=None, help='Number of candidate sentences to process (for testing)')
    args = parser.parse_args()

    log.info("Loading unlabeled documents...")
    docs = load_unlabeled_documents(args.unlabeled)
    log.info("Extracting candidate sentences...")
    df_unlabeled = extract_candidate_sentences(docs, max_sentences=args.num_sentences)
    log.info("Deduplicating against labeled data...")
    df_unlabeled = deduplicate_unlabeled(df_unlabeled, args.labeled)
    log.info("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(args.model, args.vectorizer)
    log.info("Computing uncertainty and model labels...")
    df_unlabeled = compute_uncertainty_and_label(df_unlabeled, model, vectorizer)
    log.info(f"Splitting auto-labeled (threshold={args.threshold}) and human review samples...")
    df_auto, df_human = split_auto_and_human(df_unlabeled, threshold=args.threshold)
    # Select top-n most uncertain for human review
    df_human = df_human.sort_values('uncertainty').head(args.n)
    log.info(f"Exporting {len(df_auto)} auto-labeled and {len(df_human)} human review samples...")
    export_for_annotation(df_auto, args.output_auto)
    export_for_annotation(df_human, args.output_human)
    export_active_learning_report(df_unlabeled, df_auto, df_human, args.report)
    log.info("Semi-automated active learning batch ready.")

# %%
if __name__ == "__main__":
    main() 