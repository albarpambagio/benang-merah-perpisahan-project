# %%
# Divorce Reason Classifier Training Script
# Standalone, notebook-style (%% cells), Gradient Boosting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from typing import Tuple, List
import logging
from tqdm import tqdm
import os
from datetime import datetime

# Import shared utilities
from divorce_utils import load_data, get_tfidf_features, get_transformer_embeddings, combine_features, plot_evaluation

# %%
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("divorce_reason_classifier")

# %%
def load_data_step(data_path):
    log.info("Loading data...")
    df = load_data(data_path)
    log.info(f"Loaded {len(df)} rows.")
    return df

# %%
def split_data_step(df):
    log.info("Splitting data into train and test sets...")
    X = df['reason_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# %%
def extract_features_step(X_train, X_test):
    log.info("Extracting TF-IDF features...")
    vectorizer, X_train_tfidf, X_test_tfidf = get_tfidf_features(X_train, X_test)
    log.info("Extracting transformer embeddings...")
    X_train_emb, X_test_emb = get_transformer_embeddings(X_train, X_test)
    log.info("Combining features...")
    X_train_comb = combine_features(X_train_tfidf, X_train_emb)
    X_test_comb = combine_features(X_test_tfidf, X_test_emb)
    return vectorizer, X_train_comb, X_test_comb

# %%
def train_model_step(X_train_comb, y_train):
    log.info("Starting hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_comb, y_train)
    log.info(f"Best params: {grid.best_params_}")
    clf_best = grid.best_estimator_
    return clf_best

# %%
def export_model_report(y_test, y_pred, y_proba, report_dir="report"):
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"model_report_{timestamp}.csv")
    # Get classification report as dict
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    rows = []
    # Per-class metrics
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                rows.append({"metric": f"{label}_{metric}", "value": value})
        else:
            rows.append({"metric": label, "value": metrics})
    # Add confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    rows.append({"metric": "confusion_matrix", "value": str(cm.tolist())})
    # Add ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    rows.append({"metric": "roc_auc", "value": roc_auc})
    # Save as CSV
    pd.DataFrame(rows).to_csv(report_path, index=False)
    log.info(f"Model report exported to {report_path}")
    return report_path

# %%
def evaluate_model_step(clf_best, X_test_comb, y_test):
    log.info("Evaluating model...")
    y_pred = clf_best.predict(X_test_comb)
    y_proba = clf_best.predict_proba(X_test_comb)[:,1]
    log.info("Classification Report:")
    print(classification_report(y_test, y_pred))
    log.info("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    log.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    plot_evaluation(y_test, y_pred, y_proba)
    export_model_report(y_test, y_pred, y_proba)
    return y_pred, y_proba

# %%
def save_model_step(clf_best, vectorizer, model_path, vectorizer_path):
    log.info("Saving model and vectorizer...")
    joblib.dump(clf_best, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    log.info("Model and vectorizer saved.")

# %%
def main(data_path: str = None) -> None:
    """
    Train and evaluate divorce reason classifier with enhanced features and evaluation.
    If data_path is None, use the default CSV path.
    """
    DATA_PATH = data_path or 'annotated_divorce_reason_sample_for_labelstudio.csv'
    MODEL_PATH = 'divorce_reason_rf_classifier.joblib'
    VECTORIZER_PATH = 'divorce_reason_rf_vectorizer.joblib'
    # Step 1: Load data
    df = load_data_step(DATA_PATH)
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data_step(df)
    # Step 3: Feature extraction
    vectorizer, X_train_comb, X_test_comb = extract_features_step(X_train, X_test)
    # Step 4: Train model
    clf_best = train_model_step(X_train_comb, y_train)
    # Step 5: Evaluate
    evaluate_model_step(clf_best, X_test_comb, y_test)
    # Step 6: Save
    save_model_step(clf_best, vectorizer, MODEL_PATH, VECTORIZER_PATH)

# %%
if __name__ == "__main__":
    main() 
# %%
