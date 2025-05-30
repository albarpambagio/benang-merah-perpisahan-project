import pandas as pd

# === CONFIGURATION ===
AUTO_LABELED_PATH = 'active_learning_auto_labeled.csv'   # Path to your auto-labeled CSV
DOCCANO_EXPORT_PATH = 'doccano_export.csv'               # Path to your doccano-reviewed CSV
OUTPUT_PATH = 'combined_labeled_dataset.csv'             # Output path

# === LOAD DATA ===
auto_df = pd.read_csv(AUTO_LABELED_PATH)
doccano_df = pd.read_csv(DOCCANO_EXPORT_PATH)

# === OPTIONAL: Harmonize column names if needed ===
# For example, if doccano uses 'text' and your auto-labeled uses 'reason_text':
# doccano_df = doccano_df.rename(columns={'text': 'reason_text'})

# === CONCATENATE AND DEDUPLICATE ===
combined_df = pd.concat([auto_df, doccano_df], ignore_index=True)

# Remove duplicates based on the main text column (adjust as needed)
if 'reason_text' in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=['reason_text'])
elif 'text' in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=['text'])

# === SAVE RESULT ===
combined_df.to_csv(OUTPUT_PATH, index=False)
print(f"Combined dataset saved to {OUTPUT_PATH}. Total rows: {len(combined_df)}")