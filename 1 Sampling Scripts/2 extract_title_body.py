import pandas as pd

# =========================
# CONFIG
# =========================
ORIGINAL_DATASET = "csv/dataset.csv"

CORRECT_SAMPLE_FILE = "test_csv/sample_correct_combined_perplexity_bodywords.csv"
MISS_SAMPLE_FILE = "test_csv/sample_miss_combined_perplexity_bodywords.csv"

OUT_CORRECT_SUBDATASET = "in_csv/subdataset_correct_minimal.csv"
OUT_MISS_SUBDATASET = "in_csv/subdataset_miss_minimal.csv"

# chiavi di join
ORIGINAL_URL_COL = "issue_url"
SAMPLE_URL_COL = "issue_URL"

# colonne minime da recuperare dal dataset originale
ORIGINAL_TEXT_COLUMNS = [
    "issue_url",
    "issue_title",
    "issue_body",
]

# =========================
# LOAD
# =========================
df_original = pd.read_csv(ORIGINAL_DATASET)
df_correct = pd.read_csv(CORRECT_SAMPLE_FILE)
df_miss = pd.read_csv(MISS_SAMPLE_FILE)

# =========================
# CHECK COLONNE
# =========================
if ORIGINAL_URL_COL not in df_original.columns:
    raise ValueError(f"Colonna '{ORIGINAL_URL_COL}' non trovata in {ORIGINAL_DATASET}")

if SAMPLE_URL_COL not in df_correct.columns:
    raise ValueError(f"Colonna '{SAMPLE_URL_COL}' non trovata in {CORRECT_SAMPLE_FILE}")

if SAMPLE_URL_COL not in df_miss.columns:
    raise ValueError(f"Colonna '{SAMPLE_URL_COL}' non trovata in {MISS_SAMPLE_FILE}")

missing_original_cols = [c for c in ORIGINAL_TEXT_COLUMNS if c not in df_original.columns]
if missing_original_cols:
    raise ValueError(f"Colonne mancanti nel dataset originale: {missing_original_cols}")

# =========================
# PREPARA DATASET ORIGINALE
# =========================
df_original_text = df_original[ORIGINAL_TEXT_COLUMNS].copy()
df_original_text = df_original_text.drop_duplicates(subset=[ORIGINAL_URL_COL])

# =========================
# MERGE MINIMALE CORRECT
# =========================
df_correct_urls = df_correct[[SAMPLE_URL_COL]].copy()

df_correct_sub = df_correct_urls.merge(
    df_original_text,
    how="left",
    left_on=SAMPLE_URL_COL,
    right_on=ORIGINAL_URL_COL
)

df_correct_sub = df_correct_sub.drop(columns=[ORIGINAL_URL_COL])

# =========================
# MERGE MINIMALE MISS
# =========================
df_miss_urls = df_miss[[SAMPLE_URL_COL]].copy()

df_miss_sub = df_miss_urls.merge(
    df_original_text,
    how="left",
    left_on=SAMPLE_URL_COL,
    right_on=ORIGINAL_URL_COL
)

df_miss_sub = df_miss_sub.drop(columns=[ORIGINAL_URL_COL])

# =========================
# REPORT
# =========================
print("=== CORRECT SUBDATASET REPORT ===")
print("Rows:", len(df_correct_sub))
print("Matched title:", df_correct_sub["issue_title"].notna().sum())
print("Matched body:", df_correct_sub["issue_body"].notna().sum())
print()

print("=== MISS SUBDATASET REPORT ===")
print("Rows:", len(df_miss_sub))
print("Matched title:", df_miss_sub["issue_title"].notna().sum())
print("Matched body:", df_miss_sub["issue_body"].notna().sum())
print()

# =========================
# SAVE
# =========================
df_correct_sub.to_csv(OUT_CORRECT_SUBDATASET, index=False)
df_miss_sub.to_csv(OUT_MISS_SUBDATASET, index=False)

print("Done.")
print(f"Saved: {OUT_CORRECT_SUBDATASET}")
print(f"Saved: {OUT_MISS_SUBDATASET}")