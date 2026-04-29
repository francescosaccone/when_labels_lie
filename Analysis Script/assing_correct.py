import pandas as pd
import os

GROUND_TRUTH_FILE = "csv/dataset.csv"

LLM_DATASET = "out_csv/model_output_correct_mistralv2.csv"

OUTPUT_DATASET = "out_p/model_output_correct_with_groundtruth_mistralv2.csv"

GROUND_URL_COL = "issue_url"
GROUND_LABEL_COL = "issue_label"

LLM_URL_COL = "issue_URL"
LLM_TITLE_COL = "issue_title"
LLM_BODY_COL = "issue_body"
LLM_ASSIGNED_COL = "assigned_label"



df_ground = pd.read_csv(GROUND_TRUTH_FILE)
df_llm = pd.read_csv(LLM_DATASET)

required_ground = [GROUND_URL_COL, GROUND_LABEL_COL]
required_llm = [LLM_URL_COL, LLM_TITLE_COL, LLM_BODY_COL, LLM_ASSIGNED_COL]

missing_ground = [c for c in required_ground if c not in df_ground.columns]
missing_llm = [c for c in required_llm if c not in df_llm.columns]

if missing_ground:
    raise ValueError(f"Colonne mancanti nel ground truth: {missing_ground}")

if missing_llm:
    raise ValueError(f"Colonne mancanti nel dataset LLM: {missing_llm}")

df_ground[GROUND_URL_COL] = df_ground[GROUND_URL_COL].astype(str).str.strip()
df_ground[GROUND_LABEL_COL] = df_ground[GROUND_LABEL_COL].astype(str).str.strip()

df_llm[LLM_URL_COL] = df_llm[LLM_URL_COL].astype(str).str.strip()
df_llm[LLM_TITLE_COL] = df_llm[LLM_TITLE_COL].fillna("").astype(str)
df_llm[LLM_BODY_COL] = df_llm[LLM_BODY_COL].fillna("").astype(str)
df_llm[LLM_ASSIGNED_COL] = df_llm[LLM_ASSIGNED_COL].fillna("").astype(str).str.strip()

df_ground = df_ground[[GROUND_URL_COL, GROUND_LABEL_COL]].drop_duplicates(subset=[GROUND_URL_COL])

df_merged = df_llm.merge(
    df_ground,
    how="left",
    left_on=LLM_URL_COL,
    right_on=GROUND_URL_COL
)

df_final = df_merged[[LLM_URL_COL, LLM_TITLE_COL, LLM_BODY_COL, LLM_ASSIGNED_COL, GROUND_LABEL_COL]].copy()
df_final.columns = ["issue_URL", "issue_title", "issue_body", "assigned_label", "correct_label"]

df_final.to_csv(OUTPUT_DATASET, index=False)

print(f"Saved: {OUTPUT_DATASET}")
print(f"Rows: {len(df_final)}")
print(f"Matched correct labels: {df_final['correct_label'].notna().sum()}")
print(f"Unmatched correct labels: {df_final['correct_label'].isna().sum()}")