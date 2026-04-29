import pandas as pd
from scipy.stats import fisher_exact

GROUND_TRUTH_FILE = "csv/dataset.csv"
LLM_CORRECT_FILE = "out_csv/model_output_correct_mistralv2.csv"
LLM_MISS_FILE = "out_csv/model_output_correct_miss_mistralv2.csv"

GROUND_URL_COL = "issue_url"
GROUND_LABEL_COL = "issue_label"

LLM_URL_COL = "issue_URL"
LLM_ASSIGNED_COL = "assigned_label"

df_ground = pd.read_csv(GROUND_TRUTH_FILE)
df_correct = pd.read_csv(LLM_CORRECT_FILE)
df_miss = pd.read_csv(LLM_MISS_FILE)

required_ground = [GROUND_URL_COL, GROUND_LABEL_COL]
required_llm = [LLM_URL_COL, LLM_ASSIGNED_COL]

missing_ground = [c for c in required_ground if c not in df_ground.columns]
missing_correct = [c for c in required_llm if c not in df_correct.columns]
missing_miss = [c for c in required_llm if c not in df_miss.columns]

if missing_ground:
    raise ValueError(f"Colonne mancanti nel ground truth: {missing_ground}")

if missing_correct:
    raise ValueError(f"Colonne mancanti nel dataset correct: {missing_correct}")

if missing_miss:
    raise ValueError(f"Colonne mancanti nel dataset miss: {missing_miss}")

df_ground[GROUND_URL_COL] = df_ground[GROUND_URL_COL].astype(str).str.strip()
df_ground[GROUND_LABEL_COL] = df_ground[GROUND_LABEL_COL].fillna("").astype(str).str.strip().str.lower()

df_correct[LLM_URL_COL] = df_correct[LLM_URL_COL].astype(str).str.strip()
df_correct[LLM_ASSIGNED_COL] = df_correct[LLM_ASSIGNED_COL].fillna("").astype(str).str.strip().str.lower()

df_miss[LLM_URL_COL] = df_miss[LLM_URL_COL].astype(str).str.strip()
df_miss[LLM_ASSIGNED_COL] = df_miss[LLM_ASSIGNED_COL].fillna("").astype(str).str.strip().str.lower()

df_ground = df_ground[[GROUND_URL_COL, GROUND_LABEL_COL]].drop_duplicates(subset=[GROUND_URL_COL])

df_correct = df_correct.merge(
    df_ground,
    how="left",
    left_on=LLM_URL_COL,
    right_on=GROUND_URL_COL
)

df_miss = df_miss.merge(
    df_ground,
    how="left",
    left_on=LLM_URL_COL,
    right_on=GROUND_URL_COL
)

df_correct = df_correct[df_correct[GROUND_LABEL_COL] != ""].copy()
df_miss = df_miss[df_miss[GROUND_LABEL_COL] != ""].copy()

df_correct["llm_miss"] = (df_correct[LLM_ASSIGNED_COL] != df_correct[GROUND_LABEL_COL]).astype(int)
df_miss["llm_miss"] = (df_miss[LLM_ASSIGNED_COL] != df_miss[GROUND_LABEL_COL]).astype(int)

a = (df_miss["llm_miss"] == 1).sum()
b = (df_miss["llm_miss"] == 0).sum()

c = (df_correct["llm_miss"] == 1).sum()
d = (df_correct["llm_miss"] == 0).sum()

table = [
    [a, b],
    [c, d]
]

odds_ratio, p_value = fisher_exact(table)

print("=== CONTINGENCY TABLE ===")
print("Rows: [dataset_missclassified, dataset_correct]")
print("Cols: [LLM miss = 1, LLM miss = 0]")
print(table)
print()

print("=== CELL VALUES ===")
print(f"a (miss dataset, LLM miss=1): {a}")
print(f"b (miss dataset, LLM miss=0): {b}")
print(f"c (correct dataset, LLM miss=1): {c}")
print(f"d (correct dataset, LLM miss=0): {d}")
print()

print("=== FISHER EXACT TEST ===")
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")