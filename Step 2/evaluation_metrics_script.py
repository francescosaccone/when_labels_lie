import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

CORRECT_FILE = "out_step_3/judge_output_correct_mistralv3.csv"
MISS_FILE = "out_step_3/judge_output_miss_mistralv3.csv"

JUDGMENT_COL = "risposta_LLM"

df_correct = pd.read_csv(CORRECT_FILE)
df_miss = pd.read_csv(MISS_FILE)

if JUDGMENT_COL not in df_correct.columns:
    raise ValueError(f"Colonna '{JUDGMENT_COL}' non trovata in {CORRECT_FILE}")

if JUDGMENT_COL not in df_miss.columns:
    raise ValueError(f"Colonna '{JUDGMENT_COL}' non trovata in {MISS_FILE}")

df_correct[JUDGMENT_COL] = df_correct[JUDGMENT_COL].fillna("").astype(str).str.strip().str.lower()
df_miss[JUDGMENT_COL] = df_miss[JUDGMENT_COL].fillna("").astype(str).str.strip().str.lower()

df_correct = df_correct[df_correct[JUDGMENT_COL].isin(["yes", "no"])].copy()
df_miss = df_miss[df_miss[JUDGMENT_COL].isin(["yes", "no"])].copy()

df_correct["gold_judgment"] = "yes"
df_miss["gold_judgment"] = "no"

df_all = pd.concat([df_correct, df_miss], ignore_index=True)

y_true = df_all["gold_judgment"]
y_pred = df_all[JUDGMENT_COL]

labels = ["yes", "no"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="yes")
recall = recall_score(y_true, y_pred, pos_label="yes")
f1 = f1_score(y_true, y_pred, pos_label="yes")

print("=== CONFUSION MATRIX ===")
print("Rows: true labels [yes, no]")
print("Cols: predicted labels [yes, no]")
print(cm)
print()

print("=== CELL VALUES ===")
print(f"TP: {tp}")
print(f"FN: {fn}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print()

print("=== METRICS ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")