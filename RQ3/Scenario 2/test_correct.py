"""
Evaluate yes/no LLM judgment outputs for correct and misclassified issue datasets.

This script reads two CSV files containing the outputs of an LLM-as-a-judge
pipeline: one file for issues from the correctly classified dataset and one file
for issues from the misclassified dataset.

The judgment column is defined by JUDGMENT_COL and is expected to contain either
"yes" or "no". For the correct dataset, "yes" is treated as the correct judgment.
For the misclassified dataset, "no" is treated as the correct judgment.

The script filters out invalid or empty judgments, computes true positives,
false positives, and false negatives for both "yes" and "no", and then reports
precision, recall, F1-score, accuracy, macro precision, macro recall, and macro
F1-score.

The input CSV paths can be changed through CORRECT_FILE and MISS_FILE, for
example when evaluating a different model.
"""

import pandas as pd

CORRECT_FILE = "out_step_3_cs/judge_output_correct_mistralv3.csv"
MISS_FILE = "out_step_3_cs/judge_output_miss_mistralv3.csv"
JUDGMENT_COL = "risposta_LLM"

df_correct = pd.read_csv(CORRECT_FILE)
df_miss = pd.read_csv(MISS_FILE)

if JUDGMENT_COL not in df_correct.columns:

    raise ValueError(f"Column '{JUDGMENT_COL}' was not found in {CORRECT_FILE}")

if JUDGMENT_COL not in df_miss.columns:
    raise ValueError(f"Column '{JUDGMENT_COL}' was not found in {MISS_FILE}")

df_correct[JUDGMENT_COL] = df_correct[JUDGMENT_COL].fillna("").astype(str).str.strip().str.lower()
df_miss[JUDGMENT_COL] = df_miss[JUDGMENT_COL].fillna("").astype(str).str.strip().str.lower()

df_correct = df_correct[df_correct[JUDGMENT_COL].isin(["yes", "no"])].copy()
df_miss = df_miss[df_miss[JUDGMENT_COL].isin(["yes", "no"])].copy()

yes_tp = (df_correct[JUDGMENT_COL] == "yes").sum()
yes_fp = (df_miss[JUDGMENT_COL] == "yes").sum()
yes_fn = (df_correct[JUDGMENT_COL] == "no").sum()

no_tp = (df_miss[JUDGMENT_COL] == "no").sum()
no_fp = (df_correct[JUDGMENT_COL] == "no").sum()
no_fn = (df_miss[JUDGMENT_COL] == "yes").sum()

yes_precision = yes_tp / (yes_tp + yes_fp) if (yes_tp + yes_fp) > 0 else 0.0
yes_recall = yes_tp / (yes_tp + yes_fn) if (yes_tp + yes_fn) > 0 else 0.0
yes_f1 = 2 * yes_precision * yes_recall / (yes_precision + yes_recall) if (yes_precision + yes_recall) > 0 else 0.0

no_precision = no_tp / (no_tp + no_fp) if (no_tp + no_fp) > 0 else 0.0
no_recall = no_tp / (no_tp + no_fn) if (no_tp + no_fn) > 0 else 0.0
no_f1 = 2 * no_precision * no_recall / (no_precision + no_recall) if (no_precision + no_recall) > 0 else 0.0

total = len(df_correct) + len(df_miss)
correct_predictions = yes_tp + no_tp
accuracy = correct_predictions / total if total > 0 else 0.0

macro_precision = (yes_precision + no_precision) / 2
macro_recall = (yes_recall + no_recall) / 2
macro_f1 = (yes_f1 + no_f1) / 2

print("COUNTS")
print(f"YES TP: {yes_tp}")
print(f"YES FP: {yes_fp}")
print(f"YES FN: {yes_fn}")
print()
print(f"NO TP: {no_tp}")
print(f"NO FP: {no_fp}")
print(f"NO FN: {no_fn}")
print()

print("YES METRICS")
print(f"Precision: {yes_precision:.4f}")
print(f"Recall:    {yes_recall:.4f}")
print(f"F1-score:  {yes_f1:.4f}")
print()

print("NO METRICS")
print(f"Precision: {no_precision:.4f}")
print(f"Recall:    {no_recall:.4f}")
print(f"F1-score:  {no_f1:.4f}")
print()

print("GLOBAL METRICS")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1-score:  {macro_f1:.4f}")