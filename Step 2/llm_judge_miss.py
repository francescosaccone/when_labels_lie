import pandas as pd
import requests
import json
import time
import os
import re

INPUT_FILE = "out_step_2/judge_input_miss.csv"
OUTPUT_FILE = "out_step_3/judge_output_miss_mistralv3.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct-v0.3-q8_0"


 
URL_COL = "issue_URL"

TITLE_COL = "issue_title"

BODY_COL = "issue_body"

LABEL_GIVEN_COL = "label_given_to_model"

GROUNDTRUTH_COL = "correct_label"

WRONG_LABEL_COL = "wrong_label"

REQUEST_TIMEOUT = 300
SLEEP_SECONDS = 0.1

os.makedirs("out_step_3", exist_ok=True)

LABEL_LIST = ["bug", "enhancement", "question"]

LABEL_DESCRIPTION = """
The "bug" label is used to identify an issue report that describes a problem or error within the software or codebase. It indicates that something is not functioning as intended or producing unexpected results. Bug reports help developers identify and fix issues to improve the overall quality and reliability of the software.

The "enhancement" label indicates new feature requests. It indicates that the reported issue is not related to a clear problem or bug, but rather an opportunity to add functionality or improve the existing behavior, usability, or performance. Enhancement requests help developers prioritize valuable additions and evolve the software over time.

The "question" label is used when a user or contributor has a query or needs clarification regarding the software or codebase. It indicates that the issue report is seeking information, guidance, or assistance. Questions can be related to functionality, usage, best practices, or any other topic that requires clarification.
"""

def build_prompt(title: str, body: str, assigned_label: str) -> str:
    prompt = f"""
You are provided with a GitHub issue in this format:
TITLE: \"\"\"{title}\"\"\"

BODY: \"\"\"{body}\"\"\"

ASSIGNED LABEL: \"\"\"{assigned_label}\"\"\"

You have to judge whether the assigned label is appropriate for the issue.
Possible labels are: {LABEL_LIST}

{LABEL_DESCRIPTION}

Title: \"\"\"{title}\"\"\"
Body: \"\"\"{body}\"\"\"
Assigned label: \"\"\"{assigned_label}\"\"\"

Return ONLY a markdown JSON snippet exactly in this format:

```json
{{
  "judgment": "yes"
}}
Use:

“yes” if the assigned label is appropriate
“no” if the assigned label is not appropriate
""".strip()
    return prompt


# CHIAMATA OLLAMA
def query_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    
    response.raise_for_status()

    data = response.json()
    import pprint

    
    #print("\n=== FULL RESPONSE JSON ===")

    #pprint.pprint(data)
    
    print("\n=== MODEL TEXT RESPONSE ===")

    print(data.get("response", ""))


    return data.get("response", "").strip()





def extract_judgment(raw_output: str) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ""

    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    json_text = match.group(1) if match else raw_output.strip()

    try:
        parsed = json.loads(json_text)
        judgment = parsed.get("judgment", "")
        if isinstance(judgment, str):
            judgment = judgment.strip().lower()
            if judgment in {"yes", "no"}:
                return judgment
        return ""
    except Exception:
        return ""

def main():

    df = pd.read_csv(INPUT_FILE)
    required_cols = [URL_COL, TITLE_COL, BODY_COL, LABEL_GIVEN_COL, GROUNDTRUTH_COL, WRONG_LABEL_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataset input: {missing_cols}")

    results = []

    for idx, row in df.iterrows():
        issue_url = row.get(URL_COL, "")
        issue_title = row.get(TITLE_COL, "")
        issue_body = row.get(BODY_COL, "")
        label_given = row.get(LABEL_GIVEN_COL, "")
        correct_label = row.get(GROUNDTRUTH_COL, "")
        wrong_label = row.get(WRONG_LABEL_COL, "")

        title = "" if pd.isna(issue_title) else str(issue_title)
        body = "" if pd.isna(issue_body) else str(issue_body)
        assigned_label_input = "" if pd.isna(label_given) else str(label_given).strip().lower()
        correct_label_value = "" if pd.isna(correct_label) else str(correct_label).strip().lower()
        wrong_label_value = "" if pd.isna(wrong_label) else str(wrong_label).strip().lower()

        prompt = build_prompt(title, body, assigned_label_input)

        result_row = {
            URL_COL: issue_url,
            TITLE_COL: title,
            BODY_COL: body,
            LABEL_GIVEN_COL: assigned_label_input,
            GROUNDTRUTH_COL: correct_label_value,
            WRONG_LABEL_COL: wrong_label_value,
        }

        try:
            raw_output = query_ollama(prompt)
            llm_judgment = extract_judgment(raw_output)

            result_row["risposta_LLM"] = llm_judgment
            result_row["model_error"] = "" if llm_judgment else "Unable to parse judgment from model output"

        except Exception as e:
            result_row["risposta_LLM"] = ""
            result_row["model_error"] = str(e)

        results.append(result_row)

        if idx % 25 == 0:
            print(f"Processed {idx} rows")

        time.sleep(SLEEP_SECONDS)

    out_df = pd.DataFrame(results)[
        [URL_COL, TITLE_COL, BODY_COL, LABEL_GIVEN_COL, GROUNDTRUTH_COL, WRONG_LABEL_COL, "risposta_LLM", "model_error"]
    ]
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("Done.")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


