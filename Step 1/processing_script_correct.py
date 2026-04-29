import pandas as pd
import requests
import json
import time
import os
import re


# CONFIG
#INPUT_FILE = "in_csv/unassigned_bug_issues.csv" 

INPUT_FILE = "in_csv/subdataset_correct_minimal.csv"   # cambia file se vuoi usare l'altro dataset
OUTPUT_FILE = "out_csv/model_output_correct_mistralv2.csv"


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct-v0.2-q8_0"

URL_COL = "issue_URL"
TITLE_COL = "issue_title"
BODY_COL = "issue_body"

REQUEST_TIMEOUT = 300
SLEEP_SECONDS = 0.1


# crea automaticamente la cartella output
import os
os.makedirs("out_csv", exist_ok=True)



LABEL_LIST = ["bug", "enhancement", "question"]

LABEL_DESCRIPTION = """
The "bug" label is used to identify an issue report that describes a promblem or error within the software or codebase. It indicates that something is not functioning as intended or producing unexpected results. Bug reports help developers identify and fix issues to improve the overall quality and reliability of the software. 

The "enhancement" label indicates new feature requests. It indicates that the reported issue is not related to a clear problem or bug, but rather an opportunity to add functionality or improve the existing behavior, usability, or performance. Enhancements requests help developers prioritize valuable additions and evolve the software over time. 

The "question" label is used when a user or contributor has a query or needs clarification regarding the software or codebase. It indicates that the issue report is seekeing information, guidance, or assistance. Questions can be related to functionality, usage, best practices, or any other topic that requires clarification. 

"""


def build_prompt(title: str, body: str) -> str:
    prompt = f"""
    You are provided with a GitHub issue in this format: 
    TITLE: \"\"\"{title}\"\"\"

    BODY: \"\"\"{body}\"\"\"

    You have to assign it a label. 
    Possible label are: {LABEL_LIST}

    {LABEL_DESCRIPTION}

    Title: \"\"\"{title}\"\"\"
    Body: \"\"\"{body}\"\"\"

    #INSTRUCTION
    Select the label strictly and exclusively from the following label list: bug, enhancement, question.
    The output must be ONLY a markdown code snippet formatted exactly like this:
   

    ```json
    {{
    "label": "string"
    }}

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





def extract_label(raw_output: str) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ""

    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    json_text = match.group(1) if match else raw_output.strip()

    try:
        parsed = json.loads(json_text)
        label = parsed.get("label", "")
        return label.strip() if isinstance(label, str) else ""
    except Exception:
        return ""


def main():
    df = pd.read_csv(INPUT_FILE)

    required_cols = [URL_COL, TITLE_COL, BODY_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataset input: {missing_cols}")

    results = []

    for idx, row in df.iterrows():
        issue_url = row.get(URL_COL, "")
        issue_title = row.get(TITLE_COL, "")
        issue_body = row.get(BODY_COL, "")

        title = "" if pd.isna(issue_title) else str(issue_title)
        body = "" if pd.isna(issue_body) else str(issue_body)

        prompt = build_prompt(title, body)

        result_row = {
            URL_COL: issue_url,
            TITLE_COL: title,
            BODY_COL: body,
        }

        try:
            raw_output = query_ollama(prompt)
            assigned_label = extract_label(raw_output)

            result_row["assigned_label"] = assigned_label
            result_row["model_error"] = "" if assigned_label else "Unable to parse label from model output"

        except Exception as e:
            result_row["assigned_label"] = ""
            result_row["model_error"] = str(e)

        results.append(result_row)

        if idx % 25 == 0:
            print(f"Processed {idx} rows")

        time.sleep(SLEEP_SECONDS)

    out_df = pd.DataFrame(results)[[URL_COL, TITLE_COL, BODY_COL, "assigned_label", "model_error"]]
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("Done.")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


"""
=== MODEL TEXT RESPONSE ===
 Based on the information provided in the title and body of the issue, it appears that the issue is related to the DRM (Digital Rights Management) system and involves fetching keys ahead of the playback position. However, there is no clear indication in the text that this is a bug or an enhancement request. It could potentially be a question if the reporter is seeking clarification on how the DRM system handles key fetching in relation to the playback position.

However, since the body of the issue mentions that it was "spun out" from another issue (#3780), it's possible that the original issue contained more context or details that would help determine if this is a bug or an enhancement request. In the absence of that information, it's difficult to assign a label definitively without more context.

Therefore, based on the limited information provided, I cannot strictly and exclusively assign a label from the following label list: "bug", "enhancement", "question". The output must be ONLY a markdown code snippet formatted exactly like this:

```json
{
"label": "unknown"
}
```
=== MODEL TEXT RESPONSE ===
 I'm an AI language model and don't have the ability to directly select or manipulate code. However, based on the given context, I don't see any indication of a bug report, enhancement suggestion, or question being mentioned in the provided code snippet. Therefore, there should be no label for this code snippet according to the instructions.

```json
{
"label": ""
}
```
It
"""