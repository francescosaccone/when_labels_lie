# when_bug_lie

Replication package for the study on bug issue mislabeling and LLM-based support for issue labeling.

## Overview

This repository contains the code, scripts, and experimental workflow used to investigate whether Large Language Models (LLMs) can support developers in detecting and mitigating issue mislabeling in GitHub bug reports.

The study is organized around three research questions:

- RQ1 — analysis of developer-corrected mislabeling in GitHub issue reports
- RQ2 — extraction and analysis of linguistic, structural, and semantic features from issue reports
- RQ3 — evaluation of LLMs for:
  - direct label assignment
  - label validation in an LLM-as-a-judge setting

## Ground truth dataset

The ground truth used in this study is based on the issue report classification dataset released for the **NLBSE 2022 Tool Competition on Issue Report Classification**. The competition provided a dataset of more than **800k labeled issue reports** annotated as **bug**, **enhancement**, and **question**. citeturn761167view0

Please download the original dataset from the official NLBSE 2022 website:

- NLBSE 2022 Tool Competition page**: https://nlbse2022.github.io/tools/

After downloading it, place the original CSV file as:

```bash
dataset.csv
```

in the root directory of the project, unless you adapt the scripts to a different path.

## Repository logic

The workflow is divided into three main parts:

1. RQ1 — building the subsets of correctly labeled and missclassified bug issues
2. RQ2 — extracting textual and structural metrics from issue titles and bodies
3. RQ3 — running LLM-based experiments on the derived subsets

---

## RQ1 — Mislabeling analysis on bug issues

The scripts for RQ1 analyze GitHub issue event histories in order to identify:

- issues whose labels remain unchanged
- issues whose labels are removed or replaced
- the temporal window associated with label changes

In particular, the workflow:

- reads the original issue dataset
- filters English issue reports
- queries GitHub issue events through the GitHub API
- distinguishes between:
  - **correctly classified issues**
  - **missclassified issues**
- computes the `window` associated with the removal or replacement of a label

### Main outputs

RQ1 produces two datasets:

- `correctly_class.csv`
- `miss_classified.csv`

These files contain bug issue instances together with their extracted metadata.

### Requirement

RQ1 requires a GitHub personal access token to query issue event histories.

Set it as an environment variable:

```bash
export GITHUB_TOKEN=your_token_here
```

---

## RQ2 — Feature extraction

RQ2 extracts a set of linguistic, readability, discourse, and semantic features from issue titles and bodies.

The extracted features include, among others:

- title/body length
- polarity and subjectivity
- ambiguity index
- informality index
- readability scores
- vagueness index
- bullet point proportion
- alpha ratio
- mean word length
- stop word count
- ellipsis proportion
- first-order coherence
- entropy
- perplexity
- POS proportions
- lexical diversity

### Main outputs

The metrics extracted in RQ2 are used to enrich the two subsets generated in RQ1 and to support the subsequent analyses in RQ3.

---

## RQ3 — LLM-based experiments

RQ3 is divided into two scenarios.

### Scenario 1 — Label assignment

In this scenario, LLMs are asked to directly assign one label to each issue, choosing among:

- `bug`
- `enhancement`
- `question`

The objective is to verify:

- whether LLMs correctly assign labels to issues in the **correct** subset
- whether LLMs also fail on the **missclassified** subset in a way similar to human annotators

### Scenario 2 — LLM as judge

In this scenario, LLMs do not assign a label from scratch. Instead, they receive:

- issue title
- issue body
- an assigned label

and must evaluate whether that label is appropriate, returning a binary judgment:

- `yes`
- `no`

The objective is to assess:

- whether LLMs validate correctly assigned labels
- whether LLMs reject labels that are actually wrong

---

## Models used

The following LLMs were used in the experiments:

- Qwen2
- Mistral v0.2
- Mistral v0.3
- Llama 3 / Llama 3.1 Instruct**

## Ollama commands

Run the models with Ollama using:

```bash
ollama run qwen2:7b-instruct
ollama run mistral:7b-instruct-v0.3-q8_0
ollama run mistral:7b-instruct-v0.2-q8_0
ollama run linbeiJiang/Llama-3.1-8B-Instruct
ollama run koesn/llama3-8b-instruct
```

---

## Expected outputs

The repository produces datasets for:

- correctly labeled issues
- missclassified issues
- LLM label assignments
- LLM judgments
- statistical evaluation results

These outputs are then used to compute:

- Fisher’s exact test
- Precision
- Recall
- F1-score
- Accuracy
- label-specific analyses
- temporal analyses on mislabeling windows

---

## Notes

- The scripts assume CSV-based inputs.
- Some scripts require a valid GitHub token.
- Some LLM scripts assume Ollama is installed and running locally.
- Scenario 1 and Scenario 2 are independent experimental steps and should be interpreted separately.

## Citation

If you use this repository or build upon this replication package, please cite the corresponding paper once available.
