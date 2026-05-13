# when_bug_lie

Replication package for the study on bug issue labeling and mislabeling analysis with LLMs.

## Overview

This repository contains the code and experimental pipeline used to study whether Large Language Models (LLMs) can support developers in detecting mislabeled GitHub bug issues and in validating issue labels.

The study is organized into two main scenarios:

### Scenario 1 — LLMs for label assignment
In this scenario, LLMs are asked to directly assign one label to each issue, choosing among:

- `bug`
- `enhancement`
- `question`

The objective is to assess:

- whether LLMs correctly assign labels to issues belonging to the **correct** subset
- whether LLMs also tend to fail on the **misclassified** subset, similarly to human annotators

### Scenario 2 — LLMs as judge
In this scenario, LLMs do not assign a label from scratch. Instead, they receive:

- issue title
- issue body
- an assigned label

and they must judge whether that label is appropriate, returning a binary output:

- `yes`
- `no`

The objective is to assess:

- whether LLMs correctly validate labels that are actually correct
- whether LLMs correctly reject labels that are actually wrong

## Repository structure

The repository is organized around the two scenarios and the preprocessing steps required to build the datasets.

A typical workflow includes:

1. extraction of the subsets from the original data
2. construction of the input datasets for the experiments
3. execution of Scenario 1
4. execution of Scenario 2
5. statistical evaluation of the results

## Ground truth dataset

The ground truth used in this study is based on the issue classification dataset released in the context of the **NLBSE 2022 Tool Competition on Issue Report Classification**. The NLBSE 2022 competition focuses on automatic classification of GitHub issue reports into categories such as bug, enhancement, and question.  [oai_citation:0‡nlbse2022.github.io](https://nlbse2022.github.io/tools/?utm_source=chatgpt.com)

To replicate the study, you should download the original issue classification dataset from the NLBSE 2022 resources and use it as the starting point for building the ground truth and the derived subsets. The official competition website is available here:  [oai_citation:1‡nlbse2022.github.io](https://nlbse2022.github.io/tools/?utm_source=chatgpt.com)

## Models used

The following LLMs were used in the experiments:

- Qwen2
- Mistral v0.2
- Mistral v0.3
- Llama 3 / Llama 3.1 Instruct

## Ollama commands

Run the models with Ollama using commands such as:

```bash
ollama run qwen2:7b-instruct
ollama run mistral:7b-instruct-v0.3-q8_0
ollama run mistral:7b-instruct-v0.2-q8_0
ollama run linbeiJiang/Llama-3.1-8B-Instruct
ollama run koesn/llama3-8b-instruct
