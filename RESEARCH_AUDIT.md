# Research Audit — AutoTune Research Assistant

## What this project is

An LLM-assisted research-planning tool that extracts fine-tuning requirements, generates search queries, searches Hugging Face/arXiv/Kaggle, and synthesizes a report.

## Issues found

- A Gemini API key was hard-coded in main.py.
- The implementation warned users to set GEMINI_API_KEY but ignored that environment variable.
- JSON parsing logic was duplicated and fragile.
- Search quality was not objectively evaluated.
- Generated reports are synthesis artifacts, not proof that recommendations are correct.

## Hardening in this branch

- Credentials are read from GEMINI_API_KEY.
- Gemini model is configurable with GEMINI_MODEL.
- JSON parsing is centralized and unit-tested.
- .env.example documents configuration without secrets.
- The research status separates tool functionality from recommendation quality.

## Evaluation gap

A future benchmark should use fixed fine-tuning scenarios with expert relevance labels and measure Recall@k, source coverage and factual consistency.