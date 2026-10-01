# AutoTune Research Assistant

LLM-assisted research planning for fine-tuning projects.

## Workflow

`Requirements → Query generation → HF/arXiv/Kaggle search → Report synthesis`

## Configuration

Set:

```bash
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-2.0-flash"
```

Never put API keys in source code.

## Evidence status

This repository demonstrates an end-to-end research-assistant prototype. It does not yet establish that generated recommendations are more accurate than manual search.

The next research milestone is a fixed evaluation set with expert relevance labels, retrieval Recall@k, source/citation coverage and factual consistency checks.

## Security

A provider credential was previously present in the public Git history. Active code in this branch no longer contains it, but the affected credential should still be revoked/rotated.