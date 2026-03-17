# AIC Drug Repurposing Platform

Interactive dashboard for exploring LLM-prioritized drug candidates for Anthracycline-Induced Cardiotoxicity.

## Quick Start

```bash
pip install streamlit pandas plotly
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Data

The app reads directly from `clinicaltrials_cardiomyopathy_drugs_results_aggregated.csv` — the production output from your LLM pipeline (scripts 0–5).

89 drug entries across 3 graph models (TxGNN, CompGCN, RLR), with:
- LLM classification (Criterion 1 + Criterion 2)
- Study type breakdown (Human / Animal / In vitro / Review / Computational)
- ClinicalTrials.gov matches

## Key Features

- **Drug Table**: Sortable, filterable, color-coded by LLM classification
- **Visualizations**: Model comparison, study type distribution, rank vs. Rp scatter
- **Drug Detail**: Click any drug for full evidence profile + clinical trial info
- **Novel Candidates**: One-click filter reproducing the exact logic that found Cysteine, Dasatinib, Tranilast, and Tretinoin

## Updating Data

Replace the CSV with a new export from your pipeline. The app auto-detects columns.
