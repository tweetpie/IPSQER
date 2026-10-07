# IPSQER

Implementation for Intent-Preserving Search Query Expansion and Reranking for Ambiguous Document Retrieval using LLMs.

IPSQER addresses *intent collapse*: a single LLM-expanded query can over-focus on one meaning of an ambiguous query and lose other valid interpretations. The system keeps multiple plausible intents active through expansion, retrieval, and reranking.

## Pipeline

1. Generate candidate intents for the query.
2. Validate intents with corpus evidence from BM25 and a cross-encoder.
3. Expand each retained intent separately.
4. Retrieve per intent with BM25 and dense E5/FAISS search.
5. Pool, deduplicate, and rerank candidates with an intent-aware score.

## Repository Layout

```text
src/          Pipeline stages
queries/      Input queries
data/         Wikipedia corpus and BM25 index notebook
embeddings/   Dense embedding and FAISS index notebook
output/       Pipeline outputs
```

## Setup

Create the local Wikipedia indexes with:

- `data/wikipedia_corpus.ipynb`
- `embeddings/dense_retrieval_embeddings.ipynb`

Set an OpenAI-compatible API key for the LLM stages:

```bash
export OPENAI_API_KEY="your-key"
```

Optional:

```bash
export OPENAI_BASE_URL="https://openai.rc.asu.edu/v1"
export IPSQER_MODEL="qwen3-235b-a22b-instruct-2507"
```

## Run

```bash
python3 run_pipeline.py
```

Useful options:

```bash
python3 run_pipeline.py --dry-run
python3 run_pipeline.py --query "python"
python3 run_pipeline.py --clean-before-run
```

Run only part of the pipeline:

```bash
python3 run_pipeline.py \
  --from-stage intent_specific_query_expansion \
  --to-stage reranking
```

Skip stages whose expected outputs already exist:

```bash
python3 run_pipeline.py --skip-existing
```

Clean selected stage outputs without running:

```bash
python3 run_pipeline.py --from-stage reranking --clean-output
```

Common experiment knobs:

```bash
python3 run_pipeline.py \
  --max-generated-intents 3 \
  --top-pair-fraction 0.2 \
  --lambda-param 0.7
```

Model overrides:

```bash
python3 run_pipeline.py \
  --llm-model "qwen3-235b-a22b-instruct-2507" \
  --cross-encoder-model "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Use `--multiintent-model`, `--expansion-model`, `--grounding-model`, or `--reranker-model` to override only one stage. Use `--time-file` to choose the timing CSV path and `--profile` to save Scalene profiles in `output/profiles`.
