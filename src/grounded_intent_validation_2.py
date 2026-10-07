import os
import json
import glob
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import CrossEncoder

# =================================================
# Paths
# =================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "output" / "grounded_intent_validation_1" / "retrieval_grounding"
OUTPUT_DIR = REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores"
SUMMARY_CSV = REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores_summary.csv"
FILTERED_CSV = REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores_filtered.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================
# Config
# =================================================
MODEL_NAME = os.environ.get(
    "IPSQER_GROUNDING_MODEL_NAME",
    os.environ.get("IPSQER_CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 0.5
TOP_PAIR_FRACTION = float(os.environ.get("IPSQER_TOP_PAIR_FRACTION", "0.25"))
MIN_SELECTED_PAIRS = 2

print("Grounding scores config:")
print(f"  model_name: {MODEL_NAME}")
print(f"  device: {DEVICE}")
print(f"  alpha: {ALPHA}")
print(f"  top_pair_fraction: {TOP_PAIR_FRACTION}")
print(f"  min_selected_pairs: {MIN_SELECTED_PAIRS}")
print(f"Loading cross-encoder on {DEVICE}...")
cross_encoder = CrossEncoder(MODEL_NAME, device=DEVICE)
print("Cross-encoder loaded.")


# =================================================
# Helpers
# =================================================
def is_heading(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    words = text.split()
    if len(words) > 6:
        return False
    if not text.endswith("."):
        return False
    if any(x in text for x in [",", ";", ":", "?", "!"]):
        return False
    return True


def split_into_paragraphs(title: str, contents: str):
    """
    Split article into paragraphs, merge headings with next paragraph,
    and prepend title to each paragraph.
    """
    body = (contents or "").strip()
    if not body:
        return []

    raw_paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    paragraphs = []
    pending_headings = []

    for p in raw_paragraphs:
        if is_heading(p):
            pending_headings.append(p)
            continue

        if pending_headings:
            merged = "\n\n".join(pending_headings) + "\n\n" + p
            pending_headings = []
        else:
            merged = p

        paragraphs.append(merged)

    if pending_headings:
        paragraphs.append("\n\n".join(pending_headings))

    return [title + "\n\n" + p for p in paragraphs]


def zscore(values):
    """
    Query-wise z normalization.
    """
    if not values:
        return []

    mean_val = sum(values) / len(values)
    var_val = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = math.sqrt(var_val)

    if std_val == 0:
        return [0.0 for _ in values]

    return [(v - mean_val) / std_val for v in values]


def score_article_ce_max(intent_text: str, title: str, contents: str) -> float:
    """
    Cross-encoder paragraph scoring.
    Article CE score = max paragraph score.
    """
    paragraphs = split_into_paragraphs(title, contents)

    if not paragraphs:
        fallback = (title + "\n\n" + contents).strip()
        if not fallback:
            return 0.0
        paragraphs = [fallback]

    pairs = [(intent_text, p) for p in paragraphs]

    scores = cross_encoder.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    if isinstance(scores, (float, int)):
        scores = [float(scores)]
    else:
        try:
            scores = scores.tolist()
        except Exception:
            scores = list(scores)

    if not scores:
        return 0.0

    return float(max(scores))


# =================================================
# Main processing
# =================================================
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
print(f"Found {len(files)} files.")

summary_rows = []

for file in files:
    print(f"Processing {file}")

    with open(file, "r") as f:
        data = json.load(f)

    query_number_full = data["query_number"]
    query_number = query_number_full.split("-")[-1]
    query = data["query"]

    intents = data["intents"]

    # -------------------------------------------------
    # Build all (intent, document) pairs for this query
    # -------------------------------------------------
    pair_rows = []
    intent_meta = {}

    for intent in intents:
        intent_id = intent["intent_id"]
        title = intent["title"]
        description = intent.get("description", "")
        confidence = intent.get("confidence", None)
        docs = intent.get("top_documents", [])

        intent_text = description.strip() if description and description.strip() else title.strip()

        intent_meta[intent_id] = {
            "intent_id": intent_id,
            "title": title,
            "description": description,
            "confidence": confidence,
            "intent_text": intent_text,
            "docs": docs,
        }

        for doc in docs:
            docid = doc.get("docid", "")
            rank = doc.get("rank", None)
            doc_title = doc.get("title", "")
            doc_contents = doc.get("contents", "")
            bm25_raw = float(doc.get("score", 0.0))

            ce_raw = score_article_ce_max(intent_text, doc_title, doc_contents)

            pair_rows.append({
                "intent_id": intent_id,
                "docid": docid,
                "rank": rank,
                "doc_title": doc_title,
                "bm25_raw": bm25_raw,
                "ce_raw": ce_raw,
            })

    # -------------------------------------------------
    # Query-wise z-normalization over all pairs
    # -------------------------------------------------
    bm25_values = [p["bm25_raw"] for p in pair_rows]
    ce_values = [p["ce_raw"] for p in pair_rows]

    bm25_z_values = zscore(bm25_values)
    ce_z_values = zscore(ce_values)

    for p, bz, cz in zip(pair_rows, bm25_z_values, ce_z_values):
        p["bm25_z"] = bz
        p["ce_z"] = cz
        p["combined_score"] = ALPHA * bz + (1.0 - ALPHA) * cz

    # -------------------------------------------------
    # Sort all pairs and select top 25%
    # -------------------------------------------------
    pair_rows_sorted = sorted(pair_rows, key=lambda x: x["combined_score"], reverse=True)
    selected_k = max(1, int(math.ceil(TOP_PAIR_FRACTION * len(pair_rows_sorted))))
    selected_pairs = pair_rows_sorted[:selected_k]

    selected_counts = Counter(p["intent_id"] for p in selected_pairs)
    selected_docids_by_intent = defaultdict(list)

    for p in selected_pairs:
        selected_docids_by_intent[p["intent_id"]].append(p["docid"])

    # Retain intent if at least MIN_SELECTED_PAIRS selected pairs belong to it
    retained_intents = {
        intent_id: (selected_counts.get(intent_id, 0) >= MIN_SELECTED_PAIRS)
        for intent_id in intent_meta.keys()
    }

    # Fallback: if none retained, keep the best intent by selected count,
    # then by sum of combined score
    if not any(retained_intents.values()) and len(intent_meta) > 0:
        print("fallback")
        best_intent = max(
            intent_meta.keys(),
            key=lambda iid: (
                selected_counts.get(iid, 0),
                sum(p["combined_score"] for p in pair_rows if p["intent_id"] == iid)
            )
        )
        retained_intents[best_intent] = True

    # Query-level threshold used for selection
    intent_threshold = selected_pairs[-1]["combined_score"] if selected_pairs else None

    # -------------------------------------------------
    # Build detailed JSON output
    # -------------------------------------------------
    output = {
        "query_number": query_number,
        "query": query,
        "selection_fraction": TOP_PAIR_FRACTION,
        "min_selected_pairs": MIN_SELECTED_PAIRS,
        "alpha": ALPHA,
        "intent_threshold": round(intent_threshold, 6) if intent_threshold is not None else None,
        "intents": []
    }

    for intent_id, meta in intent_meta.items():
        intent_pairs = [p for p in pair_rows if p["intent_id"] == intent_id]

        document_scores = []
        for p in intent_pairs:
            document_scores.append({
                "docid": p["docid"],
                "rank": p["rank"],
                "doc_title": p["doc_title"],
                "bm25_raw": round(p["bm25_raw"], 6),
                "bm25_z": round(p["bm25_z"], 6),
                "ce_raw": round(p["ce_raw"], 6),
                "ce_z": round(p["ce_z"], 6),
                "combined_score": round(p["combined_score"], 6),
                "selected": p in selected_pairs
            })

        output["intents"].append({
            "intent_id": intent_id,
            "title": meta["title"],
            "description": meta["description"],
            "confidence": meta["confidence"],
            "selected_pair_count": int(selected_counts.get(intent_id, 0)),
            "retained_intent": bool(retained_intents.get(intent_id, False)),
            "document_scores": document_scores
        })

    out_json = os.path.join(OUTPUT_DIR, f"{query_number}.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {out_json}")

    # -------------------------------------------------
    # Build intent-level summary rows for CSV
    # -------------------------------------------------
    for intent_id, meta in intent_meta.items():
        intent_pairs = [p for p in pair_rows if p["intent_id"] == intent_id]

        if intent_pairs:
            combined_scores = [p["combined_score"] for p in intent_pairs]
            max_score = max(combined_scores)
            min_score = min(combined_scores)
        else:
            max_score = None
            min_score = None

        summary_rows.append({
            "query_number": query_number,
            "query": query,
            "intent_id": intent_id,
            "intent_title": meta["title"],
            "intent_description": meta["description"],
            "confidence": meta["confidence"],
            "selected_pair_count": int(selected_counts.get(intent_id, 0)),
            "retained_intent": bool(retained_intents.get(intent_id, False)),
            "max_score": round(max_score, 6) if max_score is not None else None,
            "min_score": round(min_score, 6) if min_score is not None else None,
            "intent_threshold": round(intent_threshold, 6) if intent_threshold is not None else None
        })

# =================================================
# Save CSVs
# =================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

filtered_df = summary_df[summary_df["retained_intent"].astype(bool)].copy()
filtered_df.to_csv(FILTERED_CSV, index=False)

print()
print(f"Saved summary CSV:  {SUMMARY_CSV}")
print(f"Saved filtered CSV: {FILTERED_CSV}")
print(f"Total intents: {len(summary_df)}")
print(f"Retained intents: {len(filtered_df)}")
print("Done.")
