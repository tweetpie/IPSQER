import os
import json
from pathlib import Path

# -------------------------------------------------
# Bypass Pyserini API check
# -------------------------------------------------
os.environ["OPENAI_API_KEY"] = "sk-dummy"

from pyserini.search.lucene import LuceneSearcher

# -------------------------------------------------
# Paths
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = REPO_ROOT / "output" / "multiintent_generation" / "query_subtopics_list.json"

INDEX_PATH = REPO_ROOT / "wikipedia_bm25_index"

OUTPUT_DIR = REPO_ROOT / "output" / "grounded_intent_validation_1" / "retrieval_grounding"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load Searcher
# -------------------------------------------------
print("Loading BM25 index...")

searcher = LuceneSearcher(str(INDEX_PATH))
searcher.set_bm25(k1=0.9, b=0.4)

print("Index Loaded.")

# -------------------------------------------------
# Read Queries
# -------------------------------------------------
with open(INPUT_JSON, "r") as f:
    queries = json.load(f)

print(f"Loaded {len(queries)} queries.")

# -------------------------------------------------
# Process each query
# -------------------------------------------------
i=0
for q in queries:
    i=i+1
    print(i)
    query_number = q["query_number"]
    original_query = q["query"]

    output_name = query_number.split("-")[-1] + ".json"

    output = {
        "query_number": query_number.split("-")[-1],
        "query": original_query,
        "intents": []
    }

    for intent in q["information_needs"]:

        title = intent["title"]
        description = intent.get("description", "")
        confidence = intent.get("confidence", None)

        keywords = intent.get("retrieval_keywords", [])

        # -----------------------------------------
        # Corpus Grounding Query
        # -----------------------------------------
        grounding_query = (
            original_query
            + " "
            + title
            + " "
            + " ".join(keywords)
        )

        hits = searcher.search(grounding_query, k=20)

        retrieved_docs = []

        for rank, hit in enumerate(hits, start=1):

            doc = searcher.doc(hit.docid)

            try:
                raw = json.loads(doc.raw())

                retrieved_docs.append({
                    "rank": rank,
                    "docid": hit.docid,
                    "score": float(hit.score),
                    "title": raw.get("title", ""),
                    "contents": raw.get("contents", "")
                })

            except Exception:
                print("error")
                print(hit.docid)
                retrieved_docs.append({
                    "rank": rank,
                    "docid": hit.docid,
                    "score": float(hit.score)
                })

        output["intents"].append({
            "intent_id": intent["id"],
            "title": title,
            "description": description,
            "confidence": confidence,
            "retrieval_keywords": keywords,
            "grounding_query": grounding_query,
            "top_documents": retrieved_docs
        })

    out_file = os.path.join(OUTPUT_DIR, output_name)

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {out_file}")

print("Done.")
