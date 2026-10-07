import json
import os
from pathlib import Path
from typing import Any

# -------------------------------------------------
# Bypass Pyserini API check
# -------------------------------------------------
os.environ["OPENAI_API_KEY"] = "sk-dummy"

from pyserini.search.lucene import LuceneSearcher

# -------------------------------------------------
# Paths
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = REPO_ROOT / "output" / "intent_specific_query_expansion" / "intent_specific_expansions.json"
INDEX_PATH = REPO_ROOT / "wikipedia_bm25_index"
OUTPUT_JSON = REPO_ROOT / "output" / "per_intent_retrieval_1" / "bm25_results.json"

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
K = 20
BM25_K1 = 0.9
BM25_B = 0.4


def clean_query(text: Any) -> str:
    """Remove commas and normalize whitespace."""
    return " ".join(str(text).replace(",", " ").split()).strip()


def main() -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading BM25 index...")
    searcher = LuceneSearcher(str(INDEX_PATH))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)
    print("Index loaded.")

    with INPUT_JSON.open("r", encoding="utf-8") as fin:
        queries = json.load(fin)

    if not isinstance(queries, list):
        raise ValueError("Input JSON must contain a list of query objects.")

    print(f"Loaded {len(queries)} queries.")

    all_results: list[dict[str, Any]] = []

    for query_index, query_item in enumerate(queries, start=1):
        query_number = query_item["query_number"]
        original_query = query_item["query"]

        print(
            f"Processing BM25 query {query_index}/{len(queries)} "
            f"(query_number={query_number})"
        )

        query_output = {
            "query_number": query_number,
            "query": original_query,
            "intents": [],
        }

        for intent in query_item.get("intents", []):
            intent_title = intent.get("intent_title", "")
            intent_description = intent.get("intent_description", "")
            expanded_query = intent.get("expanded_query", "")

            retrieval_query = clean_query(expanded_query)

            if not retrieval_query:
                hits = []
            else:
                hits = searcher.search(retrieval_query, k=K)

            top_documents: list[dict[str, Any]] = []

            for rank, hit in enumerate(hits, start=1):
                document_output: dict[str, Any] = {
                    "rank": rank,
                    "title": "",
                    "link": "",
                    "text": "",
                    "bm25_score": float(hit.score),
                }

                try:
                    stored_document = searcher.doc(hit.docid)

                    if stored_document is None:
                        raise RuntimeError(
                            f"No stored document found for docid={hit.docid}"
                        )

                    raw = json.loads(stored_document.raw())

                    document_output["title"] = raw.get("title", "")
                    document_output["link"] = raw.get("url", raw.get("link", ""))
                    document_output["text"] = raw.get("contents", raw.get("text", ""))

                except Exception as exc:
                    print(
                        f"Warning: failed to parse BM25 document "
                        f"docid={hit.docid}: {exc}"
                    )

                top_documents.append(document_output)

            query_output["intents"].append(
                {
                    "intent_title": intent_title,
                    "intent_description": intent_description,
                    "expanded_query": expanded_query,
                    "top_documents": top_documents,
                }
            )

        all_results.append(query_output)

    with OUTPUT_JSON.open("w", encoding="utf-8") as fout:
        json.dump(all_results, fout, ensure_ascii=False, indent=2)

    print(f"Saved BM25 results to {OUTPUT_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()
