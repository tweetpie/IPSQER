import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

BM25_JSON = REPO_ROOT / "output" / "per_intent_retrieval_1" / "bm25_results.json"
DEEP_JSON = REPO_ROOT / "output" / "per_intent_retrieval_2" / "deep_results.json"
COMBINED_JSON = REPO_ROOT / "output" / "per_intent_retrieval_3" / "combined_results.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    return data


def index_queries(
    data: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed = {}

    for query_item in data:
        key = str(query_item["query_number"])

        if key in indexed:
            raise ValueError(f"Duplicate query_number found: {key}")

        indexed[key] = query_item

    return indexed


def index_intents(
    query_item: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed = {}

    for intent in query_item.get("intents", []):
        key = (
            str(intent.get("intent_title", "")),
            str(intent.get("intent_description", "")),
            str(intent.get("expanded_query", "")),
        )

        if key in indexed:
            raise ValueError(
                "Duplicate intent found for query_number="
                f"{query_item.get('query_number')}: {key[0]}"
            )

        indexed[key] = intent

    return indexed


def to_common_document(
    document: dict[str, Any],
    retrieval_type: str,
) -> dict[str, Any]:
    return {
        "rank": document.get("rank"),
        "title": document.get("title", ""),
        "link": document.get("link", ""),
        "text": document.get("text", ""),
        "retrieval_type": retrieval_type,
    }


def main() -> None:
    bm25_data = load_json(BM25_JSON)
    deep_data = load_json(DEEP_JSON)

    bm25_queries = index_queries(bm25_data)
    deep_queries = index_queries(deep_data)

    bm25_query_numbers = set(bm25_queries)
    deep_query_numbers = set(deep_queries)

    if bm25_query_numbers != deep_query_numbers:
        missing_from_deep = sorted(bm25_query_numbers - deep_query_numbers)
        missing_from_bm25 = sorted(deep_query_numbers - bm25_query_numbers)

        raise ValueError(
            "BM25 and deep files contain different query numbers.\n"
            f"Missing from deep: {missing_from_deep}\n"
            f"Missing from BM25: {missing_from_bm25}"
        )

    combined_results: list[dict[str, Any]] = []

    # Preserve the BM25 file's query order.
    for bm25_query in bm25_data:
        query_number = str(bm25_query["query_number"])
        deep_query = deep_queries[query_number]

        if bm25_query.get("query") != deep_query.get("query"):
            raise ValueError(
                f"Query text mismatch for query_number={query_number}"
            )

        bm25_intents = index_intents(bm25_query)
        deep_intents = index_intents(deep_query)

        if set(bm25_intents) != set(deep_intents):
            raise ValueError(
                f"Intent mismatch for query_number={query_number}"
            )

        combined_query = {
            "query_number": bm25_query["query_number"],
            "query": bm25_query["query"],
            "intents": [],
        }

        # Preserve the BM25 file's intent order.
        for bm25_intent in bm25_query.get("intents", []):
            intent_key = (
                str(bm25_intent.get("intent_title", "")),
                str(bm25_intent.get("intent_description", "")),
                str(bm25_intent.get("expanded_query", "")),
            )
            deep_intent = deep_intents[intent_key]

            combined_documents = [
                to_common_document(document, "bm25")
                for document in bm25_intent.get("top_documents", [])
            ]
            combined_documents.extend(
                to_common_document(document, "deep_retrieval")
                for document in deep_intent.get("top_documents", [])
            )

            combined_query["intents"].append(
                {
                    "intent_title": bm25_intent.get("intent_title", ""),
                    "intent_description": bm25_intent.get(
                        "intent_description",
                        "",
                    ),
                    "expanded_query": bm25_intent.get("expanded_query", ""),
                    "top_documents": combined_documents,
                }
            )

        combined_results.append(combined_query)

    COMBINED_JSON.parent.mkdir(parents=True, exist_ok=True)

    with COMBINED_JSON.open("w", encoding="utf-8") as fout:
        json.dump(combined_results, fout, ensure_ascii=False, indent=2)

    print(f"Saved combined results to {COMBINED_JSON}")


if __name__ == "__main__":
    main()
