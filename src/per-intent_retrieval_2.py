import bisect
import json
import mmap
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
 
# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_JSON = REPO_ROOT / "output" / "intent_specific_query_expansion" / "intent_specific_expansions.json"
OUTPUT_JSON = REPO_ROOT / "output" / "per_intent_retrieval_2" / "deep_results.json"

FAISS_INDEX_PATH = REPO_ROOT / "wiki_faiss_gpu" / "wiki_e5_ivfpq.faiss"
OFFSETS_PATH = REPO_ROOT / "wiki_faiss_gpu" / "shard_offsets.json"
ORIGINAL_ARTICLE_DIR = REPO_ROOT / "wiki_output"
GLOBAL_METADATA_PATH = REPO_ROOT / "wiki_faiss_gpu" / "all_metadata.jsonl"
GLOBAL_METADATA_OFFSETS_PATH = REPO_ROOT / "wiki_faiss_gpu" / "all_metadata_offsets.npy"

MODEL_NAME = "intfloat/e5-base-v2"

SEARCH_BACKEND = "auto"  # "auto", "gpu", or "cpu"
MODEL_DEVICE = "auto"    # "auto", "cuda", or "cpu"
GPU_ID = 0
NPROBE = 64

# Number of unique articles returned per intent.
K = 20

# Number of passage candidates retrieved before article deduplication.
PASSAGE_CANDIDATES = 100

ARTICLE_FILE_CACHE_SIZE = 16


def clean_query(text: Any) -> str:
    """Remove commas and normalize whitespace."""
    return " ".join(str(text).replace(",", " ").split()).strip()


# ============================================================
# Validate required files
# ============================================================

for required_path, label in (
    (INPUT_JSON, "Input JSON"),
    (FAISS_INDEX_PATH, "FAISS index"),
    (OFFSETS_PATH, "Shard-offset mapping"),
    (ORIGINAL_ARTICLE_DIR, "Original article directory"),
    (GLOBAL_METADATA_PATH, "Global metadata file"),
    (GLOBAL_METADATA_OFFSETS_PATH, "Global metadata offset index"),
):
    if not required_path.exists():
        raise FileNotFoundError(f"{label} was not found:\n{required_path.resolve()}")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load FAISS-to-metadata mapping
# ============================================================

with OFFSETS_PATH.open("r", encoding="utf-8") as fin:
    offset_data = json.load(fin)

metadata_shards = offset_data["shards"]
total_vectors = int(offset_data["total_vectors"])
embedding_dimension = int(offset_data["dimension"])

global_end_boundaries = [
    int(shard["global_end"])
    for shard in metadata_shards
]

print(f"Mapped vectors   : {total_vectors:,}")
print(f"Metadata shards  : {len(metadata_shards):,}")
print(f"Vector dimension : {embedding_dimension}")

# Load the global metadata offset table once and keep the metadata file
# memory-mapped for the lifetime of the retrieval process.
global_metadata_offsets = np.load(GLOBAL_METADATA_OFFSETS_PATH, mmap_mode="r")

if global_metadata_offsets.dtype != np.uint64 or global_metadata_offsets.shape != (total_vectors,):
    raise RuntimeError(
        "Global metadata offset index does not match total_vectors: "
        f"shape={global_metadata_offsets.shape}, dtype={global_metadata_offsets.dtype}, "
        f"expected shape=({total_vectors},), dtype=uint64"
    )

global_metadata_file = GLOBAL_METADATA_PATH.open("rb")
global_metadata_mmap = mmap.mmap(
    global_metadata_file.fileno(),
    0,
    access=mmap.ACCESS_READ,
)

# ============================================================
# Select query-embedding device
# ============================================================


def select_model_device() -> str:
    if MODEL_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL_DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE='cuda', but CUDA is unavailable.")
        return "cuda"

    if MODEL_DEVICE == "cpu":
        return "cpu"

    raise ValueError("MODEL_DEVICE must be 'auto', 'cuda', or 'cpu'.")


embedding_device = select_model_device()
print(f"Query model device: {embedding_device}")

query_model = SentenceTransformer(MODEL_NAME, device=embedding_device)
query_model.max_seq_length = 512
query_model.eval()

if embedding_device == "cuda":
    query_model.half()

# ============================================================
# Load FAISS index
# ============================================================

print(f"Loading FAISS index: {FAISS_INDEX_PATH}")
cpu_index = faiss.read_index(str(FAISS_INDEX_PATH))

if not cpu_index.is_trained:
    raise RuntimeError("The FAISS index is not trained.")

if int(cpu_index.ntotal) != total_vectors:
    raise RuntimeError(
        "FAISS index and offset mapping disagree:\n"
        f"Index vectors: {cpu_index.ntotal:,}\n"
        f"Mapped vectors: {total_vectors:,}"
    )

cpu_index.nprobe = NPROBE


def faiss_gpu_is_available() -> bool:
    return (
        hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "get_num_gpus")
        and faiss.get_num_gpus() > GPU_ID
    )


def create_search_index():
    if SEARCH_BACKEND not in {"auto", "cpu", "gpu"}:
        raise ValueError("SEARCH_BACKEND must be 'auto', 'cpu', or 'gpu'.")

    if SEARCH_BACKEND == "cpu":
        print("FAISS search backend: CPU")
        return cpu_index, "cpu", None

    gpu_available = faiss_gpu_is_available()

    if SEARCH_BACKEND == "gpu" and not gpu_available:
        raise RuntimeError(
            "SEARCH_BACKEND='gpu', but GPU-enabled FAISS is unavailable."
        )

    if not gpu_available:
        print("GPU FAISS unavailable; using CPU search.")
        return cpu_index, "cpu", None

    try:
        print(f"Copying FAISS index to GPU {GPU_ID}...")

        gpu_resources = faiss.StandardGpuResources()
        clone_options = faiss.GpuClonerOptions()
        clone_options.useFloat16 = True

        if hasattr(faiss, "INDICES_64_BIT"):
            clone_options.indicesOptions = faiss.INDICES_64_BIT

        gpu_index = faiss.index_cpu_to_gpu(
            gpu_resources,
            GPU_ID,
            cpu_index,
            clone_options,
        )
        gpu_index.nprobe = NPROBE

        print("FAISS search backend: GPU")
        return gpu_index, "gpu", gpu_resources

    except Exception as exc:
        if SEARCH_BACKEND == "gpu":
            raise

        print("Could not copy the FAISS index to GPU. Using CPU instead.")
        print(f"Reason: {exc}")
        return cpu_index, "cpu", None


search_index, active_backend, gpu_resources = create_search_index()

# ============================================================
# Query encoding
# ============================================================


def encode_queries(queries: list[str]) -> np.ndarray:
    if not queries:
        raise ValueError("At least one query is required.")

    formatted_queries = []

    for query in queries:
        cleaned = str(query).strip()

        if not cleaned:
            raise ValueError("Query text cannot be empty.")

        if cleaned.lower().startswith("query:"):
            formatted_queries.append(cleaned)
        else:
            formatted_queries.append("query: " + cleaned)

    with torch.inference_mode():
        query_vectors = query_model.encode(
            formatted_queries,
            batch_size=min(64, len(formatted_queries)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    expected_shape = (len(queries), embedding_dimension)

    if query_vectors.shape != expected_shape:
        raise RuntimeError(
            "Unexpected query-embedding shape:\n"
            f"Received: {query_vectors.shape}\n"
            f"Expected: {expected_shape}"
        )

    return query_vectors

# ============================================================
# Global FAISS ID -> metadata-shard position
# ============================================================


def locate_global_id(global_id: int) -> tuple[int, int]:
    if global_id < 0 or global_id >= total_vectors:
        raise IndexError(
            f"Global ID {global_id} is outside the valid range "
            f"0..{total_vectors - 1}."
        )

    shard_index = bisect.bisect_right(global_end_boundaries, global_id)
    shard = metadata_shards[shard_index]
    local_row = global_id - int(shard["global_start"])
    return shard_index, local_row

# ============================================================
# Read passage metadata for FAISS results
# ============================================================


def resolve_passage_metadata(
    identifiers: list[int],
) -> list[Optional[dict[str, Any]]]:
    resolved: list[Optional[dict[str, Any]]] = [None for _ in identifiers]

    valid_requests = []

    for result_position, global_id in enumerate(identifiers):
        if global_id < 0:
            continue

        if global_id >= total_vectors:
            raise IndexError(
                f"Global ID {global_id} is outside the valid range "
                f"0..{total_vectors - 1}."
            )

        valid_requests.append((int(global_id), result_position))

    # Read records in physical file order to reduce random I/O, then place
    # each record back into its original FAISS result position.
    ordered_requests = sorted(
        valid_requests,
        key=lambda item: int(global_metadata_offsets[item[0]]),
    )

    for global_id, result_position in ordered_requests:
        offset = int(global_metadata_offsets[global_id])
        global_metadata_mmap.seek(offset)
        raw_line = global_metadata_mmap.readline()

        if not raw_line:
            raise RuntimeError(
                f"Could not read global metadata record for global ID {global_id}."
            )

        try:
            metadata = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid global metadata record for global ID {global_id}."
            ) from exc

        # Preserve the metadata fields produced by the original resolver.
        shard_index, local_row = locate_global_id(global_id)
        shard = metadata_shards[shard_index]
        metadata["global_id"] = global_id
        metadata["metadata_shard"] = shard["shard_name"]
        metadata["metadata_row"] = local_row

        resolved[result_position] = metadata

    return resolved

# ============================================================
# Passage retrieval
# ============================================================


def search_passages(
    query: str,
    top_k: int = PASSAGE_CANDIDATES,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_vectors = encode_queries([query])
    scores, identifiers = search_index.search(query_vectors, top_k)

    result_ids = [int(global_id) for global_id in identifiers[0]]
    metadata_rows = resolve_passage_metadata(result_ids)

    passage_results = []

    for rank, (global_id, score, metadata) in enumerate(
        zip(result_ids, scores[0], metadata_rows),
        start=1,
    ):
        if global_id < 0 or metadata is None:
            continue

        passage_results.append(
            {
                "passage_rank": rank,
                "score": float(score),
                "global_id": global_id,
                "passage_id": metadata.get("passage_id"),
                "article_id": metadata.get("article_id"),
                "paragraph_id": metadata.get("paragraph_id"),
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "contents": metadata.get("contents"),
                "source_file": metadata.get("source_file"),
                "source_line": metadata.get("source_line"),
            }
        )

    return [
        passage
        for passage in passage_results
        if len((passage.get("contents") or "").strip()) >= 10
    ]

# ============================================================
# Original full-article store
# ============================================================


class FullArticleStore:
    def __init__(
        self,
        original_article_dir: Path,
        max_cached_files: int = 16,
    ):
        self.original_article_dir = original_article_dir
        self.max_cached_files = max_cached_files
        self._file_cache: OrderedDict[
            str,
            dict[str, dict[str, Any]],
        ] = OrderedDict()

    def _load_source_file(
        self,
        source_file: str,
    ) -> dict[str, dict[str, Any]]:
        if not source_file:
            raise ValueError("source_file cannot be empty.")

        source_file = str(source_file)

        if source_file in self._file_cache:
            cached = self._file_cache.pop(source_file)
            self._file_cache[source_file] = cached
            return cached

        original_path = self.original_article_dir / source_file

        if not original_path.exists():
            raise FileNotFoundError(
                f"Original article file was not found:\n{original_path.resolve()}"
            )

        articles_by_id: dict[str, dict[str, Any]] = {}

        with original_path.open("r", encoding="utf-8") as fin:
            for source_line, line in enumerate(fin):
                clean_line = line.strip()

                if not clean_line:
                    continue

                try:
                    article = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue

                article_id = article.get("id")

                if article_id is None:
                    continue

                article_id = str(article_id)
                article["_source_file"] = source_file
                article["_source_line"] = source_line
                articles_by_id[article_id] = article

        self._file_cache[source_file] = articles_by_id

        while len(self._file_cache) > self.max_cached_files:
            self._file_cache.popitem(last=False)

        return articles_by_id

    def get_article(
        self,
        article_id: str | int,
        source_file: str,
    ) -> Optional[dict[str, Any]]:
        articles = self._load_source_file(source_file)
        return articles.get(str(article_id))


article_store = FullArticleStore(
    original_article_dir=ORIGINAL_ARTICLE_DIR,
    max_cached_files=ARTICLE_FILE_CACHE_SIZE,
)

# ============================================================
# Deduplicate passages into ranked articles
# ============================================================


def passages_to_top_articles(
    passage_results: list[dict[str, Any]],
    top_n: int = K,
    load_full_articles: bool = True,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    seen_article_ids = set()
    top_articles = []

    for passage in passage_results:
        raw_article_id = passage.get("article_id")

        if raw_article_id is None:
            continue

        article_id = str(raw_article_id)

        if article_id in seen_article_ids:
            continue

        seen_article_ids.add(article_id)

        source_file = passage.get("source_file")
        full_article = None

        if load_full_articles and source_file:
            try:
                full_article = article_store.get_article(
                    article_id=article_id,
                    source_file=source_file,
                )
            except Exception as exc:
                print(
                    f"Warning: full article lookup failed for "
                    f"article_id={article_id}: {exc}"
                )

        full_title = ""
        full_link = ""
        full_text = ""

        if full_article is not None:
            full_title = full_article.get("title", "")
            full_link = full_article.get("url", full_article.get("link", ""))
            full_text = full_article.get("contents", full_article.get("text", ""))

        top_articles.append(
            {
                "rank": len(top_articles) + 1,
                "title": full_title or passage.get("title") or "",
                "link": full_link or passage.get("url") or "",
                "text": full_text,
                "deep_score": float(passage["score"]),
                "best_paragraph_text": passage.get("contents") or "",
            }
        )

        if len(top_articles) >= top_n:
            break

    return top_articles

# ============================================================
# Complete query -> top unique articles
# ============================================================


def search_top_articles(
    query: str,
    top_articles: int = K,
    passage_candidates: int = PASSAGE_CANDIDATES,
    load_full_articles: bool = True,
) -> list[dict[str, Any]]:
    if passage_candidates < top_articles:
        passage_candidates = top_articles

    passage_results = search_passages(
        query=query,
        top_k=passage_candidates,
    )

    return passages_to_top_articles(
        passage_results=passage_results,
        top_n=top_articles,
        load_full_articles=load_full_articles,
    )

# ============================================================
# Batch run over intent JSON
# ============================================================


def main() -> None:
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
            f"Processing deep query {query_index}/{len(queries)} "
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

            if retrieval_query:
                top_documents = search_top_articles(
                    query=retrieval_query,
                    top_articles=K,
                    passage_candidates=PASSAGE_CANDIDATES,
                    load_full_articles=True,
                )
            else:
                top_documents = []

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

    print(f"Saved deep retrieval results to {OUTPUT_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()