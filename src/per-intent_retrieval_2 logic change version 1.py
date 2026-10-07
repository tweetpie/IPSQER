import bisect
from time import perf_counter
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Optional
from time import perf_counter

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

TOTAL_METADATA_TIME = 0.0
TOTAL_ARTICLE_LOOKUP_TIME = 0.0
TOTAL_FULL_FILE_LOAD_TIME = 0.0
TOTAL_ARTICLE_LOOKUPS = 0
TOTAL_FULL_FILE_LOADS = 0
METADATA_OFFSET_FILENAME = "metadata_offsets.npy"


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

_model_load_start = perf_counter()
query_model = SentenceTransformer(MODEL_NAME, device=embedding_device)
query_model.max_seq_length = 512
query_model.eval()

if embedding_device == "cuda":
    query_model.half()
_model_load_elapsed = perf_counter() - _model_load_start

# ============================================================
# Load FAISS index
# ============================================================

print(f"Loading FAISS index: {FAISS_INDEX_PATH}")
_faiss_load_start = perf_counter()
cpu_index = faiss.read_index(str(FAISS_INDEX_PATH))
_faiss_load_elapsed = perf_counter() - _faiss_load_start

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


_faiss_backend_start = perf_counter()
search_index, active_backend, gpu_resources = create_search_index()
_faiss_backend_elapsed = perf_counter() - _faiss_backend_start

# ============================================================
# Query encoding
# ============================================================


TOTAL_ENCODE_TIME = 0.0
TOTAL_FAISS_SEARCH_TIME = 0.0
TOTAL_SEARCH_PASSAGES_TIME = 0.0
ENCODE_CALLS = 0
FAISS_SEARCH_CALLS = 0

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

    global TOTAL_ENCODE_TIME, ENCODE_CALLS

    _encode_start = perf_counter()
    with torch.inference_mode():
        query_vectors = query_model.encode(
            formatted_queries,
            batch_size=min(64, len(formatted_queries)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    TOTAL_ENCODE_TIME += perf_counter() - _encode_start
    ENCODE_CALLS += 1

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
    global TOTAL_METADATA_TIME

    metadata_total_start = perf_counter()
    resolved: list[Optional[dict[str, Any]]] = [None for _ in identifiers]

    requests_by_shard = defaultdict(lambda: defaultdict(list))

    for result_position, global_id in enumerate(identifiers):
        if global_id < 0:
            continue

        shard_index, local_row = locate_global_id(global_id)
        requests_by_shard[shard_index][local_row].append(result_position)

    shard_timings = []

    for shard_index, requested_rows in requests_by_shard.items():
        shard_start = perf_counter()

        shard = metadata_shards[shard_index]
        metadata_path = Path(shard["metadata_path"])
        if not metadata_path.is_absolute():
            metadata_path = REPO_ROOT / metadata_path

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file was not found:\n{metadata_path.resolve()}"
            )

        offset_path = metadata_path.parent / METADATA_OFFSET_FILENAME
        if not offset_path.exists():
            raise FileNotFoundError(
                "Metadata offset index was not found.\n"
                f"Expected: {offset_path.resolve()}\n"
                "Run build_metadata_offsets.py first."
            )

        # Measure loading/opening the offset index separately.
        t0 = perf_counter()
        offsets = np.load(offset_path, mmap_mode="r")
        offset_load_time = perf_counter() - t0

        expected_rows = int(shard["rows"])

        t0 = perf_counter()
        if offsets.shape != (expected_rows,) or offsets.dtype != np.uint64:
            raise RuntimeError(
                f"Invalid metadata offset index: {offset_path.resolve()}\n"
                f"Expected shape=({expected_rows},), dtype=uint64; "
                f"got shape={offsets.shape}, dtype={offsets.dtype}."
            )
        offset_validation_time = perf_counter() - t0

        # Measure opening the metadata file separately.
        t0 = perf_counter()
        fin = metadata_path.open("rb")
        file_open_time = perf_counter() - t0

        seek_read_time = 0.0
        json_decode_time = 0.0
        metadata_assembly_time = 0.0
        records_read = 0

        try:
            for row_number, result_positions in requested_rows.items():
                if row_number < 0 or row_number >= expected_rows:
                    raise IndexError(
                        f"Metadata row {row_number} is outside shard bounds "
                        f"0..{expected_rows - 1}: {metadata_path}"
                    )

                t0 = perf_counter()
                fin.seek(int(offsets[row_number]))
                raw_line = fin.readline()
                seek_read_time += perf_counter() - t0
                records_read += 1

                if not raw_line:
                    raise RuntimeError(
                        f"Could not read metadata row {row_number} "
                        f"from {metadata_path}"
                    )

                t0 = perf_counter()
                try:
                    metadata = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "Invalid metadata record at:\n"
                        f"{metadata_path}\n"
                        f"Row: {row_number + 1}"
                    ) from exc
                json_decode_time += perf_counter() - t0

                t0 = perf_counter()
                global_id = int(shard["global_start"]) + row_number
                metadata["global_id"] = global_id
                metadata["metadata_shard"] = shard["shard_name"]
                metadata["metadata_row"] = row_number

                for result_position in result_positions:
                    resolved[result_position] = metadata
                metadata_assembly_time += perf_counter() - t0
        finally:
            fin.close()

        shard_elapsed = perf_counter() - shard_start
        shard_timings.append(
            {
                "shard_index": shard_index,
                "shard_name": shard["shard_name"],
                "rows_requested": len(requested_rows),
                "records_read": records_read,
                "offset_load": offset_load_time,
                "offset_validate": offset_validation_time,
                "file_open": file_open_time,
                "seek_read": seek_read_time,
                "json_decode": json_decode_time,
                "metadata_assembly": metadata_assembly_time,
                "total": shard_elapsed,
            }
        )

    metadata_total = perf_counter() - metadata_total_start
    TOTAL_METADATA_TIME += metadata_total

    print("\n  metadata resolver breakdown:")
    print(f"    shards touched:       {len(shard_timings)}")
    print(f"    unique rows read:     {sum(x['records_read'] for x in shard_timings)}")
    print(f"    offset index load:    {sum(x['offset_load'] for x in shard_timings):.6f}s")
    print(f"    offset validation:    {sum(x['offset_validate'] for x in shard_timings):.6f}s")
    print(f"    metadata file open:   {sum(x['file_open'] for x in shard_timings):.6f}s")
    print(f"    seek + readline:      {sum(x['seek_read'] for x in shard_timings):.6f}s")
    print(f"    JSON decode:          {sum(x['json_decode'] for x in shard_timings):.6f}s")
    print(f"    metadata assembly:    {sum(x['metadata_assembly'] for x in shard_timings):.6f}s")
    print(f"    resolver total:       {metadata_total:.6f}s")

    # Show the slowest shards so we can see whether a few files dominate.
    for item in sorted(shard_timings, key=lambda x: x["total"], reverse=True)[:5]:
        print(
            "    slow shard: "
            f"{item['shard_name']} | rows={item['records_read']} | "
            f"total={item['total']:.6f}s | "
            f"offset_load={item['offset_load']:.6f}s | "
            f"open={item['file_open']:.6f}s | "
            f"seek_read={item['seek_read']:.6f}s | "
            f"json={item['json_decode']:.6f}s"
        )

    return resolved

# ============================================================
# Passage retrieval
# ============================================================


def search_passages(
    query: str,
    top_k: int = PASSAGE_CANDIDATES,
) -> list[dict[str, Any]]:
    global TOTAL_FAISS_SEARCH_TIME, TOTAL_SEARCH_PASSAGES_TIME, FAISS_SEARCH_CALLS
    _search_passages_start = perf_counter()

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    t0 = perf_counter()
    query_vectors = encode_queries([query])
    encode_time = perf_counter() - t0

    t0 = perf_counter()
    scores, identifiers = search_index.search(query_vectors, top_k)
    faiss_time = perf_counter() - t0
    TOTAL_FAISS_SEARCH_TIME += faiss_time
    FAISS_SEARCH_CALLS += 1

    t0 = perf_counter()
    result_ids = [int(global_id) for global_id in identifiers[0]]
    id_conversion_time = perf_counter() - t0

    t0 = perf_counter()
    metadata_rows = resolve_passage_metadata(result_ids)
    metadata_time = perf_counter() - t0

    t0 = perf_counter()
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
    result_build_time = perf_counter() - t0

    t0 = perf_counter()
    final_results = [
        passage
        for passage in passage_results
        if len((passage.get("contents") or "").strip()) >= 10
    ]
    filtering_time = perf_counter() - t0

    total_time = perf_counter() - _search_passages_start
    TOTAL_SEARCH_PASSAGES_TIME += total_time

    print(
        f"\n  search_passages breakdown:"
        f"\n    encode:          {encode_time:.6f}s"
        f"\n    FAISS search:    {faiss_time:.6f}s"
        f"\n    ID conversion:   {id_conversion_time:.6f}s"
        f"\n    metadata:        {metadata_time:.6f}s"
        f"\n    result building: {result_build_time:.6f}s"
        f"\n    filtering:       {filtering_time:.6f}s"
        f"\n    TOTAL:           {total_time:.6f}s"
    )

    return final_results

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

        global TOTAL_FULL_FILE_LOAD_TIME, TOTAL_FULL_FILE_LOADS
        load_start = perf_counter()
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

        load_elapsed = perf_counter() - load_start
        TOTAL_FULL_FILE_LOAD_TIME += load_elapsed
        TOTAL_FULL_FILE_LOADS += 1
        print(
            f"  Full-file load: {load_elapsed:.3f}s file={source_file}"
        )

        self._file_cache[source_file] = articles_by_id

        while len(self._file_cache) > self.max_cached_files:
            self._file_cache.popitem(last=False)

        return articles_by_id

    def get_article(
        self,
        article_id: str | int,
        source_file: str,
    ) -> Optional[dict[str, Any]]:
        global TOTAL_ARTICLE_LOOKUP_TIME, TOTAL_ARTICLE_LOOKUPS
        lookup_start = perf_counter()
        articles = self._load_source_file(source_file)
        article = articles.get(str(article_id))
        lookup_elapsed = perf_counter() - lookup_start
        TOTAL_ARTICLE_LOOKUP_TIME += lookup_elapsed
        TOTAL_ARTICLE_LOOKUPS += 1
        print(
            f"  Article lookup: {lookup_elapsed:.3f}s "
            f"article_id={article_id} source_file={source_file}"
        )
        return article


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

    print()
    print("========== DENSE RETRIEVAL PERFORMANCE ==========")
    print(f"Model load time          : {_model_load_elapsed:.3f}s")
    print(f"FAISS file load time     : {_faiss_load_elapsed:.3f}s")
    print(f"FAISS backend setup time : {_faiss_backend_elapsed:.3f}s")
    print(f"Active FAISS backend     : {active_backend}")
    print(f"Encode calls             : {ENCODE_CALLS}")
    print(f"Query encode total       : {TOTAL_ENCODE_TIME:.3f}s")
    print(f"FAISS search calls       : {FAISS_SEARCH_CALLS}")
    print(f"FAISS search total       : {TOTAL_FAISS_SEARCH_TIME:.3f}s")
    print(f"search_passages total    : {TOTAL_SEARCH_PASSAGES_TIME:.3f}s")
    print("========== LOOKUP PERFORMANCE ==========")
    print(f"Metadata lookup total : {TOTAL_METADATA_TIME:.3f}s")
    print(f"Article lookup total  : {TOTAL_ARTICLE_LOOKUP_TIME:.3f}s")
    print(f"Article lookups       : {TOTAL_ARTICLE_LOOKUPS}")
    print(f"Full-file load total  : {TOTAL_FULL_FILE_LOAD_TIME:.3f}s")
    print(f"Full-file loads       : {TOTAL_FULL_FILE_LOADS}")
    print("========================================")


if __name__ == "__main__":
    main()