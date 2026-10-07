import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

# =================================================
# Config & Hyperparameters
# =================================================
# Controls the trade-off between query relevance and intent diversity.
LAMBDA_PARAM = float(os.environ.get("IPSQER_LAMBDA_PARAM", "0.5"))
print(LAMBDA_PARAM)
# The number of final reranked documents to return per query
OUTPUT_DEPTH_K = 20 

MODEL_NAME = os.environ.get(
    "IPSQER_RERANKER_MODEL_NAME",
    os.environ.get("IPSQER_CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = REPO_ROOT / "output" / "per_intent_retrieval_3" / "combined_results.json"
FILTERED_CSV = REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores_filtered.csv"
OUTPUT_JSON = REPO_ROOT / "output" / "reranking" / "final_reranked_results.json"

print("Reranking config:")
print(f"  model_name: {MODEL_NAME}")
print(f"  device: {DEVICE}")
print(f"  lambda_param: {LAMBDA_PARAM}")
print(f"  output_depth_k: {OUTPUT_DEPTH_K}")

# =================================================
# Helper Functions
# =================================================
def softmax(logits):
    """
    Applies softmax across an array of logits to convert them to probabilities that sum to 1.
    Includes a shift for numerical stability.
    """
    logits = np.array(logits)
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=0)

def is_heading(text: str) -> bool:
    """Determine if a line of text is likely a heading."""
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
    Split an article into paragraphs, merge headings with the next paragraph,
    and prepend the title to each paragraph.
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

def score_article_ce_max(model, target_text: str, title: str, contents: str) -> float:
    """
    Cross-encoder paragraph scoring. Returns the maximum logit score across all paragraphs.
    Target text is either the original query or the intent description.
    """
    paragraphs = split_into_paragraphs(title, contents)

    if not paragraphs:
        fallback = (title + "\n\n" + contents).strip()
        if not fallback:
            return -10.0 # Return a very low logit for empty text
        paragraphs = [fallback]

    pairs = [(target_text, p) for p in paragraphs]

    # Predict scores for all paragraphs
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)

    if isinstance(scores, (float, int)):
        scores = [float(scores)]
    else:
        try:
            scores = scores.tolist()
        except Exception:
            scores = list(scores)

    if not scores:
        return -10.0

    return float(max(scores))

# =================================================
# Main Reranking Logic
# =================================================
def greedy_intent_aware_rerank(candidates, p_d_q, p_d_i, p_i_q, K, lambda_param):
    """
    Executes the O(Kmn) greedy selection loop balancing relevance and intent coverage.
    Formula: score(d) = λ*P(d|q) + (1-λ) * Σ [P(i|q)*P(d|i)*Π(1-P(d_j|i))]
    """
    S_indices = []
    num_intents = len(p_i_q)
    
    # Initialize coverage term: Product of (1 - P(d_j|i)) for d_j in S
    coverage = np.ones(num_intents)
    
    # Keep track of indices that haven't been selected yet
    available_indices = list(range(len(candidates)))
    
    for step in range(min(K, len(candidates))):
        best_score = -float('inf')
        best_idx = -1
        
        for idx in available_indices:
            # Calculate Diversity/Coverage Gain
            diversity_gain = 0
            for i in range(num_intents):
                diversity_gain += p_i_q[i] * p_d_i[i][idx] * coverage[i]
            
            # Total score
            score = lambda_param * p_d_q[idx] + (1 - lambda_param) * diversity_gain
            
            if score > best_score:
                best_score = score
                best_idx = idx
                
        # 1. Select document that maximizes the score
        S_indices.append(best_idx)
        available_indices.remove(best_idx)
        
        # 2. Update the coverage discount for the next iteration based on selected doc
        for i in range(num_intents):
            coverage[i] *= (1 - p_d_i[i][best_idx])
            
    # Return actual document objects in their reranked order
    return [candidates[idx] for idx in S_indices]

def main():
    print(f"Loading Cross-Encoder model on {DEVICE}...")
    model = CrossEncoder(MODEL_NAME, device=DEVICE)
    
    print("Loading data...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    df_filtered = pd.read_csv(FILTERED_CSV)
    
    final_output = []
    
    for item in json_data:
        query_num = item.get("query_number")
        query = item.get("query")
        all_intents = item.get("intents", [])
        
        print(f"\nProcessing Query [{query_num}]: '{query}'")
        
        # --- 1. Filter Intents & Calculate P(i|q) ---
        df_query = df_filtered[df_filtered['query_number'].astype(str) == str(query_num)]
        valid_intents = []
        raw_confidences = []
        
        for intent in all_intents:
            title = intent.get("intent_title", "")
            match = df_query[df_query['intent_title'] == title]
            
            if not match.empty:
                conf_score = float(match.iloc[0]['confidence'])
                valid_intents.append(intent)
                raw_confidences.append(conf_score)
                
        if not valid_intents:
            print(f"No valid intents found in CSV for query {query_num}. Skipping.")
            continue
            
        # Normalize to create valid probability distribution P(i|q) summing to 1
        total_conf = sum(raw_confidences)
        p_i_q = [c / total_conf for c in raw_confidences] if total_conf > 0 else [1.0/len(valid_intents)] * len(valid_intents)
        
        # --- 2. Pool and Deduplicate Candidate Documents ---
        unique_docs_dict = {}
        for intent in valid_intents:
            for doc in intent.get("top_documents", []):
                doc_title = doc.get("title", "")
                if doc_title and doc_title not in unique_docs_dict:
                    unique_docs_dict[doc_title] = doc
                    
        candidates = list(unique_docs_dict.values())
        num_candidates = len(candidates)
        print(f"Total unique candidates to score: {num_candidates}")
        
        if num_candidates == 0:
            continue
            
        # --- 3. Compute P(d|q) with Softmax across documents ---
        print("Computing P(d|q)...")
        raw_d_q_logits = []
        for doc in candidates:
            # Query against Document paragraphs
            max_logit = score_article_ce_max(model, query, doc.get("title", ""), doc.get("text", ""))
            raw_d_q_logits.append(max_logit)
            
        p_d_q = softmax(raw_d_q_logits)
            
        # --- 4. Compute P(d|i) with Softmax across documents ---
        print("Computing P(d|i) matrix...")
        p_d_i = {}
        
        for i, intent in enumerate(valid_intents):
            # Intent description against Document paragraphs
            intent_desc = intent.get("intent_description", intent.get("intent_title", ""))
            raw_d_i_logits = []
            
            for doc in candidates:
                max_logit = score_article_ce_max(model, intent_desc, doc.get("title", ""), doc.get("text", ""))
                raw_d_i_logits.append(max_logit)
                
            # Normalize column so scores for this intent across all docs sum to 1
            p_d_i[i] = softmax(raw_d_i_logits)
                
        # --- 5. Execute Greedy Intent-Aware Reranking ---
        print("Executing greedy reranking...")
        reranked_docs = greedy_intent_aware_rerank(
            candidates=candidates,
            p_d_q=p_d_q,
            p_d_i=p_d_i,
            p_i_q=p_i_q,
            K=OUTPUT_DEPTH_K,
            lambda_param=LAMBDA_PARAM
        )
        
        # --- 6. Format Output ---
        output_docs = []
        for rank, doc in enumerate(reranked_docs, start=1):
            output_docs.append({
                "final_rank": rank,
                "title": doc.get("title"),
                "link": doc.get("link")
            })
            
        final_output.append({
            "query_number": query_num,
            "query": query,
            "reranked_documents": output_docs
        })
        
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nProcess complete. Reranked output saved to '{OUTPUT_JSON}'.")

if __name__ == "__main__":
    main()
