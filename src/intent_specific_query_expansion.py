import json
import os
import re
from pathlib import Path
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1")
MODEL = os.environ.get(
    "IPSQER_EXPANSION_MODEL",
    os.environ.get("IPSQER_MODEL", "qwen3-235b-a22b-instruct-2507"),
)
TEMPERATURE = 0

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'"
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores_filtered.csv"
OUTPUT_JSON = REPO_ROOT / "output" / "intent_specific_query_expansion" / "intent_specific_expansions.json"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

print("Intent-specific query expansion config:")
print(f"  model: {MODEL}")
print(f"  temperature: {TEMPERATURE}")
print(f"  base_url: {BASE_URL}")
print("  api_key: set")

# Column names
QUERY_NUM_COL = "query_number"
QUERY_COL = "query"
INTENT_TITLE_COL = "intent_title"
INTENT_DESC_COL = "intent_description"
RETAINED_COL = "retained_intent"


def extract_json(text):
    """Extract JSON from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_prompt(query, intent_title, intent_description):
    return f"""
You are an expert search query expansion system.

Your task is to generate an intent-specific query expansion for the given validated intent.

Important:
- The intent has already been selected and validated.
- Do NOT generate other meanings or other intents.
- Do NOT mix interpretations.
- Do NOT explain your reasoning.
- Do NOT mention that the query is ambiguous.
- Keep the expansion tightly focused on this intent only.

What to generate:
- Use terms that are relevant to this intent.
- Include important entities, aliases, abbreviations, and domain-specific terminology when appropriate.
- Use natural search terms and short phrases that improve document retrieval.
- Avoid generic filler words like "information", "guide", "article", "website".
- Do not simply repeat the original query.

Return ONLY valid JSON in exactly this format:

{{
  "query": "{query}",
  "intent_title": "{intent_title}",
  "expanded_query": "..."
}}

Input:
- Original query: {query}
- Intent title: {intent_title}
- Intent description: {intent_description}
""".strip()


# -----------------------------
# Load input
# -----------------------------
df = pd.read_csv(INPUT_CSV)

if RETAINED_COL in df.columns:
    df = df[df[RETAINED_COL] == True].copy()

required_cols = [
    QUERY_NUM_COL,
    QUERY_COL,
    INTENT_TITLE_COL,
    INTENT_DESC_COL
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


# -----------------------------
# Run expansion
# -----------------------------
results = []

for query_number, group in tqdm(
        df.groupby(QUERY_NUM_COL),
        total=df[QUERY_NUM_COL].nunique()):

    query = str(group.iloc[0][QUERY_COL])

    query_result = {
        #"query_number": int(query_number),
        "query_number": query_number,
        "query": query,
        "intents": []
    }

    for _, row in group.iterrows():

        intent_title = str(row[INTENT_TITLE_COL])
        intent_description = str(row[INTENT_DESC_COL])

        try:
            prompt = build_prompt(
                query,
                intent_title,
                intent_description
            )

            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            output = extract_json(
                response.choices[0].message.content
            )

            query_result["intents"].append({
                "intent_title": intent_title,
                "intent_description": intent_description,
                "expanded_query": output["expanded_query"]
            })

        except Exception as e:

            print(f"Error on query {query_number} ({intent_title}): {e}")

            query_result["intents"].append({
                "intent_title": intent_title,
                "intent_description": intent_description,
                "expanded_query": "",
                "error": str(e)
            })

    results.append(query_result)


# -----------------------------
# Save output
# -----------------------------
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} queries to {OUTPUT_JSON}")
