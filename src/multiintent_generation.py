import json
import os
from pathlib import Path
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1")
MODEL = os.environ.get(
    "IPSQER_MULTIINTENT_MODEL",
    os.environ.get("IPSQER_MODEL", "qwen3-235b-a22b-instruct-2507"),
)
TEMPERATURE = 0
N = int(os.environ.get("IPSQER_MAX_GENERATED_INTENTS", "5"))

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'"
    )

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

print("Multi-intent generation config:")
print(f"  model: {MODEL}")
print(f"  max_generated_intents: {N}")
print(f"  temperature: {TEMPERATURE}")
print(f"  base_url: {BASE_URL}")
print("  api_key: set")

# Input and output files
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "queries" / "combined_queries.csv"
OUTPUT_JSON = REPO_ROOT / "output" / "multiintent_generation" / "query_subtopics_list.json"
SINGLE_QUERY = os.environ.get("IPSQER_SINGLE_QUERY")
SINGLE_QUERY_NUMBER = os.environ.get("IPSQER_SINGLE_QUERY_NUMBER", "single-1")


def build_prompt(query, n):
    return f"""
You are an expert search query analyst.

Your task is to analyze a search query and identify the possible search topics that represent what a user might be looking for.

The query can be:

1. Ambiguous
   - The same query can refer to multiple different entities or concepts.
   - Example:
     - python → programming language, snake
     - jaguar → animal, car brand, NFL team
     - mercury → planet, chemical element, Roman god

2. Faceted
   - The query has one meaning but users may search for different aspects.
   - Example:
     - climate change → causes, effects, mitigation, policies
     - machine learning → fundamentals, algorithms, applications

3. Specific
   - The query has only one clear information need.

Decision process (VERY IMPORTANT)

Step 1.
Determine whether the query has multiple DISTINCT interpretations.
If yes, return those interpretations.
DO NOT split one interpretation into multiple aspects.

Step 2.
If the query has only one interpretation, determine whether users may search for different major aspects.
If yes, return those aspects.

Step 3.
If neither applies, return exactly one search topic.

Priority:
Distinct interpretations > Major aspects > Single information need

Rules

- Return at most {n} search topics.
- Return fewer if appropriate.
- Never invent obscure or unrealistic interpretations.
- Do not rewrite the query.
- Do not generate expansions.
- Each search topic should be independent.
- Rank topics from most likely to least likely.
- Confidence values must be between 0 and 1 and approximately sum to 1.

Retrieval Keywords

For every information need generate 5–10 retrieval keywords or short phrases.

The retrieval keywords should:

- maximize retrieval effectiveness
- contain important entities
- contain common aliases
- contain abbreviations when appropriate
- contain domain-specific terminology
- include both single words and short phrases
- avoid generic words such as "information", "guide", "article", "website", etc.
- not simply repeat the original query unless necessary

Return ONLY valid JSON.

Output format

{{
  "query": "<original query>",
  "information_needs": [
    {{
      "id": 1,
      "title": "...",
      "description": "...",
      "confidence": 0.0,
      "retrieval_keywords": [
        "...",
        "...",
        "...",
        "...",
        "..."
      ]
    }}
  ]
}}

Query:
{query}
"""


if SINGLE_QUERY:
    df = pd.DataFrame(
        [
            {
                "Query Number": SINGLE_QUERY_NUMBER,
                "Query": SINGLE_QUERY,
            }
        ]
    )
    print(f"Running single query: {SINGLE_QUERY}")
else:
    df = pd.read_csv(INPUT_CSV)

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    query_id = row["Query Number"]
    query = row["Query"]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(query, N)
                }
            ]
        )

        output = json.loads(response.choices[0].message.content)

        if "information_needs" in output:
            information_needs = output["information_needs"]
        elif "information_nees" in output:
            print(
                f"WARNING: Model returned 'information_nees' "
                f"for {query_id}. Correcting to 'information_needs'."
            )
            information_needs = output["information_nees"]
        else:
            raise ValueError(
                f"Missing 'information_needs'. Returned keys: {list(output.keys())}"
            )

        results.append({
            "query_number": query_id,
            "query": query,
            "information_needs": information_needs
        })

    except Exception as e:
        print(f"Error on {query_id}: {e}")

        results.append({
            "query_number": query_id,
            "query": query,
            "information_needs": [],
            "error": str(e)
        })

# Save JSON
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} queries to {OUTPUT_JSON}")
