import json
import os
from pathlib import Path

import pandas as pd
from google import genai


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = BASE_DIR / "optimized_learning_resource_recommendations_200.csv"
OUTPUT_FILE = (
    BASE_DIR / "optimized_learning_resource_recommendations_200_bilingual.csv"
)

TRANSLATED_COLUMNS = [
    "title",
    "topic",
    "difficulty",
    "content_type",
    "learning_style",
    "preferred_topics"
]


def parse_json_response(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def translate_values(client, column, values):
    prompt = f"""
Translate these educational resource values from English to clear,
natural Modern Standard Arabic. Return only a JSON object whose keys are
the original English values and whose values are their Arabic translations.
Preserve course names and technical meaning. Do not add, remove, or merge
keys.

Column: {column}
Values:
{json.dumps(values, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    translations = parse_json_response(response.text)

    missing = [value for value in values if value not in translations]
    if missing:
        raise ValueError(
            f"Gemini omitted translations for {column}: {missing}"
        )

    return {value: str(translations[value]).strip() for value in values}


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Set it before running this one-time translation script."
        )

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_FILE}")

    df = pd.read_csv(SOURCE_FILE)
    client = genai.Client(api_key=api_key)

    for column in TRANSLATED_COLUMNS:
        values = df[column].dropna().astype(str).unique().tolist()
        translations = translate_values(client, column, values)
        df[f"{column}_ar"] = df[column].astype(str).map(translations)
        print(f"Translated {len(values)} unique values in '{column}'.")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved bilingual dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
