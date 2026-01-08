import json
import pandas as pd

def load_and_preprocess_data(path="data/problems_data.jsonl"):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Handle missing values
    text_cols = ["title", "description", "input_description", "output_description"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Combine all text fields
    df["full_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    )

    return df
