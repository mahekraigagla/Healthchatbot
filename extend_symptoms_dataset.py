import os

import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "dataset")


def build_extended_symptoms(input_csv: str = "Training.csv",
                            output_csv: str = "symptoms_df.csv",
                            max_symptoms: int = 8) -> None:
    """
    Regenerate the symptoms dataset so that for each disease we keep
    up to `max_symptoms` of the most frequently occurring symptoms
    from Training.csv.

    The output format is:
        Disease,Symptom_1,...,Symptom_8
    with exactly one row per disease.
    """
    train_path = os.path.join(DATA_DIR, input_csv)
    out_path = os.path.join(DATA_DIR, output_csv)

    df = pd.read_csv(train_path)
    # Identify the label column and symptom columns
    label_col = "prognosis"
    symptom_cols = [c for c in df.columns if c != label_col]

    rows = []
    for disease, sub in df.groupby(label_col):
        # For each symptom, compute how often it appears (value == 1)
        freq = sub[symptom_cols].sum(axis=0)
        # Sort by frequency (descending) and take top N > 0
        top = [s for s, cnt in freq.sort_values(ascending=False).items() if cnt > 0][:max_symptoms]
        # Pad with empty strings to always have max_symptoms columns
        top += [""] * (max_symptoms - len(top))
        row = {"Disease": disease}
        for i, s in enumerate(top, start=1):
            row[f"Symptom_{i}"] = s
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    build_extended_symptoms()

