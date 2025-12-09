import os
import json
import joblib
import boto3
import pandas as pd

BUCKET = "toy-reviews-bucket"
DATA_KEY = "Toys_and_Games_5.json"
LOCAL_JSON = "Toys_and_Games_5.json"

MODEL_PATH = "toy_sentiment_model.pkl"

TOP_BOTTOM_CSV = "summary_top_bottom.csv"
ISSUES_CSV = "summary_issues.csv"


def ensure_data():
    if not os.path.exists(LOCAL_JSON):
        print(f"Downloading {DATA_KEY} from s3://{BUCKET} ...")
        s3 = boto3.client("s3")
        s3.download_file(BUCKET, DATA_KEY, LOCAL_JSON)
        print("Download complete.")


def load_sample(n_rows=50000):
    print(f"Loading first {n_rows} rows from JSON...")
    rows = []
    with open(LOCAL_JSON, "r") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            rating = obj.get("overall")
            text = obj.get("reviewText") or ""
            asin = obj.get("asin")
            title = obj.get("summary") or obj.get("reviewText", "")[:60]

            if rating is None or not text.strip() or asin is None:
                continue

            rows.append(
                {
                    "asin": asin,
                    "title": title,
                    "rating": rating,
                    "review_text": text,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows.")
    return df


def main():
    ensure_data()

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    df = load_sample()

    print("Scoring sentiment probabilities...")
    df["sentiment"] = model.predict_proba(df["review_text"]).[:, 1]

    grp = (
        df.groupby(["asin", "title"])
        .agg(
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment", "mean"),
            review_count=("rating", "count"),
        )
        .reset_index()
    )

    # normalize rating to 0–1 (divide by 5) and compute gap
    grp["rating_vs_sentiment_gap"] = grp["avg_sentiment"] - (grp["avg_rating"] / 5.0)

    # Top & bottom toys by sentiment
    top = grp.sort_values("avg_sentiment", ascending=False).head(10).copy()
    top["bucket"] = "Top"

    bottom = grp.sort_values("avg_sentiment", ascending=True).head(10).copy()
    bottom["bucket"] = "Bottom"

    top_bottom = pd.concat([top, bottom], ignore_index=True)
    top_bottom.to_csv(TOP_BOTTOM_CSV, index=False)
    print(f"Saved {TOP_BOTTOM_CSV}")

    # Toys with biggest gap between ratings and sentiment
    issues = grp.sort_values("rating_vs_sentiment_gap", key=lambda s: s.abs(), ascending=False).head(15)
    issues.to_csv(ISSUES_CSV, index=False)
    print(f"Saved {ISSUES_CSV}")

    print("Done.")


if __name__ == "__main__":
    main()