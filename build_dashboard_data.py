import json
import os
import pandas as pd
from collections import defaultdict
import joblib

DATA_FILE = "Toys_and_Games_5.json"      # same file you used for training
MODEL_PATH = "toy_sentiment_model.pkl"   # same model
OUTPUT_CSV = "dashboard_stats.csv"

# how many reviews to use for the dashboard (keep this reasonable)
MAX_ROWS = 50000

def load_reviews(n_rows=MAX_ROWS):
    rows = []
    with open(DATA_FILE, "r") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = obj.get("asin")
            rating = obj.get("overall")
            text = obj.get("reviewText") or ""

            if asin is None or rating is None or not text.strip():
                continue

            rows.append(
                {
                    "asin": asin,
                    "rating": float(rating),
                    "review_text": text.strip(),
                }
            )
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} reviews for dashboard data.")
    return df

def score_sentiment(df):
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Scoring sentiment probabilities...")
    # predict_proba returns [prob_negative, prob_positive]
    probs = model.predict_proba(df["review_text"])[:, 1]
    df = df.copy()
    df["sentiment_score"] = probs
    return df

def aggregate_by_product(df):
    print("Aggregating by product (asin)...")
    grouped = (
        df.groupby("asin")
        .agg(
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            review_count=("rating", "count"),
            example_review=("review_text", "first"),
        )
        .reset_index()
    )

    # put rating on 0–1 scale to compare to sentiment
    grouped["rating_scaled"] = grouped["avg_rating"] / 5.0
    grouped["rating_minus_sentiment"] = grouped["rating_scaled"] - grouped["avg_sentiment"]

    # nice rounding
    for col in ["avg_rating", "avg_sentiment", "rating_scaled", "rating_minus_sentiment"]:
        grouped[col] = grouped[col].round(3)

    print(f"Aggregated into {len(grouped)} products.")
    return grouped

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Make sure it is in the same folder as this script."
        )

    df = load_reviews()
    df = score_sentiment(df)
    products = aggregate_by_product(df)
    products.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dashboard data to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
