import json
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

BUCKET = "toy-reviews-bucket"           # <-- change if your bucket name is different
DATA_KEY = "Toys_and_Games_5.json"      # <-- change if key is different
LOCAL_DATA = "Toys_and_Games_5.json"
MODEL_PATH = "toy_sentiment_model.pkl"
MODEL_KEY = "models/toy_sentiment_model.pkl"

def download_data_from_s3():
    s3 = boto3.client("s3")
    if not os.path.exists(LOCAL_DATA):
        print(f"Downloading {DATA_KEY} from s3://{BUCKET} ...")
        s3.download_file(BUCKET, DATA_KEY, LOCAL_DATA)
    else:
        print("Local data file already exists, skipping download.")

def load_sample(n_rows=50000):
    print(f"Loading first {n_rows} reviews from JSON ...")
    rows = []
    with open(LOCAL_DATA, "r") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rating = obj.get("overall")
            text = obj.get("reviewText") or ""
            if rating is None or not text.strip():
                continue
            # keep only 1–5 star ratings
            if rating < 1 or rating > 5:
                continue
            rows.append({"rating": rating, "review_text": text})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows.")
    return df

def prepare_labels(df):
    # simple sentiment: 1 if rating >= 4, else 0
    df = df.copy()
    df["label"] = (df["rating"] >= 4).astype(int)
    return df

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    print("Evaluation on test set:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

def upload_model_to_s3():
    s3 = boto3.client("s3")
    print(f"Uploading model to s3://{BUCKET}/{MODEL_KEY} ...")
    s3.upload_file(MODEL_PATH, BUCKET, MODEL_KEY)
    print("Upload complete.")

if __name__ == "__main__":
    download_data_from_s3()
    df = load_sample(n_rows=50000)
    df = prepare_labels(df)
    train_model(df)
    upload_model_to_s3()
    print("All done.")
