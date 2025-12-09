import os
import json
import joblib
import boto3
import pandas as pd
from flask import Flask, request, jsonify, render_template

BUCKET = "toy-reviews-bucket"  # your S3 bucket
MODEL_KEY = "models/toy_sentiment_model.pkl"
LOCAL_MODEL = "toy_sentiment_model.pkl"

TOP_BOTTOM_CSV = "summary_top_bottom.csv"
ISSUES_CSV = "summary_issues.csv"

DATA_JSON = "Toys_and_Games_5.json"   # raw file we already used


app = Flask(__name__)


def ensure_model():
    """Load the trained model, downloading from S3 if needed."""
    if not os.path.exists(LOCAL_MODEL):
        print(f"Model {LOCAL_MODEL} not found locally. Downloading from S3...")
        s3 = boto3.client("s3")
        s3.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL)
        print("Download complete.")

    print("Loading model...")
    model = joblib.load(LOCAL_MODEL)
    print("Model loaded.")
    return model


# Load model once when the app starts
model = ensure_model()


def load_tables():
    """Load summary tables if they exist, else return empty frames."""
    if os.path.exists(TOP_BOTTOM_CSV):
        top_bottom = pd.read_csv(TOP_BOTTOM_CSV)
    else:
        top_bottom = pd.DataFrame(columns=["asin", "title", "avg_rating",
                                           "avg_sentiment", "review_count", "bucket"])

    if os.path.exists(ISSUES_CSV):
        issues = pd.read_csv(ISSUES_CSV)
    else:
        issues = pd.DataFrame(columns=["asin", "title", "avg_rating",
                                       "avg_sentiment", "rating_vs_sentiment_gap",
                                       "review_count"])

    return top_bottom, issues


@app.route("/")
def home():
    top_bottom, issues = load_tables()

    # Split into top and bottom for display
    top_toys = top_bottom[top_bottom["bucket"] == "Top"].to_dict(orient="records")
    bottom_toys = top_bottom[top_bottom["bucket"] == "Bottom"].to_dict(orient="records")

    issues_list = issues.to_dict(orient="records")

    # optional last prediction to show in the UI
    last_text = request.args.get("text", "")
    last_pred = request.args.get("pred", "")
    last_prob = request.args.get("prob", "")

    return render_template(
        "index.html",
        top_toys=top_toys,
        bottom_toys=bottom_toys,
        issues=issues_list,
        last_text=last_text,
        last_pred=last_pred,
        last_prob=last_prob,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API for programmatic access."""
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Field 'text' is required"}), 400

    proba = model.predict_proba([text])[0][1]  # positive class prob
    label = int(proba >= 0.5)

    return jsonify(
        {
            "input_text": text,
            "positive_probability": float(proba),
            "predicted_label": label,
        }
    )


@app.route("/demo", methods=["POST"])
def demo_form():
    """Handle the form from the Try a Review tab and then reload / with the result."""
    text = request.form.get("review_text", "")

    if not text.strip():
        return home()

    proba = model.predict_proba([text])[0][1]
    label = int(proba >= 0.5)

    # redirect-like behaviour by re-rendering home with query params
    top_bottom, issues = load_tables()
    top_toys = top_bottom[top_bottom["bucket"] == "Top"].to_dict(orient="records")
    bottom_toys = top_bottom[top_bottom["bucket"] == "Bottom"].to_dict(orient="records")
    issues_list = issues.to_dict(orient="records")

    return render_template(
        "index.html",
        top_toys=top_toys,
        bottom_toys=bottom_toys,
        issues=issues_list,
        last_text=text,
        last_pred=label,
        last_prob=f"{proba:.3f}",
    )


if __name__ == "__main__":
    # Listen on all interfaces so the public IP works
    app.run(host="0.0.0.0", port=5000)
