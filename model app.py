import os
import boto3
import joblib
from flask import Flask, request, jsonify

BUCKET = "toy-reviews-bucket"
MODEL_KEY = "models/toy_sentiment_model.pkl"
LOCAL_MODEL_PATH = "toy_sentiment_model.pkl"

app = Flask(__name__)

def download_model_if_needed():
    if os.path.exists(LOCAL_MODEL_PATH):
        app.logger.info("Model already present locally.")
        return
    app.logger.info("Downloading model from S3...")
    s3 = boto3.client("s3")
    s3.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
    app.logger.info("Model download complete.")

def load_model():
    download_model_if_needed()
    app.logger.info("Loading model from disk...")
    model = joblib.load(LOCAL_MODEL_PATH)
    return model

model = load_model()

@app.route("/")
def index():
    return (
        "Toy Reviews Sentiment API is running.<br>"
        "Send POST /predict with JSON {'text': 'your review here'}"
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Missing 'text' in request"}), 400

    proba = model.predict_proba([text])[0]
    pred = int(proba[1] >= 0.5)
    return jsonify({
        "input_text": text,
        "positive_probability": float(proba[1]),
        "predicted_label": pred,  # 1 = positive, 0 = not positive
    })

if __name__ == "__main__":
    # for class / demo use only; production would use gunicorn, etc.
    app.run(host="0.0.0.0", port=5000)