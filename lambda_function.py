import boto3
import json

s3 = boto3.client('s3')

BUCKET_NAME = "toy-reviews-bucket"
OBJECT_KEY = "Toys_and_Games_5.json"

def lambda_handler(event, context):
    MAX_LINES = 10000  # safety limit for big file

    response = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
    body = response["Body"]

    cleaned_reviews = []
    for i, line in enumerate(body.iter_lines()):
        if i >= MAX_LINES:
            break
        if not line:
            continue

        try:
            review = json.loads(line.decode("utf-8"))
            cleaned = {
                "product_id": review.get("asin"),
                "rating": review.get("overall"),
                "review_text": review.get("reviewText"),
                "review_date": review.get("reviewTime"),
                "verified": review.get("verified"),
                "reviewer_id": review.get("reviewerID"),
                "summary": review.get("summary"),
            }
            cleaned_reviews.append(cleaned)
        except Exception as e:
            print(f"Error parsing line {i}: {e}")

    print("Sample cleaned reviews:", cleaned_reviews[:3])
    print(f"Total reviews processed (limited): {len(cleaned_reviews)}")

    return {
        "statusCode": 200,
        "body": f"Processed {len(cleaned_reviews)} reviews (first {MAX_LINES} lines)."
    }
