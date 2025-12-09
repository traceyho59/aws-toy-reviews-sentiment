# 🎁 The Best Gift: Toy Reviews Sentiment Analysis

**Columbia University APAN5450 Cloud Computing | Fall 2025**

A cloud-native sentiment analysis system that helps toy companies identify which products to keep producing and which ones might be ready for retirement by analyzing Amazon review data beyond simple star ratings.

## 📋 Project Overview

For decades, toy companies have created thousands of memories for children. With Christmas around the corner, companies need to understand which toys to keep producing and which ones might be ready for retirement. Star ratings alone don't always reflect how customers truly feel about products.

This project builds a **binary sentiment analysis model** that:
- Classifies reviews as positive or negative using NLP
- Identifies products where ratings and sentiment disagree (hidden issues)
- Provides real-time sentiment predictions via a Flask dashboard
- Runs entirely on AWS infrastructure for scalability

## 🏗️ Architecture

The system leverages multiple AWS services following the Well-Architected Framework:

| Service | Purpose |
|---------|---------|
| **S3** | Store raw data (~1GB JSON), cleaned CSVs, and trained model artifacts |
| **Lambda** | Serverless ETL to clean and transform raw JSON to structured CSV |
| **EC2** | ML model training and Flask web application hosting |
| **RDS** | PostgreSQL database for processed review data |
| **VPC** | Network isolation with Security Groups for EC2, Lambda, and RDS |

### Cloud Architecture
```
![Cloud Architecture](aws-toy-reviews-sentiment/Cloud Architecture.png)
```

## 🤖 Machine Learning Model

**Model**: Logistic Regression with TF-IDF vectorization
- **Training Data**: 50,000 Amazon Toy & Games reviews (2014)
- **Features**: TF-IDF with 20,000 max features, unigrams + bigrams
- **Labels**: Binary sentiment (1 = rating ≥ 4 stars, 0 = rating < 4 stars)

### Performance Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative (0) | 0.82 | 0.47 | 0.60 | 1,350 |
| Positive (1) | 0.92 | 0.98 | 0.95 | 8,649 |
| **Accuracy** | | | **0.91** | 9,999 |
| Weighted Avg | 0.91 | 0.91 | 0.90 | 9,999 |

## 📊 Dashboard Features

The Flask dashboard provides three analytical views:

1. **⭐ Top & Bottom Toys** - Products ranked by predicted sentiment score
2. **⚠️ Hidden Issues** - Products with high ratings but low sentiment (quality concerns)
3. **🔍 Try a Review** - Real-time sentiment prediction for any review text

### Key Insight
Products with large gaps between star ratings and sentiment scores reveal hidden issues. For example, a 5-star review stating "not the perfect wagon, especially for the price" shows the sentiment model catches concerns that ratings miss.

## 📁 Repository Structure

```
toy-reviews-sentiment/
├── src/
│   ├── train_model.py         # ML model training pipeline
│   ├── app.py                 # Flask dashboard application
│   ├── model_app.py           # Simple prediction API
│   ├── build_dashboards.py    # Generate summary CSV files
│   └── build_dashboard_data.py # Aggregate product statistics
├── lambda/
│   └── lambda_function.py     # AWS Lambda ETL function
├── templates/
│   └── index.html             # Dashboard HTML template
├── images/
│   └── architecture.png       # AWS architecture diagram
├── docs/
│   └── project_report.pdf     # Full project documentation
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- AWS account with configured credentials
- S3 bucket with Amazon review data

### Installation

```bash
# Clone the repository
git clone https://github.com/traceyho59/toy-reviews-sentiment.git
cd toy-reviews-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
cd src
python train_model.py
```

This will:
1. Download data from S3 (if not present locally)
2. Train the sentiment model on 50,000 reviews
3. Save model to `toy_sentiment_model.pkl`
4. Upload model to S3

### Run the Dashboard

```bash
# Build summary statistics
python build_dashboards.py

# Start Flask app
python app.py
```

Access the dashboard at `http://localhost:5000`

### API Usage

```bash
# Positive review
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My kid loved this toy, it was amazing and fun!"}'

# Response: {"positive_probability": 0.97, "predicted_label": 1}

# Negative review
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This toy broke on the first day and my kid was very upset."}'

# Response: {"positive_probability": 0.37, "predicted_label": 0}
```

## 💰 Cost Analysis

Development costs over one month using AWS Learner Lab:

| AWS Service | Cost |
|-------------|------|
| RDS | $8.36 |
| VPC | $0.85 |
| EC2 Compute | $0.22 |
| EC2 Other | $0.37 |
| S3 | $0.01 |
| **Total** | **$9.81** |

### Projected Production Costs (with 2014-2024 data)

| Cloud Provider | Monthly | Annual | 3-Year |
|----------------|---------|--------|--------|
| **AWS** | $20.88 | $250.85 | $754.54 |
| Azure | $35.00 | $422.00 | $1,452.00 |
| GCP | $40.00 | $480.00 | $1,440.00 |

## 🔒 Security

- **IAM Instance Profiles**: No hardcoded credentials; EC2 uses temporary credentials via metadata service
- **Security Groups**: Least-privilege access (SSH on 22, Flask on 5000, PostgreSQL on 5432)
- **VPC Isolation**: All resources deployed within the same VPC
- **Stateless Inference**: No user data stored; predictions are ephemeral

## 👥 Team

| Member | Responsibilities |
|--------|------------------|
| **Tracey Ho** | Lambda ETL, ML Pipeline, Flask Dashboard, Security Engineering |
| **Wendy Wang** | Data Acquisition, S3 Storage, VPC Networking, RDS Database, Dashboard Design |

## 📚 References

- [Amazon Review Data (2014)](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- Ni, J., Li, J., & McAuley, J. - "Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects"

## 📄 License

This project was created for educational purposes as part of Columbia University's APAN5450 Cloud Computing course.

---

*Built with ❤️ on AWS*
