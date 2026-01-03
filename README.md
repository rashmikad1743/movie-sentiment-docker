# 🎬 Movie Sentiment Analysis API

A production-ready sentiment analysis API built with FastAPI and scikit-learn for classifying movie reviews as positive or negative.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Model Training](#-model-training)
- [Running the Application](#-running-the-application)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)
- [How It Works](#-how-it-works)
- [Performance Metrics](#-performance-metrics)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Author](#-author)

---

## ✨ Features

- 🎯 **High Accuracy**: 89.99% test accuracy with F1 score of 90.08%
- ⚡ **Fast API**: Built with FastAPI for high-performance predictions
- 🐳 **Docker Ready**: Containerized for easy deployment
- 📊 **Confidence Scores**: Returns prediction confidence for each result
- 🔧 **Easy Training**: Simple script to retrain with your own data
- 📝 **Interactive Docs**: Auto-generated Swagger UI documentation

---

## 📁 Project Structure
```
movie-sentiment-docker/
│
├── 📂 data/
│   └── IMDB Dataset.csv          # Training dataset
│
├── 📂 models/
│   └── sentiment_model.pkl       # Trained model (generated after training)
│
├── 📂 app/
│   ├── __init__.py
│   ├── model.py                  # Model loading and prediction logic
│   ├── preprocessing.py          # Text preprocessing utilities
│   └── schemas.py                # Pydantic models for request/response
│
├── main.py                       # FastAPI application entry point
├── model_training.py             # Model training script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── .dockerignore                 # Docker ignore file
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

### 📝 File Descriptions

| File/Folder | Description |
|-------------|-------------|
| `data/` | Contains the IMDB dataset for training |
| `models/` | Stores the trained model pickle file |
| `app/` | Core application logic and utilities |
| `main.py` | FastAPI application with API endpoints |
| `model_training.py` | Script to train and evaluate the model |
| `requirements.txt` | List of all Python dependencies |
| `Dockerfile` | Instructions for building Docker image |
| `README.md` | Project documentation (this file) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Clone the Repository
```bash
git clone https://github.com/rashmikad1743/movie-sentiment-docker.git
cd movie-sentiment-docker
```

---

## 📦 Installation

### 1️⃣ Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Training

Train the sentiment analysis model using the IMDB dataset:
```bash
python model_training.py --data "data/IMDB Dataset.csv"
```

### ✅ Model Performance

| Metric | Score |
|--------|-------|
| **Train Accuracy** | 93.04% |
| **Test Accuracy** | 89.99% |
| **F1 Score** | 90.08% |

> ✔️ Small train-test gap confirms no overfitting

**Trained model saved at:** `models/sentiment_model.pkl`

---

## 🚀 Running the Application

### Without Docker

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

**Access the application:**

- 📄 **Swagger UI (Interactive Docs)**: http://127.0.0.1:8000/docs
- 🌐 **API Root**: http://127.0.0.1:8000

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t sentiment-api .
```

### Run Docker Container
```bash
docker run -p 8000:8000 sentiment-api
```

**Access the containerized API:**

- 📄 **Swagger UI**: http://localhost:8000/docs
- 🌐 **API Root**: http://localhost:8000

---

## 📮 API Documentation

### Predict Sentiment Endpoint

**Endpoint:** `POST /predict`

#### Request Body
```json
{
  "text": "The movie was fantastic and very engaging."
}
```

#### Response
```json
{
  "sentiment": "positive",
  "confidence": 0.97
}
```

### Example cURL Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie was fantastic and very engaging."}'
```

---

## 🔍 How It Works

### 1️⃣ Text Preprocessing

- Convert text to lowercase
- Remove stopwords
- TF-IDF feature extraction

### 2️⃣ Model Training

- **Algorithm**: Logistic Regression
- **N-grams**: (1, 2)
- **Max Features**: 20,000
- **Dataset**: IMDB Movie Reviews

### 3️⃣ Prediction Pipeline

1. Load trained `.pkl` model
2. Preprocess input text
3. Predict sentiment
4. Return sentiment label with confidence score

---

## 📊 Performance Metrics

The model is evaluated using multiple metrics to ensure reliability:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1 Score**: Harmonic mean of precision and recall

---

## 🧪 Sample FastAPI Code
```python
from fastapi import FastAPI
import pickle

app = FastAPI()

# Load trained model
model = pickle.load(open("models/sentiment_model.pkl", "rb"))

@app.post("/predict")
def predict(text: str):
    # Make prediction
    pred = model.predict([text])[0]
    confidence = model.predict_proba([text]).max()
    
    return {
        "sentiment": pred,
        "confidence": round(confidence, 4)
    }
```

---

## 🔮 Future Enhancements

- [ ] Add **Neutral** sentiment classification
- [ ] Implement Deep Learning models (BERT, RoBERTa, DistilBERT)
- [ ] Build frontend interface using Streamlit or React
- [ ] Deploy on cloud platforms (AWS, Render, Railway)
- [ ] Store predictions in database (PostgreSQL/MongoDB)
- [ ] Add batch prediction endpoint
- [ ] Implement caching for improved performance
- [ ] Add comprehensive logging and monitoring

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🔧 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

Please feel free to open issues for bug reports or feature requests!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Rashmika Makwana**

- 🐙 GitHub: [@rashmikad1743](https://github.com/rashmikad1743)
- 📧 Email: rashmikad1743@email.com

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

<div align="center">
  <p>Made with ❤️ and Python</p>
  <p>© 2024 Rashmika Makwana</p>
</div>
