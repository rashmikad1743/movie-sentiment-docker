# 🎯 Sentiment Analysis API using FastAPI & Machine Learning (Dockerized)

A **Machine Learning–powered Sentiment Analysis REST API** built using **FastAPI** and **Docker**.  
The API analyzes **movie reviews** and predicts whether the sentiment is **Positive** or **Negative**, along with a **confidence score**.

---

## 📌 Project Overview

| Feature | Details |
|------|--------|
| **Project Name** | Sentiment Analysis API |
| **Problem Type** | Binary Text Classification |
| **Dataset** | IMDB Movie Reviews (50,000 labeled reviews) |
| **ML Algorithm** | TF-IDF + Logistic Regression |
| **Model Format** | Pickle (`.pkl`) |
| **Backend Framework** | FastAPI |
| **Server** | Uvicorn |
| **Deployment** | Docker |
| **Input** | Text (Movie Review) |
| **Output** | Sentiment (Positive / Negative) + Confidence |
| **Accuracy** | ~90% on Test Data |

---

## 🚀 Features

- ✅ Trained on IMDB Movie Reviews dataset
- ✅ TF-IDF text vectorization
- ✅ Logistic Regression classification
- ✅ Model persistence using Pickle
- ✅ FastAPI for real-time inference
- ✅ Swagger UI for API testing
- ✅ Confidence score using `predict_proba()`
- ✅ Dockerized for easy deployment

---

## 🧠 Tech Stack

| Layer | Technology |
|----|-----------|
| Language | Python |
| ML | scikit-learn, pandas |
| NLP | TF-IDF |
| Model | Logistic Regression |
| API | FastAPI |
| Server | Uvicorn |
| Containerization | Docker |
| Evaluation | Accuracy, Precision, Recall, F1 |

---

## 📁 Project Structure

movie-sentiment-docker/
│
├── data/
│ └── IMDB Dataset.csv
│
├── models/
│ └── sentiment_model.pkl
│
├── model_training.py # Model training script
├── main.py # FastAPI inference service
├── requirements.txt
├── Dockerfile
└── README.md

yaml
Copy code

---

## ⚙️ Local Installation & Setup

### 1️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🧪 Model Training
3️⃣ Train the Model
bash
Copy code
python model_training.py --data "data/IMDB Dataset.csv"
🔍 Training Output Example
yaml
Copy code
Train Accuracy : 0.9304
Test Accuracy  : 0.8999
F1 Score       : 0.9008
📁 Trained model saved at:

bash
Copy code
models/sentiment_model.pkl
🧪 Running the API (Without Docker)
1️⃣ Start FastAPI Server
bash
Copy code
uvicorn main:app --reload
2️⃣ Open in Browser
Swagger UI → http://127.0.0.1:8000/docs

API Root → http://127.0.0.1:8000

🐳 Docker Setup & Usage
1️⃣ Build Docker Image
bash
Copy code
docker build -t sentiment-analysis-api .
2️⃣ Run Docker Container
bash
Copy code
docker run -p 8000:8000 sentiment-analysis-api
3️⃣ Access API
Swagger UI → http://localhost:8000/docs

API Root → http://localhost:8000

✔️ No Python or dependencies needed locally.

📮 API Usage
🔹 Endpoint
http
Copy code
POST /predict
🔹 Request Body (JSON)
json
Copy code
{
  "text": "This movie was absolutely amazing!"
}
🔹 Response (JSON)
json
Copy code
{
  "sentiment": "positive",
  "confidence": 0.9743
}
📊 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Train vs Test evaluation

Example:
yaml
Copy code
Train Accuracy : 93.04%
Test Accuracy  : 89.99%
✔️ Small gap indicates good generalization (no overfitting).

🔍 How It Works
1️⃣ Preprocessing
Convert text to lowercase

Remove stopwords

TF-IDF feature extraction

2️⃣ Model
Logistic Regression

max_iter = 1000

ngram_range = (1,2)

max_features = 20000

3️⃣ Inference
Load .pkl model

Predict sentiment

Return sentiment + confidence

📚 Concepts Used
Logistic Regression

TF-IDF Vectorization

Binary Classification

Train-Test Split

Evaluation Metrics

REST APIs

FastAPI

Docker

🧑‍💻 Sample FastAPI Code
python
Copy code
from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("models/sentiment_model.pkl", "rb"))

@app.post("/predict")
def predict(text: str):
    prediction = model.predict([text])[0]
    confidence = model.predict_proba([text]).max()
    return {
        "sentiment": prediction,
        "confidence": round(confidence, 4)
    }
📝 Future Improvements
Neutral sentiment class

Streamlit / React frontend

Pretrained models (BERT, RoBERTa)

Cloud deployment (AWS / Render / Railway)

Database logging

Advanced NLP preprocessing

🤝 Contributing
Contributions are welcome!
Feel free to open issues or submit pull requests.

🧑‍💼 Author
Rashmika Makwana
GitHub: https://github.com/rashmikad1743
Email: rashmikad1743@email.com
