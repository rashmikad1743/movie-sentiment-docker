# 🎯 Sentiment Analysis API using FastAPI & Machine Learning

This project performs **sentiment analysis** on text (movie reviews) and exposes a REST API using **FastAPI**.  
It predicts whether a given text is **Positive** or **Negative**, along with a **confidence score**.

---
## 🧾 Project Overview

| Feature                     | Details                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| Project Name               | Sentiment Analysis API                                                    |
| Problem Type               | Binary Text Classification                                               |
| Dataset                    | IMDB Movie Reviews (50,000 labeled reviews)                              |
| Algorithm                  | TF-IDF + Logistic Regression                                             |
| Model Save Format          | Pickle (`.pkl`)                                                           |
| Deployment Framework       | FastAPI + Uvicorn                                                        |
| Input                      | Text (movie reviews)                                                     |
| Output                     | Sentiment (Positive / Negative) + Confidence Score                       |
| Accuracy Achieved          | ~90% Test Accuracy                                                        |


## 📌 Features

- ✔️ Train ML model on IMDB Movie Reviews dataset  
- ✔️ Text vectorization using **TF–IDF**
- ✔️ Classification model using **Logistic Regression**
- ✔️ Model saved as **pickle (.pkl)** file
- ✔️ FastAPI endpoint for real-time predictions
- ✔️ Swagger UI for API testing (`/docs`)
- ✔️ Handles confidence score using `predict_proba()`

---

## 🧠 Tech Stack

| Layer | Technology |
|------|-----------|
| Language | Python |
| ML Libraries | scikit-learn, pandas, pickle |
| Model | TF-IDF + Logistic Regression |
| Serving | FastAPI |
| Server | Uvicorn |
| Evaluation | Accuracy, Precision, Recall, F1 Score |

---

## 📁 Project Structure

Movie Review Recommendation/
│
├── data/
│ └── IMDB Dataset.csv
│
├── models/
│ └── sentiment_model.pkl
│
├── model_training.py # Training script (TF-IDF + Logistic Regression)
├── main.py # FastAPI app (inference service)
├── requirements.txt
└── README.md

### 1️⃣ Activate virtual environment
```bash
venv\Scripts\activate
2️⃣ Install requirements
bash
Copy code
pip install -r requirements.txt
3️⃣ Train the model
bash
Copy code
python model_training.py --data "data/IMDB Dataset.csv"
4️⃣ Output example
yaml
Copy code
Train Accuracy : 0.9304
Test Accuracy  : 0.8999
F1 Score       : 0.9008
The trained model is saved in:

bash
Copy code
models/sentiment_model.pkl
🧪 Running API (FastAPI)
1️⃣ Start server
bash
Copy code
uvicorn main:app --reload
2️⃣ Open in browser
Swagger UI → http://127.0.0.1:8000/docs

FastAPI root → http://127.0.0.1:8000

📮 API Usage
Endpoint
bash
Copy code
POST /predict
Request (JSON)
json
Copy code
{
  "text": "This movie was absolutely amazing!"
}
Response (JSON)
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

Train vs Test evaluation (to check overfitting)

Example:

yaml
Copy code
Train Accuracy : 93.04%
Test Accuracy  : 89.99%
Since the gap is small (~3%), the model is not overfitting.

🔍 How It Works
1️⃣ Preprocessing
Convert text to lowercase

Stopwords removal

TF-IDF feature extraction

2️⃣ Model
Logistic Regression with:

max_iter=1000

ngram_range=(1,2)

max_features=20000

3️⃣ Inference
Load .pkl model

Predict class

Return class + confidence

📚 Concepts Used
Logistic Regression

TF-IDF vectorization

Binary Classification

Train-Test Split

Cross-Validation (optional)

FastAPI & Pydantic schema

👨‍💻 Sample Code Snippet (FastAPI)
python
Copy code
from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open("models/sentiment_model.pkl", "rb"))

@app.post("/predict")
def predict(text: str):
    prediction = model.predict([text])[0]
    return {"sentiment": prediction}
📝 Future Improvements
Neutral sentiment class (Softmax model)

Streamlit / React frontend

Use pretrained embeddings (BERT, RoBERTa)

Deploy on AWS / Render / Railway

Database logging of predictions

Use spaCy / NLTK for better preprocessing

🤝 Contributing
PRs are welcome!
Feel free to open issues or suggest new features.

🧑‍💼 Author
Your Rashmika Makwana

GitHub: rashmikad1743

Email: rashmikad1743@email.com

