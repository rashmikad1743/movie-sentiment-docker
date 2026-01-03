# 🎬 Movie Sentiment Analysis API (FastAPI + Machine Learning)

![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **Movie Review Sentiment Classifier** that predicts whether a given movie review is **Positive** or **Negative** using **Machine Learning**.  
The trained model is deployed as a **REST API using FastAPI** and is **Dockerized** for easy deployment.

---

## 📌 About the Project

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone behind text.  
In this project, we use the **IMDB Movie Reviews Dataset** and apply **TF-IDF Vectorization** with **Logistic Regression** to classify sentiments.

The trained model is exposed via a **FastAPI endpoint**, allowing real-time predictions.

---

## 🚀 Features

- Binary sentiment classification (Positive / Negative)
- TF-IDF based text feature extraction
- Logistic Regression ML model
- Confidence score using probability prediction
- REST API using FastAPI
- Interactive API testing with Swagger UI
- Docker support for deployment

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn, Pandas  
- **NLP:** TF-IDF Vectorizer  
- **Model:** Logistic Regression  
- **API Framework:** FastAPI  
- **Server:** Uvicorn  
- **Deployment:** Docker  

---

## 📂 Project Structure

<img width="300" height="313" alt="image" src="https://github.com/user-attachments/assets/befc1796-ee53-49c5-9dad-99df4fd2e87f" />




---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rashmikad1743/movie-sentiment-docker.git
cd movie-sentiment-docker
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🧪 Model Training
Run the training script to train and save the model:

bash
Copy code
python model_training.py --data "data/IMDB Dataset.csv"
✅ Model Performance
yaml
Copy code
Train Accuracy : 93.04%
Test Accuracy  : 89.99%
F1 Score       : 90.08%
📁 Trained model saved at:

bash
Copy code
models/sentiment_model.pkl
🚀 Running the FastAPI App (Without Docker)
bash
Copy code
uvicorn main:app --reload
Swagger UI → http://127.0.0.1:8000/docs

API Root → http://127.0.0.1:8000

🐳 Docker Usage
Build Docker Image
bash
Copy code
docker build -t sentiment-api .
Run Docker Container
bash
Copy code
docker run -p 8000:8000 sentiment-api
Access API:

http://localhost:8000/docs

📮 API Endpoint
Predict Sentiment
Endpoint

http
Copy code
POST /predict
Request Body

json
Copy code
{
  "text": "The movie was fantastic and very engaging."
}
Response

json
Copy code
{
  "sentiment": "positive",
  "confidence": 0.97
}
🔍 How It Works
1️⃣ Text Preprocessing
Convert text to lowercase

Remove stopwords

TF-IDF feature extraction

2️⃣ Model Training
Logistic Regression

N-grams (1,2)

Max features: 20,000

3️⃣ Prediction
Load trained .pkl model

Predict sentiment

Return sentiment with confidence

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

✔️ Small train–test gap confirms no overfitting.

🧪 Sample FastAPI Code
python
Copy code
from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("models/sentiment_model.pkl", "rb"))

@app.post("/predict")
def predict(text: str):
    pred = model.predict([text])[0]
    confidence = model.predict_proba([text]).max()
    return {
        "sentiment": pred,
        "confidence": round(confidence, 4)
    }
🔮 Future Enhancements
Add Neutral sentiment class

Use Deep Learning models (BERT, RoBERTa)

Build frontend using Streamlit / React

Deploy on AWS / Render / Railway

Store predictions in database

🤝 Contributing
Contributions are welcome!
Feel free to open issues or submit pull requests.

👨‍💻 Author
Rashmika Makwana
GitHub: https://github.com/rashmikad1743
Email: rashmikad1743@email.com
