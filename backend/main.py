# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib   # ⬅️ yaha joblib use karenge, pickle nahi

# ---------- Request & Response Models ----------

class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


# ---------- FastAPI App ----------

app = FastAPI(
    title="Sentiment Analysis API",
    description="API to predict sentiment (positive/negative) using a Logistic Regression model trained on IMDB reviews.",
    version="1.0.0",
)

model = None  # will hold the loaded model


# ---------- Load model on startup ----------

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("models/sentiment_model.pkl")
    print("✅ Model loaded from models/sentiment_model.pkl")


# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API is running. Go to /docs to test the /predict endpoint."
    }


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """
    Take a text input and return predicted sentiment + confidence score.
    """
    text = request.text

    # 1) Predict label
    pred_label = model.predict([text])[0]

    # 2) Default confidence
    confidence = 0.0

    # 3) If model supports predict_proba (LogisticRegression does)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        class_index = list(model.classes_).index(pred_label)
        confidence = float(proba[class_index])

    return SentimentResponse(
        sentiment=pred_label,
        confidence=round(confidence, 4),
    )
