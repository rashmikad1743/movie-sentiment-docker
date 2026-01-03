# app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Type a movie review below and get **Positive / Negative** sentiment with confidence.")

review_text = st.text_area(
    "Enter your review:",
    height=150,
    placeholder="Example: I absolutely loved this movie. The acting was brilliant!"
)

if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": review_text}
                )
                if response.status_code != 200:
                    st.error(f"Server error: {response.status_code}")
                else:
                    data = response.json()
                    sentiment = data.get("sentiment", "unknown")
                    confidence = data.get("confidence", 0.0) * 100

                    if sentiment.lower() == "positive":
                        st.success(f"✅ Sentiment: **POSITIVE** ({confidence:.2f}% confidence)")
                    elif sentiment.lower() == "negative":
                        st.error(f"❌ Sentiment: **NEGATIVE** ({confidence:.2f}% confidence)")
                    else:
                        st.info(f"Sentiment: {sentiment} ({confidence:.2f}% confidence)")

                    with st.expander("Raw response"):
                        st.json(data)

            except Exception as e:
                st.error("Could not connect to API. Is FastAPI running?")
                st.exception(e)
