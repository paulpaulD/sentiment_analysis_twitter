import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
model = joblib.load('text_classification_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
st.title("Text Sentiment Analysis")
user_input = st.text_area("Enter the text you want to analyze:")
if st.button("Predict"):
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded")
    elif user_input.strip() == '':
        st.error("Please provide some text for sentiment analysis.")
    else:
        processed_text = preprocess_text(user_input)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        if prediction == 1:
            sentiment = "Positive"
        elif prediction == -1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        st.success(f"The predicted sentiment is: {sentiment}")
