import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
model = joblib.load('text_classification_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess the text by cleaning and normalizing it
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Streamlit App
st.title("Text Sentiment Analysis")

# Create a text input area for the user to input the text
user_input = st.text_area("Enter the text you want to analyze:")

# When the user clicks the 'Predict' button
if st.button("Predict"):
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded")
    elif user_input.strip() == '':
        st.error("Please provide some text for sentiment analysis.")
    else:
        # Preprocess and predict the sentiment
        processed_text = preprocess_text(user_input)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        
        # Display the sentiment prediction
        if prediction == 1:
            sentiment = "Positive"
        elif prediction == -1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        st.success(f"The predicted sentiment is: {sentiment}")
