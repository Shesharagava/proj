import streamlit as st
import joblib
import pandas as pd
from collections import Counter

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set up the Streamlit interface
st.set_page_config(page_title="📈 Stock Sentiment & Risk Analyzer", page_icon="📊")
st.title("📊 Stock Sentiment & Risk Analyzer")
st.write("Enter a news headline to predict its sentiment and assess investment risk.")

# User input
user_input = st.text_area("Enter a stock-related headline here:")

# Prediction
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]

        # Display sentiment
        if prediction == "positive":
            st.success("🟢 Sentiment: Positive")
        elif prediction == "negative":
            st.error("🔴 Sentiment: Negative")
        else:
            st.info("⚪ Sentiment: Neutral")

        # Risk analysis (using sample batch logic)
        sentiment_data = ["positive", "negative", "neutral", prediction]  # Later: replace with full dataset
        count = Counter(sentiment_data)
        total = len(sentiment_data)
        pos_ratio = count.get("positive", 0) / total
        neg_ratio = count.get("negative", 0) / total

        # Risk rule
        if pos_ratio > 0.7:
            risk_level = "🟢 Low Risk"
        elif neg_ratio > 0.6:
            risk_level = "🔴 High Risk"
        else:
            risk_level = "🟡 Medium Risk"

        st.markdown("### 📉 Risk Assessment")
        st.write("Sentiment Summary:", dict(count))
        st.write("Estimated Risk Level:", risk_level)

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
