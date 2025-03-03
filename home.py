import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model langsung dari Hugging Face
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Mapping label untuk model nlptown
labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return labels[predicted_class]

# Streamlit UI
st.title("Sentiment Analysis dengan BERT Multilingual")
st.write("Model ini menggunakan BERT Multilingual dari [NLPTown](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) untuk menganalisis sentimen dari teks yang dimasukkan.")
st.write("Masukkan teks di bawah untuk menganalisis sentimennya.")
st.write("Model yang digunakan hanya bisa memprediksi sentiment dari bahasa : English, Dutch, German, French, Italian, Spanish")
data = {
    "Language": ["English", "Dutch", "German", "French", "Italian", "Spanish"],
    "Accuracy (exact)": ["67%", "57%", "61%", "59%", "59%", "58%"],
    "Accuracy (off-by-1)": ["95%", "93%", "94%", "94%", "95%", "95%"]
    }

st.write("### Model Accuracy per Language")
st.table(data)  

user_input = st.text_area("Masukkan teks:")

if st.button("Analisis Sentimen"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"### Hasil Sentimen: {sentiment}")
    else:
        st.write("Harap masukkan teks terlebih dahulu!")
