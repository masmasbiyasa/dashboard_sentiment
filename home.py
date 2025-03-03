import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model sentimen
SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# Load model emosi
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)

# Mapping label untuk model sentimen
sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

# Mapping label untuk model emosi
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return sentiment_labels[predicted_class]

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return emotion_labels[predicted_class]

# Streamlit UI
st.title("Text Analysis: Sentiment & Emotion Detection")

st.write('model sentimen yang digunakan adalah [NLPTown](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)')
st.write('model emosi yang digunakan adalah [J-Hartmann](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)')

# Tampilkan tabel akurasi model di berbagai bahasa
data_lang = {
    "Language": ["English", "Dutch", "German", "French", "Italian", "Spanish"],
    "Accuracy (exact)": ["67%", "57%", "61%", "59%", "59%", "58%"],
    "Accuracy (off-by-1)": ["95%", "93%", "94%", "94%", "95%", "95%"]
}

st.write("### Model Accuracy per Language darri bert-base-multilingual-uncased-sentiment")
st.table(data_lang)

# Tampilkan tabel informasi model emosi
data_emotion = {
    "Description": ["Observations per emotion", "Total observations", "Training split", "Evaluation split", "Evaluation accuracy", "Random-chance baseline"],
    "Value": ["2,811", "~20k", "80%", "20%", "66%", "14% (1/7)"]
}

st.write("### Emotion Model Training Details dari emotion-english-distilroberta-base")
st.table(data_emotion)

st.write("Masukkan teks di bawah untuk dianalisis.")

user_input = st.text_area("Masukkan teks:")

if st.button("Analisis"):
    if user_input:
        sentiment_result = predict_sentiment(user_input)
        emotion_result = predict_emotion(user_input)

        st.write('### Input teks: ')
        st.write(user_input)

        
        st.write(f"### Hasil Sentimen: {sentiment_result}")
        st.write(f"### Hasil Emosi: {emotion_result}")
        st.error("Perhatikan bahwa model ini mungkin tidak akurat 100%.")
    else:
        st.write("Harap masukkan teks terlebih dahulu!")
