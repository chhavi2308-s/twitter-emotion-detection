"""
Twitter Emotion Detection using LSTM

How to Run the Project

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python train_model.py

3. Run the web application:
streamlit run app.py

4. Open the browser at:
http://localhost:8501
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Text cleaning (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Load dataset
df = pd.read_csv("dataset/train.txt", sep=";", header=None, names=["text","emotion"])

# CLEAN DATA (important!)
df["clean_text"] = df["text"].apply(clean_text)

texts = df["clean_text"]

# Tokenizer built on cleaned data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

# Label encoder
encoder = LabelEncoder()
encoder.fit(df["emotion"])

# Load trained model
model = load_model("model/emotion_model.h5")


st.title("Twitter Emotion Detection using LSTM")

tweet = st.text_input("Enter a tweet")

if st.button("Predict Emotion"):

    cleaned = clean_text(tweet)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(seq, maxlen=40)

    pred = model.predict(padded)

    emotion = encoder.inverse_transform([np.argmax(pred)])

    st.write("Predicted Emotion:", emotion[0])