from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create FastAPI app
app = FastAPI(title="Next Word Prediction API")

# Load model
model = tf.keras.models.load_model("model/next_word_lstm.keras")

# Load tokenizer
with open("model/tokenzier.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Same max length used during training
MAX_LEN = 56

# Request schema
class TextInput(BaseModel):
    text: str


# Predict next word
def generate_text(seed_text, next_words=20):
    text = seed_text

    for _ in range(next_words):
        # Tokenize current text
        token_list = tokenizer.texts_to_sequences([text])[0]

        # Pad sequence
        token_list = pad_sequences([token_list], maxlen=MAX_LEN, padding="pre")

        # Predict next word
        prediction = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(prediction)

        # Convert index → word
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break

        if next_word == "":
            break

        text += " " + next_word

    return text


@app.get("/")
def home():
    return {
        "message": "LSTM Next Word Prediction API"
    }


@app.post("/predict")
def predict(data: TextInput):
    generated = generate_text(data.text, next_words=20)

    return {
        "input": data.text,
        "generated_text": generated
    }