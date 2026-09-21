import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "next_word_lstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# ---------- Globals (loaded lazily) ----------
model = None
tokenizer = None
reverse_index = None

WINDOW = 10  # must match training window


def load_artifacts():
    """Load model + tokenizer once."""
    global model, tokenizer, reverse_index

    if model is None:
        model = load_model(MODEL_PATH)

    if tokenizer is None:
        tokenizer = joblib.load(TOKENIZER_PATH)
        reverse_index = {
            i: w for w, i in tokenizer.word_index.items() if w != "<OOV>"
        }

    return model, tokenizer, reverse_index


# ---------- Relevance check ----------
def is_related(text: str) -> bool:
    _, tok, _ = load_artifacts()
    words = text.lower().split()
    if not words:
        return False
    known = [w for w in words if w in tok.word_index and w != "<OOV>"]
    return len(known) > len(words) / 2


# ---------- Prediction ----------
def predict_text(text: str, n_words: int = 20) -> str:
    mdl, tok, rev = load_artifacts()

    if not is_related(text):
        return (
            f"[Not related to corpus] '{text}' has no words "
            f"from the Nepal corpus."
        )

    for _ in range(n_words):
        seq = tok.texts_to_sequences([text])[0][-WINDOW:]
        padded = pad_sequences([seq], maxlen=WINDOW, padding="pre")

        preds = mdl.predict(padded, verbose=0)[0]
        preds[0] = 0  # block padding index

        pos = int(np.argmax(preds))
        word = rev.get(pos)

        if word is None:
            break

        text += " " + word

    return text