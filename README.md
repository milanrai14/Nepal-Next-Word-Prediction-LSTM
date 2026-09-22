# About Nepal Next Text Generation Using LSTM

An end-to-end **About Nepal Text Generation** project using an **LSTM (Long Short-Term Memory)** neural network. The project starts with text preprocessing and LSTM model training in a Jupyter Notebook and continues with model deployment through a **FastAPI backend** and an interactive **Streamlit UI**.

The model learns patterns from a text corpus related to **Nepal** and generates text by predicting the next word repeatedly from a given starting sequence.

---

## 🚀 Project Overview

The project demonstrates the complete workflow of an NLP deep learning application:

```text
Text Corpus
    ↓
Preprocessing
    ↓
Tokenization
    ↓
Sequence Generation
    ↓
Padding
    ↓
LSTM Model Training
    ↓
Model Evaluation
    ↓
Save Model + Tokenizer
    ↓
FastAPI Backend
    ↓
Streamlit UI
    ↓
Text Generation
```

Given starting text such as:

```text
Nepal is
```

the model predicts the next word and continues generating text based on its previous predictions.

The application supports generating **up to 20 words**.

---

## 📂 Project Structure

```text
Nepal Text Generation/
│
├── notebook/
│   └── nepal_text_generation.ipynb
│
└── api/
    │
    ├── main.py
    ├── streamlit.py
    │
    └── model/
        ├── next_word_lstm.keras
        └── tokenizer.pkl
```

### File Description

| File                          | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `nepal_text_generation.ipynb` | Data preprocessing, tokenization, sequence creation, model training and evaluation |
| `main.py`                     | FastAPI backend and LSTM text-generation logic                                     |
| `streamlit.py`                | Streamlit user interface                                                           |
| `next_word_lstm.keras`        | Trained LSTM model                                                                 |
| `tokenizer.pkl`               | Saved tokenizer used for converting text into sequences                            |

---

# 🧠 1. Data Preprocessing

The project begins with a text corpus related to **Nepal**.

The text is cleaned and prepared before training.

The preprocessing process includes:

* Cleaning the text
* Removing unnecessary characters
* Preparing the text corpus
* Tokenizing words
* Converting words into numerical representations
* Creating input-target sequences
* Padding sequences

All of these steps are performed in:

```text
notebook/nepal_text_generation.ipynb
```

---

# 🔤 2. Tokenization

The tokenizer converts words into numerical IDs that can be processed by the neural network.

For example:

```text
Nepal is a beautiful country
```

may become:

```text
[12, 7, 4, 25, 9]
```

The tokenizer maintains a mapping between words and their numerical indices.

After training, the tokenizer is saved as:

```text
api/model/tokenizer.pkl
```

The same tokenizer is required when making predictions because the input must be converted using the same vocabulary used during training.

---

# 🔢 3. Sequence Generation

The text is transformed into sequences for next-word prediction.

For example:

```text
Nepal
Nepal is
Nepal is a
Nepal is a beautiful
```

The model learns relationships such as:

```text
Input                    Target

Nepal                    is
Nepal is                 a
Nepal is a               beautiful
Nepal is a beautiful     country
```

This allows the LSTM to learn which words are likely to follow previous words.

---

# 🧠 4. LSTM Model

The project uses an **LSTM-based language model**.

The general architecture is:

```text
Input Sequence
      ↓
Embedding Layer
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Softmax
      ↓
Next Word
```

The LSTM processes the sequence and learns contextual relationships between words.

---

# 💾 5. Saving the Model and Tokenizer

After training, the trained model is saved as:

```text
api/model/next_word_lstm.keras
```

The tokenizer is saved as:

```text
api/model/tokenizer.pkl
```

For example:

```python
model.save("api/model/next_word_lstm.keras")
```

and:

```python
import dill

with open("api/model/tokenizer.pkl", "wb") as f:
    dill.dump(tokenizer, f)
```

These two files are used by the FastAPI backend during inference.

---

# ⚡ 6. FastAPI Backend

The `main.py` file is responsible for serving the trained LSTM model.

Location:

```text
api/main.py
```

FastAPI handles the prediction process.

It:

1. Loads the trained LSTM model
2. Loads the tokenizer
3. Receives the user's starting text
4. Converts the text into token IDs
5. Pads the sequence
6. Passes it to the LSTM
7. Predicts the next word
8. Adds the predicted word to the text
9. Repeats the process
10. Returns the generated text

---

## API Endpoint

```text
POST /predict
```

### Request

```json
{
    "text": "Nepal is",
    "num_words": 20
}
```

### Response

```json
{
    "input": "Nepal is",
    "generated_text": "Nepal is a beautiful country ..."
}
```

The API limits generation to a maximum of **20 words**.

---

# 🎨 7. Streamlit UI

The `streamlit.py` file provides the application's user interface.

Location:

```text
api/streamlit.py
```

Streamlit is responsible **only for the UI**.

It does not contain the model prediction logic.

The UI allows the user to:

* Enter starting text
* Select the number of words
* Send the request to FastAPI
* Display the generated text

---

# 🔌 8. Streamlit + FastAPI Integration

The Streamlit frontend communicates with the FastAPI backend through HTTP requests.

```text
                    User
                     │
                     ▼
              ┌─────────────┐
              │  Streamlit  │
              │     UI      │
              └──────┬──────┘
                     │
                     │ POST /predict
                     ▼
              ┌─────────────┐
              │   FastAPI   │
              │  main.py    │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Tokenizer  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ LSTM Model  │
              └──────┬──────┘
                     │
                     ▼
              Generated Text
                     │
                     ▼
              ┌─────────────┐
              │  Streamlit  │
              │     UI      │
              └─────────────┘
```

Streamlit sends a request such as:

```json
{
    "text": "Nepal is",
    "num_words": 20
}
```

to:

```text
http://127.0.0.1:8000/predict
```

FastAPI processes the request and returns the generated text.

---

# 🛠️ Installation

Navigate into the `api` directory:

```bash
cd api
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install tensorflow numpy fastapi uvicorn pydantic dill streamlit requests
```

---

# ▶️ Run FastAPI

From the `api` directory:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

You can test the `/predict` endpoint directly through Swagger UI.

---

# ▶️ Run Streamlit

Open another terminal and navigate to the `api` directory:

```bash
cd api
```

Then run:

```bash
streamlit run streamlit.py
```

The Streamlit application provides the graphical interface for interacting with the text-generation API.

---

# 🧪 Example

### Input

```text
Nepal is
```

### Number of Words

```text
20
```

### Generation Process

The model repeatedly predicts the next word:

```text
Nepal is
      ↓
Nepal is a
      ↓
Nepal is a beautiful
      ↓
Nepal is a beautiful country
      ↓
...
```

The final generated text is returned by FastAPI and displayed in Streamlit.

---

# 📏 Sequence Length

The sequence length used during inference must match the sequence length used during model training.

For example:

```python
MAX_LEN = 56
```

The input is padded to this length before being passed to the LSTM model.

---

# 🧰 Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **LSTM**
* **NumPy**
* **Dill**
* **FastAPI**
* **Uvicorn**
* **Pydantic**
* **Streamlit**
* **Jupyter Notebook**

---

# 🎯 Learning Outcomes

This project demonstrates:

* NLP text preprocessing
* Tokenization
* Vocabulary creation
* Sequence generation
* Padding
* Word-level language modeling
* LSTM architecture
* Next-word prediction
* Model serialization
* FastAPI REST API development
* Streamlit UI development
* Frontend-backend integration
* ML model serving
* End-to-end NLP deployment

---

# 🔮 Future Improvements

* Implement temperature sampling
* Add Top-K sampling
* Add Top-P sampling
* Train on a larger Nepal-related corpus
* Experiment with GRU
* Experiment with Transformer-based models
* Improve text preprocessing
* Add API testing
* Dockerize the application
* Deploy FastAPI and Streamlit

---

## 👨‍💻 Author

**Milan Rai**

AI/ML Learner | Python | Deep Learning | NLP

---

## ⭐ Project Summary

**About Nepal Text Generation** demonstrates the complete journey from an NLP experiment to an integrated application:

```text
Jupyter Notebook
       ↓
Text Preprocessing
       ↓
Tokenization
       ↓
LSTM Training
       ↓
Model + Tokenizer
       ↓
FastAPI Backend
       ↓
Streamlit UI
       ↓
Nepal Text Generation
```
