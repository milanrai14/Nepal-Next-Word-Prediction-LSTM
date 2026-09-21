from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from schemas import PredictRequest, PredictResponse
from next_prediction import load_artifacts, predict_text, is_related


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + tokenizer once at startup
    load_artifacts()
    print("✅ Model & tokenizer loaded")
    yield
    print("🛑 Shutting down")


app = FastAPI(
    title="Nepal Next Word Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Nepal Next Word Predictor API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    related = is_related(req.text)
    output = predict_text(req.text, req.n_words)

    return PredictResponse(input=req.text, output=output, related=related)