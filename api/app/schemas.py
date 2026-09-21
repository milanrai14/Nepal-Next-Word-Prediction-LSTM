from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Seed text for prediction")
    n_words: int = Field(20, ge=1, le=100, description="Number of words to generate")


class PredictResponse(BaseModel):
    input: str
    output: str
    related: bool