from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load Babelscape/rebel-large
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

app = FastAPI()

class TextRequest(BaseModel):
    text: str  # Input text for the model

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Babelscape/rebel-large API!",
        "usage": "Send a POST request to /predict with 'text' in JSON format.",
        "example": {
            "text": "Elon Musk is the CEO of Tesla."
        }
    }

@app.post("/predict")
async def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs)

    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": decoded_text}
