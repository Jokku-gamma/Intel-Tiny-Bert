from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import torch

tokenizer=AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
model=AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

app=FastAPI()

class QARequest(BaseModel):
    context:str
    question:str

@app.post("/predict")
async def predict(reqeust:QARequest):
    inputs=tokenizer(reqeust.question,reqeust.context,return_tensors="pt")

    with torch.no_grad():
        outputs=model(**inputs)
    
    start_scores=outputs.start_logits
    end_scores=outputs.end_logits

    start_idx=torch.argmax(start_scores)
    end_idx=torch.argmax(end_scores)+1

    answer=tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])

    )
    return {"answer":answer}

