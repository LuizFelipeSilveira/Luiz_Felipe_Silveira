"""
Parte 1 - Questão 1
API FastAPI que gera texto a partir de uma frase usando o GPT-2 (HuggingFace).

Executar: uvicorn p1q1:app --reload
Testar:   http://127.0.0.1:8000/docs -> POST /generate
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = "openai-community/gpt2"

app = FastAPI(title="P1Q1 - Geração de texto com GPT-2")

generator = pipeline("text-generation", model=MODEL_NAME)


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(50, ge=1, le=300)
    temperature: float = Field(0.8, gt=0, le=2)


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    try:
        result = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerationResponse(
        prompt=request.prompt,
        generated_text=result[0]["generated_text"],
    )