"""
Parte 1 - Questão 2
API FastAPI que traduz textos do inglês para o francês com o modelo
Helsinki-NLP/opus-mt-en-fr. Textos longos são divididos em parágrafos e em
blocos de até 400 tokens para respeitar o limite de 512 tokens do modelo.

Executar: uvicorn p1q2:app --reload
Testar:   http://127.0.0.1:8000/docs -> POST /translate
"""
import re

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
MAX_TOKENS = 400

app = FastAPI(title="P1Q2 - Tradução EN → FR com Helsinki-NLP")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str


def split_into_chunks(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(tokenizer.encode(candidate)) <= MAX_TOKENS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def translate_chunks(chunks: list[str]) -> list[str]:
    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_text(text: str) -> str:
    translated_paragraphs = []

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            translated_paragraphs.append("")
            continue

        chunks = split_into_chunks(paragraph)
        translated_paragraphs.append(" ".join(translate_chunks(chunks)))

    return "\n".join(translated_paragraphs)


@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")

    try:
        translated = translate_text(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TranslationResponse(
        original_text=request.text,
        translated_text=translated,
    )