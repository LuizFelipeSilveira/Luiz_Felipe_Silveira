"""
Parte 2 - Questão 2
API de tradução EN → FR usando LangChain + OpenAI, com chamada assíncrona,
cache em memória, timeout e retries.

Requer um arquivo .env com OPENAI_API_KEY (veja .env.example).

Executar: uvicorn p2q2:app --reload
Testar:   http://127.0.0.1:8000/docs -> POST /translate-openai
"""
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
set_llm_cache(InMemoryCache())

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional translator. Translate the user's text "
            "from English to French. Return only the translation, preserving "
            "the original formatting and line breaks.",
        ),
        ("human", "{text}"),
    ]
)

chain = None
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        timeout=30,
        max_retries=2,
    )
    chain = prompt | llm | StrOutputParser()

app = FastAPI(title="P2Q2 - Tradução EN → FR com LangChain + OpenAI")


class OpenAITranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class OpenAITranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    model: str
    latency_ms: float


@app.post("/translate-openai", response_model=OpenAITranslationResponse)
async def translate_openai(request: OpenAITranslationRequest):
    if chain is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY não configurada no arquivo .env.",
        )

    start = time.perf_counter()

    try:
        translated = await chain.ainvoke({"text": request.text})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro ao chamar a OpenAI: {e}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return OpenAITranslationResponse(
        original_text=request.text,
        translated_text=translated.strip(),
        model=OPENAI_MODEL,
        latency_ms=latency_ms,
    )