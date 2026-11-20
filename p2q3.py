"""
Parte 2 - Questão 3
API de tradução EN → DE usando LangChain para gerenciar as chamadas ao modelo
Helsinki-NLP/opus-mt-en-de (HuggingFace), executado localmente em CPU.

Um wrapper próprio (HuggingFaceTranslationLLM) foi necessário porque a
integração pronta do LangChain (HuggingFacePipeline) não funciona com o
Transformers 5.

Executar: uvicorn p2q3:app --reload
Testar:   http://127.0.0.1:8000/docs -> POST /translate-de
"""
import re
import time
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"

set_llm_cache(InMemoryCache())


class HuggingFaceTranslationLLM(LLM):
    model_name: str = MODEL_NAME
    max_tokens_per_chunk: int = 400

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "huggingface-marian-translation"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _split_into_chunks(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        chunks = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if len(self._tokenizer.encode(candidate)) <= self.max_tokens_per_chunk:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _translate_chunks(self, chunks: list[str]) -> list[str]:
        inputs = self._tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=512)
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        translated_paragraphs = []

        for paragraph in prompt.split("\n"):
            if not paragraph.strip():
                translated_paragraphs.append("")
                continue

            chunks = self._split_into_chunks(paragraph)
            translated_paragraphs.append(" ".join(self._translate_chunks(chunks)))

        return "\n".join(translated_paragraphs)


prompt = PromptTemplate.from_template("{text}")
llm = HuggingFaceTranslationLLM()
chain = prompt | llm | StrOutputParser()

app = FastAPI(title="P2Q3 - Tradução EN → DE com LangChain + HuggingFace")


class GermanTranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class GermanTranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    model: str
    latency_ms: float


@app.post("/translate-de", response_model=GermanTranslationResponse)
def translate_de(request: GermanTranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")

    start = time.perf_counter()

    try:
        translated = chain.invoke({"text": request.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return GermanTranslationResponse(
        original_text=request.text,
        translated_text=translated,
        model=MODEL_NAME,
        latency_ms=latency_ms,
    )