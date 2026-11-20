"""
Parte 2 - Questão 1
Protótipo de chatbot com LangChain e um Fake LLM que responde a perguntas
pré-definidas por palavra-chave. Nenhum modelo real é chamado.

Executar: uvicorn p2q1:app --reload
Testar:   http://127.0.0.1:8000/docs -> POST /chat
"""
import re
import unicodedata
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

PREDEFINED_ANSWERS = {
    "ola": "Olá! Sou o FakeBot, um chatbot de testes. Como posso ajudar?",
    "nome": "Meu nome é FakeBot, um chatbot simulado com LangChain.",
    "horario": "Nosso atendimento funciona de segunda a sexta, das 9h às 18h.",
    "preco": "Nossos planos começam em R$ 49,90 por mês.",
    "langchain": "LangChain é um framework para construir aplicações com LLMs.",
    "tchau": "Até logo! Foi um prazer conversar com você.",
}

DEFAULT_ANSWER = (
    "Desculpe, ainda não sei responder isso. "
    "Tente perguntar sobre horário, preço ou LangChain."
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


class PredefinedFakeLLM(LLM):
    answers: dict[str, str] = PREDEFINED_ANSWERS
    default_answer: str = DEFAULT_ANSWER

    @property
    def _llm_type(self) -> str:
        return "predefined-fake-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        question = normalize(prompt.split("Pergunta:")[-1])
        words = re.findall(r"\w+", question)

        for keyword, answer in self.answers.items():
            if any(word.startswith(keyword) for word in words):
                return answer

        return self.default_answer


prompt = PromptTemplate.from_template(
    "Você é um assistente de atendimento ao cliente.\nPergunta: {question}"
)
llm = PredefinedFakeLLM()
chain = prompt | llm | StrOutputParser()

app = FastAPI(title="P2Q1 - Chatbot com Fake LLM (LangChain)")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    message: str
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = chain.invoke({"question": request.message})
    return ChatResponse(message=request.message, response=response)