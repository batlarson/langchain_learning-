from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)

class Eleccion(BaseModel):
    ticker: str
    recomendacion: Literal["comprar", "mantener", "vender"]
    motivo: str = Field(description="Una frase corta explicando la recomendación")
    riesgo: Literal["bajo", "medio" , "alto"]

structured_model = model.with_structured_output(Eleccion)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asesor financiero."),
    ("human", "Analiza {ticker} para inversión en dividendos")
])


chain = prompt | structured_model

respuesta = chain.invoke({"ticker": "MAIN"})
print(respuesta)
print(type(respuesta.recomendacion))