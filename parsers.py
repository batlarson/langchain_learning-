from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asesor financiero. Responde SOLO con un JSON válido con estos campos:
- ticker: string
- recomendacion: "comprar", "mantener" o "vender"
- motivo: string con una frase corta
- riesgo: "bajo", "medio" o "alto"
No incluyas nada más, solo el JSON."""),
    ("human", "Analiza {ticker} para inversión en dividendos")
])

parser = JsonOutputParser()
chain = prompt | model | parser

respuesta = chain.invoke({"ticker": "MAIN"})
print(respuesta)
print(type(respuesta))