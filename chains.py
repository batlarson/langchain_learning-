from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asesor financiero. Responde siempre en {idioma}."),
    ("human", "Dame un análisis financiero sobre {ticker}")
])

chain = prompt | model 

respuesta = chain.invoke({"ticker": "ABBV", "idioma": "inglés"})
print(respuesta.content[0]['text'])