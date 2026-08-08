from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asesor financiero. Responde siempre en {idioma}"),
    ("human", "Dime si {empresa} es buena para invertir en dividendos")
])

chain = prompt | model 

respuesta = chain.invoke({"empresa": "Royal Bank of Canada", "idioma": "inglés"})
print(respuesta.content[0]['text'])