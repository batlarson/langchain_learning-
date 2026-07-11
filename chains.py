from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asesor financiero experto en {tema}. Responde en español de forma concisa."),
    ("human", "{pregunta}")
])

chain = prompt | model 

respuesta = chain.invoke({"tema": "dividendos", "pregunta": "Qué es el YOC?"})
print(respuesta.content[0]['text'])