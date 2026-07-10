from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    api_key=api_key,
    request_timeout=30
)

historial = [
    SystemMessage(content="Eres un asesor financiero experto en dividendos e inversión en bolsa. Respondes en español de forma concisa y útil.")
]


while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    
    historial.append(HumanMessage(content=pregunta))
    respuesta = model.invoke(historial)
    historial.append(AIMessage(content=respuesta.content))
    
    print(f"IA: {respuesta.content[0]['text']}\n")