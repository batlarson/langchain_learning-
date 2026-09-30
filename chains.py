from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asesor financiero. Responde siempre en {idioma}"),
    MessagesPlaceholder("historial"),
    ("human", "{pregunta}")
])

chain = prompt | model | StrOutputParser()

historial = [

]


while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    
    respuesta = chain.invoke({"pregunta": pregunta, "idioma": "inglés", "historial": historial[-6:]})
    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=respuesta))


    print(respuesta)

