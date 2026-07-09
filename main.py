from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

messages = [
    ("system", "Eres un asistente útil que responde en español."),
    ("human", "Dime un dato curioso sobre Python en una frase.")
]

ai_msg = model.invoke(messages)
print(ai_msg.content[0]['text'])