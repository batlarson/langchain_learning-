from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

@tool
def obtener_precio(ticker: str) -> str:
    """Obtiene el precio actual de una acción dado su ticker."""
    import yfinance as yf
    precio = yf.Ticker(ticker).info.get('currentPrice', 'No disponible')
    return f"El precio actual de {ticker} es {precio}$"

@tool
def obtener_dividendo(ticker: str) -> str:
    """Obtiene el dividendo anual de una acción dado su ticker."""
    import yfinance as yf
    dividendo = yf.Ticker(ticker).info.get('dividendRate', 'No disponible')
    return f"El dividendo anual de {ticker} es {dividendo}$"

tools = [obtener_precio, obtener_dividendo]
agent = create_agent(model, tools, system_prompt="Eres un asesor financiero. Responde en español.")

respuesta = agent.invoke({"messages": [("human", "Cuál es el precio y el dividendo de KO?")]})
print(respuesta["messages"][-1].content[0]['text'])