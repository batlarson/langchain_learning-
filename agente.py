from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)

def _obtener_precio(ticker: str) -> float:
    import yfinance as yf
    return yf.Ticker(ticker).info.get("currentPrice", 0.0)

def _obtener_dividendo(ticker: str) -> float:
    import yfinance as yf
    return yf.Ticker(ticker).info.get("dividendRate", 0.0)

@tool
def obtener_precio(ticker: str) -> str:
    """Obtiene el precio actual de una acción dado su ticker."""
    precio = _obtener_precio(ticker)
    return f"El precio actual de {ticker} es {precio}$"

@tool
def obtener_dividendo(ticker: str) -> str:
    """Obtiene el dividendo anual de una acción dado su ticker."""
    dividendo = _obtener_dividendo(ticker)
    return f"El dividendo anual de {ticker} es {dividendo}$"

@tool
def calcular_yoc(ticker: str, pmc: float) -> str:
    """Calcula el YOC de una acción dado su ticker y precio medio de compra."""
    dividendo = _obtener_dividendo(ticker)
    yoc = (dividendo/pmc)*100
    return f'El YOC de {ticker} en tu cartera es de {yoc}%'

@tool
def calcular_inversion(ticker: str, dinero: float, precio: float | None = None) -> str:
    """Calcula cuántas acciones puedes comprar con cierta cantidad de dinero.
    Si el usuario indica un precio concreto, pásalo en 'precio'. Si no, se usa el precio actual."""
    if precio is None:
        precio = _obtener_precio(ticker)
    if not precio:
        return f"No he podido obtener el precio de {ticker}"
    cantidad = dinero/precio
    
    return f'Con tu dinero puedes comprar {cantidad} acciones'

tools = [obtener_precio, obtener_dividendo, calcular_yoc, calcular_inversion]
agent = create_agent(model, tools, system_prompt="Eres un asesor financiero. Responde en español.")

respuesta = agent.invoke({"messages": [("human", "¿Cuál es el YOC de ABBV si la compré a 132$ y cuánto cuesta ahora?")]})
for message in respuesta["messages"]:
    message.pretty_print()

print(respuesta["messages"][-1].text)