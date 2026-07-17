from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# 1. Cargar el documento
loader1 = TextLoader("datos_portfolio.txt", encoding="utf-8")
loader2 = TextLoader("datos_acciones.txt", encoding="utf-8")
loader3 = TextLoader("datos_estrategia.txt", encoding="utf-8")

documentos = loader1.load() + loader2.load() + loader3.load()

# 2. Dividir en trozos
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
trozos = splitter.split_documents(documentos)

# 3. Crear embeddings y guardar en FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(trozos, embeddings)
retriever = vectorstore.as_retriever()

# 4. Crear el chain de RAG
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)


historial = [
    SystemMessage(content="Eres un asesor financiero experto en dividendos e inversión en bolsa. Respondes en español de forma concisa y útil.")
]


while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    
    # Buscar contexto relevante
    docs = retriever.invoke(pregunta)
    contexto = "\n".join([doc.page_content for doc in docs])
    
    # Añadir al historial con contexto
    mensaje_con_contexto = f"Contexto: {contexto}\n\nPregunta: {pregunta}"
    historial.append(HumanMessage(content=mensaje_con_contexto))
    
    respuesta = model.invoke(historial)
    historial.append(AIMessage(content=respuesta.content))
    
    print(f"IA: {respuesta.content[0]['text']}\n")
