from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# 1. Cargar el documento
loader = TextLoader("datos_portfolio.txt", encoding="utf-8")
documentos = loader.load()

# 2. Dividir en trozos
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
trozos = splitter.split_documents(documentos)

# 3. Crear embeddings y guardar en FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(trozos, embeddings)
retriever = vectorstore.as_retriever()

# 4. Crear el chain de RAG
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responde basándote SOLO en el contexto proporcionado. Si no encuentras la respuesta, di que no tienes esa información.\n\nContexto: {contexto}"),
    ("human", "{pregunta}")
])

chain = (
    {"contexto": retriever, "pregunta": RunnablePassthrough()}
    | prompt
    | model
)

respuesta = chain.invoke("Cuántos dividendos quiero ganar?")
print(respuesta.content[0]['text'])