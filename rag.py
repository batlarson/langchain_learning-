from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import time

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

archivos = ["datos_acciones.txt", "datos_estrategia.txt", "datos_portfolio.txt"]
documentos = []
for archivo in archivos:
    with open(archivo, encoding="utf-8") as f:
        texto = f.read()

    documentos.append(Document(page_content=texto, metadata={"fuente": archivo}))


splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=200, chunk_overlap=40)
trozos = splitter.split_documents(documentos)



embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(trozos, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)

historial = []

prompt = ChatPromptTemplate([
    ("system", """Eres un asesor financiero experto en dividendos e inversión en bolsa. Respondes en español de forma concisa y útil. Si la respuesta no está en el contexto, di que no tienes esa información.
    Contexto:
    {contexto}
    """),
    MessagesPlaceholder("historial"),
    ("human", "{pregunta}")
])

prompt_reformular = ChatPromptTemplate([
    ("system", "Dada la conversación y la última pregunta, reescríbela para que se entienda sin la conversación. No la respondas, solo reescríbela. Si ya se entiende sola, devuélvela igual"),
    MessagesPlaceholder("historial"),
    ("human", "{pregunta}")
])

chain = prompt | model | StrOutputParser()
# chain_reformular = prompt_reformular | model | StrOutputParser()


while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    
    # if len(historial) > 0:
    #     pregunta_busqueda = chain_reformular.invoke({"historial": historial[-6:], "pregunta": pregunta})
    # else:
    #     pregunta_busqueda = pregunta

    # print(repr(pregunta_busqueda))

    docs = None

    for intento in range(3):
        try:
            docs = retriever.invoke(pregunta)
            break

        except Exception as e:
            print(f"Intento {intento + 1} fallido: {e}")
            time.sleep(1)

    if docs is None:                   # fallaron los 3
        print("Ha habido un error, inténtalo de nuevo")
        continue


    try:
        for doc in docs:
            print(doc.page_content, "|", doc.metadata["fuente"])
        contexto = "\n".join([doc.page_content for doc in docs])
            
            
        respuesta = chain.invoke({"pregunta": pregunta, "contexto": contexto, "historial": historial[-6:]})
        historial.append(HumanMessage(content=pregunta))
        historial.append(AIMessage(content=respuesta))
    except Exception as e:
        print(f"Ha habido un error: {e}")
        continue
    
    print(respuesta)
