import asyncio
import os
import edge_tts
import sounddevice as sd
from faster_whisper import WhisperModel
from rag import chain, retriever
from langchain_core.messages import HumanMessage, AIMessage
import time

async def hablar(texto, nombre):
    comunicador = edge_tts.Communicate(texto, "es-ES-ElviraNeural")
    await comunicador.save(nombre)

# asyncio.run(hablar("Buenos dias, ¿que tal estas?"))

modelo_stt = WhisperModel("small", device="cpu", compute_type="int8")

historial = []

def escuchar(segundos):
    print("Habla ahora")
    audio = sd.rec(int(segundos * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait() 

    audio = audio.flatten()
       
    inicio = time.time()
    segmentos, info = modelo_stt.transcribe(audio, language="es")
    texto = " ".join(segmento.text for segmento in segmentos).strip()
    print(f"STT: {time.time() - inicio:.2f} s")
    return texto

turno = 0

while True:
    turno += 1
    pregunta = escuchar(5)
    print(f"Tú: {pregunta}")
    if "salir" in pregunta.lower():
        break

    if not pregunta:
        continue


    inicio = time.time()

    try:
        docs = retriever.invoke(pregunta)

        for doc in docs:
            print(doc.page_content, "|", doc.metadata["fuente"])
        contexto = "\n".join([doc.page_content for doc in docs])
            
            
        respuesta = chain.invoke({"pregunta": pregunta, "contexto": contexto, "historial": historial[-6:]})
        historial.append(HumanMessage(content=pregunta))
        historial.append(AIMessage(content=respuesta))
    except Exception as e:
        print(f"Ha habido un error: {e}")
        continue

    print(f"RAG: {time.time() - inicio:.2f} s")

    respuesta = respuesta.replace("*", "")

    inicio = time.time()
    nombre = f"respuesta_{turno}.mp3"
    asyncio.run(hablar(respuesta, nombre))
    os.startfile(nombre)

    print(f"Voz: {time.time() - inicio:.2f} s")
    input("Pulsa Enter para hablar de nuevo...")