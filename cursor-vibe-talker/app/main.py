import speech_recognition as sr
from graph import create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver

import asyncio
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

openai = AsyncOpenAI()

MONGODB_URI = "mongodb://admin:admin@localhost:27018/?authSource=admin"
config = {"configurable": {"thread_id": "2"}}

def normalize_text(text):
    text = text.lower()

    # fix common speech issues
    text = text.replace("dot py", ".py")
    text = text.replace("dot", ".")
    text = text.replace(" ", "_")

    return text

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)

        r = sr.Recognizer()

        with sr.Microphone() as source:
            # noice cancellation
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2

            while True:
                print("Say Something")
                audio = r.listen(source)

                print("Processing audio...")
                try:
                    sst = r.recognize_google(audio)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    continue
                except sr.RequestError:
                    print("API Error")
                    continue
                sst = normalize_text(sst)
                print("You Said: ",sst)

                for event in graph.stream({ "messages": [{"role": "user", "content": sst}] }, config, stream_mode="values"):
                    if "messages" in event:
                        event["messages"][-1].pretty_print()

# async def speak(text: str):
#     async with openai.audio.speech.with_streaming_response.create(
#     model="gpt-4o-mini-tts",
#     voice="coral",
#     input="Today is a wonderful day to build something people love!",
#     instructions="Speak in a cheerful and positive tone.",
#     response_format="pcm",
#     ) as response:
#         await LocalAudioPlayer().play(response)

main()

# if __name__ == "__main__":
#     asyncio.run(speak(text="This is a sample voice. Hi Suman"))