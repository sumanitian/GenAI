import os
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
QUADRANT_HOST = "localhost"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "test1234"


config = {
    "version": "v1.1",
    "embedder": {
        "provider": "huggingface",
        "config" : {
            "model": "sentence-transformers/all-mpnet-base-v2"
        },
    },
    "llm": {
        "provider": "groq",
        "config": {
            "api_key": api_key,
            "model": "llama-3.3-70b-versatile"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
            "embedding_model_dims": 768
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        },
    },
}

mem_client = Memory.from_config(config)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def chat(message):
    mem_result = mem_client.search(query=message, user_id="s123")

    print("mem_result", )

    memories = "\n".join([m["memory"] for m in mem_result.get("results")])

    print(f"\n\nMEMORY:\n\n{memories}\n\n")

    SYSTEM_PROMPT = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        
        Memory and Score:
        {memories}
    """

    messages= [
        { "role": "system", "content": SYSTEM_PROMPT },
        {"role": "user", "content": message}
    ]

    result = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )

    messages.append(
        {"role": "assistant", "content": result.choices[0].message.content}
    )

    # storing the conversation in memory
    mem_client.add(messages, user_id="s123")

    return result.choices[0].message.content

while True:
    message = input(">> ")
    print(f"BOT: ", chat(message=message))