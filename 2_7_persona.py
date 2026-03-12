import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

system_prompt = """
You are a senior computer science professor who explains concepts clearly 
with simple examples so that beginners can understand.

Return JSON in this format:
{
 "persona": "Professor",
 "answer": "explanation"
}
"""

query = input("> ")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query}
]

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    response_format={"type": "json_object"},
    messages=messages
)

parsed = json.loads(response.choices[0].message.content)

print("Persona:", parsed["persona"])
print("Answer:", parsed["answer"])