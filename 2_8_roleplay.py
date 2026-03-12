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
You are role-playing as a senior Python developer conducting a technical interview.

Rules:
- Ask one interview question at a time.
- Evaluate the user's answer.
- Provide feedback before asking the next question.

Return JSON in this format:

{
 "role": "Interviewer",
 "question": "string",
 "feedback": "string"
}
"""

messages = [
    {"role": "system", "content": system_prompt}
]

query = input("> ")
messages.append({"role": "user", "content": query})

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    response_format={"type": "json_object"},
    messages=messages
)

parsed = json.loads(response.choices[0].message.content)

print("Role:", parsed["role"])
print("Question:", parsed["question"])
print("Feedback:", parsed["feedback"])