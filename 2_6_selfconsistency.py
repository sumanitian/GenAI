#it can be made more good

import os
import json
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

system_prompt = """
Solve the user's problem.

Return JSON in this format:
{
 "answer": "final answer"
}
"""

query = input("> ")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query}
]

answers = []
num_samples = 10

for _ in range(num_samples):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.9,
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed = json.loads(response.choices[0].message.content)

    if "answer" in parsed:
        answers.append(parsed["answer"])

print("All Answers:", answers)

if len(answers) == 0:
    print("No valid answers returned.")
else:
    most_common = Counter(answers).most_common(1)[0][0]
    print("Final Self-Consistent Answer:", most_common)