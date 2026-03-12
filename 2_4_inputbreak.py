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
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast thing 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse,you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string" }}

Example: 
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright! The user is interested in maths query and he is asking a basic arithmetic operation"}}
Output: {{ step: "think", content: "To perform the addition i must go from left t right and add all the operands.}}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

result = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "what is 3 + 4 * 5"},

        {"role": "assistant", "content": json.dumps({"step": "analyse", "content": "Alright! The user is interested in maths query and he is asking an expression involving order of operations."})},
        {"role": "assistant", "content": json.dumps({"step": "analyse", "content": "First, we need to identify the order of operations (PEMDAS) which stands for Parentheses, Exponents, Multiplication and Division, and Addition and Subtraction. In this case, we have Multiplication (4 * 5) which should be evaluated first because it comes before Addition."})},
        {"role": "assistant", "content": json.dumps({"step": "think", "content": "To perform the multiplication, we take 4 and multiply it with 5 which results in 20."})},
        {"role": "assistant", "content": json.dumps({"step": "think", "content": "Now that we have the result of the multiplication, which is 20, we can rewrite the original expression as 3 + 20."})},
    ]
)

print(result.choices[0].message.content)