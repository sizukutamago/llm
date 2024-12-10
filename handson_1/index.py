from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは!私はジョンと言います!"},
    ],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content

    if content is not None:
        print(content, end="", flush=True)
