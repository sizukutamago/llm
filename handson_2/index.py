from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()


model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

output = model.invoke("自己紹介してください")
print(output)
