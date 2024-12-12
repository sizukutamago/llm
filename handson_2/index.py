from langchain_openai import OpenAI


model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

output = model.invoke("自己紹介してください")
print(output)
