from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    ("system", "You are a history teacher"),
    ("human", "Tell me 2-3 lines about {topic}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topic": "Maharana Pratap"})

result = model.invoke(prompt)

print("GENERATED PROMPT")
print(prompt)
print()

print("RESULT")
print(result.content)
