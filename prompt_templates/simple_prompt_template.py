from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = "What is the capital of {country}"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"country": "India"})
result = model.invoke(prompt)

print(prompt)
print(result.content)
