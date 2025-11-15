from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_template(
    "Tell me 1 joke about: {topic}"
)

uppercase_words = RunnableLambda(lambda x: x.upper())

chain = prompt_template | model | StrOutputParser() | uppercase_words

result = chain.invoke("Engineers")

print(result)
