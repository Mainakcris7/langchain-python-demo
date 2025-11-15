from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_template(
    "Tell me 2-3 lines about: {topic}"
)

# chain = prompt_template | model | StrOutputParser()

prompt_lambda = RunnableLambda(lambda x: prompt_template.invoke(x))
model_lambda = RunnableLambda(lambda x: model.invoke(x))

chain = prompt_lambda | model_lambda | StrOutputParser()

result = chain.invoke("Shivaji")

print(result)


# Always remember, output of the previous component is sent as the input of the next component's invoke() method
# The components we are using in the chain thus must have a suitable invoke() method
