from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Behave like a professional who has descent knowledge in football, think deeply and answer the questions in a single sentence. I want 100 percentage definite answer. Don't way I can't predeict or neutral answer."),
        ("human", "Which team has the highest number of UCL trophies?"),
        ("ai", "Real Madrid has the highest number of UCL trophies (15)."),
        ("human", "Predict the winner of {tournament}?")
    ]
)


def agree_statement(statement):
    agree_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Write 2-3 points supporting the statement."),
        ("human", "The statement: {statement}")
    ])

    return agree_prompt_template.invoke(statement)


def disagree_statement(statement):
    disagree_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Write 2-3 points disagreeing the statement."),
        ("human", "The statement: {statement}")
    ])

    return disagree_prompt_template.invoke(statement)


agree_statement_chain = (
    RunnableLambda(agree_statement) | model | StrOutputParser()
)

disagree_statement_chain = (
    RunnableLambda(disagree_statement) | model | StrOutputParser()
)

combine_output = RunnableLambda(
    lambda x: f"SUPPORTING\n {x["agree"]} \n\n DISCARDING\n {x["disagree"]}")

chain = (
    prompt_template
    | model
    | StrOutputParser()
    # if we dont return x, it will not be available for the subsequent components of the chain
    | RunnableLambda(lambda x: print(f"The statement is: {x}") or x)
    | RunnableParallel({
        "agree": agree_statement_chain,
        "disagree": disagree_statement_chain
    })
    | combine_output
)

result = chain.invoke("World Cup 2026")

print(result)
