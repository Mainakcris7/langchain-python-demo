from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Please classify the given review in 'positive', 'negative', 'neutral'. Answer in one word only!"),
        ("human", "Review: {review}")
    ]
)


def positive_feedback(feedback):
    positive_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Please write a short professional email to the user based on the positive feedback."),
            ("human", "Feedback: {feedback}")
        ])
    return positive_prompt_template.invoke(feedback)


def negative_feedback(feedback):
    negative_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Please write a short professional email to the user based on the negative feedback."),
            ("human", "Feedback: {feedback}")
        ])
    return negative_prompt_template.invoke(feedback)


def neutral_feedback(feedback):
    neutral_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Please write a short professional email to the user based on the neutral feedback."),
            ("human", "Feedback: {feedback}")
        ])
    return neutral_prompt_template.invoke(feedback)


branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        RunnableLambda(lambda x: positive_feedback(x)
                       ) | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        RunnableLambda(lambda x: negative_feedback(x)
                       ) | model | StrOutputParser()
    ),
    RunnableLambda(lambda x: neutral_feedback(x)
                   ) | model | StrOutputParser()

)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableLambda(lambda feedback: print(f"Feedback is: {feedback}") or feedback)
    | branches
    | StrOutputParser()
)

result = chain.invoke(
    "The product exceeded all my expectations — excellent build quality, fast delivery, and it works perfectly without any issues.")

print(result)
