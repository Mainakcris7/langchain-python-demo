import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

load_dotenv()

class ModelOutput(BaseModel):
    answer: str

class UserNameInput(BaseModel):
    name: str = Field("Name of the user in lower case with snake case notation, e.g, 'Alex Simith' -> 'alex_smith'")

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_GPT4O_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_GPT4O_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_GPT4O_API_VERSION"]
)

@tool("Date_Time", description="Useful when finding out the current date and time")
def get_date_time() -> str:
    import datetime
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%A, %B %d, %Y at %I:%M %p")

user_data = {
    "alice_johnson": {
        "email": "alice.j@example.com",
        "age": 28,
        "is_active": True
    },
    "bob_smith": {
        "email": "bob.s@example.com",
        "age": 35,
        "is_active": False
    },
    "charlie_brown": {
        "email": "c.brown@example.com",
        "age": 42,
        "is_active": True
    },
    "diana_prince": {
        "email": "diana.p@example.com",
        "age": 21,
        "is_active": True
    },
    "ethan_hunt": {
        "email": "e.hunt@example.com",
        "age": 50,
        "is_active": True
    },
    "fiona_glenanne": {
        "email": "fiona.g@example.com",
        "age": 30,
        "is_active": False
    },
    "george_lucas": {
        "email": "george.l@example.com",
        "age": 65,
        "is_active": True
    },
    "hannah_montana": {
        "email": "hannah.m@example.com",
        "age": 19,
        "is_active": False
    },
    "ivan_drago": {
        "email": "ivan.d@example.com",
        "age": 45,
        "is_active": True
    },
    "jenna_coleman": {
        "email": "jenna.c@example.com",
        "age": 33,
        "is_active": True
    }
}

@tool(name_or_callable="User_Details", description="Useful when fetching details about a particular user by their name.", args_schema=UserNameInput)
def get_user_data(name: str) -> str:
    """
    Args: 
        name: str
    """
    return user_data[name]

messages = [
    ("system", "You are a helpful AI assistant."),
    # ("human", "What is the date and time for the day after tomorrow?")
    ("human", "Give me a summary for the user: 'Bob Smith'")
]

agent = create_agent(
    model=llm,
    tools=[get_date_time, get_user_data],
    response_format=ToolStrategy(ModelOutput)
)

print("Agent in progress...")
result = agent.invoke({"messages": messages})
print(result["structured_response"])