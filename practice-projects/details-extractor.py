import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json

load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_GPT4O_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_GPT4O_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_GPT4O_API_VERSION"]
)

class CandidateJsonOutputFormat(BaseModel):
    """
    Information about a candidate
    """
    name: str = Field(description = "Full name of the candidate", examples=["Mainak Mukherjee", "Alex Smith"])
    expeienceInYears: float = Field("Total experience of the candidate in years, if he/she is a fresher, it should be '0'", examples=[2.5, 6.2, 0])
    address: str = Field(description = "Full address of the candidate in the given format -> City, State, Country")
    email: str = Field("Email address of the candidate")
    skills: list[str] = Field("Comma seperated list of skills for the candidate", examples=[["Java", "Spring Boot", "React"], ["Python", "Langchain", "GenAI"]])
    projects: list[str] = Field("Comma separated list of projects by the candidate", examples=[["Philia - Social Media App", "Sentiment Analysis on Movie Review data"]])

messages = [
    ('system', "You are a helpful assistant who helps in generating JSON data about the candiate in the given fotmat : {format}. If you are unable to find any specific details, just put null (or, empty list if the field is a list) in that field in the output."),
    ("user", "Details of the candidate: {details}")
]

json_parser = JsonOutputParser(pydantic_object=CandidateJsonOutputFormat)

prompt_template = (ChatPromptTemplate
                   .from_messages(messages)
                   .partial(format = json_parser.get_format_instructions())
                )

chain = prompt_template | llm | json_parser

# user_details = """
# I am John Doe, a dedicated software engineer with 5 years and 6 months of professional experience. I currently reside in San Francisco, California, USA, and my email address is john.doe@example.com.

# My core skills are Python, Django, PostgreSQL, and AWS. I actively contributed to two major professional projects: I developed the 'E-commerce Platform Backend,' and I also built the 'Real-time Data Processing Pipeline.'
# """

# user_details = """
# I am Jane Smith, a recent university graduate with a strong background in Computer Science. I currently live in Boston, Massachusetts, USA. You can reach me via email at jane.smith@example.com.

# While I am a fresher, I possess foundational skills in several key technologies. I actively use Java, Spring, and MySQL, and I have basic proficiency in Docker.

# I currently do not have any major professional projects, but I excelled in my university coursework, which built a solid basis for my career."
# """

user_details = """
I am Alex Chen, a dynamic full-stack developer with 3 years of experience. You can contact me directly at alex.chen@dev.net.

My key skills include TypeScript, Node.js, and React. I focus heavily on creating scalable web solutions.

I have successfully delivered two significant projects: I led the development of a 'Customer Relationship Management (CRM) tool,' and I also optimized the performance of a 'Real-time Stock Tracking Dashboard.'
"""


print("AI in progress...")
result = chain.invoke({"details": user_details})
print(result)

file_path = f"./data/{'_'.join(result["name"].lower().split())}.json"
print("Saving the output in: ", file_path)

with open(file_path, 'w') as f:
    json.dump(result, f, indent = 4)

print("Saved successfully ✅")