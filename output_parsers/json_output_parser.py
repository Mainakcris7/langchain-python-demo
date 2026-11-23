import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)


class JobDetails(BaseModel):
    """
    Information about a job
    """
    title: str = Field(description="Title of the job", examples=[
                       "Software Engineer", "Data Scientist"])
    experience_required: str = Field(
        description="Experience required for the job in years", examples=["2-4 years", "5-7 years"])
    description: str = Field(
        description="Description of the job in detail and easily understandable format")
    requirements: list[str] = Field(description="List of requirements and key skills for the job", examples=[
                                    ["Python", "Django", "REST APIs"], ["Machine Learning", "Data Analysis"]])
    location: str = Field(
        description="Location of the job in the given format -> City, State, Country")
    salary_range: str = Field(
        description="Salary range in INR for the job in the gi"
        "ven format -> Min-Max (in INR)", examples=["7-8 LPA", "20-22 LPA"])


job_json_parser = JsonOutputParser(pydantic_object=JobDetails)

prompt = ChatPromptTemplate.from_messages([
    ('system',
     'You are a helpful assistant who helps in extracting job details from the given text and generating JSON data in the specified format: {format}. If you are unable to find any specific details, just put null (or, empty list if the field is a list) in that field in the output.'),
    ('user', 'Details of the job: {details}')
]).partial(format=job_json_parser.get_format_instructions())

JOB_DETAILS = """
We are looking for a talented Software Engineer to join our team. The ideal candidate will have experience in developing web applications using Python and Django. The job requires proficiency in REST APIs and a strong understanding of front-end technologies. We strongly recommend knowledge in GenAI, to be very specific, knowledge in Langchain/Langgraph, RAG etc. The position is based in Bangalore, India (Karnataka), and offers a competitive salary range of 10-12 LPA.
Responsibilities:
- Develop and maintain web applications using Python and Django.
- Collaborate with front-end developers to integrate user-facing elements with server-side logic.
- Ensure the performance, quality, and responsiveness of applications.
- Identify and correct bottlenecks and fix bugs.
- Help maintain code quality, organization, and automatization.
Experience Required:
- 3-5 years of experience in software development, with a focus on web applications.
"""

chain = prompt | llm | job_json_parser

print("Executing LLM chain...")
result = chain.invoke({'details': JOB_DETAILS})
print(result)

print("Writing to file...")
with open("./output_parsers/job_details.json", "w") as f:
    json.dump(result, f, indent=4)

print("Saved successfully ✅")
