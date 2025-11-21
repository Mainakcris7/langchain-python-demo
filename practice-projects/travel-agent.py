import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import Optional
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

travel_data = [
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '22/11/2025', 'cost': 5250.75 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '23/11/2025', 'cost': 5380.50 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '24/11/2025', 'cost': 5190.25 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '25/11/2025', 'cost': 5500.00 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '26/11/2025', 'cost': 5420.90 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '27/11/2025', 'cost': 5310.65 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '28/11/2025', 'cost': 5650.40 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '29/11/2025', 'cost': 5200.15 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '30/11/2025', 'cost': 5490.85 },
    { 'source': 'delhi', 'dest': 'mumbai', 'date': '01/12/2025', 'cost': 5350.30 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '02/12/2025', 'cost': 2150.10 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '03/12/2025', 'cost': 2280.45 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '04/12/2025', 'cost': 2190.70 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '05/12/2025', 'cost': 2350.95 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '06/12/2025', 'cost': 2210.05 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '07/12/2025', 'cost': 2400.30 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '08/12/2025', 'cost': 2290.55 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '09/12/2025', 'cost': 2100.80 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '10/12/2025', 'cost': 2320.60 },
    { 'source': 'bengaluru', 'dest': 'chennai', 'date': '11/12/2025', 'cost': 2250.75 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '12/12/2025', 'cost': 3800.50 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '13/12/2025', 'cost': 3950.85 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '14/12/2025', 'cost': 3750.10 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '15/12/2025', 'cost': 4000.35 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '16/12/2025', 'cost': 3880.60 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '17/12/2025', 'cost': 4100.90 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '18/12/2025', 'cost': 3920.15 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '19/12/2025', 'cost': 3780.45 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '20/12/2025', 'cost': 4050.70 },
    { 'source': 'kolkata', 'dest': 'hyderabad', 'date': '21/12/2025', 'cost': 3850.25 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '22/12/2025', 'cost': 2850.90 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '23/12/2025', 'cost': 2990.25 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '24/12/2025', 'cost': 3100.50 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '25/12/2025', 'cost': 3250.75 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '26/12/2025', 'cost': 2950.00 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '27/12/2025', 'cost': 3150.30 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '28/12/2025', 'cost': 3050.60 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '29/12/2025', 'cost': 3300.95 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '30/12/2025', 'cost': 3000.10 },
    { 'source': 'mumbai', 'dest': 'goa', 'date': '31/12/2025', 'cost': 3400.45 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '01/01/2026', 'cost': 3450.70 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '02/01/2026', 'cost': 3590.25 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '03/01/2026', 'cost': 3390.50 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '04/01/2026', 'cost': 3650.75 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '05/01/2026', 'cost': 3500.00 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '06/01/2026', 'cost': 3700.30 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '07/01/2026', 'cost': 3480.60 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '08/01/2026', 'cost': 3750.95 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '09/01/2026', 'cost': 3550.10 },
    { 'source': 'chennai', 'dest': 'pune', 'date': '10/01/2026', 'cost': 3680.45 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '11/01/2026', 'cost': 2650.15 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '12/01/2026', 'cost': 2780.40 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '13/01/2026', 'cost': 2600.65 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '14/01/2026', 'cost': 2850.90 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '15/01/2026', 'cost': 2720.10 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '16/01/2026', 'cost': 2900.35 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '17/01/2026', 'cost': 2680.50 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '18/01/2026', 'cost': 2950.75 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '19/01/2026', 'cost': 2750.00 },
    { 'source': 'delhi', 'dest': 'lucknow', 'date': '20/01/2026', 'cost': 2880.25 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '21/01/2026', 'cost': 4500.40 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '22/01/2026', 'cost': 4650.65 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '23/01/2026', 'cost': 4450.90 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '24/01/2026', 'cost': 4700.15 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '25/01/2026', 'cost': 4550.45 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '26/01/2026', 'cost': 4800.70 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '27/01/2026', 'cost': 4600.95 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '28/01/2026', 'cost': 4850.05 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '29/01/2026', 'cost': 4650.30 },
    { 'source': 'bengaluru', 'dest': 'jaipur', 'date': '30/01/2026', 'cost': 4750.55 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '31/01/2026', 'cost': 3950.90 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '01/02/2026', 'cost': 4100.25 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '02/02/2026', 'cost': 3900.50 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '03/02/2026', 'cost': 4150.75 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '04/02/2026', 'cost': 4000.00 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '05/02/2026', 'cost': 4200.30 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '06/02/2026', 'cost': 4050.60 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '07/02/2026', 'cost': 4250.95 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '08/02/2026', 'cost': 4100.10 },
    { 'source': 'hyderabad', 'dest': 'ahmedabad', 'date': '09/02/2026', 'cost': 4300.45 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '10/02/2026', 'cost': 2350.70 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '11/02/2026', 'cost': 2490.25 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '12/02/2026', 'cost': 2300.50 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '13/02/2026', 'cost': 2550.75 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '14/02/2026', 'cost': 2400.00 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '15/02/2026', 'cost': 2600.30 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '16/02/2026', 'cost': 2450.60 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '17/02/2026', 'cost': 2650.95 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '18/02/2026', 'cost': 2500.10 },
    { 'source': 'kolkata', 'dest': 'patna', 'date': '19/02/2026', 'cost': 2700.45 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '20/02/2026', 'cost': 2950.90 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '21/02/2026', 'cost': 3090.25 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '22/02/2026', 'cost': 2900.50 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '23/02/2026', 'cost': 3150.75 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '24/02/2026', 'cost': 3000.00 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '25/02/2026', 'cost': 3200.30 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '26/02/2026', 'cost': 3050.60 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '27/02/2026', 'cost': 3250.95 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '28/02/2026', 'cost': 3100.10 },
    { 'source': 'pune', 'dest': 'nagpur', 'date': '01/03/2026', 'cost': 3300.45 }
]

class TravelDataInput(BaseModel):
    source: Optional[str] = Field(description="Source city in lower case", default = "", examples=['kolkata', 'mumbai'])
    dest: Optional[str] = Field(description="Destination city in lower case", default="", examples=['kolkata', 'mumbai'])

class AgentOutput(BaseModel):
    answer: str

@tool(name_or_callable='Current_Date', description="Returns date in the 'dd/MM/yyyy format")
def get_current_date() -> str:
    import datetime
    date = datetime.date.today()
    return date.strftime('%d/%m/%y')


@tool(name_or_callable='Trip_Details', description="Returns flight trip details, if given proper source and destination city, if source is not present use '', if dest is not present use: '', but always call this tool to get flight details.", args_schema=TravelDataInput)
def get_trip_details(source: str, dest: str) -> str:
    trips = []

    if source and dest:
        for trip in travel_data:
            if trip['source'] == source and trip['dest'] == dest:
                trips.append(trip)
    elif source:
        for trip in travel_data:
            if trip['source'] == source:
                trips.append(trip)
    elif dest:
        for trip in travel_data:
            if trip['dest'] == dest:
                trips.append(trip)
    else:
        travel_data
            
    return trips

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_GPT4O_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_GPT4O_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_GPT4O_API_VERSION"]
)

messages = [
    ('human', """
     
     You are a highly efficient and helpful AI travel agent specializing in flight searches. Your primary goal is to provide accurate and actionable flight information based on the user's request.

        Strictly follow these steps and constraints:

        Analyze and Extract: Carefully analyze the user's request to precisely determine the following four mandatory parameters:

        Departure Location (Source, if not specified, use '' to call the tool)

        Arrival Location (Destination, if not specified use '' to call the tool)

        Travel Date(s) (Must account for one-way, round-trip, or multi-city requests)

        Specific Needs (e.g., cost, time, etc).

        Reasoning and Confirmation: Internally reason through and confirm your understanding of the extracted parameters.

        Execute Search (Use proper tools): Perform the flight search using the identified criteria.

        Reporting Constraint (Crucial):

        If flights are available: Present the available options clearly, summarizing the key details (price, dates, etc).
        If NO flights are available for the specified criteria: You must respond only with the definitive statement: 'I apologize, but no flights are currently available for the specified criteria.'

        DO NOT HALUCINATE. Do not invent flight details, prices, or availability under any circumstances.

        Always conclude with a relevant follow-up question to assist the user further (e.g., "Would you like to adjust the dates or explore nearby airports?").
        """)
]

agent = create_agent(
    model=llm,
    tools=[get_current_date, get_trip_details],
    response_format=ToolStrategy(AgentOutput)
)

while True:
    user_input = input("User: ")
    if(user_input == 'exit'):
        break
    messages.append(
        ('user', user_input)
    )
    result = agent.invoke({"messages": messages})
    print("AI: ", result["structured_response"].answer)
    messages.append(
        ('ai', result["structured_response"].answer)
    )
    
print(messages)