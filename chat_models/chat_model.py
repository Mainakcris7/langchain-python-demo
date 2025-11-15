from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# Simple query to the model
response = model.invoke("What is the capital of India?")
print(response.content)

# Simple conversation with the model
messages = [
    SystemMessage("Translate the english sentences to spanish language."),
    HumanMessage("English: One coffee for me, please!")
]

chat_response = model.invoke(messages)
print(chat_response.content)

# Continuos conversation with the model
conversation = [
    SystemMessage("I want you to talk to me as a Senior Software Engineer. I want to deploy my newly created Spring Boot + React project to any of the cloud service. Guide me. Each response should be only 2-3 lines.")
]
while (True):
    user_input = input("User: ")
    if (user_input.lower().strip() == "bye"):
        break
    conversation.append(HumanMessage(user_input))
    model_response = model.invoke(conversation)
    print(f'AI: {model_response.content}')
    conversation.append(AIMessage(model_response.content))

print(conversation)
