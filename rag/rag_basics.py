import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join("data", "story.txt")
vector_store_path = os.path.join("db", "faiss_db")


# Initialize embedding model and chat model
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if not os.path.exists(file_path):
    print(file_path)
    raise FileNotFoundError("File not found!")

# Load and process document
loader = TextLoader(file_path=file_path, encoding="utf-8")
document = loader.load()

if (os.path.exists(vector_store_path)):
    print("Vector store already exists!")
else:
    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10)
    docs = splitter.split_documents(documents=document)

    print("Document chunk information:")
    print(f"Total chunks: {len(docs)}")
    print(f"First chunk length: {len(docs[0].page_content)}")
    print(f"First chunk:\n{docs[0].page_content}")

    # Create and store vector store
    print("Stroing vector store in local...")
    store = FAISS.from_documents(documents=docs, embedding=embedding)
    store.save_local(folder_path=vector_store_path)
    print("Saved successfully!")

# Load vector store
vector_store = FAISS.load_local(
    folder_path=vector_store_path,
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1}
)

query = "What role did Zoe play in the investigation?"

# Retrieve relevant documents
retrieved_docs = retriever.invoke(query)

print("Relevant documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"DOCUMENT: {i}")
    print(doc.page_content)

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful investiagtor agent, please help to find out the answers of the queries, by looking at the relavnt context. If you are not sure, just simply say I DONT'T KNOW. Don't halucinate.\n"
     "Context: {context}"),
    ("human", "Query: {query}")
])

# Create context from retrieved documents


def create_context(docs):
    context_list = [doc.page_content for doc in docs]
    return "\n".join(context_list)

# Create prompt with context and query


def create_prompt(context):
    return prompt_template.invoke({"context": context, "query": query})


# Build the final chain
chain = (
    retriever
    | RunnableLambda(lambda x: create_context(x))
    | RunnableLambda(lambda x: create_prompt(x))
    | model
    | StrOutputParser()
)
# Invoke the chain to get the answer
print("\nANSWER:")
result = chain.invoke(query)
print(result)

# RAG (Retrieval-Augmented Generation) Basics with LangChain - Step by Step Process:
# 1. Load environment variables and initialize embedding/chat models
# 2. Load documents from file using TextLoader
# 3. Split documents into chunks using RecursiveCharacterTextSplitter
# 4. Create a vector store (FAISS) from document chunks and embeddings
# 5. Load the vector store and create a retriever with similarity search
# 6. Define a prompt template with system and user message roles
# 7. Create functions to format context and prompts from retrieved documents
# 8. Build a chain that: retrieves docs → formats context → creates prompt → generates response
# 9. Invoke the chain with a query to get the final answer
