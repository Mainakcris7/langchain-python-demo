import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

file_path = os.path.join("data", "story2.txt")

if not os.path.exists(file_path):
    raise FileNotFoundError("File not found!");

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_GPT4O_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_GPT4O_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_GPT4O_API_VERSION"]
)

text_loader = TextLoader(file_path=file_path, encoding='utf-8')
docs = text_loader.load()
whole_document = "\n\n".join([d.page_content for d in docs])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# Stuff document chain
# Takes the whole document and summarize it
def get_stuff_chain_result():
    stuff_chain_template = [
        ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the whole ducument that is passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you!"),
        ('user', 'Please summarize the document: {document}')
    ]

    stuff_chain_prompt = ChatPromptTemplate.from_messages(stuff_chain_template)

    stuff_chain = stuff_chain_prompt | llm | StrOutputParser()

    print("Summarizing using stuff chain...")
    stuff_chain_result = stuff_chain.invoke(whole_document)
    return stuff_chain_result

# Map-reduce chain
# First generates the summaries of small chunks of the document (MAP)
# Then generates the summary of all the summaries (REDUCE)
def get_map_reduce_chain_result():
    chunks = text_splitter.split_documents(documents=docs)
    map_chain_template = [
        ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the ducument chunk that is passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you. DON'T GENERATE ANY HATE, ABUSIVE CONTENT"),
        ('user', 'Please summarize the chunk: {document}')
    ]

    map_chain_prompt = ChatPromptTemplate.from_messages(map_chain_template)

    map_chain = map_chain_prompt | llm | StrOutputParser()

    print("Summarizing using map-reduce chain...")
    map_chain_result = [map_chain.invoke(chunk.page_content) for chunk in chunks]
    
    all_summaries = "\n".join(map_chain_result)

    reduce_chain_template = [
        ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the summaries (basically you need to provide summary of the summaries) that is passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you!"),
        ('user', 'Please summarize the summaries : {document}')
    ]

    reduce_chain_prompt = ChatPromptTemplate.from_messages(reduce_chain_template)
    
    reduce_chain = reduce_chain_prompt | llm | StrOutputParser()
    
    return reduce_chain.invoke(all_summaries)

# Refine chain
# First generates the summary of the 1st chunk, subsequently uses the last summary + current chunk for summarization
def get_refine_chain_result():
    chunks = text_splitter.split_documents(documents=docs)
    refine_chain_template = [
        ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the ducument chunk, based on the previous summary (and obviously using the current chunk info) that are passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you. IMPORTANT: For the first chunk the previous summary will be empty!"),
        ('user', 'Please summarize the chunk: {document}, based on the previous chunk summary: {prev_summary}')
    ]

    refine_chain_prompt = ChatPromptTemplate.from_messages(refine_chain_template)

    refine_chain = refine_chain_prompt | llm | StrOutputParser()

    print("Summarizing using refine chain...")
    
    last_summary = refine_chain.invoke({"document": chunks[0].page_content, "prev_summary": ""})
    final_summary = ""
    
    for chunk in chunks[1:]:
        final_summary = refine_chain.invoke({"document": chunk.page_content, "prev_summary": last_summary})
        last_summary = final_summary
    
    return final_summary

stuff_chain_result = get_stuff_chain_result()
print("STUFF CHAIN SUMMARY: ")
print(stuff_chain_result)

print("\n")
map_reduce_chain_result = get_map_reduce_chain_result()
print("MAP REDUCE CHAIN SUMMARY: ")
print(map_reduce_chain_result)

print("\n")
refine_chain_result = get_refine_chain_result()
print("REFINE CHAIN SUMMARY: ")
print(refine_chain_result)