import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.8,
        max_new_tokens=512,
        task="conversational",
    ) # type: ignore
    chatmodel=ChatHuggingFace(llm=llm)
    return chatmodel

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.
If the context is not enough to answer the question, just say that you dont know.
If the user's question asked for a specific piece of information, try to provide that specific piece of information.
If the user's question is not clear, ask for clarification.
If the user's question requires a specific format, try to provide that format.
If user asks for source IPs, provide the source IPs from the context of the attack attempts.
If user asks for successful attack attempts, provide the successful attack attempts from the context by checking if the success was TRUE.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain with RetrievalQA
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(
        search_kwargs={
            # 'k':1
            }
        ),
    return_source_documents=False,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Create QA chain with ConversationalRetrievalChain
conv_qa_chain=ConversationalRetrievalChain.from_llm(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    retriever=db.as_retriever(
        search_kwargs={
            'k':100
            }
        ),
    return_source_documents=False,
    combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)
# Now invoke with a single query
user_query=input("Write Query Here: ")

#answer with RetrievalQA:
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"], )
# # print("SOURCE DOCUMENTS: ", response["source_documents"])

#answer with ConversationalRetrievalChain:
# response = conv_qa_chain.invoke({'question': user_query, 'chat_history': []})
# print("RESULT: ", response["answer"], )
# print("SOURCE DOCUMENTS: ", response["source_documents"])
