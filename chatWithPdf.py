import langchain
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFDirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import warnings
warnings.filterwarnings("ignore")


os.environ['GOOGLE_API_KEY']
genai.configure(api_key= os.environ['GOOGLE_API_KEY'])

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                              temperature=0.3,
                              convert_system_message_to_human=True)
embeddings= GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
vectorStore_file_path= "faiss_index"

def create_vector_db():
    pdf_loader= PyPDFDirectoryLoader("pdf")
    data= pdf_loader.load()

    splitter= RecursiveCharacterTextSplitter(
    separators= ['\n\n', '\n', '.', ','],
    chunk_size=1000
    )
    chunks= splitter.split_documents(data)
    vectorStore= FAISS.from_documents(documents=chunks, embedding= embeddings)
    vectorStore.save_local(vectorStore_file_path)

def get_qa_chain():
    vectorStore= FAISS.load_local(vectorStore_file_path, embeddings)
    retriever= vectorStore.as_retriever()
    prompt_template= """
        Given the following context and a question, generate an answer based on this context.
        Use your reasoning abilities to understand the question
        In the answer try to provide as much text as possible from each section that contains similar meaning in the source document.
        Give the correct answer to the question.

        CONTEXT: {context}
        QUESTION: {question} """

    prompt = PromptTemplate(
        template= prompt_template,
        input_variables= ["context", "question"]
    )
    chain= RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever= retriever,
                                        input_key= "query",
                                        return_source_documents=True,
                                        chain_type_kwargs= {"prompt": prompt}
            )
    return chain

if __name__ == "__main__":
    chain= get_qa_chain()
