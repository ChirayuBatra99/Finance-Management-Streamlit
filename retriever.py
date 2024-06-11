import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
llm= OpenAI(temperature=0.9 , max_tokens=500)

loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])
data = loaders.load() 
print(len(data))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(data)
print(len(docs))


embeddings= OpenAIEmbeddings()
vectorindex_openai=FAISS.from_documents(docs, embeddings)

print(vectorindex_openai)
# file_path="vector_index.pkl"



vector_index = FAISS.from_documents(docs, embeddings)
vector_index.save_local('vector_index')





# print(pickle.format_version)
file_path="vector_index.pickle"
with open(file_path, "wb") as f:
    pickle.dump(vectorindex_openai, f)

# if os.path.exists(file_path):
#     with open(file_path, "rb") as f:
#         vectorIndex = pickle.load(f)


# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
# print(chain)