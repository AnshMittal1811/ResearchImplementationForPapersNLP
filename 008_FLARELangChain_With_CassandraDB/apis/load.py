# 1. One-off: load docs into a vector store
import os
import uuid
import json

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Cassandra
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from db import get_astra
from cassandra.concurrent import execute_concurrent_with_args

SOURCE_DIR = "sources"
FILE_SUFFIX = ".pdf"
category = "default"
DESTINATION_DIR = "completed"

def move_files(source_folder, destination_folder):
    files = os.listdir(source_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        os.rename(source_path, destination_path)
        print(f"Moved '{file}' to '{destination_folder}'")


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    
    pdf_loaders = [
        PyPDFLoader(pdf_name)
        for pdf_name in (
            f for f in (
                os.path.join(SOURCE_DIR, f2)
                for f2 in os.listdir(SOURCE_DIR)
            )
            if os.path.isfile(f)
            if f[-len(FILE_SUFFIX):] == FILE_SUFFIX
        )
    ]

    session, keyspace, table  = get_astra()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    documents = [
        doc
        for loader in pdf_loaders
        for doc in loader.load_and_split(text_splitter=text_splitter)
    ]    
    texts, metadatas = zip(*((doc.page_content, doc.metadata) for doc in documents))        
    embedding_vectors = embeddings.embed_documents(texts)
    parameters = [(embedding,text,str(metadata),category) for embedding,text,metadata in zip(embedding_vectors,texts,metadatas)]
    statement = session.prepare(f"INSERT INTO {keyspace}.{table} (document_id, embedding_vector, document, metadata_blob, category) VALUES (uuid(),?,?,?,?)")
    execute_concurrent_with_args(session, statement, parameters, concurrency=16)
    move_files(SOURCE_DIR,DESTINATION_DIR)