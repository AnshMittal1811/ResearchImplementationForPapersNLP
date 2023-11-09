from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import (
    LocalFileStore,
)
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def create_vector_db():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    loader = DirectoryLoader('data_test/', glob="**/*.txt", show_progress=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    fs = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, fs, namespace="sentence"
    )

    db = FAISS.from_documents(documents, cached_embedder)
    return db

db= create_vector_db()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
input_text = "Q: What are streaming LLMs in the context of Large Language Models? Give a brief overview of the paper."

def get_prediction_and_scores(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    outputs =  model.generate(input_ids, output_scores=True, return_dict_in_generate=True, max_length=100)

    generated_sequence = outputs.sequences[0]
    transition_scores = torch.exp(model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )[0])

    return tokenizer.decode(generated_sequence), generated_sequence, transition_scores

    

db = create_vector_db()

while True:
    print("input_text:", input_text)
    generated_sequence, tokens, scores = get_prediction_and_scores(input_text)
    print("generated_sequence:", generated_sequence)
    print("scores:", scores)
    # if any token is low in confidence, then do a RAG step
    if torch.min(scores) < 0.1:
        print("RAG step")
        query = generated_sequence
        docs = db.similarity_search(query)
        context = "\n".join([doc.page_content for doc in docs])
        new_input_text = f"{context}\n\n {query}"
        print("new_input_text:", new_input_text)
        generated_sequence, _, _ = get_prediction_and_scores(new_input_text)
        print("generated_sequence after RAG:", generated_sequence)
        input_text = f"{input_text} {generated_sequence}"
        print("input_text after RAG:", input_text)
        
    else:
        print("NO RAG step")
        input_text = f"{input_text} {generated_sequence}"
        if "</s>" in input_text:
            print("input_text:", input_text)
            break



