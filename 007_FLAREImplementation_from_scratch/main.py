from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import (
    LocalFileStore,
)
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# create a vector db from the transcripts
def create_vector_db():
    # we use sentence transformer to get the vector embeddings for the database
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # load all the transcripts stored in the data folder
    loader = DirectoryLoader('data_test/', glob="**/*.txt", show_progress=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

    # cache the embeddings for faster loadup
    fs = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, fs, namespace="sentence"
    )

    # create the vector db
    db = FAISS.from_documents(documents, cached_embedder)
    return db

# initialize the LLM and its tokenizer, we are using Flan T5 Large for this
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

# the input prompt for the LLM containing the question we want to ask
input_text = "Q: What are streaming LLMs in the context of Large Language Models? Give a brief overview of the paper.\nA:"

# create the vector db
db = create_vector_db()

# function to get the prediction and scores from the LLM, given a prompt
def get_prediction_and_scores(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs =  model.generate(input_ids, output_scores=True, return_dict_in_generate=True, max_length=100)
    generated_sequence = outputs.sequences[0]

    # get the probability scores for each generated token
    transition_scores = torch.exp(model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )[0])
    return tokenizer.decode(generated_sequence), generated_sequence, transition_scores


# keep generating tokens until we get a </s> token
while True:
    print("input_text:", input_text)
    # get the prediction and scores from the LLM, given the input and all the tokens generated so far
    generated_sequence, tokens, scores = get_prediction_and_scores(input_text)
    print("generated_sequence:", generated_sequence)
    # if any token is low in confidence, then do a RAG step
    if torch.min(scores) < 0.1:
        print("RAG step")
        # extract all tokens with high confidence as query
        high_confidence_tokens = tokens[torch.where(scores > 0.1)]
        query = tokenizer.decode(high_confidence_tokens)
        print("query: ", query)
        # get the context from the vector db
        docs = db.similarity_search(query)
        context = "\n".join([doc.page_content for doc in docs])
        new_input_text = f"Given the below context:\n{context}\n\n Answer the following \n{input_text}\n"
        print("new_input_text:", new_input_text)
        # get the prediction and scores from the LLM, given the new input
        generated_sequence, _, _ = get_prediction_and_scores(new_input_text)
        print("generated_sequence after RAG:", generated_sequence)
        input_text = f"{input_text} {generated_sequence}"
        if "</s>" in input_text:
            break
        
    else:
        # if all tokens are high in confidence, then just add the generated tokens to the input
        print("NO RAG step")
        input_text = f"{input_text} {generated_sequence}"
        if "</s>" in input_text:
            break
# print the final output
print("Final output:", input_text)


