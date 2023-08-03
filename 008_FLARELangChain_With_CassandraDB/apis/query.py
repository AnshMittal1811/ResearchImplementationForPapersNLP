import sys
import argparse
import langchain
from langchain.chains import FlareChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from cassandra_retriever import CassandraRetriever
from db import get_astra
from cache import CassandraCache, CassandraSemanticCache

if __name__ == '__main__':
    langchain.verbose = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help='The query argument')
    parser.add_argument('--category', type=str, help='The category argument')
    args = parser.parse_args()
    query = args.query
    category = args.category
    
    embeddings = OpenAIEmbeddings()
    
    session, keyspace, table = get_astra()         
    cassandraCache = CassandraCache(session=session, keyspace=keyspace)
    # cassandraCache = CassandraSemanticCache(session=session, keyspace=keyspace, embedding=embeddings)
    retriever = CassandraRetriever()
    retriever.init_values(embedding=embeddings,
                          session=session,
                          keyspace=keyspace,
                          table_name=table,
                          category_filter=category)
    
    langchain.llm_cache = cassandraCache
    
    llm = OpenAI()
    llm_result = llm(query)
    print(llm_result)

    # flare = FlareChain.from_llm(
    #     ChatOpenAI(temperature=0),
    #     retriever=retriever,
    #     max_generation_len=164,
    #     min_prob=0.3,        
    # )        
    
    # flare_result = flare.run(query)    
    # response = {
    #     'query': query,
    #     'flare_result': flare_result
    # }
    # print(f"QUERY: {query}\n\n")
    # print(f"FLARE RESULT:\n    {flare_result}\n\n")        

