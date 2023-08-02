from typing import List, Any, Optional
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import Callbacks
from cassandra.cluster import Session
from langchain.embeddings.base import Embeddings
import json 

class CassandraRetriever(BaseRetriever):    
    embedding: Embeddings = None
    session: Session = None 
    keyspace: str = None 
    table_name: str = None 
    category_filter: str = None 
    embedding_column: str = None
    category_column: str = None 
    document_column: str = None 
    metadata_column: str = None 
    ttl_seconds: str = None 

    class Config:
        arbitrary_types_allowed = True

    def init_values(
        self,
        embedding: Embeddings,
        session: Session,
        keyspace: str,
        table_name: str,
        category_filter: Optional[str] = "default",
        embedding_column: Optional[str] = "embedding_vector",
        category_column: Optional[str] = "category",
        document_column: Optional[str] = "document",
        metadata_blob: Optional[str] = "metadata_blob",
        ttl_seconds: Optional[int] = None,        
    ) -> None:        
        self.embedding = embedding
        self.session = session
        self.keyspace = keyspace 
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.category_filter = category_filter
        self.category_column = category_column
        self.embedding_column = embedding_column
        self.document_column = document_column
        self.metadata_column = metadata_blob        
        return None

    def get_relevant_documents(self, query: str) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        q = self.session.prepare(f"""SELECT * FROM {self.keyspace}.{self.table_name} 
                WHERE {self.category_column} = ?
                ORDER BY {self.embedding_column} ANN OF ? LIMIT 5
            """)
        results = self.session.execute(q, (self.category_filter,embedding_vector))
        if not results:
            return []        
        return [
            Document(page_content=row.document)
            for row in results
        ]        
    
    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        raise('Not supported')
        