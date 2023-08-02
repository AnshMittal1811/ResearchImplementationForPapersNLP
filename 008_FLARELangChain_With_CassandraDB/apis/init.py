from db import get_astra
session, keyspace, table = get_astra()
embedding_dimension = 1536

create_vector_table = f"""
CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
    document_id UUID PRIMARY KEY,
    embedding_vector VECTOR<FLOAT, {embedding_dimension}>,
    document TEXT,
    metadata_blob TEXT,
    category TEXT,
)
"""

create_vector_index = f"""
CREATE CUSTOM INDEX IF NOT EXISTS {table}_embedding_vector_index ON {keyspace}.{table} (embedding_vector)
USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' ;
"""

create_category_index = f"""
CREATE CUSTOM INDEX IF NOT EXISTS {table}_category_index ON {keyspace}.{table} (category)
USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' ;
"""

# session.execute(create_vector_table)
# session.execute(create_vector_index)
session.execute(create_category_index)

print('Schema created')