from langchain import BaseCache
from cassandra.cluster import Session
from langchain.schema import Generation
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM, get_prompts
from cassio.vector import VectorTable
import hashlib
import json
from functools import lru_cache


from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

CASSANDRA_CACHE_DEFAULT_TABLE_NAME = "langchain_response_cache"
CASSANDRA_CACHE_DEFAULT_TTL_SECONDS = None
#
CASSANDRA_SEMANTIC_CACHE_DEFAULT_NUM_ROWS_TO_FETCH = 8
CASSANDRA_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE = 16
CASSANDRA_SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD = 0.85
CASSANDRA_SEMANTIC_CACHE_TABLE_NAME_PREFIX = "semantic_cache_"
CASSANDRA_SEMANTIC_CACHE_DEFAULT_TTL_SECONDS = None
RETURN_VAL_TYPE = List[Generation]


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


def _dump_generations_to_json(generations: RETURN_VAL_TYPE) -> str:
    """Dump generations to json.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: Json representing a list of generations.
    """
    return json.dumps([generation.dict() for generation in generations])


def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    """Load generations from json.

    Args:
        generations_json (str): A string of json representing a list of generations.

    Raises:
        ValueError: Could not decode json string to list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )



class CassandraCache(BaseCache):
    """
    Cache that uses Cassandra / Astra DB as a backend.

    It uses a single table. The lookup keys (also primary keys) are:
        - prompt, a string
        - llm_string, a string deterministic representation of the model parameters.
          This is to keep collision between same prompts for two models separate.

    # TODO: should a cache hit "reset" the ttl ?
            (now: no. It can be done, not sure if 'natural')
    """

    def __init__(
        self,
        session: Session,
        keyspace: str,
        tableName: str = CASSANDRA_CACHE_DEFAULT_TABLE_NAME,
        ttl_seconds: int | None = CASSANDRA_CACHE_DEFAULT_TTL_SECONDS,
    ):
        """Initialize with a ready session and a keyspace name."""
        try:
            from cassio.keyvalue import KVCache
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )

        self.ttlSeconds = ttl_seconds
        self.kvCache = KVCache(
            session, keyspace, tableName, keys=["llm_string", "prompt"]
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        print(f'Looking up in Cache {prompt} Table:{_hash(llm_string)}')
        """Look up based on prompt and llm_string."""
        foundBlob = self.kvCache.get(
            {"llm_string": _hash(llm_string), "prompt": _hash(prompt)},
        )
        if foundBlob:
            return _load_generations_from_json(foundBlob)
        else:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        print(f'Updating Cache {prompt} Table:{_hash(llm_string)}')
        """Update cache based on prompt and llm_string."""
        blobToStore = _dump_generations_to_json(return_val)
        self.kvCache.put(
            {"llm_string": _hash(llm_string), "prompt": _hash(prompt)},
            blobToStore,
            self.ttlSeconds,
        )

    def delete_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> None:
        """
        A wrapper around `delete` with the LLM being passed.
        In case the llm(prompt) calls have a `stop` param, you should pass it here
        """
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.delete(prompt, llm_string=llm_string)

    def delete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        return self.kvCache.delete(
            {"llm_string": _hash(llm_string), "prompt": _hash(prompt)},
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. This is for all LLMs at once."""
        self.kvCache.clear()


class CassandraSemanticCache(BaseCache):
    """
    Cache that uses Cassandra as a vector-store backend,
    based on the CEP-30 drafts at the moment.
    """

    def __init__(
        self,
        session: Session,
        keyspace: str,
        embedding: Embeddings,
        distance_metric: str = "dot",
        score_threshold: float = CASSANDRA_SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD,
        num_rows_to_fetch: int = CASSANDRA_SEMANTIC_CACHE_DEFAULT_NUM_ROWS_TO_FETCH,
        table_name_prefix: str = CASSANDRA_SEMANTIC_CACHE_TABLE_NAME_PREFIX,
        ttl_seconds: int | None = CASSANDRA_SEMANTIC_CACHE_DEFAULT_TTL_SECONDS,
    ):
        """Initialize the cache with all relevant parameters.
        Args:
            session (cassandra.cluster.Session): an open Cassandra session
            keyspace (str): the keyspace to use for storing the cache
            embedding (Embedding): Embedding provider for semantic encoding and search.
            distance_metric (str, 'dot')
            score_threshold (optional float)
        The default score threshold is tuned to the default metric.
        Tune it carefully yourself if switching to another distance metric.
        """
        self.session = session
        self.keyspace = keyspace
        self.embedding = embedding
        self.score_threshold = score_threshold
        self.distance_metric = distance_metric
        self.num_rows_to_fetch = num_rows_to_fetch
        self.table_name_prefix = table_name_prefix
        self.ttlSeconds = ttl_seconds
        # A single instance of these can handle a number of 'llm_strings'.
        # We map each to a separate table. These are cached here:
        self.table_cache: Dict[
            str, VectorTable
        ] = {}  # model_str -> vector table object

        # The contract for this class has separate lookup and update:
        # in order to spare some embedding calculations we cache them between
        # the two calls.
        # Note: each instance of this class has its own `_get_embedding` with
        # its own lru.
        @lru_cache(maxsize=CASSANDRA_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        def _cache_embedding(text: str) -> List[float]:
            return self.embedding.embed_query(text=text)

        self._get_embedding = _cache_embedding
        self._embedding_dimension = self._getEmbeddingDimension()

    def _getEmbeddingDimension(self) -> int:
        return len(self._get_embedding(text="This is a sample sentence."))

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        if "llm_string" not in kwargs:
            raise ValueError("llm_string parameter must be passed to clear()")
        else:
            vectorTable = self._getVectorTable(kwargs["llm_string"])
            vectorTable.clear()

    def clear_through_llm(self, llm: LLM, stop: Optional[List[str]] = None) -> None:
        """
        A wrapper around `clear` with the LLM being passed.
        In case the llm(prompt) calls have a `stop` param, you should pass it here
        """
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        self.clear(llm_string=llm_string)

    def _getVectorTable(self, llm_string: str) -> VectorTable:
        if llm_string not in self.table_cache:
            try:
                from cassio.vector import VectorTable
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "Could not import cassio python package. "
                    "Please install it with `pip install cassio`."
                )
            #
            tableName = f"{self.table_name_prefix}{_hash(llm_string)}"
            #
            print(f'Table name: {tableName}')
            self.table_cache[llm_string] = VectorTable(
                session=self.session,
                keyspace=self.keyspace,
                table=tableName,
                embedding_dimension=self._embedding_dimension,
                primary_key_type='text'              
            )
        #
        return self.table_cache[llm_string]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:        
        print(f'Updating Cache {prompt} Table:{_hash(llm_string)}')
        """Update cache based on prompt and llm_string."""
        vectorTable = self._getVectorTable(llm_string)
        #
        embedding_vector = self._get_embedding(text=prompt)
        metadata = {
            "generations_str": _dump_generations_to_json(return_val),
        }
        documentId = _hash(prompt)
        #
        vectorTable.put(
            document=prompt,
            embedding_vector=embedding_vector,
            document_id=documentId,
            metadata=metadata,
            ttl_seconds=self.ttlSeconds,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        print(f'Lookup from cache {prompt} Table:{_hash(llm_string)}')
        """Look up based on prompt and llm_string."""
        hitWithId = self.lookup_with_id(prompt, llm_string)
        if hitWithId is not None:            
            return hitWithId[1]
        else:            
            return None

    def lookup_with_id(
        self, prompt: str, llm_string: str
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry)
        """
        vectorTable = self._getVectorTable(llm_string)
        #
        promptEmbedding: List[float] = self._get_embedding(text=prompt)
        hits = vectorTable.search(
            embedding_vector=promptEmbedding,
            top_k=1,
            metric=self.distance_metric,
            metric_threshold=self.score_threshold,
        )        
        if hits:
            hit = hits[0]
            metadata = hit["metadata"]
            return (
                hit["document_id"],
                _load_generations_from_json(metadata["generations_str"]),
            )
        else:
            return None

    def lookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.lookup_with_id(prompt, llm_string=llm_string)

    def delete_by_document_id(self, document_id: str, llm_string: str) -> None:
        """
        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        vectorTable = self._getVectorTable(llm_string)
        vectorTable.delete(document_id)

    def delete_by_document_id_through_llm(
        self, document_id: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> None:
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.delete_by_document_id(document_id, llm_string=llm_string)