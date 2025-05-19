from langchain.memory import ConversationBufferMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import faiss

class MemoryManager:
    """Manager for conversation memory and vector storage"""
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the memory manager
        
        Args:
            embedding_model (str): Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = FAISS.from_texts([], self.embeddings)
        self.memory = VectorStoreRetrieverMemory(
            retriever=self.vector_store.as_retriever()
        )

    def add_memory(self, text: str) -> None:
        """
        Add text to memory
        
        Args:
            text (str): Text to add to memory
        """
        self.vector_store.add_texts([text])

    def retrieve_memory(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieve relevant memories based on query
        
        Args:
            query (str): Query to search for
            k (int): Number of results to return
            
        Returns:
            List[str]: Retrieved memories
        """
        return self.memory.load_memory_variables({"query": query})["history"]

    def clear_memory(self) -> None:
        """Clear all stored memories"""
        self.vector_store = FAISS.from_texts([], self.embeddings)
