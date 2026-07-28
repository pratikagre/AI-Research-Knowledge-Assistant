import os
import chromadb
from typing import List, Dict, Any
from google import genai
from config.settings import settings

class VectorStoreManager:
    def __init__(self):
        # Initialize Google GenAI client
        # Loaded automatically from environment (including via ~/.env loaded in settings)
        self.client = genai.Client()
        self.embedding_model = "gemini-embedding-2"
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection("document_chunks")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calls Gemini API to generate embeddings for a list of texts.
        """
        if not texts:
            return []
        
        # Process in batches of 100 to avoid API payload limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch_texts
                )
                # Extract vector values
                embeddings = [e.values for e in response.embeddings]
                all_embeddings.extend(embeddings)
            except Exception as e:
                raise RuntimeError(f"Gemini embedding API call failed: {e}")
                
        return all_embeddings

    def add_chunks(self, doc_id: str, file_name: str, chunks: List[Dict[str, Any]]):
        """
        Adds text chunks to ChromaDB with their embeddings and metadata.
        """
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self._get_embeddings(texts)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def delete_document(self, doc_id: str):
        """
        Deletes all chunks associated with a document.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            # If nothing exists, delete might error on some versions
            pass

    def search(self, query: str, doc_ids: List[str] = None, k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches the collection using semantic similarity.
        """
        # Generate query embedding
        query_embeddings = self._get_embeddings([query])
        if not query_embeddings:
            return []

        # Prepare filter (where clause)
        where_filter = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        # Perform query
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            where=where_filter
        )

        # Format results
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
            ids = results["ids"][0]

            for i in range(len(documents)):
                formatted_results.append({
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                    # Cosine similarity representation (approximate depending on Chroma distance metric)
                    "score": round(1.0 - (distances[i] / 2.0), 4) if distances[i] is not None else 0.5
                })

        return formatted_results
