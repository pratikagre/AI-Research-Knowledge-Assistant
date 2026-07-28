import re
from math import log
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from src.database.models import Chunk, ConversationLog, ReferenceLog, Document
from src.vector_store.manager import VectorStoreManager
from config.settings import settings

class Citation(BaseModel):
    document_name: str
    page_number: int

class QAResponseSchema(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float  # Scale 0.0 to 1.0

class RAGService:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.client = genai.Client()
        self.llm_model = "gemini-2.5-flash"

    def _keyword_search(self, db: Session, query: str, doc_ids: Optional[List[str]] = None, k: int = 4) -> List[Dict[str, Any]]:
        """
        Executes TF-IDF keyword search on the chunks table in SQLite.
        """
        # Fetch chunks
        query_builder = db.query(Chunk)
        if doc_ids:
            query_builder = query_builder.filter(Chunk.doc_id.in_(doc_ids))
        chunks = query_builder.all()

        if not chunks:
            return []

        # Simple tokenization
        query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 1]
        if not query_words:
            # Fallback to first k
            return [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": {
                        "doc_id": c.doc_id,
                        "file_name": db.query(Document.file_name).filter_by(doc_id=c.doc_id).scalar() or "Unknown",
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index
                    },
                    "score": 0.1
                }
                for c in chunks[:k]
            ]

        # Compute document frequency (DF)
        doc_count = len(chunks)
        df = {}
        for chunk in chunks:
            words = set(re.findall(r'\w+', chunk.text.lower()))
            for w in query_words:
                if w in words:
                    df[w] = df.get(w, 0) + 1

        # Score chunks
        scored_chunks = []
        for chunk in chunks:
            score = 0.0
            words = chunk.text.lower()
            chunk_word_list = words.split()
            chunk_len = len(chunk_word_list) + 1
            
            for w in query_words:
                count = words.count(w)
                if count > 0:
                    idf = log(1.0 + (doc_count - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
                    tf = count / chunk_len
                    score += tf * idf

            if score > 0:
                # Fetch filename
                file_name = db.query(Document.file_name).filter_by(doc_id=chunk.doc_id).scalar() or "Unknown"
                scored_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "file_name": file_name,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index
                    },
                    "score": score
                })

        # Sort and limit
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:k]

    def _hybrid_search(self, db: Session, query: str, doc_ids: Optional[List[str]] = None, k: int = 4) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Search combining Semantic and Keyword ranks via Reciprocal Rank Fusion (RRF).
        """
        # Get semantic results (fetch up to 10 for RRF ranking)
        semantic_results = self.vector_store.search(query, doc_ids, k=10)
        
        # Get keyword results (fetch up to 10 for RRF ranking)
        keyword_results = self._keyword_search(db, query, doc_ids, k=10)

        # Apply Reciprocal Rank Fusion (RRF)
        # Score(d) = sum(1 / (60 + r))
        rrf_scores = {}
        chunk_map = {}

        # Process semantic
        for rank, res in enumerate(semantic_results, start=1):
            cid = res["chunk_id"]
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))

        # Process keyword
        for rank, res in enumerate(keyword_results, start=1):
            cid = res["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))

        # Sort chunk IDs by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Construct final hybrid chunks list
        hybrid_results = []
        for cid in sorted_cids[:k]:
            res = chunk_map[cid]
            res["score"] = round(rrf_scores[cid], 4)
            hybrid_results.append(res)

        return hybrid_results

    def retrieve_context(self, db: Session, query: str, search_mode: str = "hybrid", doc_ids: Optional[List[str]] = None, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves context chunks based on search mode ('semantic', 'keyword', 'hybrid').
        """
        search_mode = search_mode.lower()
        if search_mode == "semantic":
            return self.vector_store.search(query, doc_ids, k=k)
        elif search_mode == "keyword":
            return self._keyword_search(db, query, doc_ids, k=k)
        else:  # hybrid
            return self._hybrid_search(db, query, doc_ids, k=k)

    def answer_question(self, db: Session, query: str, session_id: Optional[str] = None, search_mode: str = "hybrid", doc_ids: Optional[List[str]] = None, k: int = 4) -> Dict[str, Any]:
        """
        Answers a user query using citation-grounded RAG and preserves conversation memory.
        """
        # Retrieve top chunks
        chunks = self.retrieve_context(db, query, search_mode, doc_ids, k)

        # Construct context string
        context_str = ""
        for i, c in enumerate(chunks):
            doc_name = c["metadata"].get("file_name", "Unknown")
            page_num = c["metadata"].get("page_number", "N/A")
            context_str += f"\n[Context Chunk {i+1}] Source: {doc_name} (Page {page_num})\n{c['text']}\n"

        # Load conversation history
        history_str = ""
        if session_id:
            history = db.query(ConversationLog).filter_by(session_id=session_id).order_by(ConversationLog.timestamp.desc()).limit(5).all()
            # Reverse history to maintain chronological order
            for log_entry in reversed(history):
                history_str += f"User: {log_entry.question}\nAssistant: {log_entry.answer}\n"

        # Structured RAG Prompt
        prompt = f"""You are a specialized AI Research Assistant.
Your task is to answer the user's question based strictly on the provided Context Chunks.

Guidelines:
1. Base your answer ONLY on the provided Context Chunks. Do not use external knowledge.
2. If the context does not contain enough information to answer the question, set the answer field exactly to: "I cannot determine the answer from the provided documents." and set confidence_score to 0.0.
3. Keep the answer clear, professional, and grounded.
4. For every claim you make, cite the corresponding source document and page number. Citing documents that are not in the context is forbidden.

Conversation History:
{history_str}

Context Chunks:
{context_str}

User Question: {query}
"""

        try:
            # Query Gemini using structured JSON output
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QAResponseSchema,
                    temperature=0.0
                )
            )
            
            # Load parsed response JSON
            res_data = QAResponseSchema.model_validate_json(response.text)
            answer = res_data.answer
            citations = [cit.model_dump() for cit in res_data.citations]
            confidence_score = res_data.confidence_score
        except Exception as e:
            # Fallback in case of structured schema failure
            print(f"RAG Structured generation failed: {e}")
            answer = "I cannot determine the answer from the provided documents."
            citations = []
            confidence_score = 0.0

        # Log conversation in DB
        if session_id:
            convo = ConversationLog(
                session_id=session_id,
                question=query,
                answer=answer
            )
            db.add(convo)
            db.commit()

        # Log references for analytics
        # Extract unique doc_ids from cited document names
        cited_doc_names = {c["document_name"] for c in citations}
        for doc_name in cited_doc_names:
            doc = db.query(Document).filter_by(file_name=doc_name).first()
            if doc:
                ref = ReferenceLog(doc_id=doc.doc_id)
                db.add(ref)
        
        db.commit()

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": [c["text"] for c in chunks],
            "confidence_score": confidence_score
        }
