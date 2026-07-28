from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from src.database.base import get_db
from src.rag.qa_chain import RAGService
from routes.document_routes import vector_store

router = APIRouter(prefix="/search", tags=["Semantic Search & Q&A"])

# Initialize RAG Service sharing the vector store instance
rag_service = RAGService(vector_store)

# Request Models
class SearchRequest(BaseModel):
    query: str
    search_mode: str = "hybrid"  # semantic, keyword, hybrid
    doc_ids: Optional[List[str]] = None
    k: int = 4

class QARequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    search_mode: str = "hybrid"  # semantic, keyword, hybrid
    doc_ids: Optional[List[str]] = None
    k: int = 4

@router.post("/query")
async def retrieve_chunks(req: SearchRequest, db: Session = Depends(get_db)):
    """
    Performs retrieval across document chunks using the chosen search strategy:
    - Semantic Search (dense cosine similarity)
    - Keyword Search (TF-IDF keyword matching)
    - Hybrid Search (Ranks combined using RRF)
    """
    try:
        results = rag_service.retrieve_context(
            db=db,
            query=req.query,
            search_mode=req.search_mode,
            doc_ids=req.doc_ids,
            k=req.k
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa")
async def ask_question(req: QARequest, db: Session = Depends(get_db)):
    """
    Answers a query using context-grounded RAG, returning citation page mappings
    and preserving multi-turn memory based on session_id.
    """
    try:
        response = rag_service.answer_question(
            db=db,
            query=req.query,
            session_id=req.session_id,
            search_mode=req.search_mode,
            doc_ids=req.doc_ids,
            k=req.k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
