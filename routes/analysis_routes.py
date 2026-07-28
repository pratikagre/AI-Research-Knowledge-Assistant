from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from src.database.base import get_db
from src.rag.summarizer import Summarizer
from src.rag.comparator import DocumentComparator
from routes.document_routes import classifier

router = APIRouter(prefix="/analysis", tags=["Document Analysis & ML"])

# Initialize services
summarizer = Summarizer()
comparator = DocumentComparator()

# Request Models
class SummarizeRequest(BaseModel):
    doc_id: str

class CompareRequest(BaseModel):
    doc_ids: List[str]

class ClassifyRequest(BaseModel):
    text: str

@router.post("/summarize")
async def summarize_document(req: SummarizeRequest, db: Session = Depends(get_db)):
    """
    Generates structured summaries for a document including:
    - Executive Summary
    - Technical Summary
    - Bullet Points
    - Key Takeaways
    """
    try:
        summary = summarizer.summarize_document(db, req.doc_id)
        return summary
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_documents(req: CompareRequest, db: Session = Depends(get_db)):
    """
    Compares two or more documents across various aspects:
    - Methodologies
    - Advantages & Disadvantages
    - Similarities & Differences
    - Conclusions
    - Implementation Approaches
    - Generates a comparison matrix
    """
    if len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Comparison requires at least two document IDs.")
        
    try:
        comparison = comparator.compare_documents(db, req.doc_ids)
        return comparison
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
async def classify_text(req: ClassifyRequest):
    """
    Runs inference using the custom TensorFlow model to predict the domain category
    of the input technical text.
    """
    try:
        predicted_category = classifier.predict_category(req.text)
        return {
            "input_text_sample": req.text[:150] + "..." if len(req.text) > 150 else req.text,
            "predicted_category": predicted_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
