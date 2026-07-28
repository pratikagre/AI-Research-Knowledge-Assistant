from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.analytics.metrics import AnalyticsManager

router = APIRouter(prefix="/analytics", tags=["System Analytics"])

@router.get("")
async def get_analytics(db: Session = Depends(get_db)):
    """
    Retrieves system analytics and statistics, including:
    - Number of processed documents
    - Number of processed chunks
    - Embedding generation counts
    - Predefined category counts (TensorFlow predicted domains)
    - Top referenced/most queried documents
    - Total questions answered
    """
    try:
        stats = AnalyticsManager.get_system_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
