from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.models import Document, Chunk, ConversationLog, ReferenceLog
from typing import Dict, Any

class AnalyticsManager:
    @staticmethod
    def get_system_stats(db: Session) -> Dict[str, Any]:
        """
        Computes system usage analytics.
        """
        # Count documents
        total_docs = db.query(Document).count()

        # Count chunks
        total_chunks = db.query(Chunk).count()

        # Total questions answered
        total_questions = db.query(ConversationLog).count()

        # Query categories distribution
        category_counts = db.query(
            Document.category, func.count(Document.doc_id)
        ).group_by(Document.category).all()
        
        categories_distribution = {cat: count for cat, count in category_counts}

        # Query top queried/referenced documents
        top_referenced = db.query(
            Document.doc_id, Document.file_name, func.count(ReferenceLog.id).label("ref_count")
        ).join(ReferenceLog, Document.doc_id == ReferenceLog.doc_id)\
         .group_by(Document.doc_id)\
         .order_by(func.count(ReferenceLog.id).desc())\
         .limit(5).all()

        most_queried = [
            {
                "doc_id": r.doc_id,
                "file_name": r.file_name,
                "reference_count": r.ref_count
            }
            for r in top_referenced
        ]

        return {
            "total_documents": total_docs,
            "total_processed_chunks": total_chunks,
            "total_embeddings_generated": total_chunks,  # 1 embedding per chunk
            "total_questions_answered": total_questions,
            "categories_distribution": categories_distribution,
            "most_queried_documents": most_queried
        }
