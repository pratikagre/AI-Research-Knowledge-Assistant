from typing import List, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from src.database.models import Chunk, Document
from config.settings import settings

class SummaryResponseSchema(BaseModel):
    executive_summary: str
    technical_summary: str
    bullet_points: List[str]
    key_takeaways: List[str]

class Summarizer:
    def __init__(self):
        self.client = genai.Client()
        self.llm_model = "gemini-2.5-flash"

    def summarize_document(self, db: Session, doc_id: str) -> Dict[str, Any]:
        """
        Generates Executive, Technical, Bullet Point summaries and Key Takeaways for a document.
        """
        # Fetch document metadata
        doc = db.query(Document).filter_by(doc_id=doc_id).first()
        if not doc:
            raise ValueError(f"Document with ID {doc_id} not found.")

        # Fetch all chunks
        chunks = db.query(Chunk).filter_by(doc_id=doc_id).order_by(Chunk.chunk_index).all()
        if not chunks:
            raise ValueError(f"No text chunks found for document {doc_id}.")

        # Concatenate text
        full_text = "\n".join([c.text for c in chunks])

        # Prompt
        prompt = f"""You are an expert technical writer and AI assistant.
Your task is to summarize the following document: "{doc.file_name}"

Please generate:
1. Executive Summary: High-level business overview.
2. Technical Summary: Detailed review of methodologies, system design, and algorithms.
3. Bullet Point Summary: Key details in structured bullet points.
4. Key Takeaways: Top actionable lessons or observations.

Document Content:
{full_text}
"""

        try:
            # Query Gemini
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SummaryResponseSchema,
                    temperature=0.2
                )
            )
            
            # Load parsed response JSON
            res_data = SummaryResponseSchema.model_validate_json(response.text)
            
            return {
                "doc_id": doc_id,
                "file_name": doc.file_name,
                "executive_summary": res_data.executive_summary,
                "technical_summary": res_data.technical_summary,
                "bullet_points": res_data.bullet_points,
                "key_takeaways": res_data.key_takeaways
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate summary via Gemini: {e}")
