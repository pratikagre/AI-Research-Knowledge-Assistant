from typing import List, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from src.database.models import Chunk, Document
from config.settings import settings

class ComparisonResponseSchema(BaseModel):
    methodologies_comparison: str
    advantages_disadvantages: str
    similarities: List[str]
    differences: List[str]
    conclusions_comparison: str
    implementation_approaches: str
    comparison_matrix_markdown: str

class DocumentComparator:
    def __init__(self):
        self.client = genai.Client()
        self.llm_model = "gemini-2.5-flash"

    def compare_documents(self, db: Session, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Compares two or more documents on methodologies, pros/cons, similarities, differences, and conclusions.
        """
        if len(doc_ids) < 2:
            raise ValueError("Comparison requires at least two documents.")

        documents_content = []
        doc_details = []

        for doc_id in doc_ids:
            doc = db.query(Document).filter_by(doc_id=doc_id).first()
            if not doc:
                raise ValueError(f"Document with ID {doc_id} not found.")

            # Get chunks
            chunks = db.query(Chunk).filter_by(doc_id=doc_id).order_by(Chunk.chunk_index).all()
            if not chunks:
                raise ValueError(f"No text found for document {doc_id}.")

            text_content = "\n".join([c.text for c in chunks])
            documents_content.append(f"<document id=\"{doc_id}\" name=\"{doc.file_name}\">\n{text_content}\n</document>")
            doc_details.append({"doc_id": doc_id, "file_name": doc.file_name})

        # Concatenate documents text
        docs_payload = "\n\n".join(documents_content)

        # Structured Prompt
        prompt = f"""You are a senior research scientist and analytical assistant.
Your task is to perform a detailed comparison of the following documents:

{docs_payload}

Analyze and compare the papers on these dimensions:
1. Methodologies: Focus on what methods or frameworks they use.
2. Advantages and Disadvantages: Compare pros and cons of their approaches.
3. Similarities: Points where their theories, experiments, or findings overlap.
4. Differences: Key points of divergence or contrast.
5. Conclusions: Summarize and contrast the final findings or conclusions of each document.
6. Implementation Approaches: Design choices, code architectures, or setups.
7. Comparison Matrix: A summary markdown table comparing key metrics/features.
"""

        try:
            # Query Gemini
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ComparisonResponseSchema,
                    temperature=0.2
                )
            )

            res_data = ComparisonResponseSchema.model_validate_json(response.text)

            return {
                "compared_documents": doc_details,
                "methodologies_comparison": res_data.methodologies_comparison,
                "advantages_disadvantages": res_data.advantages_disadvantages,
                "similarities": res_data.similarities,
                "differences": res_data.differences,
                "conclusions_comparison": res_data.conclusions_comparison,
                "implementation_approaches": res_data.implementation_approaches,
                "comparison_matrix_markdown": res_data.comparison_matrix_markdown
            }
        except Exception as e:
            raise RuntimeError(f"Comparison failed via Gemini: {e}")
