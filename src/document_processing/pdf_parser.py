import fitz  # PyMuPDF
from typing import List, Dict, Any

class PDFParser:
    @staticmethod
    def extract_text_with_metadata(pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page from a PDF document, preserving page numbers (1-indexed).
        """
        extracted_pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                extracted_pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF at {pdf_path}: {e}")
        
        return extracted_pages
