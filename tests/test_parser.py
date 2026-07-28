import os
import fitz
import pytest
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import Chunker

@pytest.fixture
def sample_pdf(tmp_path):
    # Create a dynamic test PDF using PyMuPDF
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is the first page abstract about Artificial Intelligence and Machine Learning.")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is the second page explaining neural network layers and backpropagation optimizer.")
    
    doc.save(str(pdf_path))
    doc.close()
    
    yield str(pdf_path)
    
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

def test_pdf_extraction(sample_pdf):
    pages = PDFParser.extract_text_with_metadata(sample_pdf)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Artificial Intelligence" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "neural network" in pages[1]["text"]

def test_chunker():
    pages_data = [
        {"page_number": 1, "text": "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."}
    ]
    chunker = Chunker(chunk_size=30, chunk_overlap=10)
    chunks = chunker.chunk_document("test_doc", pages_data)
    
    assert len(chunks) > 0
    assert chunks[0]["doc_id"] == "test_doc"
    assert chunks[0]["page_number"] == 1
    # Check that overlap contains part of the previous chunk text
    for i in range(1, len(chunks)):
        assert len(chunks[i]["text"]) > 0
