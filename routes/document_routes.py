import os
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.database.models import Document, Chunk
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import Chunker
from src.ml.predictor import DocumentClassifier
from src.vector_store.manager import VectorStoreManager
from config.settings import settings

router = APIRouter(prefix="/documents", tags=["Document Management"])

# Instantiate singletons for classification and vector store
classifier = DocumentClassifier()
vector_store = VectorStoreManager()
chunker = Chunker()

def process_pdf_pipeline(doc_id: str, file_path: str, file_name: str, db: Session):
    """
    Background worker function running the ingestion pipeline:
    1. Parse text page-by-page
    2. Auto-classify document using TensorFlow on first 2 pages
    3. Generate overlapping text chunks
    4. Index chunks to SQLite and ChromaDB
    """
    try:
        # Update status
        doc_record = db.query(Document).filter_by(doc_id=doc_id).first()
        if not doc_record:
            return
        doc_record.processing_status = "PROCESSING"
        db.commit()

        # Step 1: Text extraction
        pages_data = PDFParser.extract_text_with_metadata(file_path)
        total_pages = len(pages_data)
        doc_record.total_pages = total_pages
        db.commit()

        # Step 2: Auto-classification (on first 2 pages)
        first_pages = [p["text"] for p in pages_data[:2] if p["text"]]
        classification_text = " ".join(first_pages)
        category = classifier.predict_category(classification_text)
        doc_record.category = category
        db.commit()

        # Step 3: Chunking
        chunks = chunker.chunk_document(doc_id, pages_data)
        doc_record.total_chunks = len(chunks)
        db.commit()

        # Step 4: Index chunks in SQLite
        # Delete old chunks if reprocessing
        db.query(Chunk).filter_by(doc_id=doc_id).delete()
        for chk in chunks:
            db_chunk = Chunk(
                chunk_id=chk["chunk_id"],
                doc_id=chk["doc_id"],
                page_number=chk["page_number"],
                text=chk["text"],
                chunk_index=chk["chunk_index"]
            )
            db.add(db_chunk)
        db.commit()

        # Step 5: Index chunks in ChromaDB
        # Delete old vector store entries if reprocessing
        vector_store.delete_document(doc_id)
        vector_store.add_chunks(doc_id, file_name, chunks)

        # Mark complete
        doc_record.processing_status = "PROCESSED"
        db.commit()
        print(f"Ingestion successful for document {file_name} ({doc_id})")

    except Exception as e:
        db.rollback()
        doc_record = db.query(Document).filter_by(doc_id=doc_id).first()
        if doc_record:
            doc_record.processing_status = "FAILED"
            db.commit()
        print(f"Error processing pipeline for {file_name}: {e}")


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a PDF document and schedules its processing pipeline.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / f"{doc_id}_{file.filename}"

    # Save raw file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create document record in database
    doc_record = Document(
        doc_id=doc_id,
        file_name=file.filename,
        processing_status="PENDING",
        category="Unknown"
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    # Trigger background ingestion
    # Note: background tasks receive their own database sessions to avoid threading issues
    background_tasks.add_task(
        process_pdf_pipeline, 
        doc_id, 
        str(file_path), 
        file.filename, 
        next(get_db())
    )

    return {
        "message": "Document uploaded and scheduled for processing successfully.",
        "doc_id": doc_id,
        "file_name": file.filename,
        "processing_status": "PENDING"
    }


@router.get("")
async def list_documents(db: Session = Depends(get_db)):
    """
    Lists all uploaded documents with metadata and status.
    """
    documents = db.query(Document).order_by(Document.upload_timestamp.desc()).all()
    return [
        {
            "doc_id": d.doc_id,
            "file_name": d.file_name,
            "upload_timestamp": d.upload_timestamp.isoformat(),
            "total_pages": d.total_pages,
            "total_chunks": d.total_chunks,
            "processing_status": d.processing_status,
            "category": d.category
        }
        for d in documents
    ]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Deletes an uploaded document, its database metadata, chunks, and vector embeddings.
    """
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete local raw file
    file_path = settings.UPLOAD_DIR / f"{doc_id}_{doc.file_name}"
    if file_path.exists():
        try:
            os.remove(file_path)
        except Exception as e:
            # Continue delete even if local file delete fails
            print(f"Failed to delete local file: {e}")

    # Delete from ChromaDB
    try:
        vector_store.delete_document(doc_id)
    except Exception as e:
        print(f"Failed to delete ChromaDB index: {e}")

    # Delete from SQLite (cascade deletes chunks and referenced logs)
    db.delete(doc)
    db.commit()

    return {"message": f"Document '{doc.file_name}' deleted successfully."}


@router.post("/{doc_id}/reprocess")
async def reprocess_document(doc_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Reprocesses an existing document (re-parses, re-chunks, re-indexes).
    """
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = settings.UPLOAD_DIR / f"{doc_id}_{doc.file_name}"
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Source PDF file is missing from raw_documents storage.")

    # Set status back to pending
    doc.processing_status = "PENDING"
    db.commit()

    # Trigger background reprocessing
    background_tasks.add_task(
        process_pdf_pipeline, 
        doc_id, 
        str(file_path), 
        doc.file_name, 
        next(get_db())
    )

    return {
        "message": f"Document '{doc.file_name}' scheduled for reprocessing.",
        "doc_id": doc_id,
        "processing_status": "PENDING"
    }
