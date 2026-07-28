import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from src.database.base import Base

class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")  # PENDING, PROCESSING, PROCESSED, FAILED
    category = Column(String, default="Unknown")  # Auto classified category

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    references = relationship("ReferenceLog", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True, index=True)
    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="chunks")

class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class ReferenceLog(Base):
    __tablename__ = "reference_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    query_timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="references")
