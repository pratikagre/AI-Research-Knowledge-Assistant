import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app
from src.database.base import Base, get_db
from src.database.models import Document, Chunk

# Setup local SQLite test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_api.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # Populate dummy document and chunks for query testing
    doc = Document(
        doc_id="test-doc-id",
        file_name="test_paper.pdf",
        total_pages=1,
        total_chunks=1,
        processing_status="PROCESSED",
        category="Robotics"
    )
    db.add(doc)
    db.commit()

    chunk = Chunk(
        chunk_id="test-doc-id_c0",
        doc_id="test-doc-id",
        page_number=1,
        text="A robotic manipulator arm uses inverse kinematics for path planning and trajectory control.",
        chunk_index=0
    )
    db.add(chunk)
    db.commit()
    
    yield db
    
    db.close()
    Base.metadata.drop_all(bind=engine)

# Override database dependency in app
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@patch("routes.search_routes.rag_service.retrieve_context")
def test_search_query_endpoint(mock_retrieve_context, setup_db):
    # Mock retrieve context response
    mock_retrieve_context.return_value = [
        {
            "chunk_id": "test-doc-id_c0",
            "text": "A robotic manipulator arm uses inverse kinematics for path planning.",
            "metadata": {
                "doc_id": "test-doc-id",
                "file_name": "test_paper.pdf",
                "page_number": 1,
                "chunk_index": 0
            },
            "score": 0.95
        }
    ]

    # Test query retrieval
    response = client.post("/search/query", json={
        "query": "robotic arm path planning",
        "search_mode": "keyword",
        "doc_ids": ["test-doc-id"],
        "k": 1
    })
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert "robotic manipulator" in data[0]["text"]

@patch("routes.search_routes.rag_service.answer_question")
def test_rag_qa_endpoint(mock_answer_question, setup_db):
    # Mock RAG answer response
    mock_answer_question.return_value = {
        "answer": "Robotic arm path planning uses kinematics.",
        "citations": [{"document_name": "test_paper.pdf", "page_number": 1}],
        "retrieved_context": ["A robotic manipulator arm uses inverse kinematics for path planning."],
        "confidence_score": 0.95
    }

    response = client.post("/search/qa", json={
        "query": "how does the robotic arm move?",
        "session_id": "test_session",
        "search_mode": "keyword",
        "doc_ids": ["test-doc-id"],
        "k": 1
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "Robotic arm path planning" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_name"] == "test_paper.pdf"
