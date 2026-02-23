import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_raw_document():
    from src.ingestion.loaders import RawDocument
    return RawDocument(
        content="## Pod Scheduling ##\n" + "pods are scheduled by the scheduler " * 50,
        source="test.html",
        doc_type="concepts",
        file_format="html",
        metadata={"loader": "beautifulsoup"}
    )


@pytest.fixture
def sample_chunks(sample_raw_document):
    from src.ingestion.preprocessor import process_document
    return process_document(sample_raw_document)


@pytest.fixture
def sample_context():
    return [
        {
            "content": "[Pod Scheduling]\npods are scheduled based on resource requests",
            "source": "k8s_concepts.html",
            "section_title": "Pod Scheduling",
            "doc_type": "concepts",
            "score": 0.92
        },
        {
            "content": "[CrashLoopBackOff]\ncheck kubectl describe pod for exit codes",
            "source": "k8s_troubleshooting.html",
            "section_title": "CrashLoopBackOff",
            "doc_type": "troubleshooting",
            "score": 0.87
        }
    ]


@pytest.fixture
def mock_openai_response():
    """Mock לתשובת OpenAI chat completion"""
    mock = MagicMock()
    mock.choices[0].message.content = "This is a test answer about Kubernetes."
    mock.usage.total_tokens = 150
    return mock


@pytest.fixture
def mock_chroma_results():
    """Mock לתוצאות ChromaDB query"""
    return {
        "documents": [["doc content 1", "doc content 2"]],
        "metadatas": [[
            {"source": "test.html", "section_title": "Test", "doc_type": "concepts"},
            {"source": "test2.html", "section_title": "Test2", "doc_type": "book"}
        ]],
        "distances": [[0.1, 0.2]]
    }
