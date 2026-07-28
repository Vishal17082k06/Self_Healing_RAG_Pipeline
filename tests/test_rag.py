import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "mock-openai-key")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

from main import app, prompt, ChatRequest, format_docs


# UNIT & LOGIC TESTS (Fast, Offline, No OpenAI/MLflow Calls)

def test_prompt_template_variables():
    """Verify system_prompt maintains required input variables 'context' & 'question'."""
    input_vars = prompt.input_variables
    assert "context" in input_vars, "Prompt is missing required '{context}' variable!"
    assert "question" in input_vars, "Prompt is missing required '{question}' variable!"


def test_format_docs_utility():
    """Verify format_docs converts document list into double-newline separated string."""
    doc1 = MagicMock()
    doc1.page_content = "Event A starts at 10 AM in Hall 1."
    doc2 = MagicMock()
    doc2.page_content = "Event B starts at 2 PM in Hall 2."

    formatted = format_docs([doc1, doc2])
    assert formatted == "Event A starts at 10 AM in Hall 1.\n\nEvent B starts at 2 PM in Hall 2."


def test_chat_request_pydantic_schema():
    """Verify Pydantic model validates incoming chat payload schema."""
    req = ChatRequest(question="When does the coding workshop start?")
    assert req.question == "When does the coding workshop start?"

    with pytest.raises(ValueError):
        ChatRequest()


# INTEGRATION TESTS (FastAPI Endpoint & Chain Flow)

@pytest.fixture
def api_client():
    """Provides a FastAPI TestClient instance."""
    return TestClient(app)


@patch("main.retriever.invoke")
@patch("main.llm.invoke")
def test_chat_endpoint_success(mock_llm_invoke, mock_retriever_invoke, api_client):
    """Test /chat POST endpoint with mocked retriever and LCEL chain execution."""
    # Mock retrieved context document
    mock_doc = MagicMock()
    mock_doc.page_content = "The AI Hackathon starts at 10:00 AM in Hall B."
    mock_retriever_invoke.return_value = [mock_doc]

    # Mock LLM output 
    mock_llm_invoke.return_value = "The AI Hackathon starts at 10:00 AM in Hall B."

    payload = {"question": "When does the AI Hackathon start?"}
    response = api_client.post("/chat", json=payload)

    # Assertions
    assert response.status_code == 200, f"Unexpected error response: {response.json()}"
    data = response.json()

    assert "answer" in data
    assert "contexts" in data
    assert isinstance(data["contexts"], list)
    assert len(data["contexts"]) == 1
    assert data["contexts"][0] == "The AI Hackathon starts at 10:00 AM in Hall B."

    mock_retriever_invoke.assert_called_once_with("When does the AI Hackathon start?")


def test_chat_endpoint_invalid_payload(api_client):
    """Verify /chat endpoint returns 422 Unprocessable Entity for invalid JSON fields."""
    response = api_client.post("/chat", json={"wrong_key": "Where is the event?"})
    assert response.status_code == 422


@patch("main.retriever.invoke")
def test_chat_endpoint_internal_error_handling(mock_retriever_invoke, api_client):
    """Verify /chat returns a 500 status code when an exception occurs internally."""
    mock_retriever_invoke.side_effect = Exception("ChromaDB connection timeout")

    payload = {"question": "What is the schedule?"}
    response = api_client.post("/chat", json=payload)

    assert response.status_code == 500
    assert "ChromaDB connection timeout" in response.json()["detail"]