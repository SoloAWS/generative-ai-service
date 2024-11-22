import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os
from typing import List

# Test constants
TEST_API_KEY = "test-api-key"
TEST_PROMPT = "Test prompt"
MOCK_RESPONSE = "This is a test response"

# Mock Groq completion response structure
class MockMessage:
    def __init__(self, content: str):
        self.content = content

class MockChoice:
    def __init__(self, message: MockMessage):
        self.message = message

class MockCompletion:
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables before each test"""
    with patch.dict(os.environ, {'GROQ_API_KEY': TEST_API_KEY}):
        yield

@pytest.fixture
def test_client():
    from app.main import app  # Adjust import path as needed
    return TestClient(app)

@pytest.fixture
def mock_groq():
    with patch('groq.Groq') as mock:
        # Create a mock chat completions structure
        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = MockCompletion([
            MockChoice(MockMessage(MOCK_RESPONSE))
        ])
        mock.return_value = mock_instance
        yield mock

def test_generate_response_groq_api_error(test_client, mock_groq):
    """Test handling of Groq API errors"""
    # Simulate Groq API error
    mock_groq.return_value.chat.completions.create.side_effect = Exception(
        "Groq API error"
    )
    
    response = test_client.post(
        "/generative-ai/generate",
        json={"prompt": TEST_PROMPT}
    )
    
    assert response.status_code == 500
    assert "Error generating response" in response.json()["detail"]

def test_generate_response_missing_api_key(test_client):
    """Test behavior when GROQ_API_KEY is missing"""
    with patch.dict(os.environ, {}, clear=True):
        response = test_client.post(
            "/generative-ai/generate",
            json={"prompt": TEST_PROMPT}
        )
        
        assert response.status_code == 500
        assert "Error generating response" in response.json()["detail"]

@pytest.mark.asyncio
async def test_generate_response_multiple_requests(test_client, mock_groq):
    """Test multiple sequential requests"""
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    
    for prompt in prompts:
        response = test_client.post(
            "/generative-ai/generate",
            json={"prompt": prompt}
        )
        
        assert response.status_code == 200
        assert response.json() == {"response": MOCK_RESPONSE}