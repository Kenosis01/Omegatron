# Omegatron AI API

A FastAPI server that combines models from multiple AI providers, offering a unified OpenAI-compatible API interface with a clean, modular architecture.

## Features

- **70+ unique AI models** from five providers
- **Clean model names** 
- **OpenAI-compatible API** endpoints
- **Streaming and non-streaming** chat completions
- **Automatic provider routing** based on model selection
- **Modular provider architecture** for easy extension
- **Reasoning model support** (MiniMax reasoning-01 with thinking process)

## Architecture

The codebase is organized into separate provider modules:

```
omegatron/
├── main.py                 # FastAPI application
├── providers/
│   ├── __init__.py        # Provider imports
│   ├── base.py            # Base provider class
│   ├── flowith.py         # Flowith provider
│   ├── cloudflare.py      # Cloudflare provider
│   ├── typefully.py       # Typefully provider
│   ├── oivscode.py        # OIVSCODE provider
│   └── minimax.py         # MiniMax provider
├── requirements.txt
└── README.md
```

## Available Models by Provider

### Flowith Models (12)
- gpt-5-nano, gpt-5-mini, glm-4.5, gpt-oss-120b, gpt-oss-20b, kimi-k2
- gpt-4.1, gpt-4.1-mini, deepseek-chat, deepseek-reasoner
- gemini-2.5-flash, grok-3-mini

### Cloudflare Models (44)
- Various LLaMA, Mistral, Qwen, DeepSeek, and other models
- Names cleaned (e.g., `@cf/meta/llama-3-8b-instruct` → `llama-3-8b-instruct`)

### Typefully Models (1)
- claude-3.5-haiku

### OIVSCODE Models (10)
- gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- gpt-3.5-turbo, gpt-3.5-turbo-16k
- o1, o1-mini, o1-preview, o3-mini

### MiniMax Models (1)
- minimax-reasoning-01 (with reasoning/thinking capabilities)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### 1. List Models
```http
GET /v1/models
```

Returns all available models in OpenAI format with provider information.

### 2. Chat Completions
```http
POST /v1/chat/completions
```

Create chat completions with streaming or non-streaming responses.

**Request Body:**
```json
{
  "model": "minimax-reasoning-01",
  "messages": [
    {"role": "user", "content": "Solve this math problem: 2x + 5 = 15"}
  ],
  "stream": false,
  "max_tokens": 2048
}
```

### 3. Health Check
```http
GET /health
```

Returns server health status and provider information.

## Usage Examples

### Python with requests
```python
import requests

# List models
response = requests.get("http://localhost:8000/v1/models")
print(response.json())

# Chat completion with reasoning model
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "minimax-reasoning-01",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "stream": False
})
print(response.json())

# Chat completion with Claude
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "claude-3.5-haiku",
    "messages": [{"role": "user", "content": "Write a haiku about AI"}],
    "stream": False
})
print(response.json())
```

### cURL
```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false
  }'
```

## Provider Details

### Automatic Routing
The server automatically routes requests to the appropriate provider based on the model name:
- **Flowith models** → Uses the Flowith provider from `test.py`
- **Cloudflare models** → Uses the Cloudflare provider from `test2.py`
- **Typefully models** → Uses Typefully API for Claude 3.5 Haiku
- **OIVSCODE models** → Uses OIVSCODE endpoints for OpenAI models
- **MiniMax models** → Uses MiniMax API for reasoning models

### Reasoning Models
MiniMax reasoning models include thinking processes in their responses:
```json
{
  "choices": [{
    "message": {
      "content": "<thinking>\nLet me think about this step by step...\n</thinking>\n\nThe answer is..."
    }
  }]
}
```

## Adding New Providers

To add a new provider:

1. Create a new provider file in `providers/` directory
2. Inherit from `BaseProvider` class
3. Implement required methods: `get_models()` and `chat_completion()`
4. Add import to `providers/__init__.py`
5. Initialize and add to `PROVIDERS` dict in `main.py`

Example:
```python
# providers/newprovider.py
from .base import BaseProvider

class NewProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        self.provider_name = "newprovider"
        self.models = ["model-1", "model-2"]
    
    def get_models(self):
        return self.models
    
    async def chat_completion(self, request):
        # Implementation here
        pass
```

## Error Handling

The API provides detailed error messages for:
- Invalid model names
- Provider failures
- Network timeouts
- Authentication issues

All errors follow OpenAI API error format for compatibility.
