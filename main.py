"""
Omegatron AI API - FastAPI server with multiple AI model providers
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime
import uuid

# Import providers
from providers import FlowithProvider, CloudflareProvider, TypefullyProvider, OIVSCodeProvider, MinimaxProvider
from providers.base import ChatCompletionRequest, ChatCompletionResponse, Message, Usage, Choice

app = FastAPI(title="Omegatron AI API", version="2.0.0")

# Initialize providers
flowith_provider = FlowithProvider()
cloudflare_provider = CloudflareProvider()
typefully_provider = TypefullyProvider()
minimax_provider = MinimaxProvider()

# Create provider mapping
PROVIDERS = {
    "flowith": flowith_provider,
    "cloudflare": cloudflare_provider,
    "typefully": typefully_provider,
    "minimax": minimax_provider
}

# Collect all models from all providers
ALL_MODELS = []
MODEL_PROVIDER_MAP = {}

for provider_name, provider in PROVIDERS.items():
    models = provider.get_models()
    ALL_MODELS.extend(models)
    for model in models:
        MODEL_PROVIDER_MAP[model] = provider_name

# Remove duplicates while preserving order
ALL_MODELS = list(dict.fromkeys(ALL_MODELS))

class ModelInfo:
    def __init__(self, id: str, object: str = "model", created: int = None, owned_by: str = "omegatron"):
        self.id = id
        self.object = object
        self.created = created or int(datetime.now().timestamp())
        self.owned_by = owned_by

class ModelsResponse:
    def __init__(self, object: str = "list", data: List[ModelInfo] = None):
        self.object = object
        self.data = data or []

@app.get("/v1/models")
async def list_models():
    """List all available models from all providers"""
    current_time = int(datetime.now().timestamp())
    
    models_data = []
    for model in sorted(ALL_MODELS):
        models_data.append({
            "id": model,
            "object": "model",
            "created": current_time,
            "owned_by": "omegatron",
            "endpoint": "/v1/chat/completions"
        })
    
    return {
        "object": "list",
        "data": models_data
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using the appropriate provider"""
    
    if request.model not in ALL_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not found. Available models: {sorted(ALL_MODELS)}"
        )
    
    provider_name = MODEL_PROVIDER_MAP.get(request.model)
    if not provider_name or provider_name not in PROVIDERS:
        raise HTTPException(
            status_code=500, 
            detail=f"No provider found for model: {request.model}"
        )
    
    provider = PROVIDERS[provider_name]
    
    try:
        if request.stream:
            # For streaming, we'll collect the response and then stream it
            response = await provider.chat_completion(request)
            return StreamingResponse(
                stream_chat_completion(response.choices[0].message.content, request.model),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            response = await provider.chat_completion(request)
            return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating response: {str(e)}"
        )

async def stream_chat_completion(content: str, model: str):
    """Stream chat completion response in OpenAI format"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    current_time = int(datetime.now().timestamp())
    
    # Split content into chunks for streaming
    words = content.split()
    chunk_size = 3  # Send 3 words at a time
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " " + " ".join(chunk_words) if i > 0 else " ".join(chunk_words)
        
        chunk_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": current_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": chunk_content
                },
                "finish_reason": None
            }]
        }
        
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.05)  # Small delay for realistic streaming
    
    # Send final chunk
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk", 
        "created": current_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Omegatron AI API",
        "version": "1.0.0",
        "provider": "omegatron",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "endpoint": "/v1/chat/completions"
        },
        "total_models": len(ALL_MODELS),
        "description": "Unified AI API with multiple model support"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "provider": "omegatron",
        "total_models": len(ALL_MODELS),
        "service": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)