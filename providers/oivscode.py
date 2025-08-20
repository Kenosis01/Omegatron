"""
OIVSCODE provider implementation
"""

import uuid
import time
import httpx
import secrets
import string
import random
from typing import List
from .base import BaseProvider, ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage


class OIVSCodeProvider(BaseProvider):
    """OIVSCODE AI model provider (OpenAI models)"""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "oivscode"
        self.models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "o1",
            "o1-mini",
            "o1-preview",
            "o3-mini"
        ]
        self.api_endpoints = [
            "https://oi-vscode-server.onrender.com/v1/chat/completions",
            "https://oi-vscode-server-2.onrender.com/v1/chat/completions",
            "https://oi-vscode-server-5.onrender.com/v1/chat/completions",
            "https://oi-vscode-server-0501.onrender.com/v1/chat/completions"
        ]
        self.base_headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,en-GB;q=0.8,en-IN;q=0.7",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Microsoft Edge";v="132"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
        }
    
    def get_models(self) -> List[str]:
        """Get list of available OIVSCODE models"""
        return self.models
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle OIVSCODE chat completion"""
        messages = self.prepare_messages(request)
        
        # Generate user ID for headers
        userid = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(21))
        
        # Prepare headers
        headers = self.base_headers.copy()
        headers["userid"] = userid
        
        # Prepare payload
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": False
        }
        
        # Try endpoints with failover
        endpoints = self.api_endpoints.copy()
        random.shuffle(endpoints)
        
        for endpoint in endpoints:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=30.0
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract response content
                    if "choices" in data and len(data["choices"]) > 0:
                        choice_data = data["choices"][0]
                        message_content = choice_data.get("message", {}).get("content", "")
                        
                        # Extract usage if available
                        usage_data = data.get("usage", {})
                        usage = Usage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0)
                        )
                        
                        choice = Choice(
                            index=0,
                            message=Message(role="assistant", content=message_content),
                            finish_reason=choice_data.get("finish_reason", "stop")
                        )
                        
                        return ChatCompletionResponse(
                            id=f"chatcmpl-{uuid.uuid4().hex}",
                            created=int(time.time()),
                            model=request.model,
                            choices=[choice],
                            usage=usage
                        )
                    else:
                        continue
                        
            except httpx.ReadTimeout:
                continue
            except Exception as e:
                continue
        
        # If all endpoints failed
        raise Exception("All OI-VSCode API endpoints failed")