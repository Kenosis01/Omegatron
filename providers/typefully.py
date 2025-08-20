"""
Typefully provider implementation
"""

import uuid
import time
import re
import httpx
from typing import List
from .base import BaseProvider, ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage


class TypefullyProvider(BaseProvider):
    """Typefully AI model provider (Claude 3.5 Haiku)"""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "typefully"
        self.models = ["claude-3.5-haiku"]
        self.api_url = "https://typefully.com/tools/ai/api/completion"
        self.headers = {
            "authority": "typefully.com",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://typefully.com",
            "referer": "https://typefully.com/tools/ai/chat-gpt-alternative",
            "sec-ch-ua": '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        }
    
    def get_models(self) -> List[str]:
        """Get list of available Typefully models"""
        return self.models
    
    def _typefully_extractor(self, chunk: str) -> str:
        """Extracts content from Typefully stream format"""
        if isinstance(chunk, str):
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8', errors='replace')
            match = re.search(r'0:"(.*?)"', chunk)
            if match:
                try:
                    # Handle unicode escapes more safely
                    content = match.group(1)
                    # Replace common escape sequences manually to avoid unicode_escape issues
                    content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                    return content
                except Exception:
                    # If decoding fails, return the raw content
                    return match.group(1)
        return ""
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle Typefully chat completion"""
        messages = self.prepare_messages(request)
        
        # Convert messages to conversation prompt
        conversation_prompt = ""
        system_prompt = "You're a helpful assistant."
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                conversation_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_prompt += f"Assistant: {msg['content']}\n"
        
        # Remove trailing newline and add final user prompt if needed
        conversation_prompt = conversation_prompt.strip()
        if not conversation_prompt.endswith("User:"):
            # If the last message isn't from user, we need to extract it
            if messages and messages[-1]["role"] == "user":
                conversation_prompt = messages[-1]["content"]
        
        # Map claude-3.5-haiku to the Typefully model identifier
        model_identifier = "anthropic:claude-3-5-haiku-20241022"
        
        payload = {
            "prompt": conversation_prompt,
            "systemPrompt": system_prompt,
            "modelIdentifier": model_identifier,
            "outputLength": request.max_tokens or 600
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=60.0
                )
            
            if response.status_code != 200:
                raise Exception(f"Typefully API error: {response.status_code} - {response.text}")
            
            # Process streaming response
            full_response = ""
            
            # Process the response content
            content = response.text
            lines = content.split('\n')
            
            for line in lines:
                if line.strip():
                    extracted = self._typefully_extractor(line)
                    if extracted:
                        full_response += extracted
            
            # Clean up the response text
            if full_response:
                full_response = full_response.replace('\\n', '\n').replace('\\n\\n', '\n\n')
            
            # Calculate usage
            prompt_tokens = sum(len(msg["content"].split()) for msg in messages)
            completion_tokens = len(full_response.split()) if full_response else 0
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            choice = Choice(
                index=0,
                message=Message(role="assistant", content=full_response),
                finish_reason="stop"
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
        except Exception as e:
            raise Exception(f"Typefully chat failed: {str(e)}")