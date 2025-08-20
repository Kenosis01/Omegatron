"""
Flowith provider implementation
"""

import sys
import os
import uuid
import time
from typing import List
from .base import BaseProvider, ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage


class FlowithProvider(BaseProvider):
    """Flowith AI model provider"""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "flowith"
        self.models = [
            "gpt-5-nano", "gpt-5-mini", "glm-4.5", "gpt-oss-120b", "gpt-oss-20b", "kimi-k2",
            "gpt-4.1", "gpt-4.1-mini", "deepseek-chat", "deepseek-reasoner",
            "gemini-2.5-flash", "grok-3-mini"
        ]
    
    def get_models(self) -> List[str]:
        """Get list of available Flowith models"""
        return self.models
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle Flowith chat completion"""
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        try:
            from test import Flowith
            
            # Extract user message and system message
            user_message = ""
            system_message = "You are a helpful assistant."
            
            for msg in request.messages:
                if msg.role == "user":
                    user_message = msg.content
                elif msg.role == "system":
                    system_message = msg.content
            
            if not user_message:
                raise Exception("No user message found")
            
            ai = Flowith(
                model=request.model,
                system_prompt=system_message,
                max_tokens=request.max_tokens or 2048,
                is_conversation=False
            )
            
            # Get response
            if request.stream:
                # For streaming, collect all chunks
                response_gen = ai.chat(user_message, stream=True)
                full_response = ""
                for chunk in response_gen:
                    full_response += chunk
            else:
                full_response = ai.chat(user_message, stream=False)
            
            # Calculate usage
            messages = self.prepare_messages(request)
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
            raise Exception(f"Flowith provider error: {str(e)}")