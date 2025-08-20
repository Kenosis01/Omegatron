"""
MiniMax provider implementation
"""

import uuid
import time
import requests
import json
from typing import List
from .base import BaseProvider, ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage

# Import webscout utilities for MiniMax
try:
    from webscout.AIutel import sanitize_stream
except ImportError:
    # Fallback if webscout is not available
    sanitize_stream = None


class MinimaxProvider(BaseProvider):
    """MiniMax AI model provider"""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "minimax"
        self.models = ["minimax-reasoning-01"]
        self.api_url = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
        self.api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJtbyBuaSIsIlVzZXJOYW1lIjoibW8gbmkiLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTg3NjIwMDY0ODA2NDYzNTI0MiIsIlBob25lIjoiIiwiR3JvdXBJRCI6IjE4NzYyMDA2NDgwNjA0NDA5MzgiLCJQYWdlTmFtZSI6IiIsIk1haWwiOiJuaW1vQHN1YnN1cC52aXAiLCJDcmVhdGVUaW1lIjoiMjAyNS0wMS0wNyAxMToyNzowNyIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.Ge1ZnpFPUfXVdMini0P_qXbP_9VYwzXiffG9DsNQck4GtYEOs33LDeAiwrVsrrLZfvJ2icQZ4sRZS54wmPuWua_Dav6pYJty8ZtahmUX1IuhlUX5YErhhCRAIy3J1xB8FkLHLyylChuBHpkNz6O6BQLmPqmoa-cOYK9Qrc6IDeu8SX1iMzO9-MSkcWNvkvpCF2Pf9tekBVWNKMDK6IZoMEPbtkaPXdDyP6l0M0e2AlL_E0oM9exg3V-ohAi8OTPFyqM6dcd4TwF-b9DULxfIsRFw401mvIxcTDWa42u2LULewdATVRD2BthU65tuRqEiWeFWMvFlPj2soMze_QIiUA"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def get_models(self) -> List[str]:
        """Get list of available MiniMax models"""
        return self.models
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle MiniMax chat completion using streaming SSE approach"""
        messages = self.prepare_messages(request)

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "max_tokens": request.max_tokens or 40000,
            "temperature": request.temperature or 1.0,
            "top_p": 0.95
        }

        try:
            # Use requests with stream=True for SSE handling
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60.0
            )
            
            if response.status_code != 200:
                raise Exception(f"MiniMax API error: {response.status_code}")
            
            # Process streaming response using sanitize_stream
            def extract_content_and_thinking(chunk):
                if not isinstance(chunk, dict):
                    return None, None
                choice = chunk.get('choices', [{}])[0]
                delta = choice.get('delta', {})
                content = delta.get('content')
                thinking = delta.get('reasoning_content')
                return content, thinking
            
            full_response = ""
            full_thinking = ""
            
            if sanitize_stream:
                # Use sanitize_stream from webscout.AIutel
                for chunk in sanitize_stream(
                    response.iter_lines(),
                    intro_value="data:",
                    to_json=True,
                    content_extractor=lambda x: x  # Return the full chunk
                ):
                    if chunk and isinstance(chunk, dict):
                        choice = chunk.get('choices', [{}])[0]
                        delta = choice.get('delta', {})
                        
                        # Extract content
                        content = delta.get('content')
                        if content:
                            full_response += content
                        
                        # Extract thinking/reasoning
                        thinking = delta.get('reasoning_content')
                        if thinking:
                            full_thinking += thinking
            else:
                # Fallback manual processing
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            
                            if data_str.strip() == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    choice = data['choices'][0]
                                    delta = choice.get('delta', {})
                                    
                                    # Extract content
                                    content = delta.get('content')
                                    if content:
                                        full_response += content
                                    
                                    # Extract thinking/reasoning
                                    thinking = delta.get('reasoning_content')
                                    if thinking:
                                        full_thinking += thinking
                                        
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue
            
            # Calculate usage
            prompt_tokens = sum(len(msg["content"].split()) for msg in messages)
            completion_tokens = len(full_response.split()) if full_response else 0
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )

            # Ensure we have some content
            if not full_response:
                full_response = "No response generated"

            # Create message with thinking if available (for reasoning models)
            message_content = full_response
            if full_thinking:
                # For reasoning models, we can include thinking in a special format
                message_content = f"<thinking>\n{full_thinking}\n</thinking>\n\n{full_response}"

            choice = Choice(
                index=0,
                message=Message(role="assistant", content=message_content),
                finish_reason="stop"
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )

        except requests.exceptions.Timeout:
            raise Exception("MiniMax API timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"MiniMax API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process MiniMax response: {str(e)}")