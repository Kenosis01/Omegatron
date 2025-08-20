"""
Base provider class for all AI model providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class BaseProvider(ABC):
    """Base class for all AI model providers"""
    
    def __init__(self):
        self.models = []
        self.provider_name = ""
    
    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request"""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models"""
        pass
    
    def prepare_messages(self, request: ChatCompletionRequest) -> List[Dict[str, str]]:
        """Prepare messages from OpenAI format"""
        messages = []
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        return messages