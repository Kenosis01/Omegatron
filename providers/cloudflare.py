"""
Cloudflare provider implementation
"""

import sys
import os
import uuid
import time
import json
import re
from typing import List
from curl_cffi.requests import Session
from curl_cffi import CurlError
from uuid import uuid4
from .base import BaseProvider, ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage

# Import webscout utilities for Cloudflare
try:
    from webscout.AIutel import sanitize_stream
    from webscout.litagent import LitAgent
except ImportError:
    # Fallback if webscout is not available
    sanitize_stream = None
    LitAgent = None


class CloudflareProvider(BaseProvider):
    """Cloudflare AI model provider"""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "cloudflare"
        self.models = [
            "deepseek-coder-6.7b-base-awq",
            "deepseek-coder-6.7b-instruct-awq", 
            "deepseek-math-7b-instruct",
            "deepseek-r1-distill-qwen-32b",
            "discolm-german-7b-v1-awq",
            "falcon-7b-instruct",
            "gemma-3-12b-it",
            "gemma-7b-it",
            "hermes-2-pro-mistral-7b",
            "llama-2-13b-chat-awq",
            "llama-2-7b-chat-fp16",
            "llama-2-7b-chat-int8",
            "llama-3-8b-instruct",
            "llama-3-8b-instruct-awq",
            "llama-3.1-8b-instruct-awq",
            "llama-3.1-8b-instruct-fp8",
            "llama-3.2-11b-vision-instruct",
            "llama-3.2-1b-instruct",
            "llama-3.2-3b-instruct",
            "llama-3.3-70b-instruct-fp8-fast",
            "llama-4-scout-17b-16e-instruct",
            "llama-guard-3-8b",
            "llamaguard-7b-awq",
            "meta-llama-3-8b-instruct",
            "mistral-7b-instruct-v0.1",
            "mistral-7b-instruct-v0.2",
            "mistral-small-3.1-24b-instruct",
            "neural-chat-7b-v3-1-awq",
            "openchat-3.5-0106",
            "openhermes-2.5-mistral-7b-awq",
            "phi-2",
            "qwen1.5-0.5b-chat",
            "qwen1.5-1.8b-chat",
            "qwen1.5-14b-chat-awq",
            "qwen1.5-7b-chat-awq",
            "qwen2.5-coder-32b-instruct",
            "qwq-32b",
            "sqlcoder-7b-2",
            "starling-lm-7b-beta",
            "tinyllama-1.1b-chat-v1.0",
            "una-cybertron-7b-v2-bf16",
            "zephyr-7b-beta-awq"
        ]
        
        # Model mapping to original Cloudflare format
        self.model_mapping = {
            "deepseek-coder-6.7b-base-awq": "@hf/thebloke/deepseek-coder-6.7b-base-awq",
            "deepseek-coder-6.7b-instruct-awq": "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
            "deepseek-math-7b-instruct": "@cf/deepseek-ai/deepseek-math-7b-instruct",
            "deepseek-r1-distill-qwen-32b": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "discolm-german-7b-v1-awq": "@cf/thebloke/discolm-german-7b-v1-awq",
            "falcon-7b-instruct": "@cf/tiiuae/falcon-7b-instruct",
            "gemma-3-12b-it": "@cf/google/gemma-3-12b-it",
            "gemma-7b-it": "@hf/google/gemma-7b-it",
            "hermes-2-pro-mistral-7b": "@hf/nousresearch/hermes-2-pro-mistral-7b",
            "llama-2-13b-chat-awq": "@hf/thebloke/llama-2-13b-chat-awq",
            "llama-2-7b-chat-fp16": "@cf/meta/llama-2-7b-chat-fp16",
            "llama-2-7b-chat-int8": "@cf/meta/llama-2-7b-chat-int8",
            "llama-3-8b-instruct": "@cf/meta/llama-3-8b-instruct",
            "llama-3-8b-instruct-awq": "@cf/meta/llama-3-8b-instruct-awq",
            "llama-3.1-8b-instruct-awq": "@cf/meta/llama-3.1-8b-instruct-awq",
            "llama-3.1-8b-instruct-fp8": "@cf/meta/llama-3.1-8b-instruct-fp8",
            "llama-3.2-11b-vision-instruct": "@cf/meta/llama-3.2-11b-vision-instruct",
            "llama-3.2-1b-instruct": "@cf/meta/llama-3.2-1b-instruct",
            "llama-3.2-3b-instruct": "@cf/meta/llama-3.2-3b-instruct",
            "llama-3.3-70b-instruct-fp8-fast": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "llama-4-scout-17b-16e-instruct": "@cf/meta/llama-4-scout-17b-16e-instruct",
            "llama-guard-3-8b": "@cf/meta/llama-guard-3-8b",
            "llamaguard-7b-awq": "@hf/thebloke/llamaguard-7b-awq",
            "meta-llama-3-8b-instruct": "@hf/meta-llama/meta-llama-3-8b-instruct",
            "mistral-7b-instruct-v0.1": "@cf/mistral/mistral-7b-instruct-v0.1",
            "mistral-7b-instruct-v0.2": "@hf/mistral/mistral-7b-instruct-v0.2",
            "mistral-small-3.1-24b-instruct": "@cf/mistralai/mistral-small-3.1-24b-instruct",
            "neural-chat-7b-v3-1-awq": "@hf/thebloke/neural-chat-7b-v3-1-awq",
            "openchat-3.5-0106": "@cf/openchat/openchat-3.5-0106",
            "openhermes-2.5-mistral-7b-awq": "@hf/thebloke/openhermes-2.5-mistral-7b-awq",
            "phi-2": "@cf/microsoft/phi-2",
            "qwen1.5-0.5b-chat": "@cf/qwen/qwen1.5-0.5b-chat",
            "qwen1.5-1.8b-chat": "@cf/qwen/qwen1.5-1.8b-chat",
            "qwen1.5-14b-chat-awq": "@cf/qwen/qwen1.5-14b-chat-awq",
            "qwen1.5-7b-chat-awq": "@cf/qwen/qwen1.5-7b-chat-awq",
            "qwen2.5-coder-32b-instruct": "@cf/qwen/qwen2.5-coder-32b-instruct",
            "qwq-32b": "@cf/qwen/qwq-32b",
            "sqlcoder-7b-2": "@cf/defog/sqlcoder-7b-2",
            "starling-lm-7b-beta": "@hf/nexusflow/starling-lm-7b-beta",
            "tinyllama-1.1b-chat-v1.0": "@cf/tinyllama/tinyllama-1.1b-chat-v1.0",
            "una-cybertron-7b-v2-bf16": "@cf/fblgit/una-cybertron-7b-v2-bf16",
            "zephyr-7b-beta-awq": "@hf/thebloke/zephyr-7b-beta-awq"
        }
        
        self.api_url = "https://playground.ai.cloudflare.com/api/inference"
    
    def get_models(self) -> List[str]:
        """Get list of available Cloudflare models"""
        return self.models
    
    def _cloudflare_extractor(self, chunk: str) -> str:
        """Extracts content from Cloudflare stream format"""
        if isinstance(chunk, str):
            # Use re.search to find the pattern 0:"<content>"
            match = re.search(r'0:"(.*?)"(?=,|$)', chunk)
            if match:
                try:
                    # Decode potential unicode escapes like \u00e9 and handle escaped quotes/backslashes
                    content = match.group(1).encode().decode('unicode_escape')
                    return content.replace('\\\\', '\\').replace('\\"', '"')
                except UnicodeDecodeError:
                    # If unicode_escape fails, handle manually
                    content = match.group(1)
                    content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                    return content
        return ""
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle Cloudflare chat completion"""
        messages = self.prepare_messages(request)
        
        # Map clean model name to full Cloudflare model name
        cf_model = self.model_mapping.get(request.model, f"@cf/{request.model}")
        
        # Prepare payload
        payload = {
            "messages": messages,
            "lora": None,
            "model": cf_model,
            "max_tokens": request.max_tokens or 600,
            "stream": True
        }
        
        # Prepare headers
        headers = {
            'Accept': 'text/event-stream',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9,en-IN;q=0.8',
            'Content-Type': 'application/json',
            'DNT': '1',
            'Origin': 'https://playground.ai.cloudflare.com',
            'Referer': 'https://playground.ai.cloudflare.com/',
            'Sec-CH-UA': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': LitAgent().random() if LitAgent else 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        }
        
        # Prepare cookies
        cookies = {
            'cfzs_amplitude': uuid4().hex,
            'cfz_amplitude': uuid4().hex,
            '__cf_bm': uuid4().hex,
        }
        
        try:
            # Use curl_cffi Session
            session = Session()
            session.headers.update(headers)
            
            response = session.post(
                self.api_url,
                headers=headers,
                cookies=cookies,
                data=json.dumps(payload),
                stream=True,
                timeout=30,
                impersonate="chrome120"
            )
            
            response.raise_for_status()
            
            # Process streaming response
            full_response = ""
            
            if sanitize_stream:
                # Use sanitize_stream from webscout
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value=None,
                    to_json=False,
                    skip_markers=None,
                    content_extractor=self._cloudflare_extractor,
                    yield_raw_on_error=False
                )
                
                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        full_response += content_chunk
            else:
                # Fallback to manual processing
                content = response.text
                lines = content.split('\n')
                
                for line in lines:
                    if line.strip():
                        extracted = self._cloudflare_extractor(line)
                        if extracted:
                            full_response += extracted
            
            # Ensure we have some content
            if not full_response:
                full_response = "No response generated"
            
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
            
        except CurlError as e:
            raise Exception(f"Cloudflare request failed (CurlError): {e}")
        except Exception as e:
            raise Exception(f"Cloudflare chat failed: {str(e)}")