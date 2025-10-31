#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import openai
import logging
import copy
import sys
import time
import requests
import json
from typing import List, Dict, Any, Optional, Union


logger = logging.getLogger(__name__)


class LLMClient:
    """
    A unified wrapper class for sending requests to LLM endpoints.
    
    This class provides a clean interface for interacting with OpenAI APIs
    with support for direct URL configuration and multi-turn conversation management
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = None,
        timeout: int = 2400,
        default_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the LLM client wrapper.
        
        Args:
            base_url: Direct base URL for the API endpoint
            timeout: Request timeout in seconds (default: 2400)
            model_name: Which model name to use for API endpoint
            default_headers: Additional headers to include in requests
        """
        self.timeout = timeout
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.model_name = model_name
        
        # Default to OpenAI client
        if not self.base_url:
            raise ValueError("`base_url` is not provided")
            
        self.client = openai.OpenAI(
            base_url=self.base_url,
            default_headers=self.default_headers,
            timeout=self.timeout,
        )

        # Multi-turn conversation support
        self.conversation_history: List[Dict[str, str]] = []
        self.system_message: Optional[str] = None
        
        # Store last response for extended thinking access
        self._last_response = None
        
        logger.info(f"OpenAI client initialized with base URL: {self.base_url}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_tokens: int = 200,
        temperature: float = 0.6,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to the LLM endpoint.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (default: 0.6)
            max_tokens: Maximum tokens to generate (default: 200)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            String containing the assistant's response
            
        Raises:
            Exception: If the API request fails
        """
        try:
            
            return self._openai_chat_completion(messages, max_tokens, temperature, **kwargs)
            
        except Exception as e:
            logger.error(f"Chat completion request failed: {e}")
            raise
    
    def _openai_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Handle OpenAI chat completion requests."""
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        self._last_response = completion
        output_text = completion.choices[0].message.content
        
        logger.debug("OpenAI chat completion request successful")
        return output_text
    
    def simple_chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.6,
        **kwargs
    ) -> str:
        """
        Send a simple chat request with a single user prompt.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message to set context
            **kwargs: Additional parameters to pass to chat_completion
            
        Returns:
            String containing the assistant's response
            
        Raises:
            Exception: If the API request fails
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.chat_completion(messages=messages, 
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            **kwargs)
            return response
        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if the LLM endpoint is healthy and responding.
        
        Returns:
            True if the endpoint is healthy, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            self.chat_completion(
                messages=test_messages,
                max_tokens=1,
                temperature=0.0
            )
            logger.info("Health check passed")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def set_system_message(self, system_message: str) -> None:
        """
        Set the system message for multi-turn conversations.
        
        Args:
            system_message: The system message to use for conversations
        """
        self.system_message = system_message
        logger.debug("System message set for multi-turn conversations")
    
    def start_conversation(self, system_message: Optional[str] = None) -> None:
        """
        Start a new multi-turn conversation, clearing any existing history.
        
        Args:
            system_message: Optional system message to set for this conversation
        """
        self.conversation_history = []
        if system_message:
            self.system_message = system_message
        logger.debug("Started new multi-turn conversation")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('user', 'assistant', or 'system')
            content: The content of the message
        """
        if role not in ['user', 'assistant', 'system']:
            raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")
        
        self.conversation_history.append({"role": role, "content": content})
        logger.debug(f"Added {role} message to conversation history")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return copy.deepcopy(self.conversation_history)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.debug("Cleared conversation history")
    
    def continue_conversation(
        self,
        user_message: str,
        n: int = 1,
        max_tokens: int = 200,
        temperature: float = 0.6,
        **kwargs
    ) -> str:
        """
        Continue the multi-turn conversation with a new user message.
        
        Args:
            user_message: The user's message to add to the conversation
            temperature: Sampling temperature (default: 0.6)
            n: Number of completions to generate (default: 1)
            stop: Stop sequences for generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            String containing the assistant's response
            
        Raises:
            Exception: If the API request fails
        """
        # Add user message to conversation history
        self.add_message("user", user_message)
        
        # Build messages list for API call
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self.conversation_history)
        
        try:
            response = self.chat_completion(
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            assistant_response = response
            
            # Add assistant response to conversation history
            self.add_message("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Failed to get conversation response: {e}")
            raise
    
    
    def get_conversation_length(self) -> int:
        """
        Get the number of messages in the conversation history.
        
        Returns:
            Number of messages in conversation history
        """
        return len(self.conversation_history)
    
    def truncate_conversation(self, max_messages: int) -> None:
        """
        Truncate conversation history to keep only the most recent messages.
        
        Args:
            max_messages: Maximum number of messages to keep
        """
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
            logger.debug(f"Truncated conversation history to {max_messages} messages")
    
    def get_last_response_raw(self) -> Optional[Any]:
        """
        Get the raw response object from the last API call.
        
        Returns:
            The raw response object, or None if no response has been made
        """
        return self._last_response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'base_url': self.base_url,
            'timeout': self.timeout
        }
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up if needed
        pass
