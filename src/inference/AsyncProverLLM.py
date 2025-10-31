#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from src.tools.AsyncLLMClient import AsyncLLMClient
from src.inference.BaseProverLLM import BaseProverLLM
from src.tools.lean_utils import remove_import_statements
from logging import getLogger

logger = getLogger(__name__)

class AsyncProverLLM(BaseProverLLM):
    """
    Async implementation of ProverLLM that uses AsyncLLMClient.
    
    This class provides asynchronous proof generation functionality
    by extending the BaseProverLLM with async-specific implementations.
    """
    
    def __init__(self,
                 llm_client: AsyncLLMClient,
                 model_name: str,
                 prompt_strategy: str = 'non_cot',
                 max_tokens: int = 16384):
        """
        Initialize the AsyncProverLLM with async LLM client and configuration.
        
        Args:
            llm_client: The AsyncLLMClient instance for API communication
            model_name: Name of the model being used
            prompt_strategy: Either 'cot' or 'non_cot' for prompt selection
            max_tokens: Maximum tokens for proof generation
        """
        super().__init__(llm_client, model_name, prompt_strategy, max_tokens)
    
    async def generate_proof(self, formal_statement: str, useful_theorems: str = None) -> str:
        """
        Generate a single proof for the given formal statement asynchronously.
        
        Args:
            formal_statement: The formal statement to prove
            useful_theorems: Optional useful theorems to include in the prompt
            
        Returns:
            The extracted proof string
        """
        # Build the complete prompt using base class method
        prompt = self._build_full_prompt(formal_statement, useful_theorems)
        
        # Generate response using async LLM client
        response = await self.llm_client.simple_chat(prompt, max_tokens=self.max_tokens, temperature=0.6)

        # Extract proof using base class method
        proof = self._extract_proof_from_response(response)

        # Remove import statements
        proof = remove_import_statements(proof)
        
        return proof
    
    async def health_check(self) -> bool:
        """
        Check if the underlying async LLM client is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return await self.llm_client.health_check()