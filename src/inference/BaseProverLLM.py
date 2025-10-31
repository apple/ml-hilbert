#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from abc import ABC, abstractmethod
from src.tools.string import extract_proof_ds_prover_v2
from src.prompts.formal_to_formal import COT_PROMPT, NON_COT_PROMPT
from logging import getLogger
from typing import Union

logger = getLogger(__name__)

class BaseProverLLM(ABC):
    """
    Base class that abstracts the common Prover LLM functionality.
    
    This class encapsulates shared logic between sync and async versions:
    - Prompt strategy selection and formatting
    - Model-specific handling (e.g., DeepSeek Prover V2)
    - Configuration management
    - Proof extraction logic
    """
    
    def __init__(self,
                 llm_client: Union['LLMClient', 'AsyncLLMClient'],
                 model_name: str,
                 prompt_strategy: str = 'non_cot',
                 max_tokens: int = 16384):
        """
        Initialize the BaseProverLLM with LLM client and configuration.
        
        Args:
            llm_client: The LLMClient or AsyncLLMClient instance for API communication
            model_name: Name of the model being used
            prompt_strategy: Either 'cot' or 'non_cot' for prompt selection
            max_tokens: Maximum tokens for proof generation
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.prompt_strategy = prompt_strategy
        self.max_tokens = max_tokens
        self.is_ds_prover_v2 = 'DeepSeek-Prover-V2' in self.model_name
        
        # Select and validate prompt template
        self.prompt_template = self._select_prompt_template()
        
        logger.info(f"{self.__class__.__name__} initialized with model: {model_name}, strategy: {prompt_strategy}")
    
    def _select_prompt_template(self) -> str:
        """
        Select the appropriate prompt template based on strategy and model.
        
        Returns:
            The selected prompt template string
        """
        if self.prompt_strategy == 'cot':
            if not self.is_ds_prover_v2:
                logger.warning("Chain of Thought Prompting will probably only work well with DeepSeek Prover V2.")
            return COT_PROMPT
        elif self.prompt_strategy == 'non_cot':
            return NON_COT_PROMPT
        else:
            logger.warning(f"Invalid prompt strategy: {self.prompt_strategy}. Defaulting to non-Chain of Thought")
            return NON_COT_PROMPT
    
    def _format_prompt(self, formal_statement: str) -> str:
        """
        Format the prompt with the given formal statement.
        
        Args:
            formal_statement: The formal statement to include in the prompt
            
        Returns:
            The formatted prompt string
        """
        return self.prompt_template.format(formal_statement=formal_statement)
    
    def _build_full_prompt(self, formal_statement: str, useful_theorems: str = None) -> str:
        """
        Build the complete prompt including useful theorems if provided.
        
        Args:
            formal_statement: The formal statement to prove
            useful_theorems: Optional useful theorems to include in the prompt
            
        Returns:
            The complete formatted prompt string
        """
        prompt = self._format_prompt(formal_statement)
        if useful_theorems:
            prompt += f"\nYou may find the following theorems from Mathlib4 useful: {useful_theorems}"
        return prompt
    
    def _extract_proof_from_response(self, response: str) -> str:
        """
        Extract proof from LLM response using the appropriate extraction method.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The extracted proof string
        """
        return extract_proof_ds_prover_v2(response)
    
    @abstractmethod
    def generate_proof(self, formal_statement: str, useful_theorems: str = None) -> str:
        """
        Generate a single proof for the given formal statement.
        
        Args:
            formal_statement: The formal statement to prove
            useful_theorems: Optional useful theorems to include in the prompt
            
        Returns:
            The extracted proof string
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the underlying LLM client is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "prompt_strategy": self.prompt_strategy,
            "max_tokens": self.max_tokens,
            "is_ds_prover_v2": self.is_ds_prover_v2
        }