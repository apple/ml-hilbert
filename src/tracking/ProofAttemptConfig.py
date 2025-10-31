#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ProofAttemptConfig:
    """Configuration for all proof attempt limits and retry counts in HILBERTWorker.
    
    This class centralizes all the various iteration limits and attempt counts
    used throughout the proof generation pipeline, replacing the scattered
    num_passes variables with more descriptive names.
    """
    
    # Main proof generation attempts - parallel execution
    subgoal_decomp_attempts: int = 4           # For main subgoal decomposition attempts
    formal_proof_attempts: int = 4             # For formal prover LLM attempts
    
    # Error correction attempts - sequential execution with conversation context
    main_theorem_error_corrections: int = 4    # For correcting syntax/semantic errors in main theorem
    subgoal_error_corrections: int = 6         # For correcting errors in individual subgoals  
    missing_tags_error_corrections: int = 3    # For error correction when required tags/code blocks are missing from responses  
    
    # Verification and validation attempts
    parallel_subgoal_proof_attempts: int = 4    # For parallel verification of extracted subgoals
    
    # Sketch refinement attempts
    proof_sketch_corrections: int = 8          # For iteratively refining proof sketches when subgoals are invalid
    missing_subgoal_extraction_attempts: int = 3       # For attempting to extract missing subgoals from sketches
    
    # Timeout configurations
    proof_verification_timeout: int = 60       # Timeout for Lean proof verification

    # Limit LLM calls per worker
    max_prover_llm_calls: int = None  # Max calls to prover LLM per worker, None for unlimited
    max_reasoner_llm_calls: int = None  # Max calls to reasoning LLM per worker, None for unlimited
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProofAttemptConfig':
        """Create ProofAttemptConfig from dictionary, using defaults for missing keys."""
        return cls(
            subgoal_decomp_attempts=config_dict.get('subgoal_decomp_attempts', 4),
            formal_proof_attempts=config_dict.get('formal_proof_attempts', 4),
            main_theorem_error_corrections=config_dict.get('main_theorem_error_corrections', 4),
            subgoal_error_corrections=config_dict.get('subgoal_error_corrections', 6),
            missing_tags_error_corrections=config_dict.get('missing_tags_error_corrections', 3),
            parallel_subgoal_proof_attempts=config_dict.get('parallel_subgoal_proof_attempts', 4),
            proof_sketch_corrections=config_dict.get('proof_sketch_corrections', 8),
            missing_subgoal_extraction_attempts=config_dict.get('missing_subgoal_extraction_attempts', 3),
            proof_verification_timeout=config_dict.get('proof_verification_timeout', 60),
            max_prover_llm_calls=config_dict.get('max_prover_llm_calls', None),
            max_reasoner_llm_calls=config_dict.get('max_reasoner_llm_calls', None),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ProofAttemptConfig to dictionary."""
        return {
            'subgoal_decomp_attempts': self.subgoal_decomp_attempts,
            'formal_proof_attempts': self.formal_proof_attempts,
            'main_theorem_error_corrections': self.main_theorem_error_corrections,
            'subgoal_error_corrections': self.subgoal_error_corrections,
            'missing_tags_error_corrections': self.missing_tags_error_corrections,
            'parallel_subgoal_proof_attempts': self.parallel_subgoal_proof_attempts,
            'proof_sketch_corrections': self.proof_sketch_corrections,
            'missing_subgoal_extraction_attempts': self.missing_subgoal_extraction_attempts,
            'proof_verification_timeout': self.proof_verification_timeout,
            'max_prover_llm_calls': self.max_prover_llm_calls,
            'max_reasoner_llm_calls': self.max_reasoner_llm_calls
        }
