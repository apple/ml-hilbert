#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Comprehensive statistics tracking for HILBERT proof generation.

This module provides detailed tracking of LLM usage, strategy execution,
and performance metrics for proof generation.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from src.tools.directories import write_string_to_file

class StrategyType(Enum):
    """Types of proof strategies used in HILBERT."""
    FORMAL_LLM = "formal_llm"
    SHALLOW_SOLVE = "shallow_solve"
    SKETCH_GENERATION = "sketch_generation"
    SKETCH_ASSEMBLY = "sketch_assembly"
    SUBGOAL_DECOMP = "subgoal_decomposition"
    RECURSIVE = "recursive"
    ERROR_CORRECTION = "error_correction"
    SEARCH_OPERATION = "search_operation"

class LLMType(Enum):
    """Types of LLM clients."""
    INFORMAL_LLM = "informal_llm"
    PROVER_LLM = "prover_llm"

class PromptType(Enum):
    """Types of prompts/operations for LLM calls."""
    # Basic operations
    SIMPLE_CHAT = "simple_chat"
    CHAT_COMPLETION = "chat_completion"
    GENERATE_PROOF = "generate_proof"
    
    # Search operations
    SEARCH_QUERY_GENERATION = "search_query_generation"
    SEARCH_RESULT_SELECTION = "search_result_selection"
    SEARCH_SEMANTIC = "search_semantic"
    SEARCH_BY_NAME = "search_by_name"
    
    # Proof generation stages
    SKETCH_GENERATION = "sketch_generation"
    SKETCH_ASSEMBLY = "sketch_assembly"
    INFORMAL_PROOF_SKETCH = "informal_proof_sketch"
    LEAN_SKETCH_CREATION = "lean_sketch_creation"
    SHALLOW_SOLVE = "shallow_solve"
    
    # Error correction types
    ERROR_CORRECTION_MAIN_THEOREM = "error_correction_main_theorem"
    ERROR_CORRECTION_SUBGOAL = "error_correction_subgoal"
    ERROR_CORRECTION_MISSING_TAGS = "error_correction_missing_tags" 
    PROOF_SKETCH_CORRECTION = "proof_sketch_correction"
    SYNTAX_ERROR_CORRECTION = "syntax_error_correction"
    SEMANTIC_ERROR_CORRECTION = "semantic_error_correction"
    
    # Verification and validation
    SUBGOAL_CORRECTNESS_CHECK = "subgoal_correctness_check"
    LEAN_ERROR_CHECK = "lean_error_check"
    PROOF_VERIFICATION = "proof_verification"
    
    # Other specialized operations
    SUBGOAL_EXTRACTION = "subgoal_extraction"
    MISSING_SUBGOAL_EXTRACTION = "missing_subgoal_extraction"
    AUTOFORMALIZER_INITIAL = "autoformalizer_initial"
    AUTOFORMALIZER_FEEDBACK = "autoformalizer_feedback"
    AUTOFORMALIZER_CHECK = "autoformalizer_check"
    AUTOFORMALIZER_CORRECTION = "autoformalizer_correction"
    
    # Generic fallback
    UNKNOWN = "unknown"

class VerificationType(Enum):
    """Types of Lean verification operations based on HILBERTWorker contexts."""
    # Formal LLM proof verification (single_attempt_prover_llm)
    FORMAL_LLM_GENERATED_PROOF_VERIFICATION = "formal_llm_generated_proof_verification"
    
    # Shallow solve proof verification (shallow_solve)
    SHALLOW_SOLVE_GENERATED_PROOF_VERIFICATION = "shallow_solve_generated_proof_verification"
    
    # Subgoal error correction verification (_single_verify_and_correct_subgoal)
    SUBGOAL_ERROR_CORRECTION_VERIFICATION = "subgoal_error_correction_verification"
    
    # Proof sketch compilation verification (_compile_and_correct_proof_sketch)
    PROOF_SKETCH_WITH_SORRY_VERIFICATION = "proof_sketch_with_sorry_verification"
    
    # Proof sketch error correction verification (_compile_and_correct_proof_sketch)
    PROOF_SKETCH_ERROR_CORRECTION_VERIFICATION = "proof_sketch_error_correction_verification"
    
    # Sketch assembly verification (verify_and_correct_proof_sketch_with_theorems)
    SKETCH_ASSEMBLY_WITH_THEOREMS_VERIFICATION = "sketch_assembly_with_theorems_verification"
    
    # Sketch assembly error correction verification (verify_and_correct_proof_sketch_with_theorems)
    SKETCH_ASSEMBLY_ERROR_CORRECTION_VERIFICATION = "sketch_assembly_error_correction_verification"
    
    # Theorem syntax checking (_syntax_check_and_correct_single_theorem)
    EXTRACTED_THEOREM_SYNTAX_VERIFICATION = "extracted_theorem_syntax_verification"
    
    # Theorem statement error correction verification (correct_error_in_theorem)
    THEOREM_STATEMENT_ERROR_CORRECTION_VERIFICATION = "theorem_statement_error_correction_verification"
    
    # Batch verification operations
    BATCH_PROOF_VERIFICATION = "batch_proof_verification"
    
    # Generic fallback
    UNKNOWN_VERIFICATION = "unknown_verification"

@dataclass
class VerificationMetrics:
    """Metrics for a single Lean verification call."""
    verification_type: str  # VerificationType enum value
    verification_duration: float = 0.0  # in seconds
    timeout: int = 30  # timeout used for verification
    is_sorry_ok: bool = False  # whether sorry was allowed
    return_error_message: bool = False  # whether error message was requested
    success: bool = True  # whether verification succeeded (not timed out/crashed)
    proof_valid: bool = False  # whether the proof was actually valid
    has_error_message: bool = False  # whether an error message was returned
    proof_length: int = 0  # estimated length of proof being verified
    is_batch: bool = False  # whether this was a batch verification
    batch_size: int = 1  # size of batch (1 for single verification)
    context: str = ""  # Additional context (e.g., theorem name, function context)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""
    llm_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # Number of input tokens served from cache
    call_duration: float = 0.0  # in seconds
    prompt_type: str = ""  # e.g., "sketch_generation", "error_correction"
    success: bool = True
    context: str = ""  # Additional context about the call (e.g., theorem name, search query)

@dataclass
class SearchOperationMetrics:
    """Metrics for a search operation."""
    search_type: str  # "semantic_search", "name_search", etc.
    query: str
    num_results: int = 0
    search_duration: float = 0.0  # in seconds
    success: bool = True
    timestamp: float = field(default_factory=time.time)

@dataclass
class StrategyAttempt:
    """Metrics for a strategy attempt."""
    strategy: str
    start_time: float
    end_time: Optional[float] = None
    duration: float = None
    success: bool = False
    attempts_count: int = 1
    depth: int = 0
    theorem_name: str = ""
    error_message: Optional[str] = None

@dataclass
class ProofStatistics:
    """Comprehensive statistics for a proof generation session."""
    
    # Problem metadata
    problem_id: str
    problem_statement: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: float = None

    # Overall success metrics
    final_success: bool = False
    total_attempts: int = 0
    max_depth_reached: int = 0
    
    # LLM usage tracking
    llm_calls: List[LLMCallMetrics] = field(default_factory=list)
    
    # Search operations tracking
    search_operations: List[SearchOperationMetrics] = field(default_factory=list)
    
    # Strategy tracking
    strategy_attempts: List[StrategyAttempt] = field(default_factory=list)
    
    # Verification tracking
    verification_operations: List[VerificationMetrics] = field(default_factory=list)
    
    # Aggregated metrics (computed)
    total_llm_calls: Dict[str, int] = field(default_factory=dict)
    total_input_tokens: Dict[str, int] = field(default_factory=dict)
    total_output_tokens: Dict[str, int] = field(default_factory=dict)
    total_cached_input_tokens: Dict[str, int] = field(default_factory=dict)
    total_tokens: Dict[str, int] = field(default_factory=dict)
    
    # Search operation aggregates
    total_search_operations: Dict[str, int] = field(default_factory=dict)
    search_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Prompt type aggregates
    prompt_type_counts: Dict[str, int] = field(default_factory=dict)
    prompt_type_success_rates: Dict[str, float] = field(default_factory=dict)
    
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    
    # Verification operation aggregates
    total_verification_operations: Dict[str, int] = field(default_factory=dict)
    verification_success_rates: Dict[str, float] = field(default_factory=dict)
    verification_validity_rates: Dict[str, float] = field(default_factory=dict)
    total_verification_duration: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self._compute_aggregated_metrics()
    
    def add_llm_call(self, llm_type: LLMType, input_tokens: int, output_tokens: int, 
                     call_duration: float = 0.0, prompt_type: str = "", success: bool = True,
                     context: str = "", cached_input_tokens: int = 0):
        """Add an LLM call to the statistics."""
        call_metrics = LLMCallMetrics(
            llm_type=llm_type.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            call_duration=call_duration,
            prompt_type=prompt_type,
            success=success,
            context=context
        )
        self.llm_calls.append(call_metrics)
        self._update_llm_aggregates(call_metrics)
    
    def add_search_operation(self, search_type: str, query: str, num_results: int = 0,
                           search_duration: float = 0.0, success: bool = True):
        """Add a search operation to the statistics."""
        search_metrics = SearchOperationMetrics(
            search_type=search_type,
            query=query,
            num_results=num_results,
            search_duration=search_duration,
            success=success
        )
        self.search_operations.append(search_metrics)
        self._update_search_aggregates(search_metrics)
    
    def add_verification_operation(self, verification_type: VerificationType, 
                                 verification_duration: float = 0.0,
                                 timeout: int = 30, is_sorry_ok: bool = False,
                                 return_error_message: bool = False, success: bool = True,
                                 proof_valid: bool = False, has_error_message: bool = False,
                                 proof_length: int = 0, is_batch: bool = False,
                                 batch_size: int = 1, context: str = ""):
        """Add a verification operation to the statistics."""
        verification_metrics = VerificationMetrics(
            verification_type=verification_type.value,
            verification_duration=verification_duration,
            timeout=timeout,
            is_sorry_ok=is_sorry_ok,
            return_error_message=return_error_message,
            success=success,
            proof_valid=proof_valid,
            has_error_message=has_error_message,
            proof_length=proof_length,
            is_batch=is_batch,
            batch_size=batch_size,
            context=context
        )
        self.verification_operations.append(verification_metrics)
        self._update_verification_aggregates(verification_metrics)
    
    def start_strategy_attempt(self, strategy: StrategyType, depth: int = 0, 
                              theorem_name: str = "") -> int:
        """Start tracking a strategy attempt. Returns the attempt index."""
        attempt = StrategyAttempt(
            strategy=strategy.value,
            start_time=time.time(),
            depth=depth,
            theorem_name=theorem_name
        )
        self.strategy_attempts.append(attempt)
        self.strategy_counts[strategy.value] = self.strategy_counts.get(strategy.value, 0) + 1
        return len(self.strategy_attempts) - 1
    
    def complete_strategy_attempt(self, attempt_index: int, successfully_completed: bool, 
                                 error_message: Optional[str] = None):
        """Complete a strategy attempt."""
        if attempt_index < len(self.strategy_attempts):
            attempt = self.strategy_attempts[attempt_index]
            attempt.end_time = time.time()
            attempt.duration =  (attempt.end_time - attempt.start_time)
            attempt.success = successfully_completed
            attempt.error_message = error_message
    
    def increment_attempt_count(self, attempt_index: int):
        """Increment the attempt count for a strategy."""
        if attempt_index < len(self.strategy_attempts):
            self.strategy_attempts[attempt_index].attempts_count += 1
    
    def finalize_statistics(self, successfully_completed: bool):
        """Finalize the statistics when proof generation is complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.final_success = successfully_completed
        self.total_attempts = len(self.strategy_attempts)
        self.max_depth_reached = max((attempt.depth for attempt in self.strategy_attempts), default=0)
        self._compute_aggregated_metrics()
    
    def _update_llm_aggregates(self, call_metrics: LLMCallMetrics):
        """Update aggregated LLM metrics."""
        llm_type = call_metrics.llm_type
        prompt_type = call_metrics.prompt_type
        
        self.total_llm_calls[llm_type] = self.total_llm_calls.get(llm_type, 0) + 1
        self.total_input_tokens[llm_type] = self.total_input_tokens.get(llm_type, 0) + call_metrics.input_tokens
        self.total_output_tokens[llm_type] = self.total_output_tokens.get(llm_type, 0) + call_metrics.output_tokens
        self.total_cached_input_tokens[llm_type] = self.total_cached_input_tokens.get(llm_type, 0) + call_metrics.cached_input_tokens
        self.total_tokens[llm_type] = self.total_input_tokens[llm_type] + self.total_output_tokens[llm_type]
        
        # Update prompt type aggregates
        if prompt_type:
            self.prompt_type_counts[prompt_type] = self.prompt_type_counts.get(prompt_type, 0) + 1
    
    def _update_search_aggregates(self, search_metrics: SearchOperationMetrics):
        """Update aggregated search metrics."""
        search_type = search_metrics.search_type
        
        self.total_search_operations[search_type] = self.total_search_operations.get(search_type, 0) + 1
    
    def _update_verification_aggregates(self, verification_metrics: VerificationMetrics):
        """Update aggregated verification metrics."""
        verification_type = verification_metrics.verification_type
        
        self.total_verification_operations[verification_type] = \
            self.total_verification_operations.get(verification_type, 0) + 1
        self.total_verification_duration[verification_type] = \
            self.total_verification_duration.get(verification_type, 0.0) + verification_metrics.verification_duration
    
    def _compute_prompt_type_success_rates(self):
        """Compute success rates for each prompt type."""
        prompt_type_success_counts = {}
        
        # Count successes for each prompt type
        for call in self.llm_calls:
            if call.prompt_type:
                if call.success:
                    prompt_type_success_counts[call.prompt_type] = \
                        prompt_type_success_counts.get(call.prompt_type, 0) + 1
        
        # Calculate success rates
        for prompt_type, total_count in self.prompt_type_counts.items():
            success_count = prompt_type_success_counts.get(prompt_type, 0)
            self.prompt_type_success_rates[prompt_type] = success_count / total_count if total_count > 0 else 0.0
    
    def _compute_search_success_rates(self):
        """Compute success rates for each search type."""
        search_success_counts = {}
        
        # Count successes for each search type
        for search_op in self.search_operations:
            if search_op.success:
                search_success_counts[search_op.search_type] = \
                    search_success_counts.get(search_op.search_type, 0) + 1
        
        # Calculate success rates
        for search_type, total_count in self.total_search_operations.items():
            success_count = search_success_counts.get(search_type, 0)
            self.search_success_rates[search_type] = success_count / total_count if total_count > 0 else 0.0
    
    def _compute_verification_success_rates(self):
        """Compute success rates and validity rates for each verification type."""
        verification_success_counts = {}
        verification_validity_counts = {}
        
        # Count successes and validities for each verification type
        for verification_op in self.verification_operations:
            verification_type = verification_op.verification_type
            
            if verification_op.success:
                verification_success_counts[verification_type] = \
                    verification_success_counts.get(verification_type, 0) + 1
            
            if verification_op.proof_valid:
                verification_validity_counts[verification_type] = \
                    verification_validity_counts.get(verification_type, 0) + 1
        
        # Calculate success rates (did not timeout/crash)
        for verification_type, total_count in self.total_verification_operations.items():
            success_count = verification_success_counts.get(verification_type, 0)
            self.verification_success_rates[verification_type] = success_count / total_count if total_count > 0 else 0.0
            
            # Calculate validity rates (proof was actually valid)
            validity_count = verification_validity_counts.get(verification_type, 0)
            self.verification_validity_rates[verification_type] = validity_count / total_count if total_count > 0 else 0.0
    
    def _compute_aggregated_metrics(self):
        """Recompute all aggregated metrics."""
        # Reset aggregated metrics
        self.total_llm_calls.clear()
        self.total_input_tokens.clear()
        self.total_output_tokens.clear()
        self.total_tokens.clear()
        self.total_search_operations.clear()
        self.total_verification_operations.clear()
        self.total_verification_duration.clear()
        self.prompt_type_counts.clear()
        self.strategy_counts.clear()
        
        # Recompute from raw data
        for call in self.llm_calls:
            self._update_llm_aggregates(call)
        
        for search_op in self.search_operations:
            self._update_search_aggregates(search_op)
        
        for verification_op in self.verification_operations:
            self._update_verification_aggregates(verification_op)
        
        for attempt in self.strategy_attempts:
            self.strategy_counts[attempt.strategy] = self.strategy_counts.get(attempt.strategy, 0) + 1
        
        self._compute_prompt_type_success_rates()
        self._compute_search_success_rates()
        self._compute_verification_success_rates()
    
    def get_total_duration(self) -> float:
        """Get total duration of proof generation in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get a summary of key statistics."""
        total_calls = sum(self.total_llm_calls.values())
        total_tokens = sum(self.total_tokens.values())
        total_searches = sum(self.total_search_operations.values())
        total_verifications = sum(self.total_verification_operations.values())
        total_verification_time = sum(self.total_verification_duration.values())
        
        return {
            "problem_id": self.problem_id,
            "final_success": self.final_success,
            "total_duration_seconds": self.get_total_duration(),
            "total_attempts": self.total_attempts,
            "max_depth_reached": self.max_depth_reached,
            "total_llm_calls": total_calls,
            "total_tokens_used": total_tokens,
            "total_search_operations": total_searches,
            "total_verification_operations": total_verifications,
            "total_verification_duration_seconds": total_verification_time,
            "llm_breakdown": dict(self.total_llm_calls),
            "token_breakdown": dict(self.total_tokens),
            "search_operation_breakdown": dict(self.total_search_operations),
            "verification_operation_breakdown": dict(self.total_verification_operations),
            "verification_duration_breakdown": dict(self.total_verification_duration),
            "prompt_type_breakdown": dict(self.prompt_type_counts),
            "strategy_attempts": dict(self.strategy_counts),
            "prompt_type_success_rates": dict(self.prompt_type_success_rates),
            "search_success_rates": dict(self.search_success_rates),
            "verification_success_rates": dict(self.verification_success_rates),
            "verification_validity_rates": dict(self.verification_validity_rates)
        }
    
    def save_to_file(self, filepath: str):
        """Save statistics to a JSON file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to dictionary for JSON serialization
        stats_dict = asdict(self)
        
        # Add computed summary
        stats_dict["summary"] = self.get_summary_stats()
        stats_dict["generated_at"] = datetime.now().isoformat()
        
        write_string_to_file(json.dumps(stats_dict, indent=2), filepath)

    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProofStatistics':
        """Load statistics from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Remove computed fields that will be regenerated
        for field_name in ['summary', 'generated_at']:
            data.pop(field_name, None)
        
        # Convert strategy attempts back to dataclass objects
        if 'strategy_attempts' in data:
            data['strategy_attempts'] = [
                StrategyAttempt(**attempt) for attempt in data['strategy_attempts']
            ]
        
        # Convert LLM calls back to dataclass objects
        if 'llm_calls' in data:
            data['llm_calls'] = [
                LLMCallMetrics(**call) for call in data['llm_calls']
            ]
        
        # Convert search operations back to dataclass objects
        if 'search_operations' in data:
            data['search_operations'] = [
                SearchOperationMetrics(**search_op) for search_op in data['search_operations']
            ]
        
        # Convert verification operations back to dataclass objects
        if 'verification_operations' in data:
            data['verification_operations'] = [
                VerificationMetrics(**verification_op) for verification_op in data['verification_operations']
            ]
        
        stats = cls(**data)
        stats._compute_aggregated_metrics()
        return stats

class ProofStatisticsTracker:
    """Context manager for tracking proof statistics."""
    
    def __init__(self, problem_id: str, problem_statement: str, 
                 stats_save_dir: Optional[str] = None):
        self.stats = ProofStatistics(
            problem_id=problem_id, 
            problem_statement=problem_statement
        )
        self.stats_save_dir = stats_save_dir
        self.current_strategy_attempts = {}  # Track ongoing strategy attempts
    
    def __enter__(self) -> ProofStatistics:
        return self.stats
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        successfully_completed = exc_type is None
        self.stats.finalize_statistics(successfully_completed)
        
        if self.stats_save_dir:
            stats_file = os.path.join(
                self.stats_save_dir, 
                f"{self.stats.problem_id}_stats.json"
            )
            self.stats.save_to_file(stats_file)
    
    def track_llm_call(self, llm_type: LLMType, input_tokens: int, output_tokens: int,
                      call_duration: float = 0.0, prompt_type: str = "", success: bool = True,
                      context: str = "", cached_input_tokens: int = 0):
        """Helper method to track LLM calls."""
        self.stats.add_llm_call(llm_type, input_tokens, output_tokens, 
                               call_duration, prompt_type, success, context, cached_input_tokens)
    
    def track_search_operation(self, search_type: str, query: str, num_results: int = 0,
                              search_duration: float = 0.0, success: bool = True):
        """Helper method to track search operations."""
        self.stats.add_search_operation(search_type, query, num_results, 
                                      search_duration, success)
    
    def start_strategy(self, strategy: StrategyType, depth: int = 0, 
                      theorem_name: str = "") -> int:
        """Helper method to start tracking a strategy."""
        return self.stats.start_strategy_attempt(strategy, depth, theorem_name)
    
    def complete_strategy(self, attempt_index: int, success: bool, 
                         error_message: Optional[str] = None):
        """Helper method to complete a strategy attempt."""
        self.stats.complete_strategy_attempt(attempt_index, success, error_message)