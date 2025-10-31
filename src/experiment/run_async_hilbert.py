#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from src.models.AsyncHILBERT import AsyncHILBERT
from src.tracking.ProofAttemptConfig import ProofAttemptConfig
from src.inference.AsyncProverLLM import AsyncProverLLM
from src.tools.AsyncLLMClient import AsyncLLMClient
from src.tools.SemanticSearchEngine import SemanticSearchEngine
from src.inference.AsyncLeanVerifier import AsyncLeanVerifier
from logging import getLogger

logger = getLogger(__name__)

def run_async_hilbert(cfg):
    """
    Run AsyncHILBERT experiment
    The async handling is done internally by AsyncHILBERT.run_from_file().
    
    Args:
        cfg: Configuration object containing experiment parameters
        
    Returns:
        Dictionary with experiment results
    """
    # Extract config sections for cleaner access
    exp_cfg = cfg.experiment
    prover_cfg = exp_cfg.prover_llm
    informal_cfg = exp_cfg.informal_llm
    search_engine_cfg = exp_cfg.search_engine
    
    max_concurrent_requests = exp_cfg.max_concurrent_requests
    logger.info("Starting AsyncHILBERT experiment...")
    
    # Create factory functions for AsyncHILBERT
    # These ensure each worker gets fresh client instances
    
    def create_prover_llm():
        """Factory function to create AsyncProverLLM instances."""
        # Get fresh AsyncLLMClient instance from task_id
        base_url = prover_cfg.base_url
        prover_llm_client = AsyncLLMClient(base_url=base_url, model_name=prover_cfg.llm_name)

        return AsyncProverLLM(
            llm_client=prover_llm_client,
            model_name=prover_cfg.llm_name,
            prompt_strategy=prover_cfg.prompt_strategy,
            max_tokens=prover_cfg.max_tokens
        )
    
    def create_informal_llm_client():
        """Factory function to create AsyncLLMClient instances for informal reasoning."""
        return AsyncLLMClient(**informal_cfg)
    
    def create_lean_verifier():
        """Factory function to create AsyncLeanVerifier instances."""
        verifier_base_url = exp_cfg.verifier_base_url
        return AsyncLeanVerifier(base_url=verifier_base_url,
                        max_concurrent_requests=max_concurrent_requests)
    
    # Create the shared semantic search engine (thread-safe)
    search_engine_params = {k: v for k, v in search_engine_cfg.items()}
    search_engine = SemanticSearchEngine(**search_engine_params)

    # Create AsyncHILBERT instance with factory functions
    experiment = AsyncHILBERT(
        prover_llm_factory=create_prover_llm,
        informal_llm_client_factory=create_informal_llm_client,
        lean_verifier_factory=create_lean_verifier,
        search_engine=search_engine,
        verify_each_subgoal_separately=exp_cfg.verify_each_subgoal_separately,
        proof_attempt_config=exp_cfg.proof_attempt_config,
        complexity_proof_length_cutoff=exp_cfg.complexity_proof_length_cutoff,
        return_proofs=exp_cfg.save_proofs_to_disk,
        max_depth=exp_cfg.max_depth,
        max_concurrent_problems=exp_cfg.max_concurrent_problems,
        proof_save_dir=exp_cfg.proof_save_dir,
        sequential_processing=exp_cfg.get('sequential_processing', False),
        run_proof_attempts_sequentially=exp_cfg.run_proof_attempts_sequentially,
        enable_retrieval=exp_cfg.get('enable_retrieval', True)
    )
    
    # Run the experiment - run_from_file handles asyncio.run internally
    logger.info(f"Processing problems from: {cfg.data.file_path}")
    results = experiment.run_from_file(cfg.data.file_path)
    
    logger.info(f"AsyncHILBERT experiment completed successfully!")
    logger.info(f"Results: {results.get('successful_problems', 0)}/{results.get('total_problems', 0)} problems solved")
    logger.info(f"Pass rate: {results.get('pass_rate', 0):.2%}")
    
    return results
