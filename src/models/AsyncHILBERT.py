#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import asyncio
import os
import traceback
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from src.inference.AsyncProverLLM import AsyncProverLLM
from src.tools.AsyncLLMClient import AsyncLLMClient
from src.inference.AsyncLeanVerifier import AsyncLeanVerifier
from src.tools.SemanticSearchEngine import SemanticSearchEngine
from src.models.HILBERTWorker import HILBERTWorker
from src.tracking.ProofAttemptConfig import ProofAttemptConfig
from src.tracking.resource_tracking import MaxLLMCallsExceeded
from src.tools.AsyncJobPool import AsyncJobPool
from src.tools.string import extract_jsonl_contents
from logging import getLogger
from src.tools.directories import check_if_file_exists, open_file_contents

logger = getLogger(__name__)

class AsyncHILBERT:
    """
    Async coordinator class that manages multiple HILBERTWorkers for batch processing.
    
    This class uses AsyncJobPool to coordinate multiple HILBERTWorkers processing different
    problems in parallel
    """

    def __init__(self,
                 # Client factory functions to create instances per worker
                 prover_llm_factory,  # Function that returns AsyncProverLLM instance
                 informal_llm_client_factory,  # Function that returns AsyncLLMClient instance
                 lean_verifier_factory,  # Function that returns AsyncLeanVerifier instance
                 search_engine: SemanticSearchEngine,  # Shared search engine
                 return_proofs: bool,
                 proof_attempt_config: ProofAttemptConfig = None,
                 max_depth: int = 3,
                 max_concurrent_problems: int = 2,
                 verify_each_subgoal_separately: bool = True,
                 complexity_proof_length_cutoff: int = 20,
                 max_tokens: int = 16384,
                 proof_save_dir: Optional[str] = None,
                 sequential_processing: bool = False,
                 run_proof_attempts_sequentially: bool = False,
                 enable_retrieval: bool = True):
        """
        Initialize AsyncHILBERT with factory functions for creating worker instances.
        
        Args:
            prover_llm_factory: Function that returns AsyncProverLLM instance
            informal_llm_client_factory: Function that returns AsyncLLMClient instance  
            lean_verifier_factory: Function that returns AsyncLeanVerifier instance
            search_engine: Shared SemanticSearchEngine instance
            return_proofs: Whether to return generated proofs in results
            proof_attempt_config: Configuration for all proof attempt limits and retry counts.
                                If None, uses default ProofAttemptConfig values.
            max_depth: Maximum recursion depth for subgoal decomposition
            max_concurrent_problems: How many problems are solved concurrently
            verify_each_subgoal_separately: Whether to verify each subgoal separately
            complexity_proof_length_cutoff: Cutoff for proof complexity estimation
            max_tokens: Maximum tokens for LLM calls
            proof_save_dir: Optional directory to save successful proofs as .lean files
            sequential_processing: If True, process problems one by one instead of in parallel
            run_proof_attempts_sequentially: If True, run proof attempts sequentially within each worker
        """
        # Store factory functions
        self.prover_llm_factory = prover_llm_factory
        self.informal_llm_client_factory = informal_llm_client_factory
        self.lean_verifier_factory = lean_verifier_factory
        
        # Store shared resources and configuration
        self.search_engine = search_engine
        self.return_proofs = return_proofs
        self.proof_attempt_config = proof_attempt_config or ProofAttemptConfig()
        self.max_depth = max_depth
        self.verify_each_subgoal_separately = verify_each_subgoal_separately
        self.complexity_proof_length_cutoff = complexity_proof_length_cutoff
        self.max_tokens = max_tokens
        self.sequential_processing = sequential_processing
        self.run_proof_attempts_sequentially = run_proof_attempts_sequentially
        self.max_concurrent_problems = max_concurrent_problems
        # Proof saving configuration (passed to workers)
        self.proof_save_dir = proof_save_dir
        self.enable_retrieval = enable_retrieval
        
        # Track active workers for cleanup
        self._active_workers: List[HILBERTWorker] = []
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with resource cleanup."""
        await self.close()

    async def close(self):
        """Close all active workers and clean up resources."""
        if not self._closed:
            # Close all active workers
            close_tasks = []
            for worker in self._active_workers:
                if hasattr(worker, 'close'):
                    close_tasks.append(worker.close())
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self._active_workers.clear()
            self._closed = True

    async def _create_worker(self) -> HILBERTWorker:
        """
        Create a new HILBERTWorker instance with fresh clients.
        
        Returns:
            Configured HILBERTWorker instance
        """
        if self._closed:
            raise RuntimeError("AsyncHILBERT is closed")
        
        # Create fresh instances for this worker
        prover_llm = self.prover_llm_factory()
        informal_llm_client = self.informal_llm_client_factory()
        lean_verifier = self.lean_verifier_factory()
        
        # Create worker with shared configuration
        worker = HILBERTWorker(
            prover_llm=prover_llm,
            informal_llm_client=informal_llm_client,
            lean_verifier=lean_verifier,
            search_engine=self.search_engine,  # Shared
            proof_attempt_config=self.proof_attempt_config,
            max_depth=self.max_depth,
            verify_each_subgoal_separately=self.verify_each_subgoal_separately,
            complexity_proof_length_cutoff=self.complexity_proof_length_cutoff,
            max_tokens=self.max_tokens,
            proof_save_dir=self.proof_save_dir,
            run_proof_attempts_sequentially=self.run_proof_attempts_sequentially,
            enable_retrieval=self.enable_retrieval
        )
        
        # Track for cleanup
        self._active_workers.append(worker)
        return worker

    async def _single_problem_worker(self, problem_id: int, formal_statement: str, header: str) -> Tuple[int, bool, Optional[str]]:
        """
        Worker function for processing a single problem.
        
        Args:
            problem_id: Unique identifier for the problem
            formal_statement: The formal statement to prove
            header: The theorem header/context
            
        Returns:
            Tuple of (problem_id, success, proof)
        """

        # check if proof already exists
        if self.proof_save_dir is not None:
            proof_path = os.path.join(self.proof_save_dir, f"{problem_id}.lean")
            if check_if_file_exists(proof_path):
                logger.info(f"Skipping problem {problem_id} as proof already exists.")
                # read the file
                proof = open_file_contents(proof_path)
                return problem_id, True, proof
        try:
            # Create a dedicated worker for this problem
            async with await self._create_worker() as worker:
                success, proof = await worker.generate_single_proof(formal_statement, header, problem_id)
                return problem_id, success, proof
        except MaxLLMCallsExceeded as e:
            logger.error(f"Worker exceeded max LLM calls for problem {problem_id}: {e}")
            return problem_id, False, None
        except Exception as e:
            logger.error(f"Worker failed for problem {problem_id}: {e}")
            # print traceback
            traceback.print_exc()
            return problem_id, False, None

    async def batch_solve_problems(self, 
                                 formal_statements: List[str], 
                                 headers: List[str],
                                 problem_ids: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Solve multiple problems either sequentially or in parallel using HILBERTWorkers.
        
        Args:
            formal_statements: List of formal statements to prove
            headers: List of corresponding headers/contexts
            problem_ids: Optional list of problem IDs (default: use indices)
            
        Returns:
            Dictionary with results including success rates and proofs
        """
        if self.sequential_processing:
            return await self._batch_solve_sequential(formal_statements, headers, problem_ids)
        else:
            return await self._batch_solve_parallel(formal_statements, headers, problem_ids)

    async def _batch_solve_sequential(self, 
                                    formal_statements: List[str], 
                                    headers: List[str],
                                    problem_ids: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Solve multiple problems sequentially (one by one).
        
        Args:
            formal_statements: List of formal statements to prove
            headers: List of corresponding headers/contexts
            problem_ids: Optional list of problem IDs (default: use indices)
            
        Returns:
            Dictionary with results including success rates and proofs
        """
        if self._closed:
            raise RuntimeError("AsyncHILBERT is closed")
        
        if len(formal_statements) != len(headers):
            raise ValueError("formal_statements and headers must have the same length")
        
        num_problems = len(formal_statements)
        if problem_ids is None:
            problem_ids = list(range(num_problems))
        
        logger.info(f"Processing {num_problems} problems sequentially with AsyncHILBERT...")
        
        results = {}
        proofs = {}
        failure_cases = []
        completed_count = 0
        
        # Process problems one by one
        for i, (formal_statement, header) in enumerate(zip(formal_statements, headers)):
            problem_id = problem_ids[i]
            logger.info(f"Processing problem {i+1}/{num_problems} (ID: {problem_id})...")
            
            try:
                problem_id_result, success, proof = await self._single_problem_worker(
                    problem_id, formal_statement, header
                )
                
                results[problem_id_result] = success
                completed_count += 1
                
                if success and proof:
                    if self.return_proofs:
                        proofs[problem_id_result] = proof
                    logger.info(f"✓ Problem {problem_id_result} solved successfully")
                else:
                    failure_cases.append(problem_id_result)
                    logger.info(f"✗ Problem {problem_id_result} failed")
                    
            except Exception as e:
                logger.error(f"Problem {problem_id} failed with exception: {e}")
                results[problem_id] = False
                failure_cases.append(problem_id)
                completed_count += 1
                logger.info(f"✗ Problem {problem_id} failed with exception")
        
        # Calculate statistics
        success_count = sum(1 for success in results.values() if success)
        pass_rate = success_count / len(results) if results else 0.0
        
        return_dict = {
            "total_problems": num_problems,
            "completed_problems": completed_count,
            "successful_problems": success_count,
            "results": results,
            "failure_cases": failure_cases,
            "pass_rate": pass_rate,
        }
        
        if self.return_proofs:
            return_dict["proofs"] = proofs
            return_dict["formal_statements"] = {pid: stmt for pid, stmt in zip(problem_ids, formal_statements)}
        
        logger.info(f"Sequential processing complete: {success_count}/{completed_count} problems solved ({pass_rate:.2%} success rate)")
        
        return return_dict

    async def _batch_solve_parallel(self, 
                                  formal_statements: List[str], 
                                  headers: List[str],
                                  problem_ids: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Solve multiple problems in parallel using HILBERTWorkers and AsyncJobPool.
        
        Args:
            formal_statements: List of formal statements to prove
            headers: List of corresponding headers/contexts
            problem_ids: Optional list of problem IDs (default: use indices)
            
        Returns:
            Dictionary with results including success rates and proofs
        """
        if self._closed:
            raise RuntimeError("AsyncHILBERT is closed")
        
        if len(formal_statements) != len(headers):
            raise ValueError("formal_statements and headers must have the same length")
        
        num_problems = len(formal_statements)
        if problem_ids is None:
            problem_ids = list(range(num_problems))
        
        logger.info(f"Processing {num_problems} problems in parallel with AsyncHILBERT...")
        
        # Use AsyncJobPool to coordinate workers
        pool = AsyncJobPool(max_concurrent=self.max_concurrent_problems)
        
        # Submit jobs to the pool
        for i, (formal_statement, header) in enumerate(zip(formal_statements, headers)):
            pool.submit(
                self._single_problem_worker, 
                problem_ids[i], 
                formal_statement, 
                header,
                name=f"{problem_ids[i]}"
            )
        
        # Process results as they complete
        results = {}
        proofs = {}
        failure_cases = []
        completed_count = 0
        
        # Use AsyncJobPool.wait_for_all() to collect all results
        try:
            all_results = await pool.wait_for_all()
            
            # Process results
            for task_name, result in all_results:
                if isinstance(result, Exception):
                    # Extract problem_id from task_name
                    try:
                        problem_id = task_name.split('_')[1] if '_' in task_name else task_name
                        problem_id = int(problem_id) if str(problem_id).isdigit() else problem_id
                    except:
                        problem_id = task_name
                    
                    logger.error(f"Task {task_name} failed with exception: {result}")
                    results[problem_id] = False
                    failure_cases.append(problem_id)
                    completed_count += 1
                    continue
                    
                # Successful task - unpack the result tuple
                problem_id, success, proof = result
                results[problem_id] = success
                completed_count += 1
                
                if success and proof:
                    # Store in memory if return_proofs is enabled
                    if self.return_proofs:
                        proofs[problem_id] = proof
                else:
                    failure_cases.append(problem_id)
        
        except Exception as e:
            logger.error(f"Parallel batch processing failed: {e}")
            raise
        
        # Calculate statistics
        success_count = sum(1 for success in results.values() if success)
        pass_rate = success_count / len(results) if results else 0.0
        
        return_dict = {
            "total_problems": num_problems,
            "completed_problems": completed_count,
            "successful_problems": success_count,
            "results": results,
            "failure_cases": failure_cases,
            "pass_rate": pass_rate,
        }
        
        if self.return_proofs:
            return_dict["proofs"] = proofs
            return_dict["formal_statements"] = {pid: stmt for pid, stmt in zip(problem_ids, formal_statements)}
        
        logger.info(f"Parallel processing complete: {success_count}/{completed_count} problems solved ({pass_rate:.2%} success rate)")
        
        return return_dict

    def run_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process problems from a JSONL file (synchronous interface).
        Uses asyncio.run internally to handle the async execution.
        
        Args:
            file_path: Path to the JSONL file containing problems
            
        Returns:
            Dictionary with results including success rates and proofs
        """
        return asyncio.run(self._run_from_file_async(file_path))

    async def _run_from_file_async(self, file_path: str) -> Dict[str, Any]:
        """
        Internal async implementation of run_from_file.
        
        Args:
            file_path: Path to the JSONL file containing problems
            
        Returns:
            Dictionary with results including success rates and proofs
        """
        if self._closed:
            raise RuntimeError("AsyncHILBERT is closed")
        
        # Load examples from file
        examples = extract_jsonl_contents(file_path)
        # shuffle examples
        random.shuffle(examples)

        formal_statements = [example['formal_statement'] for example in examples]
        headers = [example['header'] for example in examples]
        
        # Handle ID field
        if 'name' in examples[0]:
            problem_ids = [example['name'] for example in examples]
        elif 'id' in examples[0]:
            problem_ids = [example['id'] for example in examples]
        else:
            problem_ids = list(range(len(examples)))
        
        return await self.batch_solve_problems(formal_statements, headers, problem_ids)

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of AsyncHILBERT.
        
        Returns:
            Dictionary containing configuration parameters
        """
        config_dict = {
            "max_depth": self.max_depth,
            "verify_each_subgoal_separately": self.verify_each_subgoal_separately,
            "complexity_proof_length_cutoff": self.complexity_proof_length_cutoff,
            "max_tokens": self.max_tokens,
            "return_proofs": self.return_proofs,
            "proof_save_dir": self.proof_save_dir,
            "sequential_processing": self.sequential_processing,
            "max_concurrent_problems": self.max_concurrent_problems,
            "run_proof_attempts_sequentially": self.run_proof_attempts_sequentially,
        }
        
        # Add proof attempt configuration
        config_dict.update(self.proof_attempt_config.to_dict())
        
        return config_dict
