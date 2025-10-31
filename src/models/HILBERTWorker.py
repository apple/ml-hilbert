#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import asyncio
import re
import os
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from src.inference.AsyncProverLLM import AsyncProverLLM
from src.tools.AsyncLLMClient import AsyncLLMClient
from src.inference.AsyncLeanVerifier import AsyncLeanVerifier
from src.tracking.ProofAttemptConfig import ProofAttemptConfig
from src.tracking.ProofProgressTracker import ProofTree, SubgoalNode, ProofStatus, ProofStrategy
from src.tools.string import extract_lean_block, extract_search_queries, extract_theorems_queries, extract_tag, extract_all_lean_blocks
from src.tools.lean_utils import replace_have_proofs_with_sorry, check_theorem_signature_match, \
    extract_theorem_signature, _remove_comments, extract_theorem_name, extract_proof_body_from_theorem, remove_import_statements, \
    extract_missing_identifiers, extract_all_have_names, _check_for_sorries, _extract_all_theorems_from_string, \
    remove_import_lines
from src.prompts.hilbert import INFORMAL_LLM_CREATE_LEAN_SKETCH, INFORMAL_LLM_INFORMAL_PROOF_SKETCH, SEARCH_QUERY_PROMPT, SEARCH_ANSWER_PROMPT, \
    PROOF_SKETCH_CORRECTION_PROMPT, SUBGOAL_PROOF_CORRECTION_PROMPT, ERROR_CORRECTION_SYSTEM_PROMPT, \
        ERROR_SEARCH_QUERY_PROMPT, SOLVE_SUBGOAL_PROMPT, THEOREM_CORRECTION_PROMPT, \
        DETERMINE_IF_CORRECT_SUBGOAL, CORRECT_SKETCH_BASED_ON_INCORRECT_SUBGOAL, EXTRACT_SUBGOALS_FROM_SKETCH_PROMPT, \
        EXTRACT_MISSING_SUBGOALS_PROMPT, USE_SKETCH_AND_THEOREMS_TO_PROVE, PROOF_SKETCH_ASSEMBLY_CORRECTION_PROMPT, \
        MISSING_INFORMAL_PROOF_TAG_ERROR_PROMPT, MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT, THEOREM_SIGNATURE_MISMATCH_PROMPT, \
        CHECK_FOR_LEAN_ERRORS_PROMPT, POTENTIALLY_USEFUL_THEOREMS_PROMPT, NO_THEOREM_STATEMENT_PROMPT
from src.prompts.lean_hints import GENERAL_HINTS, TACTIC_HINTS
from src.tools.SemanticSearchEngine import SemanticSearchEngine
from src.tools.AsyncJobPool import AsyncJobPool
from src.tools.directories import write_string_to_file, make_dirs
from src.tracking.ProofStatistics import ProofStatistics, StrategyType, LLMType, PromptType, VerificationType
from src.tracking.resource_tracking import TrackedAsyncLLMClient, TrackedAsyncProverLLM, StrategyTracker, TrackedAsyncLeanVerifier, MaxLLMCallsExceeded
from logging import getLogger

logger = getLogger(__name__)

class DummyContext:
    """Dummy context manager for when statistics are disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class HILBERTWorker:
    """
    A standalone async worker class for processing single HILBERT problems.
    
    This class owns its own AsyncLLMClient and AsyncLeanVerifier instances and is designed
    to work on one problem at a time. It uses AsyncJobPool internally for parallel operations
    like multiple proof attempts or subgoal solving.
    """

    def __init__(self,
                 prover_llm: AsyncProverLLM,
                 informal_llm_client: AsyncLLMClient,
                 lean_verifier: AsyncLeanVerifier,
                 search_engine: SemanticSearchEngine,
                 proof_attempt_config: ProofAttemptConfig = None,
                 max_depth: int = 3,
                 verify_each_subgoal_separately: bool = True,
                 complexity_proof_length_cutoff: int = 20,
                 max_tokens: int = 16384,
                 proof_save_dir: str = None,
                 run_proof_attempts_sequentially: bool = False,
                 enable_statistics: bool = True,
                 enable_retrieval: bool = True):
        """
        Initialize HILBERTWorker with async clients and configuration.
        
        Args:
            proof_attempt_config: Configuration for all proof attempt limits and retry counts.
                                If None, uses default ProofAttemptConfig values.
            run_proof_attempts_sequentially: If True, run proof attempts one at a time instead of in parallel.
                                            If False (default), use parallel execution for faster processing.
        """
        self.max_depth = max_depth
        self.prover_llm = prover_llm
        self.proof_config = proof_attempt_config or ProofAttemptConfig()
        self.verify_each_subgoal_separately = verify_each_subgoal_separately
        self.informal_llm_client = informal_llm_client

        self.search_engine = search_engine
        self.lean_verifier = lean_verifier
        self.complexity_proof_length_cutoff = complexity_proof_length_cutoff
        self.max_tokens = max_tokens
        self.proof_save_dir = proof_save_dir
        self.run_proof_attempts_sequentially = run_proof_attempts_sequentially
        self.enable_statistics = enable_statistics
        self.enable_retrieval = enable_retrieval
        # Setup proof directory if specified
        if self.proof_save_dir:
            self._setup_proof_directory()
        
        # Track if we're closed for resource management
        self._closed = False
        
        # Initialize proof tree for progress tracking
        self.proof_tree = None
        self._proof_tree_lock = asyncio.Lock()
        
        # Statistics tracking
        self.enable_statistics = enable_statistics
        self.stats = None
        self.original_prover_llm = prover_llm
        self.original_informal_llm_client = informal_llm_client
        self.original_lean_verifier = lean_verifier

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with resource cleanup."""
        await self.close()

    async def close(self):
        """Close async clients and clean up resources."""
        if not self._closed:
            # Close clients that support async context management
            if hasattr(self.informal_llm_client, 'close'):
                await self.informal_llm_client.close()
            if hasattr(self.lean_verifier, 'close'):
                await self.lean_verifier.close()
            self._closed = True

    async def _remove_import_statements(self, theorem: str, prev_prompt: str = None, prev_response: str = None):
        """
        Remove import statements from the theorem.
        If the resulting theorem is empty, ask the LLM to give the full theorem
        """
        no_import_theorem = remove_import_statements(theorem)
        if not no_import_theorem:
            for i in range(self.proof_config.missing_tags_error_corrections):
                logger.info(100*"*")
                logger.info("Trying to obtain the full theorem statement, attempt %d/%d", i+1, self.proof_config.missing_tags_error_corrections)
                logger.info(100*"*")

                # Create messages list with previous context if available
                messages = []
                if prev_prompt and prev_response:
                    messages.append({"role": "user", "content": prev_prompt})
                    messages.append({"role": "assistant", "content": prev_response})
                
                # Add the request for full theorem statement
                messages.append({"role": "user", "content": NO_THEOREM_STATEMENT_PROMPT})
                
                # Get response from informal LLM
                response = await self.informal_llm_client.chat_completion(
                    messages, 
                    max_tokens=16384, 
                    reasoning={'effort': 'high'},
                    prompt_type=PromptType.ERROR_CORRECTION_MISSING_TAGS.value,
                    context="Requesting full theorem statement"
                )
                
                logger.info(80*"*")
                logger.info("Full theorem statement response (attempt %d)", i+1)
                logger.info(80*"*")
                logger.info(response)
                logger.info(80*"*")
                
                # Extract theorem from response
                extracted_theorem = extract_lean_block(response)
                if extracted_theorem:
                    # Remove import statements from the extracted theorem
                    final_theorem = remove_import_statements(extracted_theorem)
                    if final_theorem:
                        return final_theorem
                
                # If extraction failed, add the failed response to conversation for next attempt
                messages.append({"role": "assistant", "content": response})
            
            # If all attempts failed, return None
            return None
        
        return no_import_theorem

    def _setup_proof_directory(self):
        """Create the proof save directory if it doesn't exist."""
        if self.proof_save_dir:
            make_dirs(self.proof_save_dir)
            # Also create proof_stats directory
            stats_dir = os.path.join(self.proof_save_dir, "proof_stats")
            make_dirs(stats_dir)
            logger.info("Proof save directory set up at: %s", self.proof_save_dir)

    async def print_proof_tree(self) -> None:
        """
        Print the current progress of the proof in tree format.
        
        Shows the hierarchical structure of the problem including:
        - All subgoals and their status
        - Which subgoals have been solved/failed
        - What is currently being recursed
        - Current depth and attempt information
        """
        async with self._proof_tree_lock:
            if not self.proof_tree:
                logger.info("No proof tree available - proof not started yet.")
                return
            
            logger.info("\n%s", "="*80)
            logger.info("PROOF PROGRESS TREE")
            logger.info("="*80)
            tree_output = self.proof_tree.print_tree()
            logger.info(tree_output)
            logger.info("="*80)
            logger.info("\n")

    async def _initialize_proof_tree(self, problem: str, problem_id: str = None) -> None:
        """Initialize the proof tree for a new problem."""
        async with self._proof_tree_lock:
            theorem_name = extract_theorem_name(problem) or problem_id or "main_problem"
            root_node = SubgoalNode(
                name=theorem_name,
                theorem=problem,
                depth=0,
                status=ProofStatus.SOLVING
            )
            self.proof_tree = ProofTree(root=root_node, max_depth=self.max_depth)

    async def update_proof_tree_strategy(self, theorem: str, strategy: ProofStrategy):
        """Update the strategy for a given theorem in the proof tree."""
        if self.proof_tree:
            if bool(re.search(r'\btheorem\b', theorem)):
                theorem_name = extract_theorem_name(theorem)
            else:
                theorem_name = theorem
            async with self._proof_tree_lock:
                self.proof_tree.set_node_strategy(theorem_name, strategy)
    
    async def update_proof_tree_status(self, theorem: str, status: ProofStatus):
        """Update the status for a given theorem in the proof tree."""
        if self.proof_tree:
            if bool(re.search(r'\btheorem\b', theorem)):
                theorem_name = extract_theorem_name(theorem)
            else:
                theorem_name = theorem
            async with self._proof_tree_lock:
                self.proof_tree.update_node_status(theorem_name, status)

    def _save_proof_to_file(self, problem_id: str, header: str, proof: str):
        """
        Save a proof to a .lean file immediately when it's generated.
        
        Args:
            problem_id: Unique identifier for the problem  
            header: The theorem header/context
            proof: The generated proof
        """
        if not self.proof_save_dir:
            return
            
        try:
            # Use problem_id directly as filename
            filename = f"{problem_id}.lean"
            filepath = os.path.join(self.proof_save_dir, filename)
            full_content = ""
            # Combine header and proof
            full_content += header.strip() + "\n"
            full_content += proof.strip() + "\n"
            
            # Write to file
            write_string_to_file(full_content, filepath)

            logger.info("Proof saved for problem %s: %s", problem_id, filepath)

        except Exception as e:
            logger.error("Failed to save proof for problem %s: %s", problem_id, e)
            traceback.print_exc()

    async def _save_proof_tree_to_file(self, problem_id: str):
        """
        Save the proof progress tree to a text file after proof completion.
        
        Args:
            problem_id: Unique identifier for the problem
        """
        if not self.proof_save_dir or not problem_id:
            return
            
        try:
            async with self._proof_tree_lock:
                if not self.proof_tree:
                    logger.warning("No proof tree available to save for problem %s", problem_id)
                    return
                
                # Generate tree output
                tree_output = self.proof_tree.print_tree()
                
                # Create proof_stats directory within proof_save_dir
                stats_dir = os.path.join(self.proof_save_dir, "proof_stats")
                make_dirs(stats_dir)
                
                # Create filename and filepath
                filename = f"{problem_id}_proof_tree.txt"
                filepath = os.path.join(stats_dir, filename)
                
                # Add header with metadata
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                full_content = f"Proof Progress Tree for Problem: {problem_id}\n"
                full_content += f"Generated at: {timestamp}\n"
                full_content += "="*80 + "\n\n"
                full_content += tree_output
                full_content += "\n\n" + "="*80 + "\n"
                
                # Write to file
                write_string_to_file(full_content, filepath)

                logger.info("Proof tree saved for problem %s: %s", problem_id, filepath)

        except Exception as e:
            logger.error("Failed to save proof tree for problem %s: %s", problem_id, e)
            traceback.print_exc()

    async def _run_async_pool_and_get_first_truthy(self, worker_coro, list_of_args):
        """
        Run multiple async jobs using AsyncJobPool and return the first truthy result.
        
        This mirrors the functionality from the original HILBERT class but for the worker's
        internal parallel operations.

        Args:
            worker_coro: The async worker function to run
            list_of_args: A list of tuples, where each tuple contains the arguments for one call
            
        Returns:
            First truthy result or None if all fail
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        pool = AsyncJobPool()
        for args_tuple in list_of_args:
            # Submit the coroutine to the pool
            pool.submit(worker_coro, *args_tuple)
        
        # Await the pool's result
        return await pool.wait_for_first_truthy()

    async def _correct_response_for_missing_tags(self, original_prompt: str, original_response: str, error_prompt: str, extract_function, max_attempts: int = None) -> Optional[str]:
        """
        Correct a response that is missing required tags or code blocks.
        
        Args:
            original_prompt: The original prompt that was sent
            original_response: The original response that was missing tags/blocks
            error_prompt: Error message to send to the model
            extract_function: Function to extract the required content (e.g., extract_tag, extract_lean_block)
            max_attempts: Maximum number of correction attempts (uses config default if None)
            
        Returns:
            Extracted content or None if all attempts failed
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Use config value if max_attempts not provided
        if max_attempts is None:
            max_attempts = self.proof_config.missing_tags_error_corrections
        
        # Create the messages with the original conversation
        messages = [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": original_response}
        ]
        messages.append({"role": "user", "content": error_prompt})
        
        for attempt in range(max_attempts):
            logger.info("Error correction attempt %d/%d", attempt + 1, max_attempts)
            
            response = await self.informal_llm_client.chat_completion(
                messages, 
                max_tokens=32768, 
                reasoning={'effort': 'high'},
                prompt_type=PromptType.ERROR_CORRECTION_MISSING_TAGS.value,
                context="Correcting missing tags/code blocks"
            )
            
            logger.info(80*"*")
            logger.info("Error correction response (attempt %d)", attempt + 1)
            logger.info(80*"*")
            logger.info(response)
            logger.info(80*"*")
            
            # Try to extract the required content
            extracted_content = extract_function(response)
            if extracted_content:
                return extracted_content
            
            # Add the failed response to conversation for next attempt
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": error_prompt})
        
        return None

    async def generate_proof_sketch(self, problem: str, useful_theorems: str) -> Optional[str]:
        """Generate an informal proof sketch for the given problem."""
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")

        with StrategyTracker(self.stats, StrategyType.SKETCH_GENERATION, 0, extract_theorem_name(problem) or "main") if self.enable_statistics else DummyContext():
            logger.info(80*"*")
            logger.info("Coming up with informal proof")
            logger.info(80*"*")
            logger.info("Prompt:")

            useful_theorems_section = self._format_useful_theorems_section(useful_theorems)
            input_prompt = INFORMAL_LLM_INFORMAL_PROOF_SKETCH.format(problem=problem, useful_theorems_section=useful_theorems_section)
            informal_proof_string = await self.informal_llm_client.simple_chat(
                input_prompt, 
                max_tokens=32768, 
                reasoning={'effort': 'very_high'},
                prompt_type=PromptType.INFORMAL_PROOF_SKETCH.value,
                context=f"Problem: {problem}"
            )
            
            logger.info(80*"*")
            logger.info("Informal proof response")
            logger.info(80*"*")
            logger.info(informal_proof_string)
            logger.info(80*"*")
            
            # extract informal_proof from it
            informal_proof = extract_tag(informal_proof_string, "informal_proof")
            if not informal_proof:
                # run error correction loop
                informal_proof = await self._correct_response_for_missing_tags(
                    input_prompt,
                    informal_proof_string,
                    MISSING_INFORMAL_PROOF_TAG_ERROR_PROMPT,
                    lambda response: extract_tag(response, "informal_proof")
                )
                if not informal_proof:
                    return None
            
            useful_theorems_section = self._format_useful_theorems_section(useful_theorems)
            input_prompt = INFORMAL_LLM_CREATE_LEAN_SKETCH.format(problem=problem, useful_theorems_section=useful_theorems_section, lean_hints=GENERAL_HINTS, informal_proof=informal_proof)
            logger.info(input_prompt)
            proof_string = await self.informal_llm_client.simple_chat(
                input_prompt, 
                max_tokens=32768, 
                reasoning={'effort': 'very_high'},
                prompt_type=PromptType.LEAN_SKETCH_CREATION.value,
                context=f"Problem: {problem}"
            )

            logger.info(80*"*")
            logger.info("Printing Informal Proof Sketch Response Before Self Reflection")
            logger.info(80*"*")
            logger.info(proof_string)
            logger.info(80*"*")

            parsed_proof = extract_lean_block(proof_string)
            if not parsed_proof:
                # error correction loop
                parsed_proof = await self._correct_response_for_missing_tags(
                    input_prompt,
                    proof_string,
                    MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                    extract_lean_block
                )
                if not parsed_proof:
                    return None
            
            # remove import statements if any
            parsed_proof = await self._remove_import_statements(parsed_proof, input_prompt, proof_string)
            if not parsed_proof:
                return None
            parsed_proof = replace_have_proofs_with_sorry(parsed_proof)
            
            logger.info(80*"*")
            logger.info("Printing Informal Proof Sketch Before Self Reflection")
            logger.info(80*"*")
            logger.info(parsed_proof)
            logger.info(80*"*")
            
            return parsed_proof

    async def correct_error_and_send_proof(self,
                             error_message: str,
                             useful_theorems: str,
                             messages: List[Dict[str, str]],
                             type_of_correction: str = None,
                             problem: str = None) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Correct an error and generate a new proof attempt.
        
        Args:
            error_message: The error message to correct
            useful_theorems: Useful theorems for correction
            messages: Current conversation messages list
            sketch_correction: Whether this is sketch correction
            problem: The problem statement for sketch correction
            
        Returns:
            Tuple of (generated_proof, updated_messages)
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        if not error_message:
            # missing error message, terminate early
            logger.info(100*"$")
            logger.info("ERROR MESSAGE MISSING SO STOPPING")
            logger.info(100*"$")
            
            return None, None
        
        # add mistaken identifiers to useful theorems
        if type_of_correction == 'sketch_outline':
            useful_theorems_section = self._format_useful_theorems_section(useful_theorems)
            proof_correction_prompt = PROOF_SKETCH_CORRECTION_PROMPT.format(informal_statement=problem,
                                                                            error_message=error_message,
                                                                            useful_theorems_section=useful_theorems_section,
                                                                            lean_hints=GENERAL_HINTS)
        elif type_of_correction == 'subgoal':
            useful_theorems_section = self._format_useful_theorems_section(useful_theorems)
            proof_correction_prompt = SUBGOAL_PROOF_CORRECTION_PROMPT.format(error_message=error_message,
                                                                             useful_theorems_section=useful_theorems_section)
        elif type_of_correction == 'sketch_completion':
            proof_correction_prompt = PROOF_SKETCH_ASSEMBLY_CORRECTION_PROMPT.format(error_message=error_message,
                                                                                     lean_hints=GENERAL_HINTS)
        else:
            raise ValueError(f"Unknown type of correction: {type_of_correction}")
        # Add user message to conversation
        messages_copy = messages.copy()
        messages_copy.append({"role": "user", "content": proof_correction_prompt})
        
        final_answer = await self.informal_llm_client.chat_completion(
            messages_copy, 
            max_tokens=49152, 
            reasoning={'effort': 'high'},
            prompt_type=PromptType.ERROR_CORRECTION_SUBGOAL.value,
            context="Error: {}\n\nType:{}".format(error_message, type_of_correction)
        )
        logger.info("PRODUCED ANSWER for error message: %s", error_message)
        logger.info(final_answer)
        
        # Add assistant response to conversation
        messages_copy.append({"role": "assistant", "content": final_answer})
        
        generated_proof = extract_lean_block(final_answer)
        if not generated_proof:
            # error correction loop
            generated_proof = await self._correct_response_for_missing_tags(
                proof_correction_prompt,
                final_answer,
                MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                extract_lean_block
            )
        if generated_proof:
            generated_proof = await self._remove_import_statements(generated_proof, proof_correction_prompt, final_answer)
            if not generated_proof:
                return None, messages_copy
        return generated_proof, messages_copy

    async def correct_error_in_theorem(self,
                             error_message: str,
                             header: str,
                             problem: str) -> Optional[str]:
        """Correct syntax/semantic errors in theorem statements."""
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        if not error_message:
            logger.info(100*"$")
            logger.info("ERROR MESSAGE MISSING IN CORRECT ERROR IN THEOREM")
            logger.info(100*"$")
            return None
        
        # Create local conversation with system message
        messages = [{"role": "system", "content": ERROR_CORRECTION_SYSTEM_PROMPT}]
        useful_theorems = ""
        for i in range(self.proof_config.main_theorem_error_corrections):
            # check if we need any missing theorems
            useful_theorems = await self._augment_useful_theorems(error_message, problem, useful_theorems)
            if useful_theorems:
                useful_theorems_str = POTENTIALLY_USEFUL_THEOREMS_PROMPT.format(useful_theorems=useful_theorems)
            else:
                useful_theorems_str = ""
            proof_correction_prompt = THEOREM_CORRECTION_PROMPT.format(error_message=error_message, 
                                                                       potentially_useful_theorems=useful_theorems_str)

            # Add user message and get response
            current_messages = messages.copy()
            current_messages.append({"role": "user", "content": proof_correction_prompt})
            
            final_answer = await self.informal_llm_client.chat_completion(
                current_messages, 
                max_tokens=16384, 
                reasoning={'effort': 'high'},
                prompt_type=PromptType.SYNTAX_ERROR_CORRECTION.value,
                context=f"Error: {error_message}"
            )
            logger.info(100*"*")
            logger.info("ERROR CORRECTION RESPONSE")
            logger.info(final_answer)
            logger.info(100*"*")
            generated_statement = extract_lean_block(final_answer)

            if not generated_statement:
                logger.info(100*"$")
                logger.info("ERROR: Response did not have Lean 4 code block. Trying to correct it")
                logger.info(100*"$")
                # error correction loop
                generated_statement = await self._correct_response_for_missing_tags(
                    proof_correction_prompt,
                    final_answer,
                    MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                    extract_lean_block
                )
                if not generated_statement:
                    logger.info(100*"$")
                    logger.info("ERROR: Response did not have Lean 4 code block. Returning None")
                    logger.info(100*"$")
                    return None
                
            # Remove import statements
            generated_statement = await self._remove_import_statements(generated_statement, proof_correction_prompt, final_answer)
            if not generated_statement:
                logger.info(100*"$")
                logger.info("ERROR: Could not obtain theorem statement without imports. Returning None")
                logger.info(100*"$")
                return None
            # Add assistant response to conversation for next iteration
            messages.append({"role": "user", "content": proof_correction_prompt})
            messages.append({"role": "assistant", "content": final_answer})
            
            status, error_message = await self.lean_verifier.verify_proof(
                header + generated_statement, 
                return_error_message=True, 
                is_sorry_ok=True,
                verification_type=VerificationType.THEOREM_STATEMENT_ERROR_CORRECTION_VERIFICATION,
                context=f"Theorem statement correction attempt {i+1}/{self.proof_config.main_theorem_error_corrections}"
            )
            if status:
                logger.info(80*"*")
                logger.info("THEOREM IS CORRECTED")
                logger.info(generated_statement)
                logger.info(80*"*")
                return generated_statement
            else:
                logger.info(80*"*")
                logger.info("THEOREM IS WRONG")
                logger.info("Error message")
                logger.info(error_message)
                logger.info(80*"*")

        return None

    def _format_useful_theorems_section(self, useful_theorems: str) -> str:
        """Format the useful theorems section based on retrieval settings."""
        if not self.enable_retrieval or not useful_theorems.strip():
            return ""
        else:
            return f"Here are a list of useful theorems from Mathlib to solve the problem:\n{useful_theorems}\n"

    def get_error_system_prompt(self) -> List[Dict[str, str]]:
        """Start a new error correction conversation by returning initial messages list."""
        return [{"role": "system", "content": ERROR_CORRECTION_SYSTEM_PROMPT}]

    async def search_and_select(self, problem: str = None, error_message: str = None) -> str:
        """Search for and select relevant theorems."""
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Return empty string if retrieval is disabled
        if not self.enable_retrieval:
            return ""
        
        if error_message is not None:
            # For error-based queries, use the conversational prompt that asks for exactly 2 queries
            search_prompt = ERROR_SEARCH_QUERY_PROMPT.format(error_message=error_message)
        else:
            search_prompt = SEARCH_QUERY_PROMPT.format(problem=problem)

        search_queries = await self.informal_llm_client.simple_chat(
            search_prompt,
            max_tokens=16384,
            reasoning={'effort': 'low'},
            prompt_type=PromptType.SEARCH_QUERY_GENERATION.value,
            context=f"Problem: {problem}" if problem else f"Error: {error_message}"
        )
            
        search_answer = await self._search_and_select_queries(problem, search_queries)
        return search_answer

    def get_search_answers(self, search_query_answer: str) -> str:
        """Extract the relevant theorems from the search query answer"""
        search_queries = extract_search_queries(search_query_answer)
        search_answer = ""
        for query in search_queries:
            search_answer += f"Results for query: {query}"
            search_answer += self.search_engine.get_search_results(query)
            search_answer += "\n"
        return search_answer
    
    async def _search_and_select_queries(self, problem: str, search_query_answer: str) -> str:
        """Process search queries and select most relevant theorems."""
        search_answer = self.get_search_answers(search_query_answer)
        logger.info(80*"*")
        logger.info("Extracted search results")
        logger.info(search_answer)
        logger.info(80*"*")

        # select the most relevant ones
        select_prompt = SEARCH_ANSWER_PROMPT.format(problem=problem, theorems=search_answer)
        selected_theorems_answer = await self.informal_llm_client.simple_chat(
            select_prompt,
            max_tokens=16384,
            reasoning={'effort': 'low'},
            prompt_type=PromptType.SEARCH_RESULT_SELECTION.value,
            context=f"Problem: {problem}"
        )
        logger.info(100*"&")
        logger.info("Selected theorems answer")
        logger.info(100*"&")
        logger.info(selected_theorems_answer)
        logger.info(100*"&")
        
        selected_theorems = extract_theorems_queries(selected_theorems_answer)
        logger.info(80*"*")
        logger.info("Selected Theorems")
        logger.info(selected_theorems)
        logger.info(80*"*")

        useful_theorems = self.search_engine.search_by_name(selected_theorems, exact_match=True)
        logger.info(80*"*")
        logger.info("Useful Theorems")
        logger.info(useful_theorems)
        logger.info(80*"*")
        return useful_theorems

    async def _single_attempt_prover_llm(self, theorem: str, header: str) -> Optional[str]:
        """Make a single attempt to prove a theorem using the formal prover LLM."""
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        proof = await self.prover_llm.generate_proof(header + theorem)
        logger.info(80*"*")
        logger.info("Generated Formal Proof:")
        logger.info(proof)
        logger.info(80*"*")
        if proof:
            # verify the proof and see what happens
            result, error_message = await self.lean_verifier.verify_proof(
                header + proof, 
                timeout=self.proof_config.proof_verification_timeout, 
                return_error_message=True, 
                is_sorry_ok=False,
                verification_type=VerificationType.FORMAL_LLM_GENERATED_PROOF_VERIFICATION,
                context=f"Formal LLM proof for theorem: {extract_theorem_name(theorem) or 'unknown'}"
            ) 
        else:
            result = False
            error_message = "No proof generated by Prover LLM"       
        logger.info("RESULT: %s", result)
        logger.info("ERROR MESSAGE: %s", error_message)

        if result:
            # check if proved theorem matches the input theorem
            if check_theorem_signature_match(theorem, proof):
                logger.info("Success! Proved theorem matches the input theorem!")
                return proof
            else:
                logger.info("WARNING: Theorem signatures did NOT match")
                logger.info(100*"*")
                logger.info("Generated Formal Proof:")
                logger.info(proof)
                logger.info(100*"*")
                logger.info("Input Theorem:")
                logger.info(theorem)
                logger.info(100*"*")
                logger.info("Proof Signature:")
                logger.info(extract_theorem_signature(proof))
                logger.info(100*"*")
                logger.info("Input Theorem Signature:")
                logger.info(extract_theorem_signature(theorem))
                logger.info(100*"*")
                return None
        else:
            logger.info("Proof verification failed!")
            logger.info(80*"*")
            logger.info("Error Message:")
            logger.info(error_message)
            logger.info(80*"*")
            return None

    async def _attempt_proof_with_prover_llm(self, theorem: str, header: str) -> Optional[str]:
        """
        Attempt proof generation with multiple parallel tries using AsyncJobPool.
        
        This mirrors the _run_async_pool_and_get_first_truthy pattern from original HILBERT.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Prepare the arguments for each parallel job
        args_for_each_job = [(theorem, header) for _ in range(self.proof_config.formal_proof_attempts)]

        # Use AsyncJobPool to run parallel attempts and get first successful result
        return await self._run_async_pool_and_get_first_truthy(
            self._single_attempt_prover_llm,
            args_for_each_job
        )

    async def _check_subgoal_is_correct(self, theorem: str) -> Tuple[bool, Optional[str]]:
        """Check if a subgoal theorem statement is mathematically correct."""
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        correctness_prompt = DETERMINE_IF_CORRECT_SUBGOAL.format(problem=theorem, 
                                                                 lean_hints=GENERAL_HINTS)
        informal_response = await self.informal_llm_client.simple_chat(
            correctness_prompt, 
            max_tokens=32768, 
            reasoning={'effort': 'high'},
            prompt_type=PromptType.SUBGOAL_CORRECTNESS_CHECK.value,
            context=f"Theorem: {theorem}"
        )
        
        if 'YES' in informal_response:
            logger.info("The subgoal is mathematically correct! Full response:")
            logger.info(informal_response)
        else:
            logger.info("Invalid theorem! Full response")
            logger.info(theorem)
            logger.info(informal_response)
            # extract the justification from <justification> tags
            justification = extract_tag(informal_response, 'justification')
            return False, justification

        # check for Lean correctness
        correctness_prompt = CHECK_FOR_LEAN_ERRORS_PROMPT.format(theorem_statement = theorem)
        informal_response = await self.informal_llm_client.simple_chat(
            correctness_prompt, 
            max_tokens=32768, 
            reasoning={'effort': 'high'},
            prompt_type=PromptType.LEAN_ERROR_CHECK.value,
            context=f"Theorem: {theorem}"
        )
        
        if 'YES' in informal_response:
            logger.info("The subgoal is correctly formed! Full response:")
            logger.info(informal_response)
        else:
            logger.info("Invalid theorem! Full response")
            logger.info(theorem)
            logger.info(informal_response)
            # extract the justification from <justification> tags
            justification = extract_tag(informal_response, 'justification')
            return False, justification
        return True, None
    
    def _estimate_proof_length(self, proof: str) -> int:
        """Estimate the number of lines in the proof."""
        # remove comments
        no_comments_proof = _remove_comments(proof)
        theorem_body = extract_proof_body_from_theorem(no_comments_proof)
        lines = theorem_body.split("\n")
        # remove empty lines
        lines = [line for line in lines if line.strip()]
        return len(lines)

    def _setup_statistics_tracking(self, problem_id: str, problem_statement: str):
        """Set up statistics tracking for a proof generation session."""
        if not self.enable_statistics:
            return
        
        self.stats = ProofStatistics(
            problem_id=problem_id or "unknown",
            problem_statement=problem_statement
        )
        
        # Wrap LLM clients with tracking wrappers
        self.prover_llm = TrackedAsyncProverLLM(self.original_prover_llm, self.stats, self.proof_config.max_prover_llm_calls)
        self.informal_llm_client = TrackedAsyncLLMClient(
            self.original_informal_llm_client, self.stats, LLMType.INFORMAL_LLM, self.proof_config.max_reasoner_llm_calls
        )
        # Wrap lean verifier with tracking wrapper
        self.lean_verifier = TrackedAsyncLeanVerifier(self.original_lean_verifier, self.stats)
    
    def _finalize_statistics(self, successfully_completed: bool, problem_id: str = None):
        """Finalize and save statistics."""
        if not self.enable_statistics or not self.stats:
            return
        
        self.stats.finalize_statistics(successfully_completed)
        
        # Save statistics file
        if self.proof_save_dir and problem_id:
            stats_file = os.path.join(
                self.proof_save_dir, "proof_stats", f"{problem_id}_stats.json"
            )
            self.stats.save_to_file(stats_file)
            logger.info("Statistics saved to: %s", stats_file)

        # Print summary
        summary = self.stats.get_summary_stats()
        logger.info("Proof generation completed. Summary: %s", summary)

    async def generate_single_proof(self, problem: str, header: str, problem_id: str = None) -> Tuple[bool, Optional[str]]:
        """
        Generate a single proof for the given problem.
        
        Args:
            problem: The formal problem statement
            header: The theorem header/context
            problem_id: Optional problem identifier for saving proofs to disk
            
        Returns:
            Tuple of (success, proof)
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Setup statistics tracking
        self._setup_statistics_tracking(problem_id, problem)
        
        try:
            # Initialize proof tree for progress tracking
            await self._initialize_proof_tree(problem, problem_id)
            
            await self.update_proof_tree_strategy(problem, ProofStrategy.FORMAL_LLM)
            
            logger.info("Starting proof generation...")
            await self.print_proof_tree()
            
            # First, try to solve the problem with the formal LLM
            with StrategyTracker(self.stats, StrategyType.FORMAL_LLM, 0, extract_theorem_name(problem) or "main") if self.enable_statistics else DummyContext():
                proof = await self._attempt_proof_with_prover_llm(problem, header)

            if proof is not None:
                await self.update_proof_tree_status(problem, ProofStatus.SOLVED)
                success = True
                logger.info("Formal LLM succeeded!")
                await self.print_proof_tree()
            else:
                # Formal proof did not work, try generating an informal proof and outline
                await self.update_proof_tree_strategy(problem, ProofStrategy.SUBGOAL_DECOMP)
                logger.info("Formal LLM failed, switching to subgoal decomposition...")
                await self.print_proof_tree()
                
                with StrategyTracker(self.stats, StrategyType.SUBGOAL_DECOMP, 0, extract_theorem_name(problem) or "main") if self.enable_statistics else DummyContext():
                    success, proof = await self.subgoal_decomp(problem, header)
                    
            # Update final status
            if success:
                await self.update_proof_tree_status(problem, ProofStatus.SOLVED)
                logger.info("Subgoal decomposition succeeded!")
            else:
                await self.update_proof_tree_status(problem, ProofStatus.FAILED)
                logger.info("Subgoal decomposition failed!")
     
            logger.info("Final proof status:")
            await self.print_proof_tree()
            
            # Save proof tree to file after proof completion
            await self._save_proof_tree_to_file(problem_id)
            
            # Save proof immediately if successful
            if success and proof and problem_id and self.proof_save_dir:
                self._save_proof_to_file(problem_id, header, proof)
                
            return success, proof
            
        finally:
            # Finalize statistics
            self._finalize_statistics(success if 'success' in locals() else False, problem_id)

    async def _single_subgoal_decomp_attempt(self, problem: str, header: str, depth: int, attempt_id: int = None) -> Optional[str]:
        """
        Single attempt at subgoal decomposition.
        
        Args:
            problem: The formal problem statement
            header: The theorem header/context  
            depth: Current recursion depth
            attempt_id: Optional identifier for this attempt
            
        Returns:
            The successful proof or None if this attempt failed
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        # whether the current attempt produced a valid proof sketch
        generated_valid_sketch = False
        # First, get the relevant theorems from the search engine
        useful_theorems = await self.search_and_select(problem)
        
        # get a candidate proof sketch
        generated_proof_sketch = await self.generate_proof_sketch(problem, useful_theorems)
        if not generated_proof_sketch:
            logger.info(100*"$")
            logger.info("ERROR: Response did not have Lean 4 code block (attempt %d)", attempt_id)
            logger.info(100*"$")

            return None, False
            
        # first, verify that the proof sketch makes sense
        status, corrected_proof_sketch, extracted_theorems, proved_theorems = await self._correct_proof_sketch_and_extract_theorems(
            generated_proof_sketch, header, problem, useful_theorems
        )
        
        logger.info(80*"*")
        logger.info("Corrected proof sketch (attempt %d)", attempt_id)
        logger.info(80*"*")
        logger.info(corrected_proof_sketch)

        if status:
            # we have generated a valid proof sketch
            generated_valid_sketch = True
            status, _, updated_proof = await self.solve_subgoals(
                corrected_proof_sketch, extracted_theorems, proved_theorems, header, useful_theorems, depth
            )
            
            if status:
                logger.info("FINAL CORRECTED PROOF (attempt %d)", attempt_id)
                logger.info(80*"*")
                logger.info(updated_proof)
                logger.info(80*"*")
                return updated_proof, generated_valid_sketch
        
        return None, generated_valid_sketch

    async def subgoal_decomp(self, problem: str, header: str, depth: int = 0) -> Tuple[bool, Optional[str]]:
        """
        Decompose the problem into subgoals and solve recursively.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        if depth > self.max_depth:
            # too deep, flag failure
            return False, None
        
        # Prepare arguments for attempts
        args_for_each_job = [(problem, header, depth, i+1) for i in range(2*self.proof_config.subgoal_decomp_attempts)]
        valid_sketch_attempts = 0
        # Choose execution strategy based on configuration
        if self.run_proof_attempts_sequentially:
            logger.info("SEQUENTIAL ATTEMPTS")
            for args_tuple in args_for_each_job:
                logger.info(100*"^")
                logger.info("STARTING Sketch ATTEMPT %d Valid sketch attempts so far: %d", args_tuple[-1], valid_sketch_attempts)
                logger.info(100*"^")
                if valid_sketch_attempts >= self.proof_config.subgoal_decomp_attempts:
                    logger.info(100*"*")
                    logger.info("We have exhausted all possible proof attempts")
                    logger.info(100*"*")
                    break

                problem = args_tuple[0]
                await self.update_proof_tree_status(problem, ProofStatus.SOLVING)
                await self.print_proof_tree()
                result, generated_valid_proof_sketch = await self._single_subgoal_decomp_attempt(*args_tuple)
                if result:  # Return first truthy result
                    return True, result
                elif self.proof_tree:
                    # update the proof tree
                    await self.update_proof_tree_status(problem, ProofStatus.FAILED)
                    node_name = extract_theorem_name(problem)
                    async with self._proof_tree_lock:
                        self.proof_tree.remove_children(node_name)
                    await self.print_proof_tree()
                # result is not valid
                if generated_valid_proof_sketch:
                    valid_sketch_attempts += 1
        else:
            logger.info("PARALLEL ATTEMPTS")
            logger.info("TURNING OFF PROOF TREE SUPPORT")
            self.proof_tree = None 
            result, generated_valid_proof_sketch = await self._run_async_pool_and_get_first_truthy(
                self._single_subgoal_decomp_attempt,
                args_for_each_job
            )
        
        if result is not None:
            return True, result
        else:
            # If all attempts failed
            return False, None

    async def shallow_solve(self, theorem: str, header: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Attempt to solve a theorem with a shallow approach using informal LLM.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")

        with StrategyTracker(self.stats, StrategyType.SHALLOW_SOLVE, 0, extract_theorem_name(theorem) or "subgoal") if self.enable_statistics else DummyContext():
            useful_theorems = await self.search_and_select(theorem, error_message=None)
            useful_theorems_section = self._format_useful_theorems_section(useful_theorems)
            shallow_solve_prompt = SOLVE_SUBGOAL_PROMPT.format(problem=theorem, 
                                                               useful_theorems_section=useful_theorems_section, 
                                                               lean_hints=GENERAL_HINTS, 
                                                               tactic_hints=TACTIC_HINTS)
            response = await self.informal_llm_client.simple_chat(
                shallow_solve_prompt, 
                max_tokens=16384, 
                reasoning={'effort': 'high'},
                prompt_type=PromptType.SHALLOW_SOLVE.value,
                context=f"Theorem: {theorem}"
            )
            proof = extract_lean_block(response)
            if not proof:
                logger.info(100*"$")
                logger.info("ERROR: Response did not have Lean 4 code block. Generating new response")
                logger.info(100*"$")
                logger.info("Full response")
                logger.info(response)
                logger.info(100*"$")
                proof = await self._correct_response_for_missing_tags(
                    shallow_solve_prompt,
                    response,
                    MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                    extract_lean_block
                )
                if not proof:
                    return False, None, None
            # remove imports
            proof = await self._remove_import_statements(proof, shallow_solve_prompt, response)
            if not proof:
                logger.info(100*"$")
                logger.info("ERROR: Response did not have theorem statement so abandoning.")
                logger.info(100*"$")
                return False, None, None
            logger.info(80*"*")
            logger.info("Generated Informal Proof:")
            logger.info(proof)
            logger.info(80*"*")

            result, error_message = await self.lean_verifier.verify_proof(
                header + proof, 
                return_error_message=True, 
                is_sorry_ok=False,
                timeout=self.proof_config.proof_verification_timeout,
                verification_type=VerificationType.SHALLOW_SOLVE_GENERATED_PROOF_VERIFICATION,
                context=f"Shallow solve for theorem: {extract_theorem_name(theorem) or 'unknown'}"
            )

            if result:
                result, error_message = self._check_theorem_signature_match_and_get_error(theorem, proof)
                if result:
                    return True, proof, None
                else:
                    return False, proof, error_message
            else:
                return False, proof, error_message

    async def _augment_useful_theorems(self, error_message: str, problem: str, useful_theorems: str="") -> str:
        # Return unchanged if retrieval is disabled
        if not self.enable_retrieval:
            return useful_theorems
            
        missing_identifiers = extract_missing_identifiers(error_message)
        if missing_identifiers:
            logger.info(80*"*")
            logger.info("Using conversational search for missing identifiers")
            logger.info("Missing identifiers: %s", missing_identifiers)
            logger.info(80*"*")
            
            # Use search_and_select with the error message to get relevant theorems
            additional_theorems = await self.search_and_select(problem=problem, error_message=error_message)
            
            logger.info("Additional theorems found:")
            logger.info(additional_theorems)
            logger.info(80*"*")
            
            if additional_theorems and additional_theorems not in useful_theorems:
                useful_theorems += "\n" + additional_theorems
        return useful_theorems

    async def _single_verify_and_correct_subgoal(self, theorem: str, header: str, useful_theorems: str, depth: int) -> Optional[str]:
        """
        Single attempt to verify and correct a subgoal.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Find the subgoal node and update its strategy
        await self.update_proof_tree_strategy(theorem, ProofStrategy.SHALLOW_SOLVE)
        
        # try with informal LLM
        status, proof, error_message = await self.shallow_solve(theorem, header)
        if status:
            return proof
        if not proof:
            return None
        # Create local conversation for error correction
        messages = self.get_error_system_prompt()
        corrected_proof = None
        
        for j in range(self.proof_config.subgoal_error_corrections):
            logger.info(80*"-")
            logger.info("Correction passes %d/%d", j + 1, self.proof_config.subgoal_error_corrections)
            logger.info(80 * "-")
            logger.info("Error Message")
            logger.info("%s", error_message)
            logger.info(80 * "-")

            # add more useful theorems using conversational search approach
            useful_theorems = await self._augment_useful_theorems(error_message, theorem, useful_theorems)
            
            logger.info(80*"*")
            logger.info("OBTAINED USEFUL THEOREMS")
            logger.info(useful_theorems)
            logger.info(80*"*")

            corrected_proof, messages = await self.correct_error_and_send_proof(error_message=error_message,
                                                        useful_theorems=useful_theorems,
                                                        messages=messages,
                                                        type_of_correction='subgoal')
            if not corrected_proof:
                logger.info(100*"$")
                logger.info("ERROR: Response did not have Lean 4 code block. Generating new response")
                logger.info(100*"$")
                break
            result, error_message = await self.lean_verifier.verify_proof(
                header + corrected_proof, 
                return_error_message=True, 
                is_sorry_ok=False, 
                timeout=self.proof_config.proof_verification_timeout,
                verification_type=VerificationType.SUBGOAL_ERROR_CORRECTION_VERIFICATION,
                context=f"Subgoal error correction pass {j+1}/{self.proof_config.subgoal_error_corrections} for: {extract_theorem_name(theorem) or 'unknown'}"
            )
            logger.info("RESULT: %s", result)
            if result:
                result, error_message = self._check_theorem_signature_match_and_get_error(theorem, corrected_proof)
                if result:
                    return corrected_proof

            # check if length exceeds cutoff
            if depth < self.max_depth:
                proof_length = self._estimate_proof_length(corrected_proof)
                if proof_length >= self.complexity_proof_length_cutoff:
                    logger.info(100*"*")
                    logger.info("Ending proof attempt early because of complexity")
                    logger.info(100*"*")
                    return None
        return None

    async def _verify_and_correct_subgoal(self, theorem: str, header: str, useful_theorems: str, depth: int) -> Optional[str]:
        """
        Verify and correct a subgoal using parallel attempts, with recursive fallback.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Use AsyncJobPool for parallel attempts
        args_for_each_job = [(theorem, header, useful_theorems, depth) for _ in range(self.proof_config.parallel_subgoal_proof_attempts)]
        result = await self._run_async_pool_and_get_first_truthy(
            self._single_verify_and_correct_subgoal,
            args_for_each_job
        )
        
        if result:
            # update status
            await self.update_proof_tree_status(theorem, ProofStatus.SOLVED)
        await self.print_proof_tree()
        # If all parallel attempts failed, try recursive decomposition
        if result is None:
            logger.info(80 * "*")
            logger.info("PARALLEL ATTEMPTS FAILED, TRYING RECURSIVE DECOMPOSITION")
            logger.info("RECURSIVELY SOLVING THEOREM: %s", theorem)
            logger.info(80 * "*")

            if depth >= self.max_depth:
                logger.info("MAX DEPTH REACHED, CANNOT RECURSE FURTHER")
                return None
            
            # Find the current subgoal node in the tree and set up for recursion
            await self.update_proof_tree_strategy(theorem, ProofStrategy.RECURSIVE)
            await self.print_proof_tree()

            # Recursively solve the theorem
            status, proof = await self.subgoal_decomp(theorem, header, depth + 1)

            if status:
                await self.update_proof_tree_status(theorem, ProofStatus.SOLVED)
                logger.info("Recursive proof succeeded")
                await self.print_proof_tree()
                return proof
            else:
                await self.update_proof_tree_status(theorem, ProofStatus.FAILED)
                logger.info("RECURSIVE PROOF FAILED")
                await self.print_proof_tree()
                return None
        
        return result

    async def solve_subgoals(self, correct_proof_sketch: str, extracted_theorems: List[str],
                           proved_theorems: Dict, header: str,
                           useful_theorems: str, depth: int) -> Tuple[bool, Optional[List[str]], Optional[str]]:
        """
        Solve the extracted subgoals from the proof sketch.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")

        # Add subgoals to proof tree
        async with self._proof_tree_lock:
            if self.proof_tree:
                parent_name = extract_theorem_name(correct_proof_sketch)
                self.proof_tree.add_subgoals(parent_name, extracted_theorems)
                logger.info("Added %d subgoals to proof tree", len(extracted_theorems))
        await self.print_proof_tree()

        # check each theorem statement and run the correction algorithm on it
        correct_proofs = [None] * len(extracted_theorems)  # Pre-allocate with correct size
        correct_proofs_by_job_name = {}
        logger.info("ALL EXTRACTED THEOREMS")
        logger.info(extracted_theorems)

        # Create AsyncJobPool for parallel subgoal verification
        pool = AsyncJobPool()
        theorem_jobs = []  # Track which jobs correspond to which theorems
        
        # Submit jobs for theorems that need verification
        for idx, theorem in enumerate(extracted_theorems):
            logger.info(80*"*")
            logger.info("Verifying Theorem")
            logger.info(theorem)
            
            if theorem in proved_theorems:
                logger.info(80*"*")
                logger.info("Theorem already proved")
                logger.info(80*"*")
                correct_proofs[idx] = proved_theorems[theorem]
                correct_proofs_by_job_name[extract_theorem_name(theorem)] = correct_proofs[idx]
                # Update proof tree if available
                await self.update_proof_tree_status(theorem, ProofStatus.SOLVED)
                continue

            # Update proof tree - set subgoal as solving
            await self.update_proof_tree_status(theorem, ProofStatus.SOLVING)
            
            # Submit theorem verification job
            theorem_name = extract_theorem_name(theorem)
            pool.submit(
                self._verify_and_correct_subgoal, 
                theorem, header, useful_theorems, depth,
                name=theorem_name
            )
            theorem_jobs.append(idx)  # Track which index this job corresponds to
        
        logger.info("Subgoals being solved in parallel:")
        await self.print_proof_tree()
        
        # Wait for all verification jobs to complete
        if theorem_jobs:  # Only if we have jobs to wait for
            all_results = await pool.wait_until_first_failure_or_all_success()
            
            # Process results and populate correct_proofs
            for job_idx, (job_name, result) in enumerate(all_results):
                theorem_idx = theorem_jobs[job_idx]  # Get the original theorem index
                
                if isinstance(result, Exception):
                    logger.info("Job %s failed with exception: %s", job_name, result)
                    correct_proofs[theorem_idx] = None
                    await self.update_proof_tree_status(job_name, ProofStatus.FAILED)
                else:
                    correct_proofs[theorem_idx] = result
                    # Update proof tree
                    await self.update_proof_tree_status(job_name, ProofStatus.SOLVED)
                    
        logger.info("Subgoal verification completed:")
        await self.print_proof_tree()
        
        # Check if any proofs are still None (failed)
        for idx, proof in enumerate(correct_proofs):
            if proof is None:
                theorem = extracted_theorems[idx]
                logger.info(80*"*")
                logger.info("GIVING UP ON SOLVING THE SUBGOAL - ALL ATTEMPTS INCLUDING RECURSION FAILED")
                logger.info(correct_proof_sketch)
                logger.info(80*"*")
                return False, None, None
        
        logger.info("ALL PROOFS CORRECTED!")
        for proof in correct_proofs:
            logger.info(proof)
            
        # combine the proofs
        updated_proof = ""
        for proof in correct_proofs:
            updated_proof += proof + "\n\n"
        updated_proof += correct_proof_sketch
        
        # remove any stray import-related lines
        updated_proof = remove_import_lines(updated_proof)
        return True, correct_proofs, updated_proof

    def _deduplicate_theorems(self, theorems: List[str]) -> List[str]:
        """
        Deduplicate theorems based on their theorem names using extract_theorem_name function.
        
        Args:
            theorems: List of theorem strings to deduplicate
            
        Returns:
            List of unique theorems (first occurrence is kept for duplicates)
        """
        if not theorems:
            return []
        
        seen_names = set()
        unique_theorems = []
        
        for theorem in theorems:
            theorem_name = extract_theorem_name(theorem)
            
            if theorem_name is None:
                # If we can't extract a name, include the theorem as is
                unique_theorems.append(theorem)
                continue
            
            if theorem_name not in seen_names:
                seen_names.add(theorem_name)
                unique_theorems.append(theorem)
        
        return unique_theorems

    def extract_theorems_from_string(self, extracted_theorems: list):
        # extract all theorems from the extracted strings (in case the LLM returned multiple theorems in a single string)
        new_theorems = []
        for theorem in extracted_theorems:
            new_theorems.extend(_extract_all_theorems_from_string(theorem))
        return new_theorems

    async def extract_subgoals_from_sketch(self, proof_sketch: str, header: str) -> List[str]:
        """
        Extract have statements as standalone theorem statements from a proof sketch.
        
        Args:
            proof_sketch: The proof sketch containing have statements
            
        Returns:
            List of theorem statements extracted from have statements
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Prompt the LLM to extract have statements as independent theorems
        extract_prompt = EXTRACT_SUBGOALS_FROM_SKETCH_PROMPT.format(proof_sketch=proof_sketch, lean_hints=GENERAL_HINTS)
        
        initial_response = await self.informal_llm_client.simple_chat(
            extract_prompt, 
            max_tokens=16384, 
            reasoning={'effort': 'high'},
            prompt_type=PromptType.SUBGOAL_EXTRACTION.value,
            context="Extracting subgoals from proof sketch"
        )
        logger.info("INITIAL RESPONSE")
        logger.info(initial_response)
        logger.info("\n")
        # Extract theorem statements from the response
        extracted_theorems = extract_all_lean_blocks(initial_response)
        if extracted_theorems is None:
            extracted_theorems = await self._correct_response_for_missing_tags(extract_prompt,
                                                    initial_response,
                                                    MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                                                    extract_all_lean_blocks)
            if extracted_theorems is None:
                return []
        
        extracted_theorems = self.extract_theorems_from_string(extracted_theorems)
        
        logger.info("ALL EXTRACTED THEOREMS")
        for idx, theorem in enumerate(extracted_theorems):
            logger.info(idx)
            logger.info(theorem)
            logger.info(100*"-")
        # Check if all have statements were extracted by comparing with original sketch
        status, missing_theorems = await self._check_and_extract_missing_subgoals(proof_sketch, extracted_theorems)
        if not status:
            logger.info(100*"*")
            logger.info("NOT ALL MISSING THEOREMS COULD BE EXTRACTED, ABANDONING CURRENT ATTEMPT")
            logger.info(100*"*")
            return []
        
        # Combine initial and missing theorems
        all_theorems = extracted_theorems + missing_theorems
        
        # De-duplicate the theorems
        all_theorems = self._deduplicate_theorems(all_theorems)

        # Correct any syntax errors
        corrected_theorems = await self._verify_and_correct_syntax_errors_in_theorems(all_theorems, header)
        final_theorems = []
        if corrected_theorems:
            # check for theorem keyword in all of them
            for theorem in corrected_theorems:
                if bool(re.search(r'\btheorem\b', theorem)):
                    final_theorems.append(theorem)

        return final_theorems
    
    async def _check_and_extract_missing_subgoals(self, 
                                                  original_sketch: str, 
                                                  extracted_theorems: List[str]) -> List[str]:
        """
        Check if all have statements were extracted and extract missing ones using a conversational loop.
        
        Args:
            original_sketch: Original proof sketch
            extracted_theorems: Already extracted theorems
            
        Returns:
            List of missing theorem statements
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Count have statements in original sketch
        all_have_names = extract_all_have_names(original_sketch)
        
        # Count sorry occurrences in original sketch
        sorry_count = original_sketch.lower().count('sorry')
        logger.info("SORRY COUNT: %d", sorry_count)
        
        if not all_have_names:
            return True, []  # No have statements to extract
        logger.info("ALL HAVE NAMES: %s", all_have_names)

        # Extract theorem names from extracted theorems
        extracted_names = []
        for theorem in extracted_theorems:
            theorem_name = extract_theorem_name(theorem)
            if theorem_name:
                extracted_names.append(theorem_name)
        logger.info("EXTRACTED NAMES: %s", extracted_names)
        # Find missing have statement names
        missing_names = [name for name in all_have_names if name not in extracted_names]
        logger.info("MISSING NAMES: %s", missing_names)

        # Check if there are fewer subgoals than sorries
        # if len(extracted_names) < sorry_count:
        #     logger.info(f"WARNING: There are {sorry_count} sorry statements but only {len(extracted_names)} subgoals. This may indicate missing subgoals.")
        
        if not missing_names and len(extracted_names) >= sorry_count:
            return True, []  # All have statements were extracted
        
        # Start a conversation to extract missing subgoals
        messages = []
        all_missing_theorems = []
        
        for _ in range(self.proof_config.missing_subgoal_extraction_attempts):
            missing_names_message = ""

            if len(missing_names) > 0:
                missing_names_message += "Missing have statement names: " + ", ".join(missing_names) + "\n"
            # if len(extracted_names) < sorry_count:
            #     missing_names_message += f"There are {sorry_count} sorry statements but only {len(extracted_names)} subgoals. This indicates there are missing subgoals.\n"
            
            already_extracted_theorems_str = "\n".join(extracted_theorems)
            # Prompt LLM to extract the missing subgoals
            missing_prompt = EXTRACT_MISSING_SUBGOALS_PROMPT.format(
                already_extracted_theorems=already_extracted_theorems_str,
                missing_names=missing_names_message,
                original_sketch=original_sketch
            )
            messages.append({'role': 'user', 'content': missing_prompt})
            missing_response = await self.informal_llm_client.chat_completion(
                messages, 
                max_tokens=16384, 
                reasoning={'effort': 'high'},
                prompt_type=PromptType.MISSING_SUBGOAL_EXTRACTION.value,
                context="Extracting missing subgoals"
            )
            messages.append({'role': 'assistant', 'content': missing_response})
            # Parse the missing theorems
            missing_theorems = extract_all_lean_blocks(missing_response)
            
            if missing_theorems is None:
                continue  # Try again in next attempt
            logger.info("Obtained missing theorems:")
            logger.info(missing_theorems)
            # get some theorems which may have appeared in a single string
            missing_theorems = self.extract_theorems_from_string(missing_theorems)
            
            # Add newly extracted theorems
            all_missing_theorems.extend(missing_theorems)
            
            # Update the list of extracted names
            for theorem in missing_theorems:
                theorem_name = extract_theorem_name(theorem)
                if theorem_name:
                    extracted_names.append(theorem_name)
            
            # Update missing names list
            missing_names = [name for name in all_have_names if name not in extracted_names]
            
            if not missing_names:
                break  # All subgoals have been extracted
        if not missing_names:
            # All subgoals have been extracted
            return True, all_missing_theorems
        else:
            # Not all subgoals could be extracted
            return False, None

    async def _refine_proof_sketch(self, proof_sketch: str, justification: str) -> Optional[str]:
        """
        Refine a proof sketch based on incorrect subgoal feedback.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        correct_sketch_prompt = CORRECT_SKETCH_BASED_ON_INCORRECT_SUBGOAL.format(proof_sketch=proof_sketch,
                                                         issues=justification,
                                                         lean_hints=GENERAL_HINTS)
        proof_sketch_response_str = await self.informal_llm_client.simple_chat(
            correct_sketch_prompt, 
            max_tokens=16384, 
            reasoning={'effort': 'high'},
            prompt_type=PromptType.PROOF_SKETCH_CORRECTION.value,
            context=f"Issues: {justification}"
        )
        proof_sketch_response = extract_lean_block(proof_sketch_response_str)
        if not proof_sketch_response:
            proof_sketch_response = await self._correct_response_for_missing_tags(
                    correct_sketch_prompt,
                    proof_sketch_response_str,
                    MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                    extract_lean_block
                )
            
        if proof_sketch_response:
            proof_sketch_response = await self._remove_import_statements(proof_sketch_response, correct_sketch_prompt, proof_sketch_response_str)
        return proof_sketch_response

    def _check_theorem_signature_match_and_get_error(self, problem: str, candidate_proof: str):
        if not check_theorem_signature_match(candidate_proof, problem):
            # Theorem signatures did not match
            logger.info(100*"()")
            logger.info("Theorem signatures did not match")
            sig1 = extract_theorem_signature(problem)
            sig2 = extract_theorem_signature(candidate_proof)
            logger.info("sig1: %s", sig1)
            logger.info("sig2: %s", sig2)
            error_message = THEOREM_SIGNATURE_MISMATCH_PROMPT.format(proof=candidate_proof, 
                                                        theorem=problem, 
                                                        proof_signature=sig2, 
                                                        theorem_signature=sig1) 
            logger.info(100*"()")
            return False, error_message
        logger.info("THEOREM SIGNATURE MATCHES")
        return True, None
    
    async def _compile_and_correct_proof_sketch(self, proof_sketch: str, header: str, useful_theorems: str, problem: str) -> Tuple[bool, Optional[str]]:
        """
        Compile and correct a proof sketch by replacing have statements with sorry.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # replace the have statements with sorry
        corrected_proof_sketch = replace_have_proofs_with_sorry(proof_sketch)
        
        logger.info(80*"*")
        logger.info("REPLACING HAVE STATEMENTS WITH SORRY")
        logger.info(80*"*")
        logger.info(corrected_proof_sketch)
        logger.info(80*"*")

        # verify the proof sketch
        result, error_message = await self.lean_verifier.verify_proof(
            header + corrected_proof_sketch, 
            return_error_message=True, 
            is_sorry_ok=True,
            verification_type=VerificationType.PROOF_SKETCH_WITH_SORRY_VERIFICATION,
            context=f"Initial proof sketch compilation for: {extract_theorem_name(problem) or 'unknown'}"
        )
        if result:
            # check if signatures match
            result, error_message = self._check_theorem_signature_match_and_get_error(problem, corrected_proof_sketch)

        logger.info("RESULT: %s", result)
        logger.info("ERROR MESSAGE: %s", error_message)

        if not result:
            # Create local conversation for error correction
            messages = self.get_error_system_prompt()
            for j in range(self.proof_config.main_theorem_error_corrections):
                logger.info(80*"-")
                logger.info("Correction passes %d/%d", j+1, self.proof_config.main_theorem_error_corrections)
                logger.info(80*"-")
                logger.info("Error Message: %s", error_message)
                logger.info(80*"-")
                # augment useful theorems
                useful_theorems = await self._augment_useful_theorems(error_message, problem, useful_theorems)
                # correct syntax/tactic errors
                logger.info("NEW USEFUL THEOREMS FOR CORRECTION")
                logger.info(useful_theorems)
                logger.info(80*"-")
                generated_proof, messages = await self.correct_error_and_send_proof(error_message=error_message,
                                                                    useful_theorems=useful_theorems,
                                                                    messages=messages,
                                                                    type_of_correction='sketch_outline')
                if not generated_proof:
                    logger.info("ERROR! The sketch did not have a Lean 4 block")
                    return False, None
                # extract sketch
                corrected_proof_sketch = replace_have_proofs_with_sorry(generated_proof)
                result, error_message = await self.lean_verifier.verify_proof(
                    header + corrected_proof_sketch, 
                    return_error_message=True, 
                    is_sorry_ok=True,
                    verification_type=VerificationType.PROOF_SKETCH_ERROR_CORRECTION_VERIFICATION,
                    context=f"Proof sketch error correction pass {j+1}/{self.proof_config.main_theorem_error_corrections} for: {extract_theorem_name(problem) or 'unknown'}"
                )

                logger.info("RESULT: %s", result)
                logger.info("ERROR MESSAGE: %s", error_message)
                if result:
                    result, error_message = self._check_theorem_signature_match_and_get_error(problem, corrected_proof_sketch)
                    if result:
                        break
        
        if result:
            return True, corrected_proof_sketch
        else:
            return False, None

    async def verify_and_correct_proof_sketch_with_theorems(self, 
                                                            proof_sketch: str, 
                                                            all_theorems: str, 
                                                            header: str,
                                                            problem: str):
        """
        Verify the proof sketch and correct any errors using theorems.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        logger.info("VERIFYING PROOF")
        logger.info("Generated proof")
        logger.info(header + all_theorems + proof_sketch)
        status, error_message = await self.lean_verifier.verify_proof(
            header + all_theorems + proof_sketch, 
            return_error_message=True, 
            is_sorry_ok=True,
            verification_type=VerificationType.SKETCH_ASSEMBLY_WITH_THEOREMS_VERIFICATION,
            context=f"Initial sketch assembly with theorems for: {extract_theorem_name(problem) or 'unknown'}"
        )
        logger.info("Status: %s", status)
        logger.info("ERROR MESSAGE: %s", error_message)
        if status:
            status, error_message = self._check_theorem_signature_match_and_get_error(problem, proof_sketch)
            if status:
                return proof_sketch
                  
        # Create local conversation for error correction
        messages = self.get_error_system_prompt()
        
        # Try to correct the proof sketch using error correction loop
        for j in range(self.proof_config.main_theorem_error_corrections):
            logger.info(80*"-")
            logger.info("Proof sketch correction passes %d/%d", j + 1, self.proof_config.main_theorem_error_corrections)
            logger.info(80 * "-")
            logger.info("Error Message")
            logger.info("%s", error_message)
            logger.info(80 * "-")

            # Correct the proof sketch using error message and theorems
            corrected_proof, messages = await self.correct_error_and_send_proof(
                error_message=error_message,
                useful_theorems=None,
                messages=messages,
                type_of_correction='sketch_completion'
            )
            if not corrected_proof:
                logger.info("ERROR: Response did not have Lean 4 code block")
                continue
                
            # Verify the corrected proof sketch
            result, error_message = await self.lean_verifier.verify_proof(
                header + all_theorems + corrected_proof, 
                return_error_message=True, 
                is_sorry_ok=True,
                verification_type=VerificationType.SKETCH_ASSEMBLY_ERROR_CORRECTION_VERIFICATION,
                context=f"Sketch assembly error correction pass {j+1}/{self.proof_config.main_theorem_error_corrections} for: {extract_theorem_name(problem) or 'unknown'}"
            )

            logger.info("RESULT: %s", result)
            logger.info("ERROR MESSAGE: %s", error_message)
            if result:
                result, error_message = self._check_theorem_signature_match_and_get_error(problem, corrected_proof)
                if result:
                    return corrected_proof
                    
        # If all correction attempts failed, return None
        logger.info("Failed to correct proof sketch after all attempts")
        return None
    
    async def _use_sketch_and_theorems_to_generate_proof(self, 
                                                         proof_sketch: str, 
                                                         theorems: List[str], 
                                                         header: str,
                                                         problem: str):
        """
        Use the given proof sketch and theorems to generate a proof.
        """

        with StrategyTracker(self.stats, StrategyType.SKETCH_ASSEMBLY, 0, extract_theorem_name(problem) or "main") if self.enable_statistics else DummyContext():
            all_theorems = ""
            for theorem in theorems:
                all_theorems += theorem + "\n\n"
                    
            get_sketch_prompt = USE_SKETCH_AND_THEOREMS_TO_PROVE.format(theorems_string=all_theorems, 
                                                                        proof_sketch=proof_sketch)
            response = await self.informal_llm_client.simple_chat(
                get_sketch_prompt, 
                max_tokens=32768, 
                reasoning={'effort': 'high'},
                prompt_type=PromptType.SKETCH_ASSEMBLY.value,
                context="Assembling proof from sketch and theorems"
            )
            logger.info(100*"*")
            logger.info("Response from model for sketch use")
            logger.info(100*"*")
            logger.info(response)
            logger.info(100*"*")
            completed_proof_sketch = extract_lean_block(response)
            if not completed_proof_sketch:
                completed_proof_sketch = await self._correct_response_for_missing_tags(
                    get_sketch_prompt,
                    response,
                    MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT,
                    extract_lean_block
                )
                if not completed_proof_sketch:
                    logger.info("ERROR: Could not extract proof sketch from response")
                    return None
            # remove import statements
            completed_proof_sketch = await self._remove_import_statements(completed_proof_sketch, get_sketch_prompt, response)
            # verify and correct errors in sketch
            corrected_proof_sketch = await self.verify_and_correct_proof_sketch_with_theorems(completed_proof_sketch, all_theorems, header, problem)
            
            return corrected_proof_sketch
    
    async def _syntax_check_and_correct_single_theorem(self, theorem: str, header: str):
        # for each theorem, first check for syntax errors
        logger.info(80*"*")
        logger.info("CHECKING SYNTAX ERRORS IN: %s", theorem)
        logger.info(80*"*")
        
        status, error_message = await self.lean_verifier.verify_proof(
            header + theorem, 
            return_error_message=True, 
            is_sorry_ok=True,
            verification_type=VerificationType.EXTRACTED_THEOREM_SYNTAX_VERIFICATION,
            context=f"Syntax check for extracted theorem: {extract_theorem_name(theorem) or 'unknown'}"
        )

        if status:
            # theorem is syntatically correct
            logger.info(80*"*")
            logger.info("THEOREM PASSES SYNTAX CHECK")
            logger.info(80*"*")
        else:
            logger.info(80*"*")
            logger.info("THEOREM IS INVALID, TRYING TO CORRECT IT")
            logger.info("Error Message:")
            logger.info(error_message)
            logger.info(80*"*")
            theorem = await self.correct_error_in_theorem(error_message, header, theorem)
            

        if not theorem: # could not correct this theorem
            logger.info(80*"*")
            logger.info("Could NOT correct theorem so giving up")
            logger.info(theorem)
            logger.info(80*"*")
            return None
        return theorem
            
    async def _verify_and_correct_syntax_errors_in_theorems(self, theorems, header: str):

        corrected_theorems = []
        pool = AsyncJobPool()
        for theorem in theorems:
            pool.submit(self._syntax_check_and_correct_single_theorem, theorem, header)

        results = await pool.wait_for_all()
        corrected_theorems = []
        for job_name, theorem in results:
            if not theorem or isinstance(theorem, Exception):
                logger.info("RETURNING NONE BECAUSE OF JOB_NAME: %s", job_name)
                return None
            corrected_theorems.append(theorem)
        
        return corrected_theorems
    
    async def _process_single_theorem(self, theorem: str, header: str, corrected_theorems: List[str], proved_theorems: Dict) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Process a single theorem from the proof sketch.
        
        Args:
            theorem: The theorem to process
            header: The theorem header/context
            corrected_theorems: List of already corrected theorems
            proved_theorems: Dict of already proved theorems
            
        Returns:
            Tuple of (success, theorem, proof_or_none, justification_or_none)
            - success: True if theorem is valid, False if invalid
            - theorem: The (possibly corrected) theorem statement
            - proof_or_none: The proof if successfully proved with prover LLM, None otherwise
            - justification_or_none: Justification if theorem is invalid, None otherwise
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        

        if theorem in corrected_theorems:
            logger.info("Skipping theorem because it was already checked: %s", theorem)
            return True, theorem, proved_theorems.get(theorem), None

        
        # at this point, attempt a proof with prover LLM.
        # If the proof passes, it is definitely valid!
        proof = await self._attempt_proof_with_prover_llm(theorem, header)
        if proof is not None:
            return True, theorem, proof, None

        # if the proof with prover LLM failed, we don't know if it is valid or not.
        logger.info(80*"*")
        logger.info("Checking for mathematical correctness...")
        logger.info(80*"*")
        subgoal_status, justification = await self._check_subgoal_is_correct(theorem)
        if subgoal_status:
            logger.info("The subgoal is correct!")
            return True, theorem, None, None
        else:
            logger.info(80*"*")
            logger.info("THEOREM IS INVALID")
            logger.info(80*"*")
            logger.info("Justification:")
            logger.info(justification)
            theorem_name = extract_theorem_name(theorem)
            full_justification = f"Sub goal {theorem_name} has an issue:\n{justification}"
            return False, theorem, None, full_justification

    async def _correct_theorems_from_sketch(self, extracted_theorems: List[str], header: str, corrected_theorems: List[str] = [], proved_theorems: Dict = {}) -> Tuple[bool, Optional[List[str]], Optional[str], Optional[Dict]]:
        """
        Correct theorems extracted from proof sketch.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        # Use AsyncJobPool to parallelize theorem processing
        pool = AsyncJobPool()
        
        # Submit all theorems for parallel processing
        for theorem in extracted_theorems:
            theorem_name = extract_theorem_name(theorem)
            pool.submit(
                self._process_single_theorem,
                theorem, header, corrected_theorems, proved_theorems,
                name=f"theorem_{theorem_name}"
            )
        
        # Wait for first failure
        all_results = await pool.wait_until_first_failure_or_all_success()
        
        # Process all completed results
        extracted_theorems = []

        for _, (success, theorem, proof, justification) in all_results:
            if success:
                # Successful theorem
                if proof is not None:
                    proved_theorems[theorem] = proof
                if theorem not in corrected_theorems:
                    corrected_theorems.append(theorem)
                extracted_theorems.append(theorem)
            else:
                # This is the failed theorem
                return False, corrected_theorems, justification, proved_theorems
        
        # If we get here, all theorems succeeded
        return True, extracted_theorems, None, proved_theorems

    def rename_theorems(self, theorems: List[str], parent_theorem_name: str):
        """
        Rename the list of theorems by appending the parent theorem name to them.
        For example, if the parent theorem is 'theorem main_theorem: ...', and the theorems are named ['theorem h0: ..', 'theorem h1: ...'],
        the renamed theorems will be ['theorem h0_main_theorem: ...', 'theorem h1_main_theorem: ...'].
        """
        renamed_theorems = []
        
        for theorem in theorems:
            try:
                # Extract the current theorem name
                current_name = extract_theorem_name(theorem)
                
                # Create the new name by appending the parent theorem name
                new_name = f"{current_name}_{parent_theorem_name}"
                
                # Replace the old name with the new name in the theorem text
                # Find the pattern "theorem {current_name}" and replace with "theorem {new_name}"
                pattern = rf"theorem(\s+){re.escape(current_name)}"
                replacement = rf"theorem {new_name}"
                renamed_theorem = re.sub(pattern, replacement, theorem, count=1)
                
                renamed_theorems.append(renamed_theorem)
                
            except (ValueError, IndexError) as e:
                # If we can't extract the theorem name, keep the original theorem
                logger.info("Warning: Could not rename theorem due to error: %s", e)
                logger.info("Original theorem: %s", theorem)
                renamed_theorems.append(theorem)
        
        return renamed_theorems    

    async def _correct_proof_sketch_and_extract_theorems(self, proof_sketch: str, header: str, problem: str, useful_theorems: str) -> Tuple[bool, Optional[str], Optional[List[str]], Optional[Dict]]:
        """
        Correct a proof sketch and extract the constituent theorems.
        """
        if self._closed:
            raise RuntimeError("HILBERTWorker is closed")
        
        correction_budget = self.proof_config.proof_sketch_corrections
        corrected_theorems = []
        proved_theorems = {}
        for k in range(correction_budget+1):
            logger.info(80 * "*")
            logger.info("Correction budget pass: %d", k + 1)
            logger.info(80 * "*")

            if proof_sketch is None:
                return False, None, None, None
            
            status, corrected_proof_sketch = await self._compile_and_correct_proof_sketch(proof_sketch, header, useful_theorems, problem)
            
            if not status:
                # the theorem has not been corrected, give up
                return False, None, None, None
            
            # then, check that each subgoal makes sense
            extracted_theorems = await self.extract_subgoals_from_sketch(proof_sketch=corrected_proof_sketch, header=header)
            
            if not extracted_theorems: # the theorems could not be extracted
                return False, None, None, None
            
            # rename the theorems
            parent_theorem_name = extract_theorem_name(problem)
            extracted_theorems = self.rename_theorems(extracted_theorems, parent_theorem_name)

            # check if you can make a full proof with them
            completed_proof_sketch = await self._use_sketch_and_theorems_to_generate_proof(corrected_proof_sketch, extracted_theorems, header, problem)
            
            if not completed_proof_sketch:
                return False, None, None, None
            
            # check that it does not have sorry
            if _check_for_sorries(completed_proof_sketch):
                logger.info(100*"*")
                logger.info("WARNING! The proof sketch is not complete. It has sorry")
                logger.info(100*"*")
                logger.info(completed_proof_sketch)
                logger.info(100*"*")
                
                return False, None, None, None
            else:
                logger.info("No sorries found in proof sketch")
                
            # correct any mistakes in the theorems, prove some of them 
            status, corrected_theorems, justification, proved_theorems = await self._correct_theorems_from_sketch(extracted_theorems, header, corrected_theorems, proved_theorems)

            if status:
                return True, completed_proof_sketch, corrected_theorems, proved_theorems
            else:
                if not justification:
                    # something has gone wrong, a subgoal was incorrect but no justification was given. so start over
                    return False, None, None, None
                
                # refine the proof sketch unless we've run out of correction budget
                if k != correction_budget:
                    proof_sketch = await self._refine_proof_sketch(corrected_proof_sketch, justification)

        # if still not valid, give up
        return False, None, None, None
