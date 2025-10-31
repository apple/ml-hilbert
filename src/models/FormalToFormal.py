#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from src.inference.ProverLLM import ProverLLM
from src.inference.LeanVerifier import LeanVerifier
from src.tools.string import extract_jsonl_contents
from src.tools.lean_utils import check_theorem_signature_match, extract_theorem_signature
import concurrent.futures
from tqdm import tqdm
import traceback
from logging import getLogger

logger = getLogger(__name__)

class FormalToFormal:

    def __init__(self,
             prover_llm: ProverLLM,
             lean_verifier: LeanVerifier,
             num_passes: int,
             return_proofs: bool,
             num_workers: int = 32):
        """
        Class designed to replicate results from the prover papers (formal theorem to formal proof)
        
        Args:
            prover_llm: ProverLLM instance for generating proofs
            lean_verifier: LeanVerifier instance for proof verification
            num_passes: Number of passes for iterative refinement
            return_proofs: Whether to return generated proofs
            num_workers: Number of worker threads for parallel processing
        """
        self.prover_llm = prover_llm
        self.num_passes = num_passes
        self.lean_verifier = lean_verifier
        self.num_workers = num_workers
        self.return_proofs = return_proofs
        
        # Get model info from ProverLLM
        model_info = self.prover_llm.get_model_info()
        self.prover_llm_name = model_info["model_name"]
        self.prompt_strategy = model_info["prompt_strategy"]

    def generate_single_proof(self, formal_statement: str):
        """Generate a single proof attempt"""
        proof = self.prover_llm.generate_proof(formal_statement)
        logger.info("FORMAL STATEMENT", formal_statement)
        logger.info("PROOF", proof)
        return proof

    def generate_proofs_for_samples(self, sample_indices, formal_statements, headers):
        """Generate proofs for multiple samples in parallel"""
        all_proofs = []
        sample_proof_mapping = []  # Track which sample each proof belongs to
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit proof generation tasks for all samples
            future_to_sample = {}
            for sample_idx in sample_indices:
                future = executor.submit(self.generate_single_proof, formal_statements[sample_idx])
                future_to_sample[future] = sample_idx
            
            # Collect results with progress bar - process in completion order for efficiency
            with tqdm(total=len(future_to_sample), desc="LLM Inference", unit="sample") as pbar:
                for future in concurrent.futures.as_completed(future_to_sample):
                    sample_idx = future_to_sample[future]
                    try:
                        proof = future.result()
                        full_proof = headers[sample_idx] + proof
                        all_proofs.append(full_proof)
                        sample_proof_mapping.append(sample_idx)
                    except Exception as e:
                        logger.error(f"Error generating proof for sample {sample_idx}: {e}")
                        # Add empty proof to maintain alignment
                        all_proofs.append("")
                        sample_proof_mapping.append(sample_idx)
                    
                    pbar.update(1)
        
        return all_proofs, sample_proof_mapping

    def run_from_file(self, file_path: str):
        examples = extract_jsonl_contents(file_path)
        formal_statements = [example['header']+example['formal_statement'] for example in examples]
        headers = [example['header'] for example in examples]

        logger.info(f"Processing {len(formal_statements)} formal statements with iterative refinement over {self.num_passes} passes...")
        
        # Handle ID field
        if 'name' in examples[0]:
            id_field = 'name'
        elif 'id' in examples[0]:
            id_field = 'id'
        else:
            for i, example in enumerate(examples):
                example['id'] = i
            id_field = 'id'
        
        # Initialize tracking variables
        num_samples = len(formal_statements)
        pending_samples = list(range(num_samples))  # Samples that still need valid proofs
        successful_samples = set()  # Samples with valid proofs
        sample_proofs = {}  # Store valid proofs for successful samples
        results = [False] * num_samples  # Final results for each sample
        pass_rates_at_pass = [] # Track pass rate across time
        # Iterative refinement over multiple passes
        for pass_num in range(1, self.num_passes + 1):
            if not pending_samples:
                # All samples have been successfully verified
                logger.info(f"\n🎉 All samples successfully verified! Stopping early after {pass_num-1} passes.")
                break
            
            # Display current pass with decorative formatting
            logger.info(f"\n{'*' * 60}")
            logger.info(f"{'*' * 20} PASS {pass_num}/{self.num_passes} {'*' * 20}")
            logger.info(f"{'*' * 60}")
            logger.info(f"Processing {len(pending_samples)} pending samples...")
            
            # Generate proofs for all pending samples with progress bar
            try:
                logger.info(f"\n🔄 Running LLM inference for {len(pending_samples)} samples...")
                all_proofs, sample_proof_mapping = self.generate_proofs_for_samples(
                    pending_samples, formal_statements, headers
                )
                
                # Batch verify all generated proofs
                if all_proofs:
                    logger.info(f"\n{'-' * 50}")
                    logger.info(f"🔍 Lean is batch verifying {len(all_proofs)} proofs...")
                    logger.info(f"{'-' * 50}")
                    
                    verification_results = self.lean_verifier.batch_verify_proofs(all_proofs)
                    
                    # Process verification results
                    newly_successful = []
                    for i, (sample_idx, is_valid) in enumerate(zip(sample_proof_mapping, verification_results)):
                        if is_valid:
                            # check if theorem signature matches
                            if not check_theorem_signature_match(all_proofs[sample_idx], formal_statements[sample_idx]):
                                logger.info(f"🚨 Warning: Proof {i} does not match theorem signature for sample {sample_idx}")
                                logger.info("Proof signature:", extract_theorem_signature(all_proofs[sample_idx]))
                                logger.info("Formal Statement signature:", extract_theorem_signature(formal_statements[sample_idx]))
                                is_valid = False
                                continue
                            # Mark sample as successful
                            successful_samples.add(sample_idx)
                            results[sample_idx] = True
                            if self.return_proofs:
                                sample_proofs[sample_idx] = all_proofs[i]
                            newly_successful.append(sample_idx)
                    
                    # Update pending samples (remove successful ones)
                    pending_samples = [idx for idx in pending_samples if idx not in newly_successful]
                    
                    # Update progress
                    current_success_rate = len(successful_samples) / num_samples
                    pass_rates_at_pass.append(current_success_rate)
                    logger.info(f"\n✅ Pass {pass_num} Results:")
                    logger.info(f"   • New successes: {len(newly_successful)}")
                    logger.info(f"   • Total successful: {len(successful_samples)}/{num_samples} ({current_success_rate:.2%})")
                    logger.info(f"   • Remaining to retry: {len(pending_samples)}")
                
            except Exception as e:
                logger.error(f"Error in pass {pass_num}: {e}")
                traceback.print_exc()
        
        # Collect failure cases
        failure_cases = [examples[i][id_field] for i in range(num_samples) if not results[i]]
        
        return_dict = {
            "prover_llm_name": self.prover_llm_name,
            "prompt_strategy": self.prompt_strategy,
            "num_passes": self.num_passes,
            "results": results,
            "failure_cases": failure_cases,
            "pass_rate": sum(results) / len(results) if results else 0.0,
            "pass_rate_history": pass_rates_at_pass
        }

        if self.return_proofs:
            proofs = [sample_proofs.get(i) for i in range(num_samples)]
            return_dict["proofs"] = proofs
            return_dict["formal_statements"] = formal_statements
        
        return return_dict
