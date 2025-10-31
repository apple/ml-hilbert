#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from src.tools.LLMClient import LLMClient
from src.inference.LeanVerifier import LeanVerifier
from src.models.FormalToFormal import FormalToFormal
from src.inference.ProverLLM import ProverLLM
from logging import getLogger

logger = getLogger(__name__)

def run_formal_to_formal(cfg):
    # connect the LLM client
    prover_llm_cfg = cfg.experiment.prover_llm
    prover_llm_name = prover_llm_cfg.llm_name
    prover_llm_prompting_strategy = prover_llm_cfg.prompt_strategy
    prover_llm_client = LLMClient(base_url=prover_llm_cfg.base_url, 
                                  model_name=prover_llm_name)
    
    # create ProverLLM instance
    prover_llm = ProverLLM(
        llm_client=prover_llm_client,
        model_name=prover_llm_name,
        prompt_strategy=prover_llm_prompting_strategy,
        max_tokens=getattr(prover_llm_cfg, 'max_tokens', 16384)
    )
    
    lean_verifier = LeanVerifier(base_url=cfg.experiment.verifier_base_url)
    # run the formal 2 formal experiment
    experiment = FormalToFormal(
        prover_llm=prover_llm,
        lean_verifier=lean_verifier,
        num_passes=cfg.experiment.num_passes,
        return_proofs=cfg.experiment.save_proofs_to_disk,
        num_workers=getattr(cfg.experiment, 'num_workers', 32)
    )
    
    return experiment.run_from_file(cfg.data.file_path)
