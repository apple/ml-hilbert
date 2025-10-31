#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import traceback
from src.tools.proof_utils import read_client_response
from src.tools.lean_utils import extract_all_error_messages
from kimina_client.sync_client import KiminaClient
import logging
logger = logging.getLogger(__name__)

class LeanVerifier:

    def __init__(self, 
                 base_url: str):

        self.client = KiminaClient(api_url=base_url)


    def verify_proof(self, proof, timeout=30, return_error_message=False, is_sorry_ok=False):
        """
        Verify a single proof.
        Args:
            proof (str): The proof to verify.
            timeout (int, optional): The timeout for the verification request. Defaults to 30s.
            return_error_message (bool, optional): Whether to return the error message if the proof is invalid. Defaults to False.
            is_sorry_ok (bool, optional): Whether to allow the proof to be valid even if it contains a "sorry" term. Mainly used during autoformalization. Defaults to False.
        Returns:
            bool: True if the proof is valid, False otherwise.
        """
        proof = proof.strip()
        response = self.client.check(proof, timeout=timeout, infotree="original")
        verification_results = read_client_response(response)[0]
        if is_sorry_ok:
            is_proof_valid = verification_results['is_correct_with_sorry']
        else:
            is_proof_valid = verification_results['is_correct_no_sorry']

        if return_error_message:
            if not is_proof_valid:
                try:
                    error_messages = extract_all_error_messages(response, [proof])
                except Exception as e:
                    logger.info("Proof is invalid...")
                    traceback.print_exc()
                    return False, f"Proof:\n{proof}\n\nError: Most likely timed out"
                return is_proof_valid, error_messages[0]
            else:
                return is_proof_valid, None
        else:
            return is_proof_valid
    
    def batch_verify_proofs(self, proofs, return_error_messages=False, timeout=30, is_sorry_ok=False):
        """
        Verify a batch of proofs.
        Args:
            proofs (list): A list of proofs to verify.
            return_error_messages (bool, optional): Whether to return the error messages if the proofs are invalid. Defaults to False.
            timeout (int, optional): The timeout for the verification request. Defaults to 30s.
            is_sorry_ok (bool, optional): Whether to allow proofs that contain a "sorry" expression. Defaults to False.
        Returns:
            list: A list of booleans indicating the validity of each proof.
        """

        responses = self.client.check(proofs, timeout=timeout)
        
        parse_results = read_client_response(responses)

        if is_sorry_ok:
            verification_results = [
                item['is_correct_with_sorry']
                for item in parse_results
            ]
        else:
            verification_results = [
                item['is_correct_no_sorry']
                for item in parse_results
            ]

        if return_error_messages:
            all_error_messages = extract_all_error_messages(responses, proofs)
            return verification_results, all_error_messages
        else:
            return verification_results
