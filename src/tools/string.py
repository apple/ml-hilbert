#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import re
import json
from typing import List, Dict, Optional
import logging
logger = logging.getLogger(__name__)

def extract_jsonl_contents(file_path: str):
    """
    Extracts the contents of a jsonl file.
    """
    with open(file_path, 'r') as f:
        examples = []
        for line in f:
            examples.append(json.loads(line))
        return examples

def extract_proof_ds_prover_v2(response: str) -> str:
    """
    Extracts a code block from a string response, handling various delimiters.

    This function searches for code blocks in a specific order of precedence to
    handle both correctly formed and malformed delimiter pairs. The search order is:
    1. Standard triple backticks (```...```)
    2. Triple single quotes ('''...''')
    3. Malformed mixed delimiters ('''...```)
    4. Malformed mixed delimiters (```...''')

    It uses a non-greedy search (`.*?`) and returns the content of the *last*
    matching block found for the first successful pattern. This makes it robust
    to responses containing conversational text followed by a code block.

    Args:
        response: The string potentially containing the code block(s).

    Returns:
        The extracted code as a string, stripped of leading/trailing whitespace.
        If no code block is found using any pattern, it prints the original
        response for debugging and returns the string "error".
    """
    # Define regex patterns in order of precedence.
    # The non-capturing group (?:...) for the optional language hint is robust.
    # The (.*?) captures the content non-greedily.
    # re.DOTALL allows '.' to match newline characters.
    patterns_in_order = [
        # 1. Primary, standard case: ```...```
        re.compile(r"```(?:[a-zA-Z0-9_-]*)\n?(.*?)\n?```", re.DOTALL),
        # 2. Secondary, alternative case: '''...'''
        re.compile(r"'''(?:[a-zA-Z0-9_-]*)\n?(.*?)\n?'''", re.DOTALL),
        # 3. Malformed case (as requested): '''...```
        re.compile(r"'''(?:[a-zA-Z0-9_-]*)\n?(.*?)\n?```", re.DOTALL),
        # 4. Malformed case (inverse, for completeness): ```...'''
        re.compile(r"```(?:[a-zA-Z0-9_-]*)\n?(.*?)\n?'''", re.DOTALL)
    ]

    # Iterate through the patterns. The first one that finds a match wins.
    for pattern in patterns_in_order:
        matches = pattern.findall(response)
        if matches:
            # If one or more blocks are found, return the content of the last one.
            # .strip() removes leading/trailing whitespace and newlines.
            return matches[-1].strip()

    # If the loop completes without finding any matches, handle the error.
    logger.info(80 * "*")
    logger.info("ERROR: Could not extract a code block with any known delimiter pattern.")
    logger.info("GENERATED RESPONSE:")
    logger.info(80 * "*")
    logger.info(response)
    return "error"

def extract_lean_block(response):  
    
    TRIPLE_TICK_PATTERN = re.compile(r"```(?:lean|lean4)\n?(.*?)\n?```", re.DOTALL)

    # Attempt to extract the content between triple ticks
    match = TRIPLE_TICK_PATTERN.findall(response)
    if len(match) == 1:
        return match[0]
    elif len(match)>1:
        logger.info("WARNING! Multiple Lean 4 blocks detected. Choosing the last one...")
        logger.info("Full response:")
        logger.info(response)
        return match[-1]
        # raise ValueError("Multiple Lean 4 code blocks detected.")
    else:
        logger.info("No Lean 4 code blocks detected.")
        return None

def extract_all_lean_blocks(response: str) -> Optional[List[str]]:
    """
    Parse all Lean blocks from string.
    
    Args:
        response: string containing Lean blocks
        
    Returns:
        List of extracted Lean blocks, or None if no Lean blocks found
    """

    # Try to extract Lean code blocks first
    lean_pattern = r'```(?:lean4?|LEAN4?)\s*\n(.*?)\n```'
    theorems = re.findall(lean_pattern, response, re.DOTALL | re.IGNORECASE)
    if not theorems:
        return None  # No lean blocks found, don't try to parse forcefully

    return theorems
  
def extract_search_queries(text):
    """Extract content from <search> tags in a string."""
    pattern = r'<search>(.*?)</search>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def extract_theorems_queries(text):
    """Extract content from <theorem> tags in a string."""
    pattern = r'<theorem>(.*?)</theorem>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_tag(text: str, tag: str):
    """
    Extract the content between given tags.
    """
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def extract_informal_proof(text: str) -> Optional[str]:
    """
    Extracts the informal proof content from between <informal_proof> tags.
    
    Args:
        text: Input text containing informal proof tags
        
    Returns:
        The informal proof content as a string, or None if not found
    """
    return extract_tag(text, 'informal_proof')

def extract_answer(text: str) -> Optional[str]:
    """
    Extracts the answer in between <answer> </answer> tags.
    
    Args:
        text: Input text containing informal proof tags
        
    Returns:
        The extracted answer as a string, or None if not found
    """
    return extract_tag(text, 'answer')

def extract_reason(text: str):
    return extract_tag(text, 'reason')

def extract_lemmas_from_outline(text: str) -> List[str]:
    """
    Extracts lemmas from the proof outline section.
    
    Args:
        text: Input text containing proof outline with lemma tags
        
    Returns:
        List of lemma strings extracted from the proof outline
    """
    # First extract the proof outline section
    outline_pattern = re.compile(r"<proof_outline>\s*(.*?)\s*</proof_outline>", re.DOTALL)
    outline_match = outline_pattern.search(text)
    
    if not outline_match:
        return []
    
    outline_content = outline_match.group(1)
    
    # Extract all lemmas from the outline
    lemma_pattern = re.compile(r"<lemma>\s*(.*?)\s*</lemma>", re.DOTALL)
    lemmas = lemma_pattern.findall(outline_content)
    
    return [lemma.strip() for lemma in lemmas]

def parse_proof_structure(text: str) -> Dict[str, any]:
    """
    Parses a structured proof text and extracts both informal proof and lemmas.
    
    Args:
        text: Input text containing informal proof and proof outline sections
        
    Returns:
        Dictionary with keys:
        - 'informal_proof': The extracted informal proof text (str or None)
        - 'lemmas': List of extracted lemmas (List[str])
    """
    informal_proof = extract_informal_proof(text)
    lemmas = extract_lemmas_from_outline(text)
    proof_sketch = extract_tag(text, 'proof_sketch')
    return {
        'informal_proof': informal_proof,
        'lemmas': lemmas,
        'proof_sketch': proof_sketch
    }

def remove_think_block(text: str) -> str:
    """
    Removes the <think> block from the given text.

    Args:
        text: Input text containing the <think> block
    Returns:
        Text with the <think> block removed
    """
    if '<think>' in text:
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        return think_pattern.sub("", text).strip()
    else:
        return text