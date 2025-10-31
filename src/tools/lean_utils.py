#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Lean Utils
"""
import re
import logging
from typing import List, Tuple
from src.prompts.error_messages import ERROR_LINE_MESSAGE, FULL_ERROR_MESSAGE

# Import for infotree-based extraction
from src.tools.proof_utils import split_header_body


logger = logging.getLogger(__name__)

def extract_all_error_messages(responses, proofs):
    num_responses = len(responses.results)
    all_error_messages = []
    for i in range(num_responses):
        i_error_msgs = []
        if not responses.results[i].response:
            error_message = f"Proof: {proofs[i]}\nError: Timed out"
            all_error_messages.append(error_message)
            continue
        if 'messages' in responses.results[i].response:
            messages = responses.results[i].response['messages']
        else:
            # probably timed out
            if 'error' in responses.results[i].response:
                error = responses.results[i].response['error']
            else:
                error = "Timed out"
            error_message = f"Proof: {proofs[i]}\nError: {error}"
            all_error_messages.append(error_message)
            continue
        # Split proof to get body for line number reference
        _, body = split_header_body(proofs[i])
        body_lines = body.split('\n')
        

        error_lines_messages = []
        for message in messages:
            if message['severity'] == 'error':
                error_message = message['data']
                # Line numbers from Lean are 1-based, but our arrays are 0-based
                start_pos_line = message['pos']['line'] - 1
                
                if 'endPos' in message and message['endPos'] is not None:
                    end_pos_line = message['endPos']['line'] - 1
                    # Ensure we don't go out of bounds
                    if start_pos_line >= 0 and start_pos_line < len(body_lines) and end_pos_line >= 0 and end_pos_line < len(body_lines):
                        # extract the proof segment corresponding to pos and endPos
                        proof_segment = "\n".join(body_lines[start_pos_line:end_pos_line+1])
                    elif start_pos_line >= 0 and start_pos_line < len(body_lines):
                        # Fallback to single line if endPos is out of bounds
                        proof_segment = body_lines[start_pos_line]
                    else:
                        proof_segment = "Error: line number out of range"
                else:
                    # Single line error
                    if start_pos_line >= 0 and start_pos_line < len(body_lines):
                        proof_segment = body_lines[start_pos_line]
                    else:
                        proof_segment = "Error: line number out of range"

                if proof_segment.strip():
                    error_lines_message = ERROR_LINE_MESSAGE.format(
                        error_lines=proof_segment, 
                        error_message=error_message,
                        current_state="", #current_state_message
                    )
                else:
                    error_lines_message = ""
                error_lines_messages.append(error_lines_message)
        
        error_lines_message = "\n".join(error_lines_messages)
        error_message = FULL_ERROR_MESSAGE.format(proof=proofs[i],
                                    error_lines_message=error_lines_message)
        i_error_msgs.append(error_message)
        i_error_msgs = "\n".join(i_error_msgs)
        all_error_messages.append(i_error_msgs)
    return all_error_messages

def extract_theorem_name(theorem_text: str) -> str:
    """
    Extract the name of a theorem from its declaration.
    """

    # remove comments if any
    text = _remove_comments(theorem_text)

    # find the keyword theorem
    if "theorem" not in text:
        raise ValueError("No theorem keyword found in theorem declaration")
    
    # find the name of the theorem
    name_start = text.find("theorem ") + len("theorem ")
    idx = name_start
    stop_chars = {':', '(', '{', '[', '⦃'}
    while idx < len(theorem_text) and text[idx] != ":" and text[idx] not in stop_chars:
        idx += 1
    name_end = idx
    name = text[name_start:name_end].strip()
    return name


def extract_proof_body_from_theorem(theorem_text: str) -> str:
    """
    Extract the proof body from a theorem, removing the theorem declaration.
    
    Args:
        theorem_text: String containing a complete theorem with proof
        
    Returns:
        String containing just the proof body (everything after := by)
    """

    # split at first assignment
    _, after = _split_at_first_assignment(theorem_text, return_idx=False)
    proof = after.strip()
    if proof.startswith("by"):
        proof = proof[2:]
    return proof


def extract_all_have_names(text_string: str):
    """
    Extract the names of all have statements in the given proof
    Args:
        text_string (str): The string to search within.

    Returns:
        list[tuple[int, int]]: A list of (start, end) index tuples for each match.
                               Returns an empty list if no matches are found.
    """
    # remove comments
    text_string = _remove_comments(text_string)
    pattern = r'\b'+'have'+r'\b'
    indices = [match.span() for match in re.finditer(pattern, text_string)]
    have_names = []
    stop_chars = {':', '(', '{', '[', '⦃'}
    for (_, end_idx) in indices:
        idx = end_idx
        while idx < len(text_string):
            if text_string[idx] == ':' or text_string[idx] in stop_chars:
                break
            idx += 1
        have_names.append(text_string[end_idx:idx].strip())
    # remove empty strings
    have_names = [name for name in have_names if name]
    return have_names

def replace_have_proofs_with_sorry(theorem_text: str) -> str:
    """
    Replace the proofs of have statements with sorry while preserving the theorem structure.
    
    This function takes a Lean 4 theorem containing have statements and replaces all
    have statement proofs with 'sorry', while keeping the declarations intact.
    
    Handles various Lean 4 syntax patterns:
    - have h : P := by proof
    - have h : P by proof  
    - have h : P := proof
    - have h (params) : P := by proof
    - · have h : P := by proof (with bullet points)
    
    Args:
        theorem_text: String containing a Lean 4 theorem with have statements
        
    Returns:
        String with have statement proofs replaced by sorry
        
    Example:
        Input:
        theorem cases_hierarchical (n : ℕ) : n = 0 ∨ n > 0 := by
          cases' n with k
          · have h1 : 0 = 0 := by rfl
            have h2 : 0 = 0 ∨ 0 > 0 := by
              left
              exact h1
            exact h2
          · have h3 : Nat.succ k > 0 := by
              exact Nat.succ_pos k
            have h4 : Nat.succ k = 0 ∨ Nat.succ k > 0 := by
              right
              exact h3
            exact h4
            
        Output:
        theorem cases_hierarchical (n : ℕ) : n = 0 ∨ n > 0 := by
          cases' n with k
          · have h1 : 0 = 0 := by sorry
            have h2 : 0 = 0 ∨ 0 > 0 := by sorry
            exact h2
          · have h3 : Nat.succ k > 0 := by sorry
            have h4 : Nat.succ k = 0 ∨ Nat.succ k > 0 := by sorry
            exact h4
    """
    if not theorem_text or not theorem_text.strip():
        return theorem_text
    
    lines = theorem_text.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        
        # Handle empty lines and comments
        if not stripped_line or stripped_line.startswith('--'):
            result_lines.append(line)
            i += 1
            continue
        
        # Check if this line contains a have statement (accounting for bullets and other prefixes)
        if 'have ' in line:
            have_pos = line.find('have ')
            # Make sure 'have' is at a word boundary (not part of another word)
            if (have_pos == 0 or not line[have_pos - 1].isalnum()) and \
               (have_pos + 4 >= len(line) or not line[have_pos + 4].isalnum()):
                
                # Parse the have statement to extract declaration and proof parts
                have_declaration, next_i = _parse_have_statement(lines, i)
                
                if have_declaration:
                    # Get the prefix before 'have' (bullets, indentation, etc.)
                    prefix = line[:have_pos]
                    
                    result_lines.append(f"{prefix}{have_declaration} := by sorry")
                    
                    # Skip to the next statement after this have block
                    i = next_i
                else:
                    # Couldn't parse the have statement, keep original line
                    result_lines.append(line)
                    i += 1
            else:
                # 'have' is part of another word, keep the line as is
                result_lines.append(line)
                i += 1
        else:
            # No have statement, keep the line as is
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)


def _parse_have_statement(lines: List[str], start_idx: int) -> Tuple[str, str, int]:
    """
    Parse a have statement to extract the declaration part.
    
    Uses an algorithm that:
    1. Goes through each line left to right
    2. Keeps track of let statements with a counter
    3. Maintains a stack for brackets
    4. After every "let", decrements counter for every ":=" when stack is empty
    5. Ends proof when encountering terminal statement (":=" or ":= by") when stack is empty and counter is 0
    
    Args:
        lines: List of all lines in the theorem
        start_idx: Index of the line containing the have statement
        
    Returns:
        Tuple of (have_declaration, next_index) where:
        - have_declaration: The declaration part (e.g., "have h1 : 0 = 0")
        - next_index: Index of the next line after this have statement block
    """
    if start_idx >= len(lines):
        return "", ":= by", start_idx + 1
    
    # Get the line with the have statement
    have_line = lines[start_idx]
    have_pos = have_line.find('have ')
    
    if have_pos == -1:
        return "", ":= by", start_idx + 1
    
    # Calculate the indentation level based on the position of 'have'
    have_indent = have_pos
    
    # Initialize flags and stacks
    in_let_statement = False
    bracket_stack = []
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    
    # Collect declaration
    declaration_parts = []
    i = start_idx
    
    # Process each line left to right
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        line_indent = len(line) - len(line.lstrip())
        
        # Stop if we hit a line at same or lesser indentation (unless it's empty)
        if stripped_line and line_indent <= have_indent and i > start_idx:
            break
        
        # For the first line, extract everything from 'have' onwards
        if i == start_idx:
            line_to_process = line[have_pos:]
        else:
            line_to_process = line
        
        # Process each character in the line left to right
        j = 0
        found_terminal = False
        while j < len(line_to_process):
            char = line_to_process[j]
            
            # Handle brackets
            if char in bracket_pairs:
                bracket_stack.append(bracket_pairs[char])
            elif char in bracket_pairs.values():
                if bracket_stack and bracket_stack[-1] == char:
                    bracket_stack.pop()
            
            # Check for "let" statements
            if j <= len(line_to_process) - 4 and line_to_process[j:j+4] == 'let ':
                # Make sure it's a word boundary
                if (j == 0 or not line_to_process[j-1].isalnum()) and \
                   (j+4 >= len(line_to_process) or not line_to_process[j+3].isalnum()):
                    in_let_statement = True
            
            # Check for ":=" when stack is empty
            if j < len(line_to_process) - 1 and line_to_process[j:j+2] == ':=':
                if in_let_statement:
                    # Reset flag after processing let statement
                    in_let_statement = False
                elif len(bracket_stack) == 0:
                    # This is a terminal statement - extract declaration up to this point
                    if line_to_process[:j].strip():
                        declaration_parts.append(line_to_process[:j])
                    found_terminal = True
                    i += 1  # Move to next line to start skipping proof
                    break
            
            j += 1
        
        # If we found a terminal statement, break out of line processing
        if found_terminal:
            break
        
        # Add this line to declaration if we haven't found terminal statement yet
        declaration_parts.append(line_to_process)
        i += 1
    
    # Join declaration parts preserving line breaks
    have_declaration = '\n'.join(part for part in declaration_parts if part).strip()
    
    # Skip remaining proof lines (lines more indented than the have statement)
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        line_indent = len(line) - len(line.lstrip())
        
        # Stop if we hit a non-empty line at same or lesser indentation than the 'have'
        if stripped_line and line_indent <= have_indent:
            break
        
        i += 1
    
    return have_declaration, i


def _split_at_first_assignment(text_input: str, start_idx: int = 0, return_idx : bool = False) -> Tuple[str, str]:
    """
    Split text at the first := that's not inside parentheses/brackets and not part of a let statement.
    
    Args:
        text_input: String to split
        start_idx: the index to start the splitting process
        return_idx: whether or not to return the index of the first :=. Defaults to False.
    Returns:
        Tuple of (before_assignment, after_assignment)
    """
    # Use a stack to track bracket depth
    stack = []
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    
    i = start_idx
    in_let_statement = False
    text = _remove_comments(text_input)
    while i < len(text):
        char = text[i]
        # Handle brackets
        if char in bracket_pairs:
            stack.append(bracket_pairs[char])
        elif char in bracket_pairs.values():
            if stack and stack[-1] == char:
                stack.pop()
        # Check for "let" statements
        if i <= len(text) - 4 and text[i:i+4] == 'let ':
            # Make sure it's a word boundary
            if (i == 0 or not text[i-1].isalnum()) and \
                (i+4 >= len(text) or not text[i+3].isalnum()):
                in_let_statement = True

        # Check for ":=" when stack is empty
        if i < len(text) - 1 and text[i:i+2] == ':=':
            if in_let_statement:
                # Reset flag after processing let statement
                in_let_statement = False
            elif len(stack) == 0:
                # This is a terminal statement - extract declaration up to this point
                before = text[start_idx:i].strip()
                after = text[i + 2:].strip()  # +2 to skip ":="
                if return_idx:
                    return before, after, i
                else:
                    return before, after

        i += 1

    # No assignment found at depth 0
    if return_idx:
        return text, "", None
    else:
        return text, ""


def _remove_all_nontheorem_lines(text):
    all_lines = text.split("\n")
    to_include = []
    for idx, line in enumerate(all_lines):
        if line.startswith('theorem'):
            to_include = all_lines[idx:]
            break
    if not to_include:
        logger.info(100*"?")
        logger.info("Theorem does not have non-theorem lines\n%s", text)
        logger.info(100*"?")
        return None
    
    new_text = "\n".join(to_include).strip()
    return new_text

def extract_theorem_signature(text):
    """
    Extracts the theorem statement from Lean 4 code, handling various proof constructs.
    Captures everything after 'theorem' up to ':='.
    """
    # remove all lines that don't begin with "theorem"
    new_text = _remove_all_nontheorem_lines(text)
    if not new_text:
        return None, None
    before, _ = _split_at_first_assignment(new_text)

    return before

def normalize_signature(signature: str) -> str:
    """
    Normalize a signature by removing extra whitespace and standardizing formatting.
    
    Args:
        signature: The signature to normalize
        
    Returns:
        The normalized signature
    """
    if not signature:
        return ""
    
    # Replace multiple whitespaces with single space
    normalized = re.sub(r'\s+', ' ', signature.strip())
    
    # Remove spaces around operators and symbols
    operators = [
        (r'\s*=\s*', '='),
        (r'\s*∈\s*', '∈'),
        (r'\s*∏\s*', '∏'),
        (r'\s*∑\s*', '∑'),
        (r'\s*→\s*', '→'),
        (r'\s*←\s*', '←'),
        (r'\s*↔\s*', '↔'),
        (r'\s*%\s*', '%'),
        (r'\s*\|\s*', '|'),
        (r'\s*∣\s*', '∣'),  # Mathematical divides symbol
        (r'\s*∤\s*', '∤'),  # Mathematical does not divide symbol
    ]
    
    for pattern, replacement in operators:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Remove spaces around parentheses, brackets, and braces
    brackets = [
        (r'\s*\(\s*', '('),
        (r'\s*\)\s*', ')'),
        (r'\s*\[\s*', '['),
        (r'\s*\]\s*', ']'),
        (r'\s*\{\s*', '{'),
        (r'\s*\}\s*', '}'),
    ]
    
    for pattern, replacement in brackets:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Remove spaces around punctuation
    punctuation = [
        (r'\s*,\s*', ','),
        (r'\s*:\s*', ':'),
        (r'\s*;\s*', ';'),
        (r'\s*\.\s*', '.'),
    ]
    
    for pattern, replacement in punctuation:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Handle factorial operator - remove space before !
    normalized = re.sub(r'\s+!', '!', normalized)
    
    # Handle function arrows and lambda expressions
    arrows = [
        (r'\s*=>\s*', '=>'),
        (r'\s*->\s*', '->'),
        (r'\s*<-\s*', '<-'),
        (r'\s*\|\-\s*', '|-'),  # Turnstile
    ]
    
    for pattern, replacement in arrows:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Handle mathematical notation spacing
    math_ops = [
        (r'\s*\*\s*', '*'),
        (r'\s*\+\s*', '+'),
        (r'\s*-\s*', '-'),
        (r'\s*/\s*', '/'),
        (r'\s*\^\s*', '^'),
    ]
    
    for pattern, replacement in math_ops:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Clean up any remaining multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Final trim
    return normalized.strip()

def check_theorem_signature_match(theorem1: str, theorem2: str) -> bool:
    """
    Check if two theorems have matching signatures, accounting for formatting discrepancies.
    
    Args:
        theorem1: First theorem statement
        theorem2: Second theorem statement
        
    Returns:
        True if signatures match, False otherwise
    """
    # Extract signatures from both theorems
    sig1 = extract_theorem_signature(theorem1)
    sig2 = extract_theorem_signature(theorem2)
    
    # If either signature couldn't be extracted, they don't match
    if sig1 is None or sig2 is None:
        return False
    
    # Normalize both signatures
    norm_sig1 = normalize_signature(sig1)
    norm_sig2 = normalize_signature(sig2)

    logger.info("SIGNATURE ONE: %s", norm_sig1)
    logger.info("SIGNATURE TWO: %s", norm_sig2)

    # Compare normalized signatures
    return norm_sig1 == norm_sig2

def _remove_comments(text: str) -> str:
    """
    Remove comments from Lean code while preserving structure.
    
    Args:
        text: Lean code with comments
        
    Returns:
        Lean code with comments removed
    """
    # First remove block comments (/- ... -/)
    result = ""
    i = 0
    while i < len(text):
        # Check for start of block comment
        if i < len(text) - 1 and text[i:i+2] == '/-':
            # Find the end of block comment
            end_pos = text.find('-/', i + 2)
            if end_pos != -1:
                # Skip the entire block comment
                i = end_pos + 2
                continue
            else:
                # Unterminated block comment, remove rest of text
                break
        else:
            result += text[i]
            i += 1
    
    # Then remove single-line comments (-- comments)
    lines = result.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if '--' in line:
            # Find the position of -- that's not inside a string
            comment_pos = line.find('--')
            if comment_pos != -1:
                line = line[:comment_pos].rstrip()
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def remove_import_statements(text: str) -> str:
    return _remove_all_nontheorem_lines(text)

def remove_import_lines(text: str) -> str:
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("import"):
            continue
        if line.startswith("open"):
            continue
        if line.startswith("set_option"):
            continue
        new_lines.append(line)

    return "\n".join(new_lines)


def extract_missing_identifiers(error_message: str) -> List[str]:
    """
    Extract all missing identifier/name from a Lean 4 error message.
    
    Looks for patterns like:
    - "unknown constant '<id>'"
    - "unknown identifier '<id>'"
    
    Args:
        error_message: The error message from Lean 4
        
    Returns:
        List of missing identifier names found in the error message
        
    Example:
        >>> extract_missing_identifiers("unknown constant 'foo' and unknown identifier 'bar'")
        ['foo', 'bar']
        >>> extract_missing_identifiers("unknown identifier 'test'")
        ['test']
        >>> extract_missing_identifiers("some other error")
        []
    """
    if not error_message:
        return []
    
    identifiers = []
    
    # Pattern for "unknown constant '<identifier>'"
    unknown_constant_pattern = r"unknown constant\s+'([^']+)'"
    matches = re.findall(unknown_constant_pattern, error_message, re.IGNORECASE)
    identifiers.extend(matches)
    
    # Pattern for "unknown identifier '<identifier>'"
    unknown_identifier_pattern = r"unknown identifier\s+'([^']+)'"
    matches = re.findall(unknown_identifier_pattern, error_message, re.IGNORECASE)
    identifiers.extend(matches)
    
    return identifiers

def _check_for_sorries(proof: str):
    # clean up comments
    no_imports_proof = remove_import_statements(proof)
    no_comments_proof = _remove_comments(no_imports_proof)
    # check if the proof contains sorries
    if re.search(r'\b(sorry|admit)\b',  no_comments_proof, re.IGNORECASE):
        return True
    else:
        return False

def _extract_all_theorems_from_string(text: str) -> List[str]:
    """
    Extract all theorem blocks from a string containing one or more Lean 4 theorems.
    
    Partitions the text based on theorem endings (':= by sorry' or ':= sorry') rather than
    theorem-to-theorem boundaries. This is more reliable since theorems are known to end
    with these patterns.
    
    Args:
        text: String containing one or more theorem blocks
        
    Returns:
        List of strings, each containing a complete theorem block
        
    Example:
        >>> text = '''theorem foo : 1 = 1 := by sorry
        ... theorem bar : 2 = 2 := sorry'''
        >>> _extract_all_theorems_from_string(text)
        ['theorem foo : 1 = 1 := by sorry', 'theorem bar : 2 = 2 := sorry']
    """
    if not text or not text.strip():
        return []
    text_no_imports = remove_import_statements(text)
    text_no_comments = _remove_comments(text_no_imports)
    match_iterator = re.finditer(r'\btheorem\b', text_no_comments)

    positions = [match.start() for match in match_iterator] + [len(text_no_comments)]

    theorems = []

    for idx, position in enumerate(positions[:-1]):
        extracted_theorem = text_no_comments[position:positions[idx+1]]
        extracted_theorem = extracted_theorem.strip()
        theorems.append(extracted_theorem)
    return theorems
