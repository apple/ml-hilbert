#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
def read_client_response(response):
    parsed_answers = []

    for r in response.results:
        curr_response = r.response
        if curr_response is None:
            parsed_answers.append({
                "is_correct_with_sorry": False,
                "is_correct_no_sorry": False
            })
            continue
        messages = curr_response.get("messages", [])
        severities = [m.get("severity") for m in messages]
        datas = [m.get("data", "") for m in messages]

        if "error" in severities:
            parsed_answers.append({
                "is_correct_with_sorry": False,
                "is_correct_no_sorry": False
            })
            continue

        has_sorry = any(
            sev == "warning" and "declaration uses 'sorry'" in data
            for sev, data in zip(severities, datas)
        )

        parsed_answers.append({
            "is_correct_with_sorry": True,
            "is_correct_no_sorry": not has_sorry
        })

    return parsed_answers

def split_header_body(proof: str) -> tuple[str, str]:
    """
    Splits `proof` into:
    - header: the consecutive `import ...` lines at the beginning of the proof.
    We remove all "import Mathlib." lines and add "import Mathlib" if necessary.
    - body: rest of the proof

    Args:
        proof (str): The proof code to split

    Returns:
        tuple[str, str]: The header and body of the proof
    """
    proof = proof.strip()
    lines = proof.splitlines()
    header_lines = []
    proof_idx = 0

    mathlib_found = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("import"):
            if line.startswith("import Mathlib."):
                mathlib_found = True
            else:
                header_lines.append(line)
            proof_idx = i + 1
        else:
            break
    if mathlib_found and 'import Mathlib' not in header_lines:
        header_lines.insert(0, 'import Mathlib')
    header = "\n".join(header_lines).strip()
    body = "\n".join(lines[proof_idx:]).strip()

    return header, body
