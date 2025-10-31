#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
##############################################################
#               Formal to Formal 
##############################################################
COT_PROMPT = """
Complete the following Lean 4 code:
```lean4
{formal_statement}
```
Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies. The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""

NON_COT_PROMPT = """
Complete the following Lean 4 code:
'''lean4
{formal_statement}
'''
"""
