#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Prompts for the HILBERT model.
"""
################################################################
#                   Search Query Prompts
################################################################

SEARCH_QUERY_PROMPT = """
You are helping solve a Lean theorem proving problem using the mathlib library. Before attempting to write the proof, you must first search for relevant theorems and tactics.

Search Process:
1. Identify key concepts: Break down the problem into mathematical concepts, operations, and structures involved.
2. Generate search queries: For each concept, create informal search strings that describe:
   - Relevant theorems or results (e.g., "associativity of addition", "existence of inverse elements")
   - Useful tactics (e.g., "simplify arithmetic expressions", "split conjunctions")
   - Properties (e.g., "group structure on integers", "metric space properties")
   - Relevant definitions useful for the proof or any used theorem (e.g. "definition of a group", "definition of a metric space")

Search Query Format:
Enclose each search query in <search> tags with your informal description. Limit yourself to a maximum of 5 search queries. Make the search queries simple, concise, and clear.

Guidelines:
- You can either search by theorem name or natural language description
- Search for theorems that might automate parts of the proof
- Consider edge cases and special conditions mentioned in the problem

Problem to Solve:
{problem}

"""

ERROR_SEARCH_QUERY_PROMPT = """
Your task is to come up with search queries to help solve the error in the given error message.

{error_message}

Instructions:
1. Come up with natural language search queries to help solve the problem. Search for theorems/definitions/tactics that will help you better solve the problem. 
2. Focus on theorems that are "unknown" from the above error message.
3. Enclose your natural language search queries in <search> tags with the description.
4. Limit yourself to a maximum of 2 queries. Make the search queries simple, concise, and clear.
5. Do NOT try to correct the error or prove the theorem. ONLY provide the search queries.
"""

SEARCH_ANSWER_PROMPT = """
You are helping to solve a Lean theorem proving problem using the mathlib library. The problem is:
{problem}

Here are some potentially relevant theorems and definitions:
{theorems}

Instructions:
1. Select important theorems and definitions necessary to solve the problem. 
2. IMPORTANT: ONLY SELECT theorems from the GIVEN list.
3. Enclose each of them in separate <theorem> tags. 
4. Only state the full names of the theorems. Do NOT include the module name.
5. Select all theorems that could be useful in the intermediate steps of the proof.
"""

EXTRACT_SUBGOALS_FROM_SKETCH_PROMPT = """From this proof sketch, extract any missing proofs (specified with `sorry`) as independent subgoals (theorems). 
Instructions:
1. Use the same name as the have statements for the theorems.
2. Each subgoal should have the relevant context from the previous subgoals needed to simplify the proof as much as possible.
3. There should be as many extracted theorems as `sorry`s in the given theorem.
4. Do NOT include any imports or open statements. Do NOT add any definitions. ONLY include the theorem statement.
5. Use a separate Lean 4 ``lean`` block for each subgoal.
6. Use sorry for the proof. Do NOT prove any theorem.
7. Do NOT change the conclusion of the theorems from the extracted subgoals. Keep them AS IT IS.
8. Do NOT change the conclusions of the preceding theorems when presenting them as hypotheses for the next subgoals. Keep them AS IT IS.
9. Do NOT duplicate theorem names. Use distinct theorem names for the different theorems.
10. Make sure the names and types of the premises/arguments in the extracted theorems MATCH the subgoals from which they are extracted.

IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on natural number literals with UNDEFINED types, unless REQUIRED by the theorem statement. For example, do NOT allow literals like `1 / 3` or `2 / 5` or `1 - 3` ANYWHERE in the theorem statement. ALWAYS specify the types. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.
ALWAYS specify types when describing fractions. For example, ((2 : ℝ) / 3) or ((2 : ℚ) / 3) instead of (2 / 3)
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on variables of type natural numbers (Nat or ℕ), unless REQUIRED by the theorem statement. For example, do NOT allow expressions like (a-b) or (a/b) where a, b are of type ℕ. ALWAYS cast the variables to a suitable type (ℤ, ℚ or ℝ) when performing arithmetic operations. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.

Lean Hints:
{lean_hints}

Proof Sketch:
```lean4
{proof_sketch}
```"""

EXTRACT_MISSING_SUBGOALS_PROMPT = """
We are trying to extract all the subgoals from the following proof sketch:
```lean4
{original_sketch}
```

However, some subgoals seem to be missing. 

{missing_names}

These are the theorems that have already been extracted:
{already_extracted_theorems}

Instructions:
1. For missing `have` statements, extract only the missing `have` statements as independent theorems with the same names.
2. If `sorry` count is higher than the number of extracted subgoals, make sure that you are extracting all the missing subgoals needed to replace the `sorry` placeholders with proofs.
3. Do NOT include the already extracted theorems in your response.
4. Do NOT try to prove the extracted theorems. Use `sorry` for the proof.
5. Do NOT include any imports or open statements. Do NOT add any definitions. ONLY include the theorem statement.
6. Use a separate Lean 4 ``lean`` block for each subgoal.

IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on natural number literals with UNDEFINED types, unless REQUIRED by the theorem statement. For example, do NOT allow literals like `1 / 3` or `2 / 5` or `1 - 3`. ALWAYS specify the types. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.
ALWAYS specify types when describing fractions. For example, ((2 : ℝ) / 3) or ((2 : ℚ) / 3) instead of (2 / 3)
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on variables of type natural numbers (Nat or ℕ), unless REQUIRED by the theorem statement. For example, do NOT allow expressions like (a-b) or (a/b) where a, b are of type ℕ. ALWAYS cast the variables to a suitable type (ℤ, ℚ or ℝ) when performing arithmetic operations. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.

"""

INFORMAL_LLM_INFORMAL_PROOF_SKETCH = """
You are a mathematical expert whose goal is to solve problems with rigorous mathematical reasoning.

{useful_theorems_section}Instructions:
1. Provide a natural language, step-by-step proof for the given problem.
2. Start from the given premises and reason step-by-step to reach the conclusion.
3. Number each step of the proof as 1, 2, and so on.
4. Be as pedantic and thorough as possible.
5. Keep each step precise, increase the number of steps if needed.
6. Do NOT gloss over any step. Make sure to be as thorough as possible. 
7. Show the explicit calculations/simplifications, theorem applications and case analysis.
8. Enclose the informal proof in <informal_proof> tags.

Problem Statement: {problem}

"""

INFORMAL_LLM_CREATE_LEAN_SKETCH = """
You are a Lean 4 expert who is trying to help write a proof in Lean 4.

Problem Statement: {problem}

{useful_theorems_section}Informal Proof:
{informal_proof}

Instructions:

Use the informal proof to write a proof sketch for the problem in Lean 4 following these guidelines:
- Break complex reasoning into logical sub-goals using `have` statements.
- The subgoals should build up to prove the main theorem.
- Make sure to include all the steps and calculations from the given proof in the proof sketch.
- Each subgoal should ideally require applying just one key theorem or lemma, or a few tactic applications.
- Base subgoals around:
  - Useful theorems mentioned in the problem context
  - Standard library theorems (like arithmetic properties, set operations, etc.)
  - The supplied premises in the theorem statement
- Do NOT create subgoals identical to any of the given hypotheses
- Do NOT create subgoals that are more complex than the original problems. The subgoals should be SIMPLER than the given problem.
- Do NOT skip over any steps. Do NOT make any mathematical leaps.

**Subgoal Structure Requirements:**
- **Simplicity**: Each subgoal proof should be achievable with 1-3 basic tactics
- **Atomic reasoning**: Avoid combining multiple logical steps in one subgoal
- **Clear progression**: Show logical flow: `premises → intermediate steps → final result`
- **Theorem-focused**: Design each subgoal to directly apply a specific theorem when possible

NOTE: Only add sub-goals that simplify the proof of the main goal.

When writing Lean proofs, maintain consistent indentation levels.

Rules:
1. Same proof level = same indentation: All tactics at the same logical level must use identical indentation
2. Consistent characters: Use either tabs OR spaces consistently (don't mix)
3. Proper nesting: Indent sub-proofs one level deeper than their parent
4. Do NOT nest `have` statements in each other. Use distinct sub-goals as much as possible. Ensure all sub goals are named. Do NOT create anonymous have statements.
5. Do NOT include any imports or open statements in your code.
6. One line = One `have` subgoal. Do NOT split subgoals across different lines.
7. Use proper Lean 4 syntax and conventions. Ensure the proof sketch is enclosed in triple backticks ```lean```
8. Use `sorry` for all subgoal proofs - focus on structure, not implementation
9. **Do NOT use `sorry` for the main goal proof** - use your subgoals to prove it
10. NEVER use `sorry` IN the theorem statement itself
11. Ensure subgoals collectively provide everything needed for the main proof
12. Make the logical dependencies between subgoals explicit. Ensure that the subgoals are valid and provable in Lean 4.
13. Do NOT change anything in the original theorem statement.

Lean Hints:
{lean_hints}

IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on natural number literals with UNDEFINED types, unless REQUIRED by the theorem statement. For example, do NOT allow literals like `1 / 3` or `2 / 5` or `1 - 3` ANYWHERE in ANY of the subgoals. ALWAYS specify the types. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.
ALWAYS specify types when describing fractions. For example, ((2 : ℝ) / 3) or ((2 : ℚ) / 3) instead of (2 / 3). Do this everywhere EXCEPT the given theorem statement.
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on variables of type natural numbers (Nat or ℕ), unless REQUIRED by the theorem statement. For example, do NOT allow expressions like (a-b) or (a/b) where a, b are of type ℕ. ALWAYS cast the variables to a suitable type (ℤ, ℚ or ℝ) when performing arithmetic operations. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.

"""

ERROR_CORRECTION_SYSTEM_PROMPT = """
You are trying to help correct Lean 4 code errors. Use the following instructions to help solve the problem.

Instructions:
1. Enclose the proof in a ```lean ``` block.
2. Do NOT change the recursion depth. If you get a maximum recursion depth limit reached error, change the proof to avoid explicit computations involving large numbers. Use tactics like:
SYMBOLIC REWRITING TACTICS:
- `rw [theorem_name]` - rewrite using equalities without computing values
- `simp only [theorem_name]` - selective simplification (avoid `simp` alone on large expressions)
- `simp_rw [theorem_name]` - rewrite under binders without computation

STRUCTURAL MANIPULATION:
- `convert theorem_name` - for goals that are definitionally equal after minor adjustments
- `apply theorem_name` - apply theorems without forcing value computation  
- `exact theorem_name` - AVOID for large numerical expressions
- `congr` - prove equality by showing arguments are equal (avoids computing functions)

ALGEBRAIC TACTICS (symbolic):
- `ring` - solve ring equations symbolically without expanding
- `ring_nf` - normalize ring expressions without computing specific values
- `abel` - for abelian group operations symbolically
- `field_simp` - simplify field expressions without numerical computation
- `linear_combination` - for linear algebra proofs

3. Do not include any imports or open statements.
4. Do not include any other Lean code blocks except for the corrected proof.

"""

PROOF_SKETCH_CORRECTION_PROMPT = """The following Lean 4 code has compilation errors. Please fix the errors while maintaining the mathematical meaning. 

Original statement: {informal_statement}

{error_message}

Lean Hints:
{lean_hints}

Instructions:
1. Analyze what the theorem is trying to prove. Then, analyze why the error is happening, step-by-step. Add a brief explanation.
2. Then, provide a corrected version of the Lean 4 code that addresses these specific errors.
3. Do NOT include any other Lean code blocks except for the proof. Do NOT include any imports or open statements.
4. Use sorry for the proof of all `have` statements.
5. Ensure there are no use of `sorry` statements outside of `have` statements. Do NOT use `sorry` while proving the main theorem.
6. Do NOT change anything in the original theorem statement.
7. Do NOT nest `have` statements in each other. Use distinct sub-goals as much as possible. Ensure all sub goals are named. Do NOT create anonymous have statements.

{useful_theorems_section}
"""

PROOF_SKETCH_ASSEMBLY_CORRECTION_PROMPT = """The following Lean 4 code has compilation errors. Please fix the errors while maintaining the mathematical meaning. 

{error_message}

Lean Hints:
{lean_hints}

Instructions:
1. Analyze what the theorem is trying to prove. Then, analyze why the error is happening, step-by-step. Add a brief explanation.
2. Then, provide a corrected version of the Lean 4 code that addresses these specific errors. 
3. You should ONLY correct the main theorem that appears at the end. Do NOT change any of the helper theorems.
3. Do NOT include any other Lean code blocks except for the proof. Do NOT include any imports or open statements.
4. Do NOT use `sorry` in any part of the proof.
5. Do NOT change anything in the original theorem statement.
6. Do NOT include the helper theorem definitions in your response.
7. Do NOT write a proof for any subgoal from scratch. ALWAYS use the supplied theorems.

"""

THEOREM_CORRECTION_PROMPT = """
The following Lean 4 theorem has compilation errors. Please fix the errors while maintaining the mathematical meaning. 

{error_message}
Instructions:
1. Analyze why the error is happening, step-by-step. Add a brief explanation.
2. Then, provide a corrected version of the Lean 4 code that addresses these specific errors.
3. Do NOT include any other Lean code blocks except for the theorem.
4. Use sorry for the proof.
5. Do NOT include any imports or open statements.
{potentially_useful_theorems}
"""

POTENTIALLY_USEFUL_THEOREMS_PROMPT = """
The following search results could be useful in correcting the error:
{useful_theorems}
"""

SUBGOAL_PROOF_CORRECTION_PROMPT = """The following Lean 4 code has compilation errors. Please fix the errors while maintaining the mathematical meaning. 

{error_message}

Instructions:
1. Analyze what the theorem is trying to prove. Then, analyze why the error is happening, step-by-step. Add a brief explanation.
2. Then, provide a corrected version of the Lean 4 code that addresses these specific errors.
3. Do NOT include any other Lean code blocks except for the proof.
4. Do NOT use sorry.
5. Do NOT include any imports or open statements.
6. Do NOT change anything in the original theorem statement.

{useful_theorems_section}
"""

SOLVE_SUBGOAL_PROMPT = """
Think step-by-step to complete the following Lean 4 proof.

{problem}

Lean Hints:
{lean_hints}

Tactic Hints:
{tactic_hints}

Rules:
1. Same proof level = same indentation: All tactics at the same logical level must use identical indentation
2. Consistent characters: Use either tabs OR spaces consistently (don't mix)
3. Proper nesting: Indent sub-proofs one level deeper than their parent
4. Do NOT include any imports or open statements.
5. Use proper Lean 4 syntax and conventions. Ensure the proof sketch is enclosed in triple backticks ```lean```. 
6. Only include a single Lean 4 code block, corresponding to the proof along with the theorem statement.
7. When dealing with large numerical quantities, avoid explicit computation as much as possible. Use tactics like rw to perform symbolic manipulation rather than numerical computation.
8. Do NOT use sorry.
9. Do NOT change anything in the original theorem statement.
{useful_theorems_section}
"""

DETERMINE_IF_CORRECT_SUBGOAL="""
You are an expert in mathematics.

Your task is to evaluate whether the given mathematical theorem statement is mathematically correct. You do NOT have to provide a proof for the theorem in Lean.

Evaluation criteria:
1. Mathematical validity: Check for logical errors, incorrect assumptions, or calculation mistakes.
2. Do NOT flag general results or helper lemmas that are true independent of the given premises. ONLY flag inaccuracies or mistakes.
5. Provability: Determine if the statement can be proven given the provided premises, or otherwise. 

Assumptions:
1. The given premises are mathematically correct. Do NOT check this.
2. The syntax is guaranteed to be correct (do not assess syntax)

Theorem Statement:
{problem}

Report your answer as either:
• YES - if the statement is mathematically correct
• NO - if the statement has mathematical errors that prevent proof

Also provide a brief justification for your decision in <justification></justification> tags, adding details about why the statement is correct or incorrect.
If it is incorrect, also provide a description of how the error can be corrected. 
If there are missing arguments, make sure to add the relevant missing proof steps.
"""


CORRECT_SKETCH_BASED_ON_INCORRECT_SUBGOAL="""
You are an expert in writing Lean 4 proofs. You are given a Lean 4 proof sketch where one of the subgoals has some issues. 
Your task is to fix the issues and write a new proof sketch.

Proof Sketch:
{proof_sketch}

Issues:
{issues}

Lean Hints:
{lean_hints}

Rules:
1. Same proof level = same indentation: All tactics at the same logical level must use identical indentation
2. Consistent characters: Use either tabs OR spaces consistently (don't mix)
3. Proper nesting: Indent sub-proofs one level deeper than their parent
4. Do NOT nest `have` statements in each other. Write different have statements for different sub goals. 
5. Ensure all sub goals are named. Do NOT create anonymous have statements.
6. Do NOT include any imports or open statements.
7. One line = One `have` subgoal. Do NOT split subgoals across different lines.
8. Use proper Lean 4 syntax and conventions. Ensure the proof sketch is enclosed in triple backticks ```lean```
9. Use `sorry` for all subgoal proofs - focus on structure, not implementation
10. **Do NOT use `sorry` for the main goal proof** - use your subgoals to prove it
11. NEVER use `sorry` IN the theorem statement itself
12. Ensure subgoals collectively provide everything needed for the main proof
13. Make the logical dependencies between subgoals explicit. Ensure that the subgoals are valid and provable in Lean 4.
14. Modify only the incorrect subgoal and everything that follows it in the proof sketch. Leave all preceding portions unchanged.
15. Either modify the problematic subgoals to fix the errors, or add additional subgoals to fill in the missing mathematical arguments.
"""

USE_SKETCH_AND_THEOREMS_TO_PROVE="""
You are a Lean 4 expert. Your goal is to write a proof in Lean 4, according to the given proof sketch, using the supplied theorems.

Proof sketch:
{proof_sketch}

Theorems:
{theorems_string}

Instructions:
1. You can assume that the theorems are correct and use them directly in your proof.
2. Do NOT modify the given theorems.
3. Do NOT prove the given theorems. 
4. Do NOT modify the given proof sketch steps. Simply apply the given theorems to complete the missing `sorry` steps.
5. Do NOT use `sorry` in your proof.
6. Do NOT include any imports or definitions or open statements.
7. Do NOT re-define the given theorems in your response.
8. Do NOT write a proof for any subgoal from scratch. ALWAYS use the supplied theorems.
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on natural number literals with UNDEFINED types, unless REQUIRED by the theorem statement. For example, do NOT allow literals like `1 / 3` or `2 / 5` or `1 - 3`. ALWAYS specify the types. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.
ALWAYS specify types when describing fractions. For example, ((2 : ℝ) / 3) or ((2 : ℚ) / 3) instead of (2 / 3). Do this everywhere EXCEPT the given theorem statement.
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on variables of type natural numbers (Nat or ℕ), unless REQUIRED by the theorem statement. For example, do NOT allow expressions like (a-b) or (a/b) where a, b are of type ℕ. ALWAYS cast the variables to a suitable type (ℤ, ℚ or ℝ) when performing arithmetic operations. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.

Your answer should be a single Lean 4 block containing the completed proof for the given theorem.
"""

################################################################
#                   Error Correction Prompts
################################################################

MISSING_INFORMAL_PROOF_TAG_ERROR_PROMPT = """Your response did not contain valid <informal_proof> tags. Please provide your response again with the informal proof properly enclosed in <informal_proof></informal_proof> tags."""

MISSING_LEAN_CODE_BLOCK_ERROR_PROMPT = """Your response did not contain Lean code in valid ```lean``` tags. Please provide your response again with the Lean code properly enclosed in ```lean``` code blocks."""

THEOREM_SIGNATURE_MISMATCH_PROMPT = """The provided proof contains a Lean code block with a theorem signature that does not match the given theorem signature. Please provide your response again with the Lean code block properly matching the given theorem signature.
Provided Proof:
{proof}.

ERROR: The theorem statement in the proof does not match the given theorem statement:
{theorem}

Signature for the provided proof: {proof_signature}
Signature for the given theorem: {theorem_signature}
"""

CHECK_FOR_LEAN_ERRORS_PROMPT="""
You are an expert Lean 4 proof assistant. 
Your goal is to check if the given statement is provable in Lean 4.

Statement:
{theorem_statement}

Instructions:
1. Verify there are no technical issues such as type mismatches, truncation errors, casting problems, or other Lean 4-specific subtleties.
2. You should ESPECIALLY flag truncation and implicit type assumptions in natural numbers. FLAG inconsistencies like `1 / 2` or `2 - 3` without type specifications. Such statements MUST have the type specified.
3. The given theorem has passed a syntax check, but flag malformed definitions that should be in let statements. Example: `theorem h0 (S := ∑ i : Fin n, i) : S = n*(n+1)/2 := sorry` should be flagged.
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on natural number literals, or literals with UNDEFINED types, unless REQUIRED by the theorem statement. For example, do NOT allow literals like `1 / 3` or `2 / 5` or `1 - 3`. ALWAYS specify the types. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.
ALWAYS specify types when describing fractions. For example, ((2 : ℝ) / 3) or ((2 : ℚ) / 3) instead of (2 / 3). Do this everywhere EXCEPT the given theorem statement.
IMPORTANT INSTRUCTION: Do NOT, under ANY circumstances, allow division and subtraction operations on variables of type natural numbers (Nat or ℕ), unless REQUIRED by the theorem statement. For example, do NOT allow expressions like (a-b) or (a/b) where a, b are of type ℕ. ALWAYS cast the variables to a suitable type (ℤ, ℚ or ℝ) when performing arithmetic operations. AVOID natural number arithmetic UNLESS NEEDED by the theorem statement.

Assumptions:
You do NOT have to assess the mathematical validity of the statement or give a proof. 

Report your answer as either:
• YES - if the statement is well-formed and provable in Lean 4
• NO - if the statement is ill-formed.

Also provide a brief justification for your decision in <justification></justification> tags, adding details about why the statement is correct or incorrect.
If it is incorrect, also provide a description of how the error can be corrected. 
"""

NO_THEOREM_STATEMENT_PROMPT="""The provided code block did not contain a complete Lean 4 theorem statement. Your previous response only contained imports or incomplete code.

Please provide a complete response that includes the full Lean 4 theorem statement with:
1. The complete theorem signature (name, parameters, and type) as provided in the question.
2. The theorem body

Make sure to:
- Include the complete theorem statement from your previous response
- Do NOT include any import statements
- Enclose the Lean 4 theorem in ```lean4``` triple backticks

Your response should contain the complete theorem that was intended in your previous message."""
