#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
TACTIC_HINTS="""
`rfl`: Use when there are definitionally equal terms
`exact h`: Use when hypothesis `h` matches the goal exactly  
`assumption`: Searches context for exact match to goal  
`intro x`: Use for `∀` or `→` in goal; names the new hypothesis  
`intros`: Introduces multiple variables/hypotheses at once  
`cases h`: Breaks down inductive hypothesis `h` into constructors  
`obtain ⟨x, y, h⟩ := h'`: Destructures existentials and conjunctions  
`rcases`: Recursive cases for nested inductive structures
`apply h`: Use when `h : P → Q` and goal is `Q`  
`refine h ?_ ?_`: Like apply but with explicit placeholders  
`rw [h]`: Rewrites using equality `h : a = b` left-to-right  
`rw [← h]`: Rewrites right-to-left  
`simp`: Simplifies using simp lemmas; use `simp only [...]` for control  
`simp_all`: Simplifies goal and all hypotheses  
`nth_rw n [h]`: Rewrites only the nth occurrence
`constructor`: Splits conjunctive goals or applies inductive constructors  
`left`/`right`: Choose side of disjunction  
`split`: Splits iff into both directions or conjunctions  
`by_contra h`: Proof by contradiction; adds `¬goal` as `h`  
`push_neg`: Pushes negation inward through quantifiers/connectives
`ring`: Solves polynomial ring equations  
`linarith`: Linear arithmetic solver  
`norm_num`: Evaluates numeric expressions  
`positivity`: Proves positivity/nonnegativity goals  
`field_simp`: Simplifies field expressions (clears denominators)
`have h : P := proof`: Introduces intermediate result  
`suffices h : P by proof`: Reduces goal to proving `P`  
`show P`: Changes goal to definitionally equal `P`  
`calc`: Chain of equations/inequalities with justifications  
`conv => ...`: Enter conv mode for targeted rewriting
`induction x`: Structural induction on `x`  
`induction x using ind_principle`: Custom induction principle  
`induction' x with ...`: More flexible case naming  
`cases x`: Case split without induction hypothesis
`classical`: Enter classical mode locally  
`by_cases h : P`: Case split on decidability of `P`  
`use x`: Provide witness for existential goal  
`choose f hf using h`: Extract choice function from proof
`·`: Focus on next goal  
`focus`: Focus on first goal  
`all_goals`: Apply tactic to all goals  
`any_goals`: Apply tactic to any matching goal  
`swap`: Swap first two goals  
`omega`: Integer linear arithmetic  
`decide`: Decision procedure for decidable propositions  
`tauto`: Propositional tautology checker  
`simp_all only [eq_self_iff_true]`: Common finishing pattern  
`repeat`: Apply tactic repeatedly until failure
"""

GENERAL_HINTS="""
1. When dealing with inequalities, equalities and arithmetic operations like subtraction or division in `ℕ` (natural numbers), beware of truncation. Use `ℝ`, `ℚ` or `ℤ` when possible for arithmetic operations. Avoid using `ℕ` unless required by the theorem statement.
2. Be ESPECIALLY careful about implicit types while defining numeric literals. AVOID patterns like `0 - 1` or `1 / 2` without specifying the types.
3. ALWAYS specify types when dealing with numeric values to avoid ambiguities and unexpected behavior.
4. Use `simp only [specific_lemmas]` rather than bare `simp` to avoid unpredictable simplifications.
5. Use `rw [← lemma]` for reverse direction. When `rw` fails, try `conv => rhs; rw [lemma]` to target specific subterms. nth_rw n [lemma] to rewrite only the nth occurrence.
6. When `ring` fails on ring expressions, try `ring_nf` first to normalize, or cast to a concrete ring like `ℝ` where the tactic works better.
7. Apply `norm_num` for concrete arithmetic calculations and `norm_cast` to simplify type coercions systematically.
8. Use `by_contra h` for proof by contradiction, which introduces the negation of the goal as hypothesis `h`.
9. If you get a `no goals to be solved` error, it means that the previous tactics already solved the goal, and you can remove the subsequent tactics.
10. When proving theorems, ALWAYS write the proof in tactic mode, starting the proof with `:= by`.
11. Do NOT use `begin`, `end` blocks in your proof. This is invalid in Lean 4.
"""