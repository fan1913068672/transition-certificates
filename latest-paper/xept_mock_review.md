# Mock Review Report
> **Target Venue:** Elsevier CAS Journal (Formal Verification / CPS) &nbsp;·&nbsp; **Overall Prediction:** Major Revision &nbsp;·&nbsp; **Date:** 2026-06-08

---

## Score Summary

| Dimension | R1 (Objective) | R2 (Strict) | R3 (Friendly) |
|-----------|:---------:|:---------:|:---------:|
| Novelty | 3/5 | 3/5 | 4/5 |
| Technical Soundness | 3/5 | 2/5 | 4/5 |
| Experimental Evaluation | 2/5 | 2/5 | 3/5 |
| Presentation | 3/5 | 2/5 | 3/5 |
| Overall | Major Revision | Major Revision | Minor Revision |
| Confidence | 4/5 | 4/5 | 4/5 |

---

## Reviewer 1 -- Objective
> Confidence: 4/5

**Summary**

This paper proposes a verification framework for LTL properties over discrete-time dynamical systems with continuous state spaces. The central contribution is a decomposition of LTL verification into two certificate types: transition safety certificates and transition persistence certificates, tied to the transition-based acceptance condition of a nondeterministic Büchi automaton (NBA). Safety certificates establish unreachability of accepting source states; persistence certificates establish that individual accepting transitions fire only finitely often. Certificate synthesis is automated via CEGIS using Z3 (polynomial templates) and dReal combined with gradient descent (neural templates). Benchmarks are reported on 1D/2D Kuramoto oscillators and a two-room temperature control model.

**Strengths**

1. The transition-level decomposition is conceptually clean and well-motivated. The disjunctive structure of Theorem 3.7 -- find either a safety or persistence certificate per accepting source state -- provides genuine flexibility over uniform approaches.

2. For the 2D Kuramoto model and two-room temperature benchmark, the proposed framework succeeds where the closure certificate baseline times out. This is a concrete empirical improvement with explicit timing data.

3. Providing both polynomial and neural template paths broadens practical applicability. The use of ReLU activation for safety certificates and squared activation for persistence certificates is principled, motivated by the respective non-negativity requirements.

4. The paper is forthright about the 1D Kuramoto case where the baseline (0.37s) outperforms the proposed method (1.76s), which demonstrates scientific integrity.

**Weaknesses**

1. <span style="color:#dc2626">**[Major]**</span> **Inaccurate characterization of the search space reduction.** The paper claims a reduction "from X×X×Q×Q to X×Q×Q." This is incorrect. Transition safety certificates B_k are defined over X×Q; transition persistence certificates B_{k,l} are defined over X alone. Neither lives in X×Q×Q. The claim as written understates the improvement for persistence certificates while suggesting an X×Q×Q domain that neither certificate type actually occupies. A precise per-type characterization is required.

2. <span style="color:#dc2626">**[Major]**</span> **Unresolved inconsistency in the two-room temperature case study.** The paper states that both safety and persistence certificates are "simultaneously synthesized" for q_1. However, the comparison section confirms that q_1 is immediately reachable from initial conditions (via the edge (q_0, {a}, q_1) combined with L(X_0) = {a}). No valid safety certificate for q_1 can exist given this reachability. The appendix provides only persistence certificate weights for this benchmark; no safety certificate is listed. The paper must explicitly acknowledge this, state that safety certification of q_1 is infeasible, and clarify what the neural synthesis time of 1461.91s actually covers.

3. <span style="color:#dc2626">**[Major]**</span> **Template degree asymmetry confounds the 1D Kuramoto comparison.** The transition certificate uses a degree-4 polynomial template; the closure certificate baseline uses a degree-2 template. The paper attributes the slower performance (1.76s vs 0.37s) solely to dReal overhead for the transcendental sin function. This is a plausible but incomplete explanation that does not account for the increased template complexity. A matched-degree comparison or an explicit acknowledgment of this confound is required.

4. <span style="color:#dc2626">**[Major]**</span> **Scalability is not empirically validated.** All three benchmarks have state dimension at most 2. The central claim -- that transition certificates scale better because they avoid the |X×X| factor -- is never demonstrated on a system where this factor actually matters. At minimum, one benchmark with dimension ≥ 3 is needed.

5. <span style="color:#d97706">**[Minor]**</span> **Completeness of the persistence certificate framework is not discussed.** Since B_{k,l} is a function of x alone and bounds enabling-region visits rather than actual transition firings, the certificate may fail to exist even when the accepting transition fires only finitely often (e.g., if the label region is visited infinitely often but the automaton is never in state q_k during those visits). This limitation should be acknowledged.

6. <span style="color:#d97706">**[Minor]**</span> **CEGIS termination is not analyzed.** No discussion of when the loop is guaranteed to terminate, or how to distinguish "no certificate of this template class exists" from "the loop has not converged."

**Questions for Authors**

1. For the two-room temperature neural experiment: was safety certificate B_1 successfully synthesized? If not, what does the 1461.91s cover? Please clarify and revise the simultaneous synthesis claim accordingly.
2. Is CEGIS guaranteed to terminate for fixed polynomial degree when a certificate of that degree exists? How should a practitioner choose the template degree?
3. The search space bound X×Q×Q does not match the actual certificate domains X×Q and X. Can the authors provide a precise, per-type comparison with closure certificates?

---

## Reviewer 2 -- Strict
> Confidence: 4/5

**Summary**

The manuscript proposes a verification framework for LTL properties over discrete-time dynamical systems with continuous state spaces. The key contribution is a decomposition of the verification problem into two certificate types operating on accepting transitions of the NBA. While the transition-level decomposition is a sound idea and the theoretical development appears correct, I have substantial concerns regarding experimental integrity, completeness, and rigor. The paper requires major revision.

**Strengths**

1. The transition-based decomposition is a principled refinement over closure certificates, and the soundness theorem (Theorem 3.7) is clearly stated and apparently correct.

2. The dual synthesis pipeline (polynomial SMT + neural gradient descent) covers a broader class of certificate functions than purely algebraic approaches and demonstrates genuine engineering effort.

**Weaknesses**

1. <span style="color:#dc2626">**[Major]**</span> **Logical inconsistency in the two-room temperature case study.** The paper states that for q_1 ∈ Acc^start, it "simultaneously synthesizes" safety certificate B_1 and persistence certificate B_{1,1}. Yet the same paper confirms (in Section 5.3) that q_1 is immediately reachable from the initial product state via (q_0, {a}, q_1), since L(X_0) = {a} for all x_0 ∈ X_0 = [21,24]². A valid safety certificate for q_1 requires B_k(x, q_1) < 0 for reachable states, while simultaneously requiring B_k(x_0, q_0) ≥ 0 at initial states. Given that q_1 is reachable in one step from all initial states, these conditions are mutually contradictory. The appendix provides only persistence certificate weights for this benchmark -- no safety certificate parameters are listed anywhere in the paper. This is a direct logical inconsistency between the experimental claims and the theoretical framework, and must be resolved with a precise, state-by-state account of what was synthesized.

2. <span style="color:#dc2626">**[Major]**</span> **Theorem 3.7 imposes an unacknowledged uniformity constraint.** For each q_k ∈ Acc^start, Theorem 3.7 requires a uniform choice: either a single safety certificate for q_k, or persistence certificates for ALL outgoing accepting transitions from q_k. Mixed strategies -- safety for one accepting transition from q_k, persistence for another -- are not permitted under the current formulation. This restricts the class of verifiable systems and must be stated explicitly as a limitation. The paper currently gives no indication that this constraint exists, which may lead readers to believe the framework is more general than it is.

3. <span style="color:#dc2626">**[Major]**</span> **The 1D Kuramoto comparison is methodologically confounded.** The closure certificate baseline uses a degree-2 polynomial template while the proposed method uses a degree-4 polynomial template (plus automaton-state terms). The runtime difference (0.37s baseline vs 1.76s proposed) is attributed solely to dReal overhead for handling the transcendental sin function. This explanation is insufficient: the template complexity asymmetry is at least as plausible a confound. A rigorous comparison requires either matching template degrees or isolating each factor through ablation.

4. <span style="color:#dc2626">**[Major]**</span> **Scalability claims are not empirically substantiated.** The paper argues that closure certificates become infeasible due to the |X×X| factor as state dimension grows. All three benchmarks have dimension at most 2. The claimed efficiency advantage at higher dimensions is entirely extrapolated. The paper must include at least one benchmark with dimension ≥ 3 to provide empirical support for this central claim.

5. <span style="color:#d97706">**[Minor]**</span> **No completeness analysis.** Under what conditions is a transition certificate guaranteed to exist? For polynomial templates, connections to Positivstellensatz or SOS completeness exist under degree bounds and compactness conditions. For neural templates, no theory applies and the CEGIS loop is heuristic. The paper provides no guidance on when to interpret CEGIS failure as "no certificate exists" versus "template insufficiency."

6. <span style="color:#d97706">**[Minor]**</span> **Persistence certificate conservatism from state-independence is unquantified.** B_{k,l}(x) bounds the number of times the label-enabling region is visited, not the number of times the automaton actually fires the accepting transition. When the automaton is rarely in state q_k, this over-approximation may be severe. The paper acknowledges the design rationale but never characterizes the completeness gap this introduces.

**Questions for Authors**

1. For the two-room temperature neural experiment: was safety certificate B_1 for q_1 successfully synthesized? If not, please state explicitly that only the persistence certificate was found and revise all claims of "simultaneous synthesis of both types" accordingly.
2. Is the per-state uniformity constraint in Theorem 3.7 a fundamental requirement for soundness, or is it an artifact of the proof technique? Could a mixed-strategy extension (safety for some transitions from q_k, persistence for others) be sound?
3. Can the authors provide at least one benchmark with state dimension ≥ 3 to validate the scalability argument?

---

## Reviewer 3 -- Friendly
> Confidence: 4/5

**Summary**

This paper presents a clean and well-motivated framework for LTL verification via transition certificates. The key idea -- decomposing the verification obligation into per-transition certificates rather than a single monolithic certificate over the full product state space -- is elegant and maps naturally onto the structure of Büchi automaton acceptance. The combination of safety and persistence certificates addresses complementary system behaviors, the CEGIS pipeline is practically grounded, and the theoretical development in Section 3 is rigorous. The work makes a genuine contribution to certificate-based formal verification and I broadly support publication after addressing the concerns below.

**Strengths**

1. The transition-level decomposition is genuinely novel. Separating reachability arguments (safety certificates, domain X×Q) from liveness arguments (persistence certificates, domain X) maps naturally onto LTL semantics and offers a principled reduction in search space compared to closure certificates (domain X×X×Q×Q).

2. The paper is forthright about the 1D Kuramoto case where the proposed method (1.76s) is slower than the baseline (0.37s). Explicitly reporting this in Table 1 and discussing it in the text demonstrates scientific integrity.

3. The activation function design is well-motivated: ReLU for safety certificates (where a barrier-like non-negativity condition must hold) and squared activations for persistence certificates (where non-negativity is needed for the ranking argument) reduces the number of constraints that dReal must verify.

4. The G¬a motivating example in Section 3 effectively illustrates the complementarity of safety and persistence strategies, giving readers the intuition needed before the formal development.

**Weaknesses**

1. <span style="color:#dc2626">**[Major]**</span> **The "greater flexibility than state-triplet methods" claim rests on a single example.** Only the two-room temperature model demonstrates this advantage empirically. To support a journal-level claim of improved flexibility, the paper should either (a) formally characterize the class of LTL properties or product-system configurations for which transition certificates succeed but state-triplet methods structurally fail, or (b) provide two or more additional benchmarks that illustrate this boundary.

2. <span style="color:#d97706">**[Minor]**</span> **Related work is embedded in the introduction rather than given a dedicated section.** For a journal submission in formal verification, a standalone related work section with a structured comparison (method, certificate domain, applicable LTL fragment, completeness, synthesis approach) is expected and would better position the contribution.

3. <span style="color:#d97706">**[Minor]**</span> **The large polynomial-vs-neural time gap warrants deeper analysis.** The gap between polynomial synthesis (2.68s) and neural synthesis (1461.91s) for the two-room temperature case is explained descriptively ("1000 gradient iterations per CEGIS round," "simultaneous synthesis of two certificates"), but no breakdown by CEGIS phase is provided. A brief analysis of how many CEGIS iterations were performed and what fraction of time was spent on gradient descent vs. dReal queries would help practitioners calibrate expectations and identify bottlenecks.

4. <span style="color:#d97706">**[Minor]**</span> **No benchmark where neural certificates are necessary.** In every reported benchmark, polynomial certificates also succeed. The neural template path is presented as a practical contribution, but no case is shown where polynomial templates fail and neural templates succeed. Adding such a case, or explicitly scoping the neural results as a proof-of-concept path for future nonlinear systems, would strengthen this part of the paper.

**Questions for Authors**

1. Is there a formal sense in which the set of LTL properties verifiable by transition certificates strictly contains those verifiable by state-triplet certificates, or is the advantage primarily one of practical tractability?
2. For the two-room temperature neural experiment, what is the breakdown of the 1461.91s between gradient descent training and dReal verification calls? How many CEGIS iterations were required?

---

## Verification

| # | Source | Claim | Verdict | Note |
|---|--------|-------|---------|------|
| 1 | R1-W2, R2-W1 | "Two-room temperature: q_1 reachable, so safety cert B_1 cannot exist; yet paper claims simultaneous synthesis of both types" | <span style="color:#16a34a">**Valid**</span> | casestudy.tex confirms L(X_0)={a} and automaton edge (q_0,{a},q_1) exists, making q_1 immediately reachable. Appendix lists only persistence cert weights -- no safety cert parameters appear anywhere |
| 2 | R1-W1, R2-W2 | "Search space stated as X×Q×Q but actual certificate domains are X×Q (safety) and X (persistence)" | <span style="color:#16a34a">**Valid**</span> | casestudy.tex comparison section: "search space of transition certificates is at most X×Q×Q." Definitions in method.tex explicitly give B_k: X×Q→R and B_{k,l}: X→R. X×Q×Q matches neither |
| 3 | R1-W3, R2-W3 | "1D Kuramoto comparison confounded: degree-4 transition cert template vs degree-2 closure cert template" | <span style="color:#16a34a">**Valid**</span> | casestudy.tex template: sum c_i x^i (i=0..4) plus q^j (j=1..3) plus cross-term = 9 monomials; closure cert uses linear template per original paper. Asymmetry unacknowledged |
| 4 | R2-W2 | "Theorem 3.7 imposes a per-state uniformity constraint (all-safety or all-persistence per q_k); mixing not possible" | <span style="color:#16a34a">**Valid**</span> | method.tex Theorem 3.7: conditions (1) and (2) are stated as alternatives per q_k, not per individual transition. The paper never discusses this constraint as a limitation |
| 5 | R1-W6, R2-W5 | "CEGIS termination and completeness not analyzed anywhere in the paper" | <span style="color:#16a34a">**Valid**</span> | synthesis.tex has no termination discussion for either polynomial or neural variants. No Positivstellensatz connection mentioned, no acknowledgment that CEGIS failure is ambiguous |
| 6 | R1-W5, R2-W6 | "Persistence cert state-independence introduces completeness gap not acknowledged" | <span style="color:#16a34a">**Valid**</span> | I_{k,l}(x)=1 forces decrease even when automaton is not in state q_k -- bounding enabling-region visits is more conservative than bounding actual transition firings. Acknowledged implicitly in method.tex ("necessary condition") but completeness implications not stated |
| 7 | R3-W1 | "Flexibility advantage over state-triplet demonstrated by only one benchmark" | <span style="color:#16a34a">**Valid**</span> | Only the two-room temperature case shows state-triplet failure. No formal characterization of the class of systems where the advantage holds |
| 8 | R3-W4 | "Grammar error in abstract: 'guarantee accepting transitions take finitely often'" | <span style="color:#d97706">**Already Fixed**</span> | Corrected in current session to "guarantee accepting transitions occur only finitely often" |

---

## Action Plan

<span style="color:#dc2626">**Must Fix**</span> -- consensus across reviewers, direct correctness or integrity issues

- [ ] **Two-room temperature experiment narrative (casestudy.tex, Section 5.2):** State explicitly that q_1 is immediately reachable from the initial product state (via L(X_0)={a} and edge (q_0,{a},q_1)), so a valid safety certificate for q_1 cannot exist. Remove or heavily qualify the phrase "simultaneously synthesize both types." Clarify what the neural synthesis time of 1461.91s covers. Revise to read: only the persistence certificate B_{1,1} verifies the LTL specification for q_1; the safety branch is structurally infeasible here.
- [ ] **Theorem 3.7 uniformity constraint (method.tex, Section 3.3):** Add a remark after Theorem 3.7 stating that the framework requires a uniform choice (all-safety or all-persistence) per source state q_k -- mixing safety for some outgoing transitions and persistence for others from the same q_k is not supported. Discuss whether this is a soundness requirement or a proof artifact and flag it as a direction for future work.
- [ ] **Search space claim correction (casestudy.tex, Comparison section):** Replace "reduces the search space from X×X×Q×Q to X×Q×Q" with a precise per-type statement: transition safety certificates operate over X×Q; transition persistence certificates operate over X. Compare these directly to the closure certificate domain X×X×Q×Q.
- [ ] **Add higher-dimensional benchmark:** Include at least one experiment with state dimension ≥ 3 to support the scalability claim. The argument that closure certificates scale quadratically in |X| cannot rest on 2D benchmarks alone.

<span style="color:#d97706">**Should Fix**</span> -- important for scientific rigor and journal-level completeness

- [ ] **1D Kuramoto template degree asymmetry (casestudy.tex, Section 5.1):** Explicitly acknowledge that the transition certificate uses a degree-4 polynomial template while the closure certificate uses degree-2. Note this as a confound alongside the dReal overhead explanation. Either add a matched-degree comparison or clearly state that the 1.76s vs 0.37s gap reflects both template complexity and solver differences.
- [ ] **Flexibility characterization (introduction.tex / method.tex):** Provide a formal or semi-formal characterization of the class of LTL properties or product-system structures for which transition certificates succeed but state-triplet methods structurally fail (not just a single illustrative example). This is needed to justify the journal-level claim of "greater flexibility."
- [ ] **CEGIS completeness discussion (synthesis.tex, Section 4):** Add a paragraph on convergence and completeness. For polynomial templates on semialgebraic systems with SOS relaxations, note the Positivstellensatz connection under fixed degree bounds. For neural templates, explicitly state that convergence is heuristic and CEGIS failure is ambiguous between template insufficiency and certificate non-existence.
- [ ] **Persistence certificate conservatism (method.tex, Section 3.2):** After the I_{k,l}(x) "necessary condition" remark, add a sentence acknowledging that bounding enabling-region visits is a conservative over-approximation: persistence certificates may fail to exist even when the accepting transition fires only finitely often (if the label-enabling region is visited infinitely often while the automaton is never in state q_k during those visits).

<span style="color:#6b7280">**Optional**</span> -- would strengthen the paper but not blocking

- [ ] **Standalone related work section:** Extract the related work from the end of the introduction into a dedicated section with a comparison table (method, certificate domain, LTL fragment, completeness, synthesis approach).
- [ ] **Neural-only benchmark:** Add or identify a case where polynomial templates are insufficient and neural templates succeed, to justify the neural synthesis contribution as more than a proof of concept.
- [ ] **CEGIS timing breakdown for two-room temperature:** Report the number of CEGIS iterations and the split between gradient descent time and dReal query time for the 1461.91s result, to help practitioners understand where the bottleneck lies.
