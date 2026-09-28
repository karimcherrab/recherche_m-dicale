# 8. Privacy–Utility–Fairness Trade-offs

## 8.1 Overall Clinical Performance and Subgroup Utility

Clinical utility cannot be represented by a single test-set score. Four dimensions should be distinguished: overall discrimination, performance within clinical or demographic subgroups, calibration of predicted probabilities, and performance at the decision threshold used in practice. ROC-AUC measures ranking across thresholds, but does not show whether a protected model retains adequate sensitivity for a rare disease, acceptable specificity for screening, or reliable probabilities for triage. PR-AUC can be informative for uncommon outcomes, although it is prevalence-sensitive and should be interpreted alongside the reported outcome prevalence. The Brier score assesses overall probabilistic accuracy, whereas calibration error and reliability curves assess calibration more directly. Clinical acceptability therefore depends not only on average ranking performance, but also on whether outputs remain trustworthy and actionable in the populations in which decisions are made.

Aggregate performance can conceal unequal losses just as aggregate privacy metrics conceal highly exposed patients. A mitigation may preserve population-level ROC-AUC while reducing sensitivity for a rare diagnosis, a small demographic subgroup, or an intersectional population that contributes little weight to the average. Conversely, a modest overall loss may coexist with clinically important subgroup effects. Reporting only the protected model's mean score therefore cannot establish equitable utility. Studies should provide subgroup sample sizes, outcome prevalence, point estimates, and confidence intervals because estimates for small populations may be unstable. They should also distinguish pre-existing performance gaps from losses introduced by protection by reporting baseline utility and the absolute change for every prespecified subgroup. When many or intersectional groups are evaluated, uncertainty estimates should be accompanied by multiplicity-aware inference or appropriately specified hierarchical analyses.

Knolle et al. provide the most direct clinical evidence connecting stronger privacy protection with predictive performance [3]. In PTB-XL and EMBED, models trained at different levels of record-level `(\varepsilon,\delta)`-DP were evaluated using macro-average AUC on unseen data alongside patient-level membership risk. Stronger protection was accompanied by lower patient-level MIA risk and declining macro-average diagnostic performance, particularly at stricter privacy levels. This demonstrates a system-level privacy–utility trade-off in two clinical modalities. Its equity implications remain unresolved, however: the DP experiments did not provide a complete subgroup-stratified analysis of diagnostic utility at every privacy level or evaluate calibration and fixed clinical thresholds. Nor did the study jointly relate patient-level privacy reductions to subgroup-stratified changes in diagnostic utility. The results therefore show that overall utility and patient-level privacy change together, but not how privacy gains and utility losses were distributed across the same clinical populations.

Evidence from subgroup and long-tail studies shows why equal utility loss should not be presumed. Bagdasaryan et al. found larger DP-SGD accuracy reductions for underrepresented or more complex groups, including darker-skinned faces, tweets labeled as African-American English, and underrepresented classes in iNaturalist [47]. These non-clinical experiments assess accuracy rather than calibration or threshold-specific clinical harm, but show that a small aggregate loss can coexist with substantial subgroup degradation. The authors traced this pattern partly to clipping and noise acting disproportionately on uncommon or complex examples. Kulynych et al. indicate that limiting privacy disparity can involve substantial accuracy costs [7], while Truex et al. connect data skewness with minority-class membership vulnerability under different DP configurations [30]. Hu et al. further show that improving long-tail accuracy does not necessarily eliminate tail-class privacy vulnerability under DP [31]. Privacy equality and utility equality are therefore distinct objectives: improvement in one does not establish improvement in the other.

A valid comparison should evaluate baseline and protected models on identical patient-disjoint test sets and report overall and subgroup ROC-AUC or PR-AUC, sensitivity and specificity at prespecified clinical thresholds, calibration error or Brier score, worst-group performance, and confidence intervals. For each group `g`, the absolute change can be written as

\[
\Delta U_g = U_g^{\mathrm{protected}}-U_g^{\mathrm{baseline}}.
\]

For measures oriented so that higher values indicate better performance, negative values of `\Delta U_g` represent utility loss. Lower-is-better measures, such as calibration error and Brier score, should be reoriented or reported separately to avoid reversing this interpretation. Thresholds should remain fixed from the clinical specification or be selected on an independent validation set; retuning them for every protected model can obscure changes in deployable performance. Repeated runs are necessary because private optimization adds training randomness. Comparisons should, where possible, match overall utility, privacy protection, or computational cost. Privacy-budget matching is meaningful only when the protected unit, adjacency definition, `\delta`, accountant, sampling scheme, and optimization steps are aligned [8].

A mitigation cannot be considered clinically acceptable merely because it reduces aggregate privacy leakage while preserving aggregate accuracy; its effects on subgroup utility, calibration, and clinically relevant operating thresholds must also be established. Clinical evidence shows that stronger DP can lower patient-level membership risk while changing overall diagnostic performance, whereas broader evidence warns that underrepresented groups may bear disproportionate accuracy losses. It does not yet establish equitable preservation of clinical utility. Overall and worst-group performance are therefore necessary, but they do not reveal whether those receiving the greatest privacy benefit incur the greatest utility cost or whether privacy gaps narrow. Section 8.2 consequently examines absolute risk reduction and disparity reduction as separate outcomes.

## 8.2 Absolute Risk Reduction and Disparity Reduction

Privacy mitigation can pursue two related but independent objectives: reducing the level of exposure faced by patients and reducing differences in exposure between patient groups. An intervention may lower average risk while leaving the gap between groups unchanged or even widening it. Conversely, a disparity may shrink for an unacceptable reason, such as increased risk in the previously less-exposed group. “Risk reduction” and “fairness improvement” should therefore not be used interchangeably [3,7]. Let `\widehat R_p^{(0)}` and `\widehat R_p^{(1)}` denote the estimated risk for patient `p` before and after protection, respectively. The statistic may be a patient-level MIA AUC, a calibrated likelihood-ratio statistic, or another attack-specific measure; it is not necessarily a posterior probability. Comparisons are meaningful only when both models use the same estimand, threat model, attack family, and direction of the risk scale, together with comparable calibration and mechanism-matched reference-model construction.

The estimated absolute benefit received by a patient can be defined as

\[
\widehat B_p=\widehat R_p^{(0)}-\widehat R_p^{(1)}.
\]

Positive, zero, and negative values indicate reduced exposure, no measured benefit, and increased exposure, respectively. Benefit alone is insufficient because a large reduction may still leave a patient at an unacceptable residual risk `\widehat R_p^{(1)}`. Studies should therefore report both the distribution of `\widehat B_p` and

\[
\Pr\!\left(\widehat R_p^{(1)}\geq\tau\right),
\]

for a threshold `\tau` specified before examining the protected model. The baseline high-risk set should likewise be fixed as

\[
\mathcal H_0=\left\{p:\widehat R_p^{(0)}\geq\tau\right\},
\]

after which the distribution of `\widehat B_p` within `\mathcal H_0` can be evaluated. Redefining the top 1% after mitigation does not measure absolute improvement because every model has a highest percentile, although it may still describe the composition of the residual tail. Relative reduction should be reported only when the risk measure has a meaningful zero and ratio-scale interpretation. It is not directly interpretable for MIA AUC, whose no-discrimination value is 0.5, or for a likelihood ratio, whose neutral value is 1. Absolute differences should therefore remain primary [3,15].

Group disparity requires separate quantities. For group `g`, define

\[
\mu_g^{(k)}=\mathbb E\!\left[R_p^{(k)}\mid G_p=g\right],\qquad k\in\{0,1\},
\]

and group benefit

\[
B_g=\mu_g^{(0)}-\mu_g^{(1)}.
\]

For two prespecified groups, signed and absolute mean gaps can be written as

\[
D_{\mathrm{signed}}^{(k)}=\mu_a^{(k)}-\mu_b^{(k)},
\qquad
D_{\mathrm{mean}}^{(k)}=\left|\mu_a^{(k)}-\mu_b^{(k)}\right|,
\]

with disparity reduction

\[
\Delta D_{\mathrm{mean}}=D_{\mathrm{mean}}^{(0)}-D_{\mathrm{mean}}^{(1)}.
\]

The signed gap identifies which group bears greater exposure, whereas the absolute gap records its magnitude. For more than two groups, prespecified pairwise contrasts, the maximum–minimum range, and worst-group risk may be reported. Means can nevertheless conceal extreme exposure. Tail disparity should also be measured through

\[
\pi_g^{(k)}(\tau)=\Pr\!\left(R_p^{(k)}\geq\tau\mid G_p=g\right),
\]

together with group-specific 95th and 99th percentiles. Importantly, `P(G=g\mid R\geq\tau)`, which describes the composition of the risk tail, is not equivalent to `P(R\geq\tau\mid G=g)`, the probability that a member of group `g` enters that tail. This distinction is essential when groups differ substantially in size [7,30,31].

Four post-mitigation patterns illustrate why average risk, tail risk, disparity, and non-harm must be assessed jointly. A positive `\Delta D_{\mathrm{mean}}` is insufficient unless exposure decreases in the relevant groups and the proportion of patients with `\widehat B_p<0` is also reported.

| Post-mitigation pattern | Overall risk | Disparity | Interpretation |
|---|---:|---:|---|
| Similar reduction across groups | Decreases | Largely unchanged | General protection without clear gap reduction |
| Larger reduction for the more-exposed group | Decreases | Narrows | Preferred joint outcome |
| Larger reduction for the less-exposed group | Decreases | Widens | Aggregate improvement with worsening inequality |
| Increased risk in the previously less-exposed group | May decrease, remain stable, or increase | May apparently narrow | Equality through harm; unacceptable |

Current evidence supports some components of this framework but not the full joint claim. Knolle et al. show clinically that stronger record-level DP shifts patient-level MIA-risk distributions downward, but do not provide a complete paired analysis of `\widehat B_p` or changes in privacy gaps across all clinical subgroups [3]. Song and Mittal demonstrate heterogeneous sample-specific privacy risk and show why an aggregate defense result cannot determine individual benefit [15]. Kulynych et al. establish that subgroup membership-inference vulnerability can differ and emphasize statistically reliable estimation for small groups [7]. Truex et al. and Hu et al. further show, in non-clinical class-level settings, that skewed or long-tailed groups may retain disproportionate vulnerability under privacy mechanisms [30,31]. Current evidence thus supports the possibility of reducing aggregate privacy risk, but provides limited clinical evidence that mitigations simultaneously reduce patient-level residual risk and disparities across clinical populations.

A minimum evaluation should compare the same patients before and after mitigation while holding the membership hypothesis, protected unit, vulnerability scale, threat model, and operating point constant. The attacker should be retrained or recalibrated with mechanism-matched references. Reports should include `\widehat B_p`, `B_g`, residual vulnerability, mean and tail gaps, the fraction with `\widehat B_p<0`, and results within the fixed baseline high-risk set. Uncertainty must reflect both audit-model repetitions and variation across patients, with multiplicity correction for numerous subgroup contrasts. Sensitivity analyses should vary attacks, seeds, thresholds, and reference construction; incompatible quantities such as MIA AUC, posterior probability, and reconstruction error must not be pooled. Section 8.3 follows the initially exposed patients and examines unintended redistribution.

## 8.3 Protection of High-Risk Patients and Unintended Effects

Protecting a high-risk patient requires more than lowering average risk. Starting from

\[
\mathcal H_0=\left\{p:\widehat R_p^{(0)}\geq\tau\right\},
\]

evaluation should determine each patient's estimated benefit,

\[
\widehat B_p=\widehat R_p^{(0)}-\widehat R_p^{(1)},
\]

whether residual risk crossed the prespecified threshold, and whether the conclusion persists across compatible attacks, using an attack-specific threshold and mechanism-matched calibration [3,15]. Partial protection means that estimated risk decreased but remained above `\tau`; no measured benefit means that no reliable reduction was detected; and an adverse response means that risk increased. Adequate protection may conservatively require the upper confidence limit `U_p^{(1)}` for residual risk to fall below `\tau`. When a statistically and practically meaningful reduction is also claimed, the lower confidence limit for `\widehat B_p` should exceed a prespecified minimum benefit `\delta_{\min}`:

\[
U_p^{(1)}<\tau,
\qquad
L_p(\widehat B_p)>\delta_{\min}.
\]

If `\mathcal H_0` is constructed from point estimates, it should be described as a baseline screening set. A more conservative confirmed high-risk set would require `L_p^{(0)}\geq\tau`.

Patient trajectories can be summarized using the post-mitigation set

\[
\mathcal H_1=\left\{p:\widehat R_p^{(1)}\geq\tau\right\}.
\]

The resulting transitions are more informative than redefining the top 1% after protection, because percentile-based lists necessarily contain the same proportion of patients and may show tail replacement even when absolute exposure has declined.

| Transition state | Definition | Interpretation |
|---|---|---|
| Persistently high risk | `\mathcal H_0\cap\mathcal H_1` | Remained above the threshold |
| Protected | `\mathcal H_0\setminus\mathcal H_1` | Left the high-risk region |
| Newly high risk | `\mathcal H_1\setminus\mathcal H_0` | Entered the region after mitigation |
| Persistently lower risk | `(\mathcal H_0\cup\mathcal H_1)^c` | Remained below the threshold |

These four transitions describe point-estimate trajectories. Patients whose uncertainty intervals overlap `\tau` should be reported separately as indeterminate rather than assigned definitively to the protected or newly high-risk categories.

Residual-risk assignments must be interpreted with estimation uncertainty. Knolle et al. showed that stronger DP shifted clinical patient-risk distributions downward, but did not establish that the same patients consistently exited the extreme tail across every protection level, attack, or model [3]. Jebreel et al. found that binary vulnerable sets defined at very low FPR could vary across training runs, even when continuous likelihood-ratio rankings were relatively more stable [26]. Appearance in `\mathcal H_1` may therefore represent genuine new exposure, a change in ordering, or estimation error near `\tau`. Studies should report confidence intervals and transition frequencies across runs. The indeterminate category prevents unstable boundary cases from being misclassified as protected or newly vulnerable.

Intervention can also change which patients are most exposed. The Privacy Onion Effect shows that memorization is relative to the surrounding training distribution: removing highly vulnerable examples may reveal another layer whose exposure was previously obscured [35]. It demonstrates that risk transfer is possible when intervention changes the training distribution, not that every mitigation creates new high-risk patients. Risk reduction means exposure falls without important new vulnerabilities; risk redistribution means targeted patients improve while others worsen; tail replacement means absolute risks decline but different patients occupy the highest ranks; and apparent transfer arises from uncertainty, calibration changes, or incompatible audits. Genuine redistribution requires paired increases for other patients on a comparable scale; altered ranks or low Jaccard overlap alone are insufficient.

Risk-targeted protection may create further harms. Downweighting rare patients could reduce their influence while degrading the model's ability to learn rare conditions; this potential utility harm requires direct evaluation. Small groups may additionally receive less stable risk estimates and apparently unequal benefits [7]. Individualized DP-SGD demonstrates feasible non-uniform privacy budgets, but uses prespecified assignments rather than audited clinical vulnerability [45]. REPS more directly uses estimated outlierness to guide differentially private synthetic-data generation [46]. However, it is record-level, concerns synthetic tabular data, depends on its risk scorer, and has not validated protection for multi-record patients in diagnostic models. Risk labels and personalized budgets may themselves be sensitive metadata: identifying someone as high risk could reveal that their data are unusual or associated with a rare condition. Unstable estimates may also produce inconsistent protection across retraining cycles.

Evaluation should report the proportions in all four definitive transition states and the indeterminate category, residual risk within `\mathcal H_0`, each `\widehat B_p` with uncertainty, and patients whose risk increased. Transitions should be tested across seeds, models, compatible attacks, mechanism-matched reference constructions, and attack-specific thresholds, using rank correlations and Jaccard overlap alongside absolute-risk changes. Results should be stratified by clinical and demographic groups and accompanied by subgroup utility changes. Uncertainty estimation should preserve correlations among records belonging to the same patient while also accounting for variation across target and reference-model repetitions, for example through a nested or model-aware bootstrap. Targeted protection should be compared with a uniform intervention matched under prespecified resource constraints; privacy-budget matching is interpretable only when the protected unit, adjacency definition, accountant, and contribution bounds are aligned. Protecting initially high-risk patients without transferring exposure or disproportionately degrading clinical utility is necessary but not sufficient for deployment. Section 8.4 therefore examines the computational cost and practical feasibility of conducting patient-level audits and implementing such protections at clinical scale.

## 8.4 Computational Cost and Practical Feasibility

Practical feasibility is not determined by GPU time alone. It includes repeated training, reference construction, calibration, queries, statistical precision, private optimization, storage, patient linkage, monitoring, governance of identifiers and scores, and expert review. The question is whether an audit can produce reliable and actionable patient-level conclusions within computational, data, governance, and time constraints.

Reliable patient-level auditing requires repeated observations under patient inclusion and exclusion. A first-order approximation of its cost is

\[
C_{\mathrm{audit}}
\approx
M_T C_T
+
M_R C_R
+
C_{\mathrm{scoring}}(M,N,A)
+
C_{\mathrm{calibration}},
\]

Here, `M_T` and `M_R` count target and reference models, `C_T` and `C_R` are their training costs, `N` is the number of evaluated units, and `A` is the number of attacks [2,3]. Knolle et al. trained 200 target models and reused them across patients, amortizing rather than repeating this cost per patient [3]. Leave-one-patient-out retraining scales poorly, whereas shared subsampling reuses models. Offline or one-shot attacks are cheaper but implement different adversarial experiments and may not provide enough counterfactual observations for patient-specific uncertainty or low-FPR estimation [2].

Reducing the number of models is not merely an engineering compromise. Too few models produce wide intervals, unstable threshold classifications, unreliable tails, and insufficient resolution at low FPR. Jälkö et al. warn that pooled efficient evaluations can hide per-sample FPR heterogeneity and introduce finite-population bias; their study remains a preprint [14]. Score storage can scale with the product of evaluated units and models, while checkpoints add model-size-dependent costs. Sufficient statistics can reduce storage if reproducibility is preserved. Influence or difficulty measures may prioritize detailed audits, but their efficiency does not make them privacy estimates.

Protection adds a second cycle. DP-SGD requires contribution clipping, noise, accounting, and tuning; its overhead depends on architecture, batching, implementation, and hardware [8]. Patient-level DP additionally samples patients, aggregates their records, handles unequal contribution sizes, and accounts under patient adjacency. Individualized DP-SGD manages heterogeneous sampling or clipping parameters [45]. Tuning may itself require accounting or separate validation data, and protected training must be followed by independent privacy and utility re-auditing.

Clinical institutions face constraints absent from benchmarks. Auditing requires stable identifiers to link records belonging to one patient while preventing patient leakage across target, test, and reference sets. Cross-institution linkage may be legally or technically restricted, and differences in devices, acquisition protocols, preprocessing, and population distributions can compromise reference-model validity. Rare and intersectional groups may be too small for precise estimation. Risk scores must also be secured because they may reveal unusual or sensitive clinical characteristics. An audit may become obsolete after changes to the dataset, architecture, preprocessing, or privacy mechanism. Re-audit triggers should therefore be specified prospectively rather than treating one result as permanent certification [3].

A tiered workflow can allocate resources more realistically, but aggregate screening must characterize system-level leakage rather than act as a gate that determines whether patient-level auditing occurs. A prespecified sentinel sample should undergo individual auditing even when aggregate results appear reassuring.

| Stage | Purpose | Relative cost | Main limitation |
|---|---|---:|---|
| System-level characterization | Measure aggregate leakage and model behavior | Low | Cannot rule out an extreme-risk tail |
| Sentinel patient-level audit | Audit a prespecified stratified sample, including rare and multi-record patients | Moderate–high | May still miss very rare vulnerabilities |
| Expanded patient-specific audit | Extend auditing to flagged strata and random controls | High | Requires repeated models and calibration |
| Targeted causal validation | Apply leave-one-patient-out checks to selected cases | Very high | Cannot readily cover the full population |
| Protected-model re-audit | Verify benefit and detect redistribution | High | Must remain independent of intervention selection |

The saving from tiering depends on the design. Shared target models amortize training across patients, whereas target-specific references or leave-one-patient-out retraining scale with the number audited. Tiering is therefore a prioritization strategy, not proof that unscreened patients are safe. Screening sensitivity must be validated against detailed audits in a prespecified stratified and random sample.

Reports should state the numbers of target and reference models, in/out observations, seeds, attacks, patients, records, and queries; compute time, peak memory, and storage; the lowest FPR estimable with prespecified precision; accounting and tuning costs; the fraction receiving detailed audit; and re-audit time after updates. Reliable protection must balance statistical rigor with clinical computational and organizational constraints. These constraints inform the design requirements developed in Section 9.
