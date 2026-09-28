# 9. Towards Mechanism-Aware, Risk-Adaptive Patient-Level Protection

Sections 4–8 established how patient-level vulnerability should be audited, what evidence exists for disparity, which mechanisms are plausible, and how current defenses trade privacy against utility and fairness. This section does not repeat those reviews. It converts their unresolved findings into five linked design requirements: reliable estimation, explanation of persistent exposure, causal mechanism testing, targeted intervention, and external validation with re-auditing.

## 9.1 Reliable Patient-Level Risk Estimation

Patient-level risk estimation becomes useful only when the estimated vulnerability is sufficiently stable and interpretable to distinguish persistently exposed patients from apparent high-risk cases produced by attack choice, model randomness, or sampling variability. Knolle et al. established that aggregate auditing can hide extreme vulnerability among individual clinical data contributors, but their patient-level estimates were obtained under a specific repeated-training design and membership hypothesis and should not be interpreted as context-free probabilities of privacy harm [3]. A reliable estimate must identify the patient as the unit of inference and risk estimation, define whether membership means any, partial, or complete contribution of the patient's records, specify the attacker's knowledge and model access, state the attack signal and operating point, and quantify uncertainty. Ye et al. show that changes to the membership game alter the quantity being estimated even when outputs are all described as membership risk [6]. Accordingly, an audit-specific value such as

\[
V_{p,c}=0.87
\]

is uninterpretable without the configuration \(c\) that generated it and the repeated observations supporting it.

Reliable estimation also requires a patient-specific counterfactual rather than a global threshold applied to raw loss or confidence. Song and Mittal demonstrated substantial sample-level heterogeneity and proposed fine-grained privacy-risk scores based on estimated member and non-member behavior [15]. LiRA similarly evaluates whether the observed target statistic is more compatible with inclusion or exclusion using sample-specific reference behavior; online LiRA estimates IN and OUT reference distributions, while offline variants approximate the comparison using OUT reference models [2]. These approaches are more informative than asking only whether a patient's loss crosses a universal threshold, but they remain dependent on how reference data and models are constructed. Watson et al. show why difficulty calibration is essential: an intrinsically easy example may receive low loss regardless of membership, causing uncalibrated attacks to confuse predictability with training-specific influence [13]. Conceptually,

\[
\text{Reliable risk}\neq\text{raw confidence or loss}.
\]

Instead, risk should reflect deviation from the behavior expected without the target patient's contribution. For multi-record patients, that counterfactual must be defined at patient level; calibrating records separately and then taking their maximum does not automatically produce a calibrated patient-level false-positive rate.

No single attack provides an attack-independent ground truth. Threshold attacks, LiRA, RMIA, and confidence- or margin-based procedures exploit related but non-identical signals and can differ in assumptions, computational budgets, operating points, and sensitivity to reference mismatch [2,6]. RMIA introduces a likelihood-ratio attack that combines reference models with population samples and achieves strong attack power with comparatively few reference models across several evaluated settings [48]. This makes it useful for practical stress testing, but its evidence is primarily record-level and non-clinical and does not establish that RMIA alone identifies persistently vulnerable patients. Disagreement between attacks may reflect estimation failure, or it may arise because different attacks instantiate different feasible adversaries. A future audit could therefore preserve an attack-indexed patient profile,

\[
\mathbf V_p=
\left(
V_p^{\mathrm{loss}},
V_p^{\mathrm{LiRA}},
V_p^{\mathrm{RMIA}},
\ldots
\right),
\]

and evaluate rank correlation, threshold agreement, and high-risk-set overlap across attacks that share a compatible membership hypothesis. Combining heterogeneous scores into one number would require explicit calibration to a common estimand rather than simple averaging.

Statistical reliability remains a separate obstacle. Patient-specific inference requires repeated target or reference models, training seeds, membership assignments, and sufficiently many IN and OUT observations to estimate uncertainty. Low-FPR claims are especially data intensive because rare false positives must be estimated from a sufficiently large non-member population. Jälkö et al. argue that efficient vulnerability evaluations can be misleading when scores from different targets are pooled: a globally calibrated threshold need not control the FPR for every target. They also identify finite-population effects that can distort sample-specific LiRA-style estimates; this evidence should be treated as methodological because the study is currently a preprint [14]. A point estimate should therefore be accompanied by a confidence interval, the numbers of IN and OUT observations, and stability across seeds, data subsamples, and reference sets. Multiple records from the same patient must be treated as correlated contributions, with patient-level resampling and prespecified combination rules. Otherwise, apparent precision may result from counting dependent records as independent evidence.

The central research gap is therefore reliability and persistence, not the mere possibility of measuring individual risk. Azize and Basu show in a fixed-target framework that leakage is target dependent, supporting the view that targets can differ in membership identifiability under a fixed learning and auditing setting [33]. However, the clinical literature reviewed here does not yet provide a validated standard for determining whether the same multi-record patient remains high risk across plausible attacks, models, seeds, and reference constructions. A future framework could report a reliable patient risk profile

\[
\mathcal P_p=
\left\{
\left(\widehat V_{p,c},CI_{p,c}\right):c\in\mathcal C
\right\},
\]

where \(c\) indexes prespecified compatible audit configurations. Persistence could then be summarized using prespecified criteria, such as the proportion of configurations in which the lower confidence bound exceeds a fixed risk threshold, while retaining the full profile whenever the underlying attack metrics are not commensurable. Such a framework could distinguish recurrent exposure from configuration-specific alarms and provide uncertainty-aware input to later intervention selection. The resulting research question is: **How should multiple attack-specific and model-specific estimates be combined to distinguish persistently vulnerable patients from patients that appear high risk only under a particular audit configuration?**

---

## 9.2 Why Are Some Patients Persistently High-Risk?

A patient who appears vulnerable under one audit configuration is not necessarily persistently vulnerable. A high score from one model, seed, or attack may reflect sampling variability, sensitivity to a particular attack signal, or an unstable position near the decision threshold. Persistent high risk instead denotes recurrent exposure across a prespecified set of compatible models, training runs, and attacks, with uncertainty sufficiently small to exclude chance threshold crossings. Knolle et al. revealed extreme patient-level vulnerability in clinical models, while Song and Mittal demonstrated broader sample-specific heterogeneity, but neither establishes that the same individuals remain extreme under every plausible audit configuration [3,15]. Vulnerability is also relational rather than purely intrinsic: the Privacy Onion Effect shows that modifying the currently most exposed examples can reveal a different vulnerable layer [35]. Persistent vulnerability should therefore be understood as a reproducible interaction between a patient's contribution, the surrounding training population, the learning procedure, and the attacker rather than as a permanent attribute of the patient.

Rarity and atypicality provide a plausible first explanation. In long-tailed distributions, a patient with few clinically similar neighbors may supply information that cannot be recovered from common examples. Feldman argues that optimal learning on long-tailed data can require memorizing rare examples [27], while Feldman and Zhang show that memorization and estimated influence vary markedly between samples and can be concentrated among atypical examples [28]. Azize and Basu likewise formalize target-dependent leakage, showing that targets can differ in membership identifiability under a fixed learning setting [33]. These mechanisms are consistent with Knolle et al.'s observation that some underrepresented clinical populations were overrepresented in the extreme-risk tail [3]. However, population underrepresentation, diagnostic rarity, and local atypicality are not equivalent. A member of a small demographic group may have many clinically similar neighbors, whereas an unusual patient can belong to a large group. Existing clinical evidence is predominantly associational and does not show that rarity independently causes recurrent exposure. A sharper research question is therefore whether persistently high-risk patients have fewer clinically similar neighbors after controlling for diagnosis, institution, record count, and model performance.

Memorization and patient influence may offer a more proximal mechanism because they ask what the patient's presence actually changed in the trained model. Overfitting can facilitate membership inference, but average generalization gaps do not explain why only particular patients remain exposed [10]. A counterfactual comparison between \(f_D\) and \(f_{D\setminus R_p}\), where \(R_p\) contains all records contributed by patient \(p\), asks whether removing that contribution changes predictions, parameters, representations, or attack-relevant statistics in a patient-specific manner. Long et al. connect membership vulnerability in well-generalized models to unusually influential targets [32], while influence functions estimate the local effect of infinitesimally upweighting a training example and are commonly used to approximate the effect of its removal [39]. Feldman and Zhang similarly connect long-tail memorization with heterogeneous sample influence [28]. Nevertheless,

\[
\text{Influence}\neq\text{privacy risk}.
\]

A patient may be influential because their data improve clinically meaningful generalization without producing a reliable membership signal, and conventional influence approximations can be inaccurate for grouped or non-infinitesimal perturbations [40]. The unresolved question is therefore whether persistent vulnerability corresponds to a particular form of stable, patient-specific influence rather than merely large influence.

Persistence may also arise from the structure and frequency of the patient's contribution rather than from patient characteristics alone. A patient contributing \(m_p\) repeated records can affect training through increased sampling weight and a shared subject-specific signature. Twenty near-duplicate images do not convey the same information as twenty diverse longitudinal visits, and the number of records available to the attacker differs from the number used for training. Statistical dependence can alter membership signals and invalidate IID assumptions [23], but record count has not been shown to have a universally monotonic effect on patient risk. Model and training choices further modify these relationships. Greater capacity may enlarge the extreme-risk tail in clinical data [3], yet the privacy effect of overparameterization can change under appropriate regularization or early stopping [36,37]. Pretraining, augmentation, training duration, and contribution control may similarly alter which patient-specific features are learned. Persistent risk should therefore be treated as an interaction,

\[
R_p=
 f\!\left(
\text{rarity},
\text{memorization},
\text{influence},
m_p,
\text{correlation},
\text{model},
\text{training}
\right),
\]

rather than attributed to rarity, multiplicity, or capacity alone.

The research gap is not that possible drivers are unknown, but that their connection to persistence has not been causally separated. Existing evidence identifies rarity, selective memorization, record multiplicity, model configuration, and influence as plausible contributors to heterogeneous vulnerability, yet does not determine which mechanism explains repeated patient-level exposure across attack families and training runs. Future studies should first identify persistently vulnerable patients using the reliability criteria developed in Section 9.1 and then test whether their risk is mediated by a stable patient-specific learning signal. This requires controlled interventions: matching patients on clinical similarity while varying record contribution, retraining with and without all records from the same patient, changing capacity or regularization one factor at a time, and testing whether reduced patient influence produces consistent risk reduction across compatible attacks. A useful conceptual hypothesis is

\[
g_p=g_p^{\mathrm{shared}}+g_p^{\mathrm{specific}},
\]

where the shared component reflects information supported by clinically similar patients and the specific component represents a residual contribution that is more distinctive to patient \(p\). This is not yet an established decomposition or protection algorithm. The central testable question is: **Do persistently vulnerable patients induce a distinctive patient-specific model influence that remains stable across training configurations and explains their repeated membership exposure?**

---

## 9.3 From Association to Mechanism

Existing evidence identifies several patient and training characteristics associated with membership vulnerability, but association alone does not reveal which change induced by a patient's inclusion creates the observable membership signal. Knolle et al. found that extreme clinical privacy risk can be concentrated among particular patients and populations, while Song and Mittal demonstrated substantial heterogeneity in sample-specific privacy scores [3,15]. Nevertheless, characteristics such as rarity, underrepresentation, record multiplicity, high loss, or estimated influence frequently co-occur. An observed relationship such as

\[
\text{rarity}\rightarrow\text{high risk}
\]

therefore does not establish rarity as the direct mechanism. The relevant pathway may instead be

\[
\text{rarity}\rightarrow
\text{patient-specific learning}\rightarrow
\text{membership distinguishability},
\]

or

\[
\text{multiple records}\rightarrow
\text{greater training influence}\rightarrow
\text{membership distinguishability}.
\]

A mechanistic explanation must consequently identify the model change caused by including the patient, demonstrate that this change carries information about membership, and show that modifying the candidate mechanism changes privacy risk while relevant alternative explanations are controlled.

The most direct starting point is a patient-level inclusion-exclusion experiment. Let \(R_p\) denote all records contributed by patient \(p\), let \(D\) be the complete training dataset, and define

\[
D_{-p}=D\setminus R_p.
\]

Models trained on these datasets, \(f_D\) and \(f_{D_{-p}}\), can then be compared to estimate the counterfactual effect of the patient's participation. Candidate outcomes include differences in loss, logits, prediction margins, learned representations, gradients, calibration, and behavior on clinically similar patients. Feldman and Zhang use inclusion-dependent notions of memorization and influence to show that training examples differ considerably in how much their presence changes learned behavior [28]. Long et al. similarly connect distinctive effects of particular training examples with membership vulnerability, including in models that generalize well on average [32]. For clinical data, however, the intervention must remove or include all records belonging to the patient; deleting only one image or visit answers a record-level question. A single inclusion-exclusion model pair is also insufficient because stochastic training variability may be comparable to, or larger than, the effect attributed to the patient. Reliable estimation therefore requires repeated paired training runs, using matched random seeds where feasible, while keeping the surrounding cohort, preprocessing, and training protocol as comparable as possible.

A large patient effect is not automatically a privacy-relevant effect. Influence functions estimate the local effect of infinitesimally upweighting a training example and are commonly used to approximate its removal [39], but the approximated quantity depends on the selected test point, loss, parameterization, and local assumptions. Koh et al. further show that influence-function approximations can become inaccurate for group effects and other non-infinitesimal perturbations [40], which is particularly important when one patient contributes many correlated records. More fundamentally, a patient may be highly influential because their records help the model learn a clinically valuable pattern, such as the presentation of a rare disease, without making their membership readily distinguishable. Conversely, a patient may have little effect on aggregate accuracy while leaving a localized signal in losses, margins, or representations that an MIA can exploit. The review therefore distinguishes **clinical influence**, which concerns changes in predictive utility or medically relevant generalization, from **privacy-relevant influence**, which concerns changes that distinguish inclusion from exclusion. The two may overlap without being equivalent: the objective is not to eliminate patient influence, but to identify the component of influence that provides membership evidence without being necessary for clinical generalization.

A testable hypothesis is that the patient's learned contribution contains both shared and patient-specific components. Conceptually,

\[
g_p=g_p^{\mathrm{shared}}+g_p^{\mathrm{specific}},
\]

where \(g_p^{\mathrm{shared}}\) represents information also supported by clinically similar patients and \(g_p^{\mathrm{specific}}\) represents a residual change attributable more distinctively to patient \(p\). For example, imaging features characteristic of a rare disease may constitute shared clinical information even when only a small subgroup carries them, whereas idiosyncratic anatomy, acquisition artifacts, repeated near-duplicates, or an unusual combination of attributes may contribute a more patient-specific residual. Long-tail learning provides a motivation for this distinction because rare examples can require selective memorization and estimated memorization and influence are highly heterogeneous [27,28]. However, the proposed decomposition is presently a conceptual hypothesis, not an established or uniquely identifiable mathematical separation. A residual defined by one representation, neighborhood, or gradient metric may differ from that defined by another. The hypothesis to test is therefore not that every patient-specific residual constitutes leakage, but whether a stable residual predicts persistent vulnerability across compatible models and attacks:

\[
g_p^{\mathrm{specific}}
\;\overset{?}{\longrightarrow}\;
\text{persistent membership distinguishability}.
\]

Moving from association to mechanism requires an intervention-based research design. Patients should first be classified using the reliability criteria developed in Sections 9.1 and 9.2 rather than selected from one attack run. For each patient, candidate explanatory measures—including local rarity, memorization, record count, within-patient correlation, total influence, and patient-specific gradient or representation residuals—can then be estimated under repeated inclusion-exclusion training. Multivariable analysis can determine whether a candidate signal predicts persistent risk after controlling for diagnosis, clinical difficulty, institution, acquisition protocol, and number of records, but this remains correlational. Stronger evidence would come from an intervention that selectively attenuates the candidate patient-specific signal while leaving the shared clinical signal as unchanged as possible, followed by an independent re-audit. Patient-balanced sampling, contribution clipping, or removal of near-duplicates can serve as additional sensitivity analyses, but they alter broader aspects of the contribution and therefore provide less specific evidence about the proposed mediator. If selective attenuation consistently reduces the same patient's vulnerability across attacks, models, and seeds while preserving shared clinical utility, the evidence for a mechanism becomes substantially stronger. The central research question is therefore: **Can patient-specific learning influence be isolated from clinically shared influence, and does experimentally reducing that patient-specific component consistently reduce membership vulnerability?**

---

## 9.4 Targeted Mitigation of Extreme-Risk Patients

Most existing privacy mitigations are designed and evaluated at the population or model level. Standard DP-SGD, for example, applies a common clipping-and-noise mechanism under a specified adjacency relation and reports a dataset-level privacy budget [8,41]. This formal uniformity is valuable, but patients can still differ substantially in their empirical vulnerability and in the utility effects associated with protection. Knolle et al. showed that a small number of patients can be far more distinguishable than the population average, while Song and Mittal demonstrated substantial heterogeneity in sample-specific privacy risk [3,15]. Uniformly strengthening population-wide protection may increase utility costs across the population and may disproportionately affect some underrepresented groups [47], even when severe empirical vulnerability is concentrated among a smaller subset of patients. Conversely, selecting a weaker population-wide configuration to preserve average performance may leave the extreme-risk tail insufficiently protected. This does not establish that targeted mitigation is inherently superior to uniform protection. Rather, it motivates a research question: **Can additional protection be concentrated where reliable empirical exposure is greatest while retaining an appropriate baseline safeguard for the entire population?**

Targeted patient-level protection requires a predefined mapping from reliable audit evidence to an intervention. The baseline high-risk set and threshold should be fixed as specified in Section 8.2 before the protected model is evaluated. Eligibility for stronger intervention should reflect uncertainty and the persistence criteria developed in Sections 9.1 and 9.2 rather than a point estimate alone. A general policy may be written as

\[
a_p=g(\widehat R_p,U_p,S_p),
\]

where \(a_p\) is the assigned intervention, \(U_p\) is estimation uncertainty, and \(S_p\) is stability across compatible audits. Persistently vulnerable patients might receive stricter contribution bounds, lower sampling frequency, stronger clipping, or modified weights. Individualized DP-SGD shows that examples can receive different sampling or clipping parameters, but its budgets arise from preferences or policy and its analysis is example-level [45]. REPS uses estimated outlier characteristics to alter record influence during private synthetic-data generation, but remains a record-level tabular proof of concept [46]. If formal DP is claimed, data-dependent scoring and selection must be included in the end-to-end privacy analysis, and released scores or assignments require protection.

Persistently vulnerable patients may also carry rare and clinically valuable information, so targeted protection should not be reduced to “more risk, more noise.” Indiscriminately suppressing a contribution could discard the signal needed to learn uncommon conditions. Section 9.3 instead motivates the schematic hypothesis

\[
g_p=g_p^{\mathrm{shared}}+g_p^{\mathrm{specific}},
\qquad
\widetilde g_p=
 g_p^{\mathrm{shared}}+
\alpha_p g_p^{\mathrm{specific}},
\qquad
0\leq\alpha_p\leq1.
\]

Under this hypothesis, \(\alpha_p<1\) would attenuate a candidate patient-specific component while preserving information shared with clinically comparable patients. Long-tail memorization and influence studies make the distinction scientifically plausible [27,28], but do not provide an operational separation between clinically useful and privacy-relevant gradients. The decomposition, its estimators, and the mapping from audit evidence to \(\alpha_p\) all require experimental validation. This is a research hypothesis—not an identified quantity, established algorithm, or formal privacy guarantee.

Targeted mitigation can also fail in several ways. First, reducing one patient's exposure may redistribute rather than eliminate risk. The Privacy Onion Effect demonstrates that altering or removing currently vulnerable examples can expose another layer of samples that previously appeared safer [35]. Every intervention must therefore be followed by an independent audit of the complete patient population, not only the selected set. Second, high-risk patients may disproportionately belong to rare or underrepresented groups. Bagdasaryan et al. show that differential privacy can impose unequal accuracy costs across groups [47], while Kulynych et al. emphasize the need for statistically reliable disparity evaluation [7]. Downweighting unusual patients could consequently reduce their measured privacy risk while degrading sensitivity, calibration, or worst-group utility for the conditions they represent. Third, targeted decisions inherit errors from the risk estimator. False-positive classifications may expose patients to unnecessary intervention, whereas false negatives leave vulnerable patients without the intended additional protection. Risk-adaptive policies should therefore incorporate uncertainty and persistence rather than rely on a single point estimate. Risk labels, intervention assignments, and patient-specific budgets should themselves be treated as sensitive metadata because they may reveal rarity or clinically sensitive status.

The unresolved gap is an end-to-end clinical pipeline connecting reliable auditing, mechanistic interpretation, selective intervention, and independent re-auditing. Using the paired framework in Section 8, success requires reduced residual exposure among the baseline high-risk patients, no material increase among others beyond uncertainty, and overall and subgroup utility within prespecified margins. Attackers must be retrained or recalibrated against the protected mechanism, and the targeted method should be compared with a uniform intervention matched on utility, computation, or formally comparable privacy constraints. Existing studies show that leakage can be reduced and non-uniform protection is technically possible [3,15,45,46], but not this complete clinical process. The central question is: **Can persistently high-risk patients be selectively protected by reducing privacy-relevant patient-specific influence while preserving shared clinical information and avoiding vulnerability transfer?**

---

## 9.5 Validation, Generalizability, and Re-Auditing of Patient-Level Protection

A reduction under one experimental configuration is insufficient to validate patient-level protection. Applying the paired definitions from Section 8, validation must determine whether baseline high-risk patients obtained a reproducible benefit, whether any remained above the threshold, and whether others became materially more vulnerable. The baseline set must remain fixed; otherwise a changed tail identity can be mistaken for absolute reduction. Knolle et al. show that stronger protection can shift clinical patient-vulnerability distributions downward, while Song and Mittal show why aggregate defense performance can conceal heterogeneous effects [3,15]. A validated defense must therefore demonstrate paired improvement, reduced residual tail exposure, limited adverse responses, and uncertainty bounds separating protection from training variability.

Protection should next be tested across compatible attacks, random seeds, architectures, and training configurations. Success against LiRA alone may indicate that the intervention weakened the particular attack statistic or reference-model separation exploited in that evaluation rather than reducing membership distinguishability more generally. Evaluation should therefore include complementary attacks, such as threshold or loss attacks, LiRA, and RMIA, while keeping the membership hypothesis and protected unit consistent [2,48]. Attack disagreement should not automatically be classified as defense failure because attacks may exploit different signals or operate at different points on the ROC curve; however, a claim of broad empirical protection requires robustness across plausible threat models. Reference models and attackers must be retrained or recalibrated for the protected mechanism, since an attacker designed only for the unprotected model can overstate defense effectiveness [15]. Repeated seeds are also essential because changes near a high-risk threshold may reflect training randomness. Jälkö et al. further caution that efficient evaluation, pooled scores, and finite reference populations can distort per-sample vulnerability estimates, particularly at low FPR [14]. Protection should therefore be considered reproducible only when its benefit persists across independently constructed reference sets and sufficiently powered audit repetitions.

Generalizability additionally requires validation beyond the dataset and modality used to design the intervention. Patient-specific signals may reflect anatomy and acquisition artifacts in radiographs, stable physiological signatures in ECG recordings, or longitudinal diagnosis and visit patterns in electronic health records. Knolle et al. demonstrated privacy heterogeneity across several clinical datasets, but replication across datasets within one study does not by itself constitute independent external validation [3]. Wang et al.'s subject-level ECG results illustrate both the importance of subject participation leakage and the difficulty of separating membership from dataset, device, or institutional origin when non-members are drawn from different cohorts [22]. Humphries et al. show that dependencies and heterogeneous data sources can alter MIA performance [23], while Shi et al. demonstrate that distribution shifts may increase or decrease attack success depending on the attack and shift [25]. External validation should therefore include, where feasible, within-institution and cross-institution audits, site-matched reference data, patient-disjoint construction, and sensitivity analyses across devices, protocols, periods, and disease prevalence. The central question is whether the targeted mechanism generalizes across domains or is specific to one modality, institution, or acquisition process.

Privacy improvement is not clinically valid if it produces unacceptable predictive harm. Validation should apply the Section 8 criteria for overall and subgroup discrimination, calibration, and prespecified clinical operating points. Aggregate accuracy cannot substitute for subgroup validation, particularly for rare, underrepresented, and intersecting populations [7,31,47]. Prospective non-inferiority margins should be specified separately for overall utility and each clinically relevant group, with confidence intervals and subgroup sample sizes reported. The question is not only whether exposure falls, but whether the gain remains clinically acceptable for the populations whose data made the model useful.

Finally, validation must be treated as a continuing lifecycle rather than a permanent certification. The required cycle is

\[
\text{audit}
\rightarrow
\text{protect}
\rightarrow
\text{independent re-audit},
\]

followed by renewed auditing after material changes to the system. Material updates can include substantial changes in the training population, acquisition protocol, preprocessing pipeline, model architecture, optimization procedure, or protection mechanism. This is particularly important because the Privacy Onion Effect shows that modifying the current vulnerable layer can reveal a different set of exposed patients [35]. Re-auditing must measure absolute risk throughout the population rather than infer transfer merely from changes in rankings or high-risk-set overlap. Individualized DP-SGD and REPS demonstrate that non-uniform or risk-directed intervention is technically possible, but neither validates an end-to-end, multi-record clinical pipeline across external institutions and repeated model updates [45,46]. Validation should also compare targeted protection with an appropriate uniform baseline under prespecified matched constraints, such as comparable clinical utility or computational cost, so that any apparent advantage cannot be attributed merely to a stronger overall intervention. A credible future method should therefore achieve lower tail risk, preserve overall and rare-group utility within prespecified margins, and avoid material risk transfer across multiple attacks, seeds, model families, datasets, and institutions. The unresolved question is: **Can patient-level protection provide reproducible benefit across attacks and models and generalize across clinical datasets, institutions, and patient populations without transferring exposure or causing clinically important subgroup utility loss?**

### Validation dimensions

- **Attacks:** complementary, compatible MIA families and operating points.
- **Seeds:** repeated training and reference-model randomness.
- **Models:** architectures and relevant training configurations.
- **Datasets:** distinct cohorts and clinical modalities.
- **Institutions:** external sites and acquisition environments where feasible.

---
