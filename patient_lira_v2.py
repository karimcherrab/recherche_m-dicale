"""Compare patient MIA scores using saved shadow models, without retraining.

Requires patient_lira.py from the previous audit in the same directory. Run:

    python patient_lira_v2.py --run-dir "D:\\Doctorat\\prototype\\results\\mia_lira\\YOUR_RUN"

The old run must contain manifest.json, patient_scores.csv, and shadows/*.npz.
For each saved shadow, score it as a pseudo-target against the other shadows;
choose a score using only the known pseudo-target IN/OUT assignments. Then
evaluate the fixed choice against the original target-model train/test split.

The original test has already been used for the initial attacks. All further
target-model test results remain exploratory even with shadow-only selection.
This attack has privileged access to candidate records/labels via the saved
reference models; it is not representative of a label-unknown outside attacker.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from patient_lira import auc_ci, calibrated_threshold, evaluate, patient_lira


CANDIDATES = ("lira_shared", "out_z", "out_gap")
FPRS = (0.01, 0.05)


def moments(scores, masks, leave_out=None):
    """Per-patient mean and unbiased variance of reference IN and OUT scores."""
    if leave_out is not None:
        keep = np.arange(len(scores)) != leave_out
        scores, masks = scores[keep], masks[keep]
    n_in = np.sum(masks, axis=0)
    n_out = len(scores) - n_in
    if min(n_in.min(), n_out.min()) < 6:
        raise ValueError("Insufficient reference IN/OUT observations")
    sum_in = np.sum(np.where(masks, scores, 0.), axis=0)
    sum_out = np.sum(np.where(~masks, scores, 0.), axis=0)
    mean_in, mean_out = sum_in / n_in, sum_out / n_out
    var_in = np.sum(np.where(masks, (scores - mean_in) ** 2, 0.), axis=0) / (n_in - 1)
    var_out = np.sum(np.where(~masks, (scores - mean_out) ** 2, 0.), axis=0) / (n_out - 1)
    return mean_in, mean_out, var_in, var_out


def attack_scores(target, means_and_vars):
    mean_in, mean_out, var_in, var_out = means_and_vars
    adjusted_mean_in = np.maximum(mean_in, mean_out)
    shared_var = np.maximum((var_in + var_out) / 2, 0.1 ** 2)
    scores = {
        "lira_shared": ((target - mean_out) ** 2 -
                        (target - adjusted_mean_in) ** 2) / (2 * shared_var),
        "out_z": (target - mean_out) / np.sqrt(np.maximum(var_out, 0.1 ** 2)),
        "out_gap": target - mean_out,
    }
    if not all(np.isfinite(a).all() for a in scores.values()):
        raise ValueError("Non-finite scores")
    return scores


def load_run(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    patients = pd.read_csv(run_dir / "patient_scores.csv", dtype={"patient_nbr": str})
    required = {"patient_nbr", "split", "n_records", "target_stat", "attack_score"}
    if not required.issubset(patients.columns):
        raise ValueError("patient_scores.csv is missing columns from patient_lira.py")
    count, number = manifest["patient_count"], manifest["n_shadows"]
    if len(patients) != count or patients.patient_nbr.duplicated().any():
        raise ValueError("The patient table does not match the manifest")
    if set(patients.split) != {"train", "auxiliary", "test"}:
        raise ValueError("Unexpected target split labels")
    stats, masks = [], []
    for j in range(number):
        with np.load(run_dir / "shadows" / f"shadow_{j:03d}.npz", allow_pickle=False) as shadow:
            a = np.asarray(shadow["stat"], dtype=np.float64)
            m = np.asarray(shadow["mask"], dtype=bool)
        if a.shape != (count,) or m.shape != (count,) or int(m.sum()) != manifest["n_target_train"]:
            raise ValueError(f"Invalid shadow file {j}")
        stats.append(a)
        masks.append(m)
    stats, masks = np.stack(stats), np.stack(masks)
    old_score, *_ = patient_lira(patients.target_stat.to_numpy(), stats, masks)
    if not np.allclose(old_score, patients.attack_score.to_numpy(), atol=1e-8, rtol=1e-8):
        raise ValueError("Shadow files and target patient scores are not from the same run")
    return manifest, patients, stats, masks


def pseudo_target_results(stats, masks):
    rows = []
    for j in range(len(stats)):
        trial_scores = attack_scores(stats[j], moments(stats, masks, leave_out=j))
        actual_membership = masks[j]
        for name in CANDIDATES:
            s = trial_scores[name]
            auc = float(roc_auc_score(actual_membership.astype(int), s))
            for fpr in FPRS:
                threshold = calibrated_threshold(s[~actual_membership], fpr)
                result = evaluate(s[actual_membership], s[~actual_membership], threshold)
                rows.append({"pseudo_target": j, "candidate": name,
                             "nominal_fpr": fpr, "auc": auc,
                             "tpr": result["tpr"], "observed_fpr": result["observed_fpr"],
                             "advantage": result["advantage"]})
        print(f"Pseudo-target {j + 1}/{len(stats)} completed", flush=True)
    return pd.DataFrame(rows)


def select_candidate(pseudo):
    """Prespecified rule: highest mean shadow TPR-FPR at 1%; AUC breaks ties."""
    one_percent = pseudo[pseudo.nominal_fpr == 0.01]
    summary = (one_percent.groupby("candidate", as_index=False)
               .agg(mean_advantage_1pct=("advantage", "mean"),
                    mean_tpr_1pct=("tpr", "mean"), mean_auc=("auc", "mean")))
    selected = (summary.sort_values(["mean_advantage_1pct", "mean_auc", "candidate"],
                                    ascending=[False, False, True])
                .iloc[0].candidate)
    return str(selected), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Prior patient_lira.py output")
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    run_dir = args.run_dir.resolve()
    manifest, patients, stats, masks = load_run(run_dir)
    pseudo = pseudo_target_results(stats, masks)
    chosen, summary = select_candidate(pseudo)
    target_scores = attack_scores(patients.target_stat.to_numpy(), moments(stats, masks))
    s = target_scores[chosen]
    member_mask = (patients.split == "train").to_numpy()
    test_mask = (patients.split == "test").to_numpy()
    auxiliary_mask = (patients.split == "auxiliary").to_numpy()
    thresholds = {str(f): calibrated_threshold(s[auxiliary_mask], f) for f in FPRS}
    rng = np.random.default_rng(9018)
    global_auc = auc_ci(s[member_mask], s[test_mask], args.bootstrap, rng)
    operating = []
    group_results = []
    for fpr in FPRS:
        threshold = thresholds[str(fpr)]
        result = evaluate(s[member_mask], s[test_mask], threshold)
        result.update({"nominal_auxiliary_fpr": fpr,
                       "auxiliary_observed_fpr": float(np.mean(s[auxiliary_mask] > threshold)),
                       "threshold": threshold})
        operating.append(result)
    counts = patients.n_records.to_numpy()
    for group, selected_rows in (("1_record", counts == 1),
                                 ("2_to_4_records", (counts >= 2) & (counts <= 4)),
                                 ("5_plus_records", counts >= 5)):
        a, b = s[member_mask & selected_rows], s[test_mask & selected_rows]
        auc = auc_ci(a, b, args.bootstrap, rng)
        for fpr in FPRS:
            row = evaluate(a, b, thresholds[str(fpr)])
            row.update({"group": group, "auc": auc["auc"],
                        "auc_ci": auc["bootstrap_95_ci"],
                        "nominal_global_fpr": fpr})
            group_results.append(row)
    output = run_dir / "shadow_selected_v2"
    output.mkdir(exist_ok=True)
    pseudo.to_csv(output / "shadow_validation.csv", index=False)
    summary.to_csv(output / "shadow_score_comparison.csv", index=False)
    result_table = patients[["patient_nbr", "split", "n_records", "target_stat"]].copy()
    result_table["selected_score"] = s
    result_table["selected_method"] = chosen
    for fpr in FPRS:
        result_table[f"decision_fpr_{fpr}"] = (s > thresholds[str(fpr)]).astype(int)
    result_table.to_csv(output / "patient_scores.csv", index=False)
    pd.DataFrame(group_results).to_csv(output / "subgroup_metrics.csv", index=False)
    results = {"selected_on_shadow_models_only": chosen,
               "selection_rule": "Highest mean pseudo-target TPR-FPR at 1% FPR; tie-break mean AUC",
               "number_of_shadows": manifest["n_shadows"],
               "global": global_auc, "operating_points": operating,
               "notes": [
                   "The original target test split has already been inspected: this evaluation is exploratory.",
                   "Candidate selection used shadow-model IN/OUT membership labels, never target-model train/test labels.",
                   "Shadow pseudo-targets overlap in training examples; shadow-selection estimates are correlated.",
                   "Patient scores are attack statistics, not calibrated individual risk probabilities.",
                   "The same target-preprocessed features are used throughout; preprocessing leakage is not audited."]}
    (output / "mia_metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print("Shadow-only candidate ranking:", flush=True)
    print(summary.sort_values("mean_advantage_1pct", ascending=False).to_string(index=False))
    print(f"Chosen: {chosen} | Target AUC {global_auc['auc']:.4f} | CI {global_auc['bootstrap_95_ci']}")
    for row in operating:
        print(f"Auxiliary FPR {row['nominal_auxiliary_fpr']:.0%} | "
              f"Test FPR {row['observed_fpr']:.2%} | TPR {row['tpr']:.2%}")
    print(f"Exploratory results: {output}")


if __name__ == "__main__":
    main()
