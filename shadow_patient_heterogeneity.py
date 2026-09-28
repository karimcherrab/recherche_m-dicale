"""Fast pilot of individual patient MIA separability from SAVED shadow models.

Run on the ORIGINAL 32-shadow LiRA directory (takes seconds, no retraining):
  python shadow_patient_heterogeneity.py --lira-run LIRA_RUN

For each patient, compare the shadow-model score when ALL the patient's visits
were included with the score when NONE were included. The empirical AUC is the
fraction of IN-vs-OUT shadow-score pairs ranked in the expected direction.
As a noise check, select the top 5% on half the shadows and report their mean
AUC on the other, held-out half; repeat with independent random half-splits.

These are exploratory AUCs for the SHADOW MODEL TRAINING PROCEDURE. They are
not target-checkpoint membership probabilities and are noisy with 32 models.
The halves/repeats share underlying shadow models and patients, so they are
not independent replications. This tool does not query the target's Test.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc_each_model(values: np.ndarray, masks: np.ndarray):
    """Rank statistic across models for EVERY patient, with 0.5 for ties."""
    n_in = masks.sum(axis=0)
    n_out = len(values) - n_in
    good = (n_in >= 4) & (n_out >= 4)
    ranks = rankdata(values, axis=0, method="average")
    u = np.where(masks, ranks, 0.).sum(axis=0) - n_in*(n_in+1)/2
    auc = np.full(values.shape[1], np.nan)
    auc[good] = u[good] / (n_in[good]*n_out[good])
    return auc, n_in, n_out


def read_lira(run: Path):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    patients = pd.read_csv(run / "patient_scores.csv", dtype={"patient_nbr": str})
    required = {"patient_nbr", "split", "n_records", "target_stat"}
    if (not required.issubset(patients.columns) or
        len(patients) != manifest["patient_count"] or
        patients.patient_nbr.isna().any() or patients.patient_nbr.duplicated().any() or
        set(patients.split) != {"train", "auxiliary", "test"}):
        raise ValueError("LiRA patient_scores.csv and manifest.json do not match")
    values, masks = [], []
    for i in range(manifest["n_shadows"]):
        with np.load(run / "shadows" / f"shadow_{i:03d}.npz",
                     allow_pickle=False) as saved:
            stat = np.asarray(saved["stat"], dtype=float)
            mask = np.asarray(saved["mask"], dtype=bool)
        if (stat.shape != (len(patients),) or mask.shape != stat.shape or
            mask.sum() != manifest["n_target_train"] or
            not np.isfinite(stat).all()):
            raise ValueError(f"Invalid shadow scores or IN/OUT assignments: {i}")
        values.append(stat)
        masks.append(mask)
    values, masks = np.stack(values), np.stack(masks)
    if np.min(np.minimum(masks.sum(0), (~masks).sum(0))) < 6:
        raise ValueError("Insufficient shadow IN/OUT repetitions")
    return manifest, patients, values, masks


def diagnostic(values, masks, rng, selection_fraction=.05):
    permutation = rng.permutation(len(values))
    discovery = permutation[:len(values)//2]
    check = permutation[len(values)//2:]
    a, *_ = auc_each_model(values[discovery], masks[discovery])
    b, *_ = auc_each_model(values[check], masks[check])
    valid = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
    if len(valid) < 100:
        raise ValueError("Too few patients with >=4 IN/OUT models in both halves")
    top_n = max(1, round(selection_fraction*len(valid)))
    # Stable order resolves equal AUC values without using the holdout scores.
    top = valid[np.argsort(-a[valid], kind="stable")[:top_n]]
    return {"valid_patients": len(valid), "selected_count": len(top),
            "discovery_mean_auc_selected": float(np.mean(a[top])),
            "holdout_mean_auc_selected": float(np.mean(b[top])),
            "holdout_mean_auc_all": float(np.mean(b[valid])),
            "holdout_uplift": float(np.mean(b[top])-np.mean(b[valid]))}, top, a, b


def permuted_membership(masks, rng):
    """Negative control: keep each patient's IN count, break model alignment."""
    order = np.argsort(rng.random(masks.shape), axis=0)
    return np.take_along_axis(masks, order, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lira-run", required=True, type=Path,
                        help="Original patient_lira.py folder with shadows/ and manifest.json")
    parser.add_argument("--output-root", type=Path,
                        help="Default is LIRA_RUN/shadow_patient_heterogeneity")
    parser.add_argument("--half-splits", type=int, default=10)
    parser.add_argument("--null-repeats", type=int, default=3,
                        help="Per-patient permuted-IN/OUT negative-control repetitions")
    parser.add_argument("--seed", type=int, default=44029)
    args = parser.parse_args()
    if min(args.half_splits, args.null_repeats, args.seed) < 1:
        parser.error("--half-splits, --null-repeats and --seed must be positive")
    manifest, patients, scores, masks = read_lira(args.lira_run)
    auc, n_in, n_out = auc_each_model(scores, masks)
    if not np.isfinite(auc).all():
        raise ValueError("Some patients have inadequate full shadow counts")
    rng = np.random.default_rng(args.seed)
    selected = np.zeros(len(patients), dtype=int)
    pilot = []
    first_a = first_b = None
    for j in range(args.half_splits):
        row, top, a, b = diagnostic(scores, masks, rng)
        row["half_split"] = j
        pilot.append(row)
        selected[top] += 1
        if j == 0:
            first_a, first_b = a, b
    null_results = []
    for j in range(args.null_repeats):
        shuffled = permuted_membership(masks, rng)
        null_auc, _, _ = auc_each_model(scores, shuffled)
        row, _, _, _ = diagnostic(scores, shuffled, rng)
        null_results.append({"repeat": j,
                             "q99_auc": float(np.nanquantile(null_auc, .99)),
                             "fraction_auc_at_least_0.95": float(
                                 np.nanmean(null_auc >= .95)),
                             "holdout_top5_uplift": row["holdout_uplift"]})
    output = (args.output_root or args.lira_run / "shadow_patient_heterogeneity") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    result = patients[["patient_nbr", "split", "n_records"]].copy()
    result["n_in"] = n_in
    result["n_out"] = n_out
    result["shadow_patient_auc"] = auc
    result["discovery_auc_first_half"] = first_a
    result["validation_auc_first_half"] = first_b
    result["times_selected_top_5pct"] = selected
    result.to_csv(output / "patient_shadow_auc.csv", index=False)
    pd.DataFrame(pilot).to_csv(output / "half_split_diagnostics.csv", index=False)
    pd.DataFrame(null_results).to_csv(output / "permuted_membership_control.csv",
                                      index=False)
    subgroups = []
    for name, condition in (
        ("1_record", patients.n_records.to_numpy() == 1),
        ("2_to_4", patients.n_records.to_numpy().astype(int) >= 2),
        ("5_plus", patients.n_records.to_numpy() >= 5),
    ):
        if name == "2_to_4":
            condition &= patients.n_records.to_numpy() <= 4
        sample = auc[condition]
        subgroups.append({"subgroup": name, "patients": int(len(sample)),
                          "mean_shadow_auc": float(np.mean(sample)),
                          "median_shadow_auc": float(np.median(sample)),
                          "q90_shadow_auc": float(np.quantile(sample, .90)),
                          "fraction_at_least_0.8": float(np.mean(sample >= .8)),
                          "fraction_at_least_0.95": float(np.mean(sample >= .95))})
    pd.DataFrame(subgroups).to_csv(output / "subgroup_summary.csv", index=False)
    aggregate = {
        "models": len(scores), "pool_patients": len(patients),
        "n_in_range": [int(n_in.min()), int(n_in.max())],
        "n_out_range": [int(n_out.min()), int(n_out.max())],
        "mean_patient_auc": float(np.mean(auc)),
        "median_patient_auc": float(np.median(auc)),
        "q95_patient_auc": float(np.quantile(auc, .95)),
        "q99_patient_auc": float(np.quantile(auc, .99)),
        "fraction_auc_at_least_0.8": float(np.mean(auc >= .8)),
        "fraction_auc_at_least_0.95": float(np.mean(auc >= .95)),
        "mean_holdout_top5_auc": float(np.mean(
            [p["holdout_mean_auc_selected"] for p in pilot])),
        "mean_holdout_all_auc": float(np.mean(
            [p["holdout_mean_auc_all"] for p in pilot])),
        "mean_holdout_top5_uplift": float(np.mean(
            [p["holdout_uplift"] for p in pilot])),
        "min_holdout_top5_uplift": float(min(p["holdout_uplift"] for p in pilot)),
        "null_mean_q99_auc": float(np.mean([p["q99_auc"] for p in null_results])),
        "null_mean_top5_uplift": float(np.mean(
            [p["holdout_top5_uplift"] for p in null_results])),
        "notes": [
            "Scores compare the same entire patient IN vs OUT across saved shadow models.",
            "These are estimates for the shadow-model training distribution, NOT for the single original target checkpoint.",
            "With only 32 shadows, high individual AUC estimates are noisy and selecting from 64k patients inflates the extreme tail.",
            "Top 5% were chosen using only first-half shadow scores; holdout AUC comes from the other half.",
            "Repeated half-splits reuse the same 32 correlated shadow models; this does not supply independent uncertainty intervals.",
            "Negative controls permute shadow membership labels independently for each patient; they are diagnostics, not an exact p-value under correlated shadow training.",
            "The frozen original preprocessing may include patients called OUT for a particular shadow model.",
            "Do not treat these AUC scores as individual membership probabilities or clinical privacy-risk estimates.",
        ],
    }
    (output / "shadow_metrics.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{len(scores)} shadow models | {len(patients)} patients")
    print(f"Mean shadow patient AUC: {aggregate['mean_patient_auc']:.4f} "
          f"| q99: {aggregate['q99_patient_auc']:.4f}")
    print(f"Top 5% holdout mean AUC: {aggregate['mean_holdout_top5_auc']:.4f} "
          f"| holdout all: {aggregate['mean_holdout_all_auc']:.4f} "
          f"| uplift: {aggregate['mean_holdout_top5_uplift']:.4f} "
          f"| permuted-label control uplift: {aggregate['null_mean_top5_uplift']:.4f}")
    print(f"Exploratory SHADOW-MODEL diagnostic: {output}")
    print("No claim about the original target's individual privacy risk.")


if __name__ == "__main__":
    main()
