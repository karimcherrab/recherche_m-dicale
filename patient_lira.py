"""Patient-level, online LiRA-style membership audit for train_baseline.py.

Run in the original project, with data/processed/{train,validation,auxiliary,test}
and an existing baseline model directory. Requires the same Python packages as
train_baseline.py: torch numpy pandas scipy scikit-learn.

    python patient_lira_fixed.py --model-dir "D:\\Doctorat\\prototype\\results\\baseline\\20260928_120542_756287"

Defaults may be edited directly below; --model-dir is optional.
Use a NEW run directory: old cached references are incompatible.
Each reference checkpoint is saved for future diagnostics.

Optional: --data-dir PATH --output-root PATH --n-shadows 32 --device cpu
To resume, use --run-dir PATH printed by the previous run, with identical options.

Threat model: an internal auditor has the feature vectors and outcome labels for
all candidate patients and may train reference models using these records. Each
reference model trains on entire patients, with exactly as many training patients
as the target. The target's validation patients are reserved for model selection.
The target's test patients MAY enter reference training according to the fixed
in/out design; their TARGET-model membership label is never used to fit a
threshold or choose attack settings. The independent TEST claim is therefore
conditional on this prespecified online attack protocol, and this particular
test split was already inspected in the earlier mean-loss audit.

This is a patient-level adaptation of LiRA, not a literal reproduction of the
original record-level algorithm. The preprocessing is frozen from the target
training run, so the audit estimates leakage in the fitted neural network
conditional on that feature representation; it does not audit preprocessing.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# Chemins modifiables directement; les options CLI restent facultatives.
DEFAULT_MODEL_DIR = Path(r"D:\Doctorat\prototype\results\baseline\20260928_120542_756287")
DEFAULT_DATA_DIR = Path(r"D:\Doctorat\prototype\data\processed")
DEFAULT_OUTPUT_ROOT = Path(r"D:\Doctorat\prototype\results\mia_lira_fixed")
N_SHADOWS = 32
ATTACK_VERSION = "patient_signed_logit_global_variance_v2"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_data(root: Path, split: str, expected_ids: set[str], dimension: int):
    folder = root / split
    x = sparse.load_npz(folder / "X.npz").tocsr().astype(np.float32)
    y = np.load(folder / "y.npy", allow_pickle=False)
    meta = pd.read_csv(folder / "metadata.csv",
                       dtype={"patient_nbr": str, "encounter_id": str})
    if x.shape != (len(y), dimension) or len(meta) != len(y) or y.ndim != 1:
        raise ValueError(f"{split}: X, y, metadata or dimension mismatch")
    if not np.isfinite(x.data).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{split}: invalid features or binary labels")
    if not np.array_equal(meta.row_index.to_numpy(), np.arange(len(y))):
        raise ValueError(f"{split}: metadata rows not aligned")
    if meta.patient_nbr.isna().any() or meta.encounter_id.isna().any():
        raise ValueError(f"{split}: missing IDs")
    if meta.encounter_id.duplicated().any():
        raise ValueError(f"{split}: duplicate encounters")
    if not np.array_equal(y, (meta.readmitted == "<30").to_numpy()):
        raise ValueError(f"{split}: outcomes not aligned with metadata")
    if set(meta.patient_nbr) != expected_ids:
        raise ValueError(f"{split}: different patient assignments")
    return x, y.astype(np.float32), meta


def patient_groups(ids: np.ndarray):
    patient_ids, codes = np.unique(ids.astype(str), return_inverse=True)
    counts = np.bincount(codes)
    ordered_rows = np.argsort(codes, kind="stable")
    groups = np.split(ordered_rows, np.cumsum(counts)[:-1])
    return patient_ids, codes, counts, groups


@torch.inference_mode()
def patient_statistic(model, x, y, codes, counts, device, batch_size):
    """Mean true-label logit per patient; a patient-level LiRA adaptation."""
    model.eval()
    sums = np.zeros(len(counts), dtype=np.float64)
    for start in range(0, len(y), batch_size):
        stop = min(start + batch_size, len(y))
        inputs = torch.from_numpy(x[start:stop].toarray()).to(device)
        logits = model(inputs).cpu().numpy().astype(np.float64)
        statistic = (2.0 * y[start:stop] - 1.0) * logits
        if not np.isfinite(statistic).all():
            raise ValueError("Non-finite signed logits")
        sums += np.bincount(codes[start:stop], weights=statistic, minlength=len(counts))
    return sums / counts



@torch.inference_mode()
def validation_ap(model, data, device, batch_size):
    x, y, codes, counts = data
    model.eval()
    probabilities = []
    for start in range(0, len(y), batch_size):
        input_tensor = torch.from_numpy(x[start:start + batch_size].toarray()).to(device)
        logits = model(input_tensor).cpu().numpy().astype(np.float64)
        probabilities.append(np.exp(-np.logaddexp(0, -logits)))
    scores = np.concatenate(probabilities)
    weights = 1.0 / counts[codes]
    return float(average_precision_score(y, scores, sample_weight=weights))


def make_masks(n_shadows, total, n_train, seed=2026, minimum=6):
    """Fresh uniform fixed-size sample per shadow, then small coverage repairs.

    Repairs preserve each shadow's training size, use no labels or scores,
    and induce limited dependence; the design is not claimed fully independent.
    """
    if not (minimum * total <= n_shadows * n_train <= (n_shadows-minimum)*total):
        raise ValueError("Increase reference count: IN/OUT coverage infeasible")
    rng = np.random.default_rng(seed)
    masks = np.zeros((n_shadows, total), dtype=bool)
    for j in range(n_shadows):
        masks[j, rng.choice(total, n_train, replace=False)] = True
    counts = masks.sum(axis=0)
    repairs = 0
    for patient in np.flatnonzero(counts < minimum):
        while counts[patient] < minimum:
            row = int(rng.choice(np.flatnonzero(~masks[:, patient])))
            donors = np.flatnonzero(masks[row] & (counts > minimum))
            donor = int(rng.choice(donors))
            masks[row, donor], masks[row, patient] = False, True
            counts[donor] -= 1; counts[patient] += 1; repairs += 1
    for patient in np.flatnonzero(counts > n_shadows-minimum):
        while counts[patient] > n_shadows-minimum:
            row = int(rng.choice(np.flatnonzero(masks[:, patient])))
            receivers = np.flatnonzero(~masks[row] & (counts < n_shadows-minimum))
            receiver = int(rng.choice(receivers))
            masks[row, patient], masks[row, receiver] = False, True
            counts[patient] -= 1; counts[receiver] += 1; repairs += 1
    if not np.all(masks.sum(axis=1) == n_train):
        raise RuntimeError("Incorrect shadow size")
    differences = [int(np.count_nonzero(masks[i] != masks[j]))
                   for i in range(n_shadows) for j in range(i)]
    diagnostics = {"coverage_repairs": repairs,
                   "min_pairwise_different_patients": min(differences),
                   "median_pairwise_different_patients": float(np.median(differences)),
                   "min_in": int(counts.min()), "max_in": int(counts.max())}
    return masks, diagnostics


def train_shadow(index, mask, groups, x, y, validation, config, device, batch_size):
    seed = 43001 + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = MLP(config["input_dim"], config["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    included = np.flatnonzero(mask)
    best_state, best_ap = None, -float("inf")
    patience_score, stale, best_epoch = -float("inf"), 0, 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = rng.permutation(included)
        for offset in range(0, len(order), config["patients_per_batch"]):
            chosen = order[offset:offset + config["patients_per_batch"]]
            pieces = [groups[i] for i in chosen]
            rows = np.concatenate(pieces)
            weights = np.concatenate([np.full(len(part), 1.0 / len(part),
                                               dtype=np.float32) for part in pieces])
            inputs = torch.from_numpy(x[rows].toarray()).to(device)
            targets = torch.from_numpy(y[rows]).to(device)
            weights_tensor = torch.from_numpy(weights).to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.binary_cross_entropy_with_logits(
                model(inputs), targets, reduction="none")
            objective = (losses * weights_tensor).sum() / len(pieces)
            if not torch.isfinite(objective):
                raise ValueError(f"Shadow {index}: non-finite training loss")
            objective.backward()
            optimizer.step()
        ap = validation_ap(model, validation, device, batch_size)
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = copy.deepcopy(model.state_dict())
        if ap > patience_score + config["min_delta"]:
            patience_score, stale = ap, 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_ap


def patient_lira(target, shadows, membership, variance_pool):
    """Gaussian log p(stat|IN) - log p(stat|OUT); no sign clipping.

    Patient-specific means; two global residual variances estimated ONLY from
    auxiliary patients across reference models (not from target membership).
    Equal patient weighting when pooling variances. Numerical floor only.
    """
    n_shadows, n_patients = shadows.shape
    if target.shape != (n_patients,) or membership.shape != shadows.shape:
        raise ValueError("Incorrect reference score dimensions")
    if not np.isfinite(shadows).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite statistics")
    if variance_pool.shape != target.shape or not variance_pool.any():
        raise ValueError("Invalid auxiliary variance pool")
    n_in = membership.sum(axis=0)
    n_out = n_shadows - n_in
    if min(n_in.min(), n_out.min()) < 6:
        raise ValueError("Insufficient IN/OUT reference coverage")
    mean_in = np.sum(np.where(membership, shadows, 0.), axis=0) / n_in
    mean_out = np.sum(np.where(~membership, shadows, 0.), axis=0) / n_out
    residual_in = np.sum(np.where(membership, (shadows-mean_in)**2, 0.), axis=0)/(n_in-1)
    residual_out = np.sum(np.where(~membership, (shadows-mean_out)**2, 0.), axis=0)/(n_out-1)
    raw_in = float(np.mean(residual_in[variance_pool]))
    raw_out = float(np.mean(residual_out[variance_pool]))
    floor = max(1e-12, 1e-8 * max(raw_in, raw_out))
    var_in, var_out = max(raw_in, floor), max(raw_out, floor)
    score = .5 * (np.log(var_out/var_in) + (target-mean_out)**2/var_out
                  - (target-mean_in)**2/var_in)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite likelihood ratio")
    diagnostics = {"global_variance_in": var_in, "global_variance_out": var_out,
                   "numerical_variance_floor": floor,
                   "fraction_negative_mean_effect": float(np.mean(mean_in < mean_out)),
                   "fraction_exact_zero_scores": float(np.mean(score == 0))}
    return score, n_in, n_out, mean_in, mean_out, diagnostics


def calibrated_threshold(nonmembers: np.ndarray, fpr: float):
    ordered = np.sort(nonmembers)
    allowed = int(math.floor(fpr * len(ordered)))
    return float(ordered[len(ordered) - allowed - 1])


def wilson(success: int, total: int):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = success / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return [max(0., center-half), min(1., center+half)]


def evaluate(members, nonmembers, threshold):
    tp = int(np.sum(members > threshold))
    fp = int(np.sum(nonmembers > threshold))
    tpr = tp / len(members)
    fpr = fp / len(nonmembers)
    return {"members": len(members), "nonmembers": len(nonmembers),
            "true_positives": tp, "false_positives": fp,
            "tpr": tpr, "observed_fpr": fpr, "advantage": tpr-fpr,
            "tpr_95_ci": wilson(tp, len(members)),
            "fpr_95_ci": wilson(fp, len(nonmembers))}


def auc_ci(members, nonmembers, repeats, rng):
    if len(members) < 2 or len(nonmembers) < 2:
        return {"auc": None, "bootstrap_95_ci": [None, None]}
    labels = np.r_[np.ones(len(members)), np.zeros(len(nonmembers))]
    auc = float(roc_auc_score(labels, np.r_[members, nonmembers]))
    samples = []
    for _ in range(repeats):
        scores = np.r_[rng.choice(members, len(members), replace=True),
                       rng.choice(nonmembers, len(nonmembers), replace=True)]
        samples.append(roc_auc_score(labels, scores))
    return {"auc": auc, "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Baseline folder containing best_model.pt and config.json")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, help="Resume an existing audit directory")
    parser.add_argument("--n-shadows", type=int, default=N_SHADOWS)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.n_shadows < 24 or min(args.batch_size, args.cpu_threads, args.bootstrap) < 1:
        parser.error("Use >=24 shadows and positive batch/thread/bootstrap counts")
    root, model_dir = args.data_dir.resolve(), args.model_dir.resolve()
    config_path, checkpoint_path = model_dir / "config.json", model_dir / "best_model.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_path = root / "feature_names.json"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if config["preparation_report"] != report or config["feature_names_sha256"] != sha256(feature_path):
        raise ValueError("Processed data and target training run do not match")
    assignments_path = root / "patient_splits.csv"
    assignments = pd.read_csv(assignments_path, dtype=str)
    if assignments.patient_nbr.isna().any() or assignments.patient_nbr.duplicated().any():
        raise ValueError("Invalid patient_splits.csv")
    names = ("train", "auxiliary", "test", "validation")
    expected = {split: set(assignments.loc[assignments.split == split, "patient_nbr"])
                for split in names}
    if any(not ids for ids in expected.values()) or sum(map(len, expected.values())) != len(assignments):
        raise ValueError("Unexpected split names or an empty split")
    if len(set().union(*expected.values())) != len(assignments):
        raise ValueError("Patients shared across splits")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["feature_names_sha256"] != config["feature_names_sha256"] or
        checkpoint["input_dim"] != config["input_dim"] or
        list(checkpoint["hidden_dims"]) != list(config["hidden_dims"])):
        raise ValueError("Checkpoint and target config do not match")
    target = MLP(config["input_dim"], config["hidden_dims"]).to(device)
    target.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target.eval()
    data = {split: split_data(root, split, expected[split], config["input_dim"])
            for split in names}
    all_encounters = pd.concat([data[s][2].encounter_id for s in names], ignore_index=True)
    if all_encounters.duplicated().any():
        raise ValueError("Encounter appears in multiple splits")
    pool_x = sparse.vstack([data[s][0] for s in names[:3]], format="csr")
    pool_y = np.concatenate([data[s][1] for s in names[:3]])
    pool_ids = np.concatenate([data[s][2].patient_nbr.to_numpy() for s in names[:3]])
    patient_ids, codes, counts, groups = patient_groups(pool_ids)
    if len(patient_ids) != sum(len(expected[s]) for s in names[:3]):
        raise ValueError("Inconsistent patient grouping")
    lookup = {patient: split for split in names[:3] for patient in expected[split]}
    patient_splits = np.array([lookup[p] for p in patient_ids])
    val_x, val_y, val_meta = data["validation"]
    _, val_codes, val_counts, _ = patient_groups(val_meta.patient_nbr.to_numpy())
    validation = val_x, val_y, val_codes, val_counts
    n_train = len(expected["train"])
    masks, mask_diagnostics = make_masks(args.n_shadows, len(patient_ids), n_train)
    print("Reference sampling:", mask_diagnostics, flush=True)
    signature_paths = [checkpoint_path, config_path, root / "report.json", feature_path,
                       assignments_path] + [root / name / filename for name in names
                                            for filename in ("X.npz", "y.npy", "metadata.csv")]
    signature = {str(p.resolve()): sha256(p) for p in signature_paths}
    manifest = {"attack_version": ATTACK_VERSION, "script_sha256": sha256(Path(__file__)),
                "mask_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
                "cpu_threads": args.cpu_threads, "data_files_sha256": signature, "n_shadows": args.n_shadows,
                "seed": 2026, "shadow_seed_start": 43001,
                "torch_version": str(torch.__version__), "device": str(device),
                "patient_count": len(patient_ids), "n_target_train": n_train,
                "target_epoch": checkpoint["epoch"], "batch_size": args.batch_size}
    if args.run_dir:
        output = args.run_dir.resolve()
        if json.loads((output / "manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cannot resume: source files or attack settings changed")
    else:
        output = args.output_root.resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S") +
                                               "_" + uuid.uuid4().hex[:6])
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    shadow_dir = output / "shadows"
    shadow_dir.mkdir(exist_ok=True)
    print(f"Target epoch {checkpoint['epoch']}; patient pool {len(patient_ids)}; "
          f"training size/shadow {n_train}; references {args.n_shadows}; {device}", flush=True)
    print(f"Result directory / resume: {output}", flush=True)
    reference = np.empty((args.n_shadows, len(patient_ids)), dtype=np.float64)
    epochs = []
    np.savez_compressed(output / "reference_design.npz", patient_ids=patient_ids, masks=masks)
    for j in range(args.n_shadows):
        file = shadow_dir / f"shadow_{j:03d}.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as saved:
                if saved["stat"].shape != (len(patient_ids),) or not np.array_equal(saved["mask"], masks[j]):
                    raise ValueError(f"Invalid cached reference model {j}")
                reference[j] = saved["stat"]
                epoch = int(saved["best_epoch"])
        else:
            model, epoch, ap = train_shadow(j, masks[j], groups, pool_x, pool_y, validation,
                                            config, device, args.batch_size)
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "input_dim": config["input_dim"], "hidden_dims": config["hidden_dims"],
                        "best_epoch": epoch, "attack_version": ATTACK_VERSION},
                       shadow_dir / f"shadow_{j:03d}.pt")
            reference[j] = patient_statistic(model, pool_x, pool_y, codes, counts,
                                              device, args.batch_size)
            temporary = Path(str(file) + ".tmp.npz")
            np.savez_compressed(temporary, stat=reference[j], mask=masks[j],
                                best_epoch=epoch, validation_ap=ap)
            os.replace(temporary, file)
            del model
        epochs.append(epoch)
        print(f"Reference {j+1}/{args.n_shadows}: best epoch {epoch}", flush=True)
    target_stat = patient_statistic(target, pool_x, pool_y, codes, counts, device, args.batch_size)
    attack, n_in, n_out, mean_in, mean_out, score_diagnostics = patient_lira(
        target_stat, reference, masks, patient_splits == "auxiliary")
    np.save(output / "target_stat.npy", target_stat, allow_pickle=False)
    auxiliary = attack[patient_splits == "auxiliary"]
    members = attack[patient_splits == "train"]
    test = attack[patient_splits == "test"]
    thresholds = {str(fpr): calibrated_threshold(auxiliary, fpr) for fpr in (0.01, 0.05)}
    ci_rng = np.random.default_rng(9018)
    global_auc = auc_ci(members, test, args.bootstrap, ci_rng)
    operating = []
    subgroups = []
    for fpr in (0.01, 0.05):
        threshold = thresholds[str(fpr)]
        row = evaluate(members, test, threshold)
        row.update({"nominal_auxiliary_fpr": fpr, "threshold": threshold,
                    "auxiliary_observed_fpr": float(np.mean(auxiliary > threshold))})
        operating.append(row)
    for name, condition in (("1_record", counts == 1),
                            ("2_to_4_records", (counts >= 2) & (counts <= 4)),
                            ("5_plus_records", counts >= 5)):
        a = attack[condition & (patient_splits == "train")]
        b = attack[condition & (patient_splits == "test")]
        group_auc = auc_ci(a, b, args.bootstrap, ci_rng)
        for fpr in (0.01, 0.05):
            if len(a) and len(b):
                group = evaluate(a, b, thresholds[str(fpr)])
                group.update({"group": name, "nominal_global_auxiliary_fpr": fpr,
                              "auc": group_auc["auc"],
                              "auc_ci": group_auc["bootstrap_95_ci"]})
                subgroups.append(group)
    output_scores = pd.DataFrame({"patient_nbr": patient_ids, "split": patient_splits,
                                  "n_records": counts, "target_stat": target_stat,
                                  "shadow_mean_in": mean_in, "shadow_mean_out": mean_out,
                                  "n_shadow_in": n_in, "n_shadow_out": n_out,
                                  "attack_score": attack})
    for fpr in (0.01, 0.05):
        output_scores[f"decision_fpr_{fpr}"] = (attack > thresholds[str(fpr)]).astype(int)
    output_scores.to_csv(output / "patient_scores.csv", index=False)
    pd.DataFrame(subgroups).to_csv(output / "subgroup_metrics.csv", index=False)
    metrics = {"attack": ATTACK_VERSION,
               "mask_diagnostics": mask_diagnostics, "score_diagnostics": score_diagnostics,
               "checkpoint_epoch": int(checkpoint["epoch"]), "n_shadows": args.n_shadows,
               "shadow_best_epochs": epochs,
               "in_out_count_ranges": {"in": [int(n_in.min()), int(n_in.max())],
                                       "out": [int(n_out.min()), int(n_out.max())]},
               "patient_counts": {s: int(np.sum(patient_splits == s)) for s in names[:3]},
               "global": global_auc, "operating_points": operating,
               "notes": [
                   "Unit: whole patient, all stays IN or all stays OUT in each reference model.",
                   "Fresh uniform fixed-size reference samples followed by label-blind swaps to enforce minimum coverage.",
                   "Attack statistic and variance rule fixed before evaluation; never select variants by this Test AUC.",
                   "Patient statistic: mean true-label signed logit; two auxiliary-pooled residual variances; no sign clipping.",
                   "Membership score is NOT a calibrated probability of individual privacy risk.",
                   "Reference models use target-frozen features; preprocessing privacy is outside this audit.",
                   "Auditor knows all candidate patient records and outcomes; some target test patients are included in reference training, never target training.",
                   "The original test split has already been inspected for a mean-loss attack; treat this new result as exploratory, not an independent final test.",
                   "Bootstrap and Wilson intervals condition on a single target checkpoint and fixed thresholds; do not include model or shadow retraining uncertainty.",
                   "Subgroups use the GLOBAL auxiliary thresholds; subgroup FPRs may differ."]}
    (output / "mia_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False),
                                              encoding="utf-8")
    labels = np.r_[np.ones(len(members)), np.zeros(len(test))]
    observed_fpr, tpr, _ = roc_curve(labels, np.r_[members, test])
    pd.DataFrame({"observed_fpr": observed_fpr, "tpr": tpr}).to_csv(
        output / "roc_points.csv", index=False)
    print(f"LiRA AUC = {global_auc['auc']:.4f} | bootstrap 95% CI {global_auc['bootstrap_95_ci']}")
    for row in operating:
        print(f"Auxiliary target FPR {row['nominal_auxiliary_fpr']:.0%} | "
              f"Test FPR {row['observed_fpr']:.2%} | TPR {row['tpr']:.2%}")
    print("Scores are attack statistics, NOT patient risk probabilities.")


if __name__ == "__main__":
    main()
