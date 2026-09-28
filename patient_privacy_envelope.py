"""Patient-level multi-attack membership audit of the FROZEN diabetes MLP.

Requires Python >=3.10, numpy, pandas, scipy, scikit-learn; full mode also
requires torch (the same environment used to train the model).

Full audit, with fresh queries of the saved target checkpoint (one line):
    python patient_privacy_envelope.py --model-dir MODEL_DIR --processed-dir PROCESSED_DIR --lira-run LIRA_RUN

Smoke/analysis mode when the processed matrices and shadows are unavailable:
    python patient_privacy_envelope.py --scores-only --raw-csv diabetic_data.csv \
      --model-dir MODEL_DIR --lira-run LIRA_RUN

The full mode verifies the checkpoint and preprocessing, queries every patient's
observed stays, loads IN/OUT shadow scores, and checks that it reproduces the
previous LiRA result. The scores-only mode uses SAVED target outputs and skips
the best-record and OUT-z candidates. It is explicitly labelled in the report.

Attacker's privileged knowledge: all candidate visits and readmission labels,
plus membership of a disjoint subset of target-training patients for the
supervised attack. No protected attribute or patient ID enters the classifier.
Scores, tail p-values and flags are NOT probabilities of individual privacy
loss, nor estimates of data reconstruction. This single checkpoint's Test has
already been examined: every resulting performance claim is exploratory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


SEED = 2028
FPRS = (0.01, 0.05)
STRATA = ("1_record", "2_plus_records")
BASE_FEATURES = ("log_n_records", "positive_fraction", "late_fraction")
MODEL_FEATURES = BASE_FEATURES + (
    "target_stat", "shadow_mean_out", "shadow_mean_in", "shadow_effect",
    "lira_shared", "positive_target_interaction")
EXTRA_FEATURES = ("best_record_score", "p90_loss", "sd_loss", "mean_margin",
                  "mean_entropy", "positive_mean_loss", "negative_mean_loss")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")


def verify_checkpoint(model_dir: Path, lira_manifest: dict) -> dict:
    config_path = model_dir / "config.json"
    ckpt_path = model_dir / "best_model.pt"
    if not config_path.is_file() or not ckpt_path.is_file():
        raise FileNotFoundError(
            f"--model-dir needs BOTH config.json and best_model.pt: {model_dir}")
    recorded = lira_manifest["data_files_sha256"]
    # The original manifest contains absolute Windows paths; compare basenames
    # without treating them as local POSIX paths.
    for filename, path in (("config.json", config_path), ("best_model.pt", ckpt_path)):
        matches = [digest for name, digest in recorded.items()
                   if name.replace("/", "\\").endswith("\\" + filename)]
        if len(matches) != 1 or matches[0] != sha256(path):
            raise ValueError(f"{filename} does not match the LiRA run's target")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_prior(run: Path) -> tuple[dict, pd.DataFrame]:
    if not (run / "manifest.json").is_file():
        raise FileNotFoundError(
            f"--lira-run must point to a patient_lira.py output with manifest.json: {run}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(run / "patient_scores.csv", dtype={"patient_nbr": str})
    require_columns(scores, ("patient_nbr", "split", "n_records", "target_stat",
                             "attack_score", "shadow_mean_in", "shadow_mean_out"),
                    "LiRA patient_scores.csv")
    if (len(scores) != manifest["patient_count"] or
        scores.patient_nbr.isna().any() or scores.patient_nbr.duplicated().any() or
        set(scores.split) != {"train", "auxiliary", "test"} or
        (scores.n_records < 1).any()):
        raise ValueError("Malformed or mismatched LiRA patient_scores.csv")
    cols = ["target_stat", "attack_score", "shadow_mean_in", "shadow_mean_out"]
    if not np.isfinite(scores[cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite LiRA scores")
    return manifest, scores


def shadow_candidates(run: Path, manifest: dict, saved: pd.DataFrame) -> dict:
    n = len(saved)
    scores, masks = [], []
    for i in range(manifest["n_shadows"]):
        file = run / "shadows" / f"shadow_{i:03d}.npz"
        if not file.is_file():
            raise FileNotFoundError(f"Missing {file}; use --scores-only for a limited audit")
        with np.load(file, allow_pickle=False) as obj:
            stat = np.asarray(obj["stat"], dtype=float)
            mask = np.asarray(obj["mask"], dtype=bool)
        if (stat.shape != (n,) or mask.shape != (n,) or
            mask.sum() != manifest["n_target_train"] or not np.isfinite(stat).all()):
            raise ValueError(f"Invalid shadow data: {file}")
        scores.append(stat)
        masks.append(mask)
    scores, masks = np.stack(scores), np.stack(masks)
    n_in = masks.sum(axis=0)
    n_out = len(scores) - n_in
    if min(n_in.min(), n_out.min()) < 6:
        raise ValueError("Fewer than 6 reference scores IN or OUT for a patient")
    mean_in = np.where(masks, scores, 0).sum(axis=0) / n_in
    mean_out = np.where(~masks, scores, 0).sum(axis=0) / n_out
    var_in = np.where(masks, (scores - mean_in) ** 2, 0).sum(axis=0) / (n_in - 1)
    var_out = np.where(~masks, (scores - mean_out) ** 2, 0).sum(axis=0) / (n_out - 1)
    target = saved.target_stat.to_numpy(dtype=float)
    adjusted_in = np.maximum(mean_in, mean_out)
    var_shared = np.maximum((var_in + var_out) / 2, .1 ** 2)
    lira = ((target - mean_out) ** 2 - (target - adjusted_in) ** 2) / (2 * var_shared)
    for name, actual, expected in (
        ("LiRA score", lira, saved.attack_score),
        ("shadow IN mean", mean_in, saved.shadow_mean_in),
        ("shadow OUT mean", mean_out, saved.shadow_mean_out),
    ):
        if not np.allclose(actual, expected.to_numpy(dtype=float), atol=1e-8, rtol=1e-8):
            raise ValueError(f"{name} disagrees with saved run (check file order/run)")
    return {"lira_shared": lira,
            "out_z": (target - mean_out) / np.sqrt(np.maximum(var_out, .1 ** 2)),
            "out_gap": target - mean_out}


def raw_patient_summary(raw_path: Path, saved: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, usecols=["patient_nbr", "encounter_id", "readmitted"],
                      dtype={"patient_nbr": str, "encounter_id": str, "readmitted": str})
    if (raw.isna().any().any() or raw.encounter_id.duplicated().any() or
        not set(raw.readmitted).issubset({"NO", ">30", "<30"})):
        raise ValueError("Invalid IDs or readmitted labels in raw CSV")
    raw["positive"] = (raw.readmitted == "<30").astype(int)
    raw["late"] = (raw.readmitted == ">30").astype(int)
    group = raw.groupby("patient_nbr", sort=True).agg(
        n_records_raw=("encounter_id", "size"),
        positive_records=("positive", "sum"), late_records=("late", "sum"))
    joined = saved[["patient_nbr", "n_records"]].join(group, on="patient_nbr")
    if joined.n_records_raw.isna().any() or not np.array_equal(
            joined.n_records.to_numpy(), joined.n_records_raw.to_numpy()):
        raise ValueError("Raw patient visit counts do not match saved LiRA scores")
    return joined[["patient_nbr", "positive_records", "late_records"]]


def direct_patient_summary(root: Path, model_dir: Path, config: dict,
                           saved: pd.DataFrame, manifest: dict,
                           batch_size: int, device_name: str) -> pd.DataFrame:
    import torch
    from scipy import sparse
    from torch import nn

    if (json.loads((root / "report.json").read_text(encoding="utf-8")) !=
            config["preparation_report"] or
            sha256(root / "feature_names.json") != config["feature_names_sha256"]):
        raise ValueError("Processed data do not match the target preprocessing")
    # Reject a processed data folder from a different audit, even if patient
    # counts happen to match. The original run records hashes of every input.
    recorded = manifest["data_files_sha256"]
    for relative in ("report.json", "feature_names.json", "patient_splits.csv"):
        matches = [digest for original, digest in recorded.items()
                   if original.replace("/", "\\").endswith("\\" + relative)]
        if len(matches) != 1 or sha256(root / relative) != matches[0]:
            raise ValueError(f"Processed {relative} differs from LiRA run")
    for split in ("train", "auxiliary", "test"):
        for filename in ("X.npz", "y.npy", "metadata.csv"):
            relative = f"{split}\\{filename}"
            matches = [digest for original, digest in recorded.items()
                       if original.replace("/", "\\").endswith("\\" + relative)]
            if len(matches) != 1 or sha256(root / split / filename) != matches[0]:
                raise ValueError(f"Processed {split}/{filename} differs from LiRA run")
    assignments = pd.read_csv(root / "patient_splits.csv", dtype=str)
    require_columns(assignments, ("patient_nbr", "split"), "patient_splits.csv")
    if assignments.patient_nbr.duplicated().any() or assignments.patient_nbr.isna().any():
        raise ValueError("Duplicated/missing patient assignment")
    expected = {split: set(assignments.loc[assignments.split == split, "patient_nbr"])
                for split in ("train", "auxiliary", "test", "validation")}
    if (not all(expected.values()) or
        set().union(*expected.values()) != set(assignments.patient_nbr)):
        raise ValueError("Missing or unexpected patient splits")
    for split in ("train", "auxiliary", "test"):
        if set(saved.loc[saved.split == split, "patient_nbr"]) != expected[split]:
            raise ValueError(f"LiRA saved patient assignments differ: {split}")

    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu",
                            weights_only=True)
    if (checkpoint["feature_names_sha256"] != config["feature_names_sha256"] or
        checkpoint["input_dim"] != config["input_dim"] or
        list(checkpoint["hidden_dims"]) != list(config["hidden_dims"]) or
        checkpoint["epoch"] != manifest["target_epoch"]):
        raise ValueError("Checkpoint, configuration and LiRA run differ")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, previous = [], checkpoint["input_dim"]
            for width in checkpoint["hidden_dims"]:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs):
            return self.net(inputs).squeeze(-1)

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    model = MLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    grouped = []
    seen_encounters = set()
    with torch.inference_mode():
        for split in ("train", "auxiliary", "test"):
            folder = root / split
            x = sparse.load_npz(folder / "X.npz").tocsr().astype(np.float32)
            y = np.load(folder / "y.npy", allow_pickle=False)
            meta = pd.read_csv(folder / "metadata.csv",
                               dtype={"patient_nbr": str, "encounter_id": str})
            if (x.shape != (len(y), checkpoint["input_dim"]) or len(meta) != len(y) or
                y.ndim != 1 or not np.isfinite(x.data).all() or
                not np.isin(y, (0, 1)).all() or meta.patient_nbr.isna().any() or
                meta.encounter_id.isna().any() or meta.encounter_id.duplicated().any() or
                not np.array_equal(meta.row_index.to_numpy(), np.arange(len(y))) or
                not np.array_equal(y, (meta.readmitted == "<30").to_numpy()) or
                set(meta.patient_nbr) != expected[split]):
                raise ValueError(f"Processed records/labels misaligned: {split}")
            ids = set(meta.encounter_id)
            if seen_encounters & ids:
                raise ValueError("Encounter ID repeated across splits")
            seen_encounters |= ids
            logits = []
            for start in range(0, len(y), batch_size):
                batch = torch.from_numpy(x[start:start+batch_size].toarray()).to(device)
                logits.append(model(batch).cpu().numpy().astype(float))
            z = np.concatenate(logits)
            loss = np.logaddexp(0, z) - y * z
            if not np.isfinite(loss).all():
                raise ValueError(f"Non-finite target losses: {split}")
            prob = np.exp(-np.logaddexp(0, -z))
            prob = np.clip(prob, 1e-12, 1-1e-12)
            rows = pd.DataFrame({
                "patient_nbr": meta.patient_nbr.to_numpy(),
                "loss": loss, "margin": (2*y-1)*z,
                "entropy": -prob*np.log(prob)-(1-prob)*np.log(1-prob),
                "positive": y.astype(int),
                "late": (meta.readmitted == ">30").to_numpy().astype(int),
            })
            g = rows.groupby("patient_nbr", sort=True)
            part = g.agg(n_records_direct=("loss", "size"), mean_loss=("loss", "mean"),
                         positive_records=("positive", "sum"), late_records=("late", "sum"),
                         best_record_score=("loss", lambda v: -v.min()),
                         p90_loss=("loss", lambda v: v.quantile(.9)),
                         sd_loss=("loss", "std"), mean_margin=("margin", "mean"),
                         mean_entropy=("entropy", "mean"))
            part["positive_mean_loss"] = rows.loc[rows.positive == 1].groupby(
                "patient_nbr").loss.mean()
            part["negative_mean_loss"] = rows.loc[rows.positive == 0].groupby(
                "patient_nbr").loss.mean()
            part[["sd_loss", "positive_mean_loss", "negative_mean_loss"]] = part[
                ["sd_loss", "positive_mean_loss", "negative_mean_loss"]].fillna(0.)
            part["split"] = split
            grouped.append(part.reset_index())
            print(f"Checkpoint queried: {split}, {len(part)} patients", flush=True)
    direct = pd.concat(grouped, ignore_index=True)
    aligned = saved[["patient_nbr", "split", "n_records", "target_stat"]].merge(
        direct, on=["patient_nbr", "split"], how="left", validate="one_to_one")
    if (aligned.mean_loss.isna().any() or
        not np.array_equal(aligned.n_records, aligned.n_records_direct) or
        not np.allclose(-np.log(np.maximum(aligned.mean_loss, 1e-9)),
                        aligned.target_stat, rtol=1e-5, atol=1e-5)):
        raise ValueError("Fresh target scores disagree with LiRA run; check checkpoint/data")
    return aligned.drop(columns=["n_records", "target_stat", "n_records_direct"])


def partitions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    result = {}
    train = rng.permutation(np.flatnonzero(frame.split.to_numpy() == "train"))
    aux = rng.permutation(np.flatnonzero(frame.split.to_numpy() == "auxiliary"))
    result["member_fit"], result["member_audit"] = np.array_split(train, 2)
    result["aux_fit"], result["aux_rank"], result["aux_cal"] = np.array_split(aux, 3)
    result["test"] = np.flatnonzero(frame.split.to_numpy() == "test")
    if min(map(len, result.values())) < 100:
        raise ValueError("An attack/calibration partition has fewer than 100 patients")
    return result


def fit_attack(frame: pd.DataFrame, parts: dict, features) -> np.ndarray:
    fit = np.r_[parts["member_fit"], parts["aux_fit"]]
    label = np.r_[np.ones(len(parts["member_fit"])), np.zeros(len(parts["aux_fit"]))]
    n = len(fit)
    weights = np.r_[np.full(len(parts["member_fit"]), n/(2*len(parts["member_fit"]))),
                    np.full(len(parts["aux_fit"]), n/(2*len(parts["aux_fit"]))) ]
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=160, max_leaf_nodes=12,
        min_samples_leaf=80, l2_regularization=5.,
        early_stopping=False, random_state=SEED)
    model.fit(frame.iloc[fit][list(features)].to_numpy(dtype=float), label,
              sample_weight=weights)
    # Predictions for fit patients are intentionally not used in the audit.
    output = np.full(len(frame), np.nan)
    audit = np.r_[parts["member_audit"], parts["aux_rank"],
                  parts["aux_cal"], parts["test"]]
    output[audit] = model.predict_proba(
        frame.iloc[audit][list(features)].to_numpy(dtype=float))[:, 1]
    return output


def tail_p(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Conservative one-sided empirical tail with >= for ties and +1 correction."""
    ordered = np.sort(np.asarray(reference, dtype=float))
    return (1 + len(ordered) - np.searchsorted(ordered, values, side="left")) / (
        len(ordered) + 1)


def wilson(k: int, n: int) -> list[float]:
    z = 1.95996398454
    p = k/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [float(max(0, center-half)), float(min(1, center+half))]


def evaluate(frame: pd.DataFrame, parts: dict, score: np.ndarray,
             raw_cal: np.ndarray, bootstrap: int, rng: np.random.Generator) -> dict:
    member = score[parts["member_audit"]]
    test = score[parts["test"]]
    labels = np.r_[np.ones(len(member)), np.zeros(len(test))]
    auc = float(roc_auc_score(labels, np.r_[member, test]))
    boot = []
    for _ in range(bootstrap):
        boot.append(roc_auc_score(
            labels, np.r_[rng.choice(member, len(member), replace=True),
                          rng.choice(test, len(test), replace=True)]))
    points = []
    for alpha in FPRS:
        # Valid new-patient p = (1 + # calibration scores >= observed)/(n+1).
        # With ties, this is conservative. No threshold tuning uses Test.
        p_member, p_test = tail_p(raw_cal, member), tail_p(raw_cal, test)
        tp, fp = int(np.sum(p_member <= alpha)), int(np.sum(p_test <= alpha))
        points.append({"nominal_fpr": alpha, "tpr": tp/len(member),
                       "test_fpr": fp/len(test), "tp": tp, "fp": fp,
                       "member_count": len(member), "test_count": len(test),
                       "tpr_95_wilson": wilson(tp, len(member)),
                       "fpr_95_wilson": wilson(fp, len(test)),
                       "advantage": tp/len(member)-fp/len(test)})
    return {"auc": auc,
            "auc_95_bootstrap": list(map(float, np.quantile(boot, [.025, .975]))),
            "operating_points": points}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Baseline folder with config.json and best_model.pt")
    parser.add_argument("--processed-dir", type=Path,
                        help="data/processed with train/auxiliary/test matrices")
    parser.add_argument("--lira-run", type=Path, required=True,
                        help="patient_lira.py result directory with manifest.json and shadows/")
    parser.add_argument("--scores-only", action="store_true",
                        help="Use saved patient scores; requires --raw-csv; omits record/shadow attacks")
    parser.add_argument("--raw-csv", type=Path,
                        help="Original raw diabetic_data.csv (needed only for --scores-only)")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent /
                        "results" / "privacy_envelope")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.scores_only and args.raw_csv is None:
        parser.error("--scores-only requires --raw-csv")
    if not args.scores_only and args.processed_dir is None:
        parser.error("full mode requires --processed-dir; see --help")
    if args.batch_size < 1 or args.bootstrap < 1:
        parser.error("--batch-size and --bootstrap must be positive")

    manifest, frame = load_prior(args.lira_run)
    config = verify_checkpoint(args.model_dir, manifest)
    frame = frame.copy()
    full = not args.scores_only
    if full:
        candidate = shadow_candidates(args.lira_run, manifest, frame)
        extra = direct_patient_summary(args.processed_dir, args.model_dir, config, frame,
                                       manifest, args.batch_size, args.device)
    else:
        candidate = {"lira_shared": frame.attack_score.to_numpy(dtype=float),
                     "out_gap": (frame.target_stat - frame.shadow_mean_out).to_numpy(dtype=float)}
        extra = raw_patient_summary(args.raw_csv, frame)
        if config["preparation_report"]["input_sha256"] != sha256(args.raw_csv):
            raise ValueError("Raw CSV does not match the target's original data")
    frame = frame.merge(extra, on="patient_nbr", how="left", validate="one_to_one",
                        suffixes=("", "_direct"))
    if "split_direct" in frame.columns and not np.array_equal(frame.split, frame.split_direct):
        raise ValueError("Direct target patient splits disagree with LiRA run")
    if frame.positive_records.isna().any() or frame.late_records.isna().any():
        raise ValueError("Missing patient outcome summaries")
    frame["log_n_records"] = np.log1p(frame.n_records)
    frame["positive_fraction"] = frame.positive_records / frame.n_records
    frame["late_fraction"] = frame.late_records / frame.n_records
    frame["shadow_effect"] = frame.shadow_mean_in - frame.shadow_mean_out
    frame["lira_shared"] = candidate["lira_shared"]
    frame["positive_target_interaction"] = frame.positive_fraction * frame.target_stat
    features = MODEL_FEATURES + EXTRA_FEATURES if full else MODEL_FEATURES
    if not np.isfinite(frame[list(features)].to_numpy(dtype=float)).all():
        raise ValueError("Missing/non-finite supervised attack features")

    parts = partitions(frame)
    informed = fit_attack(frame, parts, features)
    control = fit_attack(frame, parts, BASE_FEATURES)
    candidate = {"mean_loss": frame.target_stat.to_numpy(dtype=float),
                 **candidate, "supervised": informed}
    if full:
        candidate["best_record"] = frame.best_record_score.to_numpy(dtype=float)
    audited = np.r_[parts["member_audit"], parts["aux_rank"],
                    parts["aux_cal"], parts["test"]]
    if not all(np.isfinite(values[audited]).all() for values in candidate.values()):
        raise ValueError("Missing/non-finite candidate scores for audit patients")

    # Calibrate each attack on a FIRST auxiliary slice, conditioning on visit
    # count (one visit vs several). Do not compare unscaled raw scores.
    strata = np.where(frame.n_records.to_numpy() == 1, STRATA[0], STRATA[1])
    p_individual = {}
    all_indices = np.arange(len(frame))
    for name, values in candidate.items():
        p = np.full(len(frame), np.nan)
        # Raw/shadow scores are independent of the supervised attack fitting
        # partition, and can audit ALL target-training patients. A supervised
        # score is meaningful only for patients withheld from its own fit.
        scope = audited if name == "supervised" else all_indices
        for stratum in STRATA:
            ref = parts["aux_rank"][strata[parts["aux_rank"]] == stratum]
            targets = scope[strata[scope] == stratum]
            if len(ref) < 100:
                raise ValueError(f"Not enough auxiliary ranking patients: {stratum}")
            p[targets] = tail_p(values[ref], values[targets])
        p_individual[name] = p
    # The envelope's second, independent auxiliary slice corrects the fact
    # that taking a maximum across methods creates additional false alarms.
    min_p = np.min(np.column_stack([p_individual[k][audited]
                                    for k in candidate]), axis=1)
    evidence = np.full(len(frame), np.nan)
    evidence[audited] = -np.log(min_p)
    final_p = np.full(len(frame), np.nan)
    final_p[audited] = tail_p(evidence[parts["aux_cal"]], evidence[audited])
    unsupervised_names = [name for name in candidate if name != "supervised"]
    unsupervised_evidence = -np.log(np.min(
        np.column_stack([p_individual[k] for k in unsupervised_names]), axis=1))
    unsupervised_p = tail_p(unsupervised_evidence[parts["aux_cal"]],
                           unsupervised_evidence)
    all_member_parts = dict(parts)
    all_member_parts["member_audit"] = np.flatnonzero(frame.split.to_numpy() == "train")
    rng = np.random.default_rng(SEED + 11)
    result = {
        "mode": "checkpoint_queries_with_shadows" if full else "saved_scores_only",
        "target_checkpoint_sha256": sha256(args.model_dir / "best_model.pt"),
        "lira_manifest_sha256": sha256(args.lira_run / "manifest.json"),
        "seed": SEED, "partition_counts": {name: len(idx) for name, idx in parts.items()},
        "methods": list(candidate), "ranking_strata": list(STRATA),
        "evaluations": {},
        "notes": [
            "Entire patients, not visits, are the independent audit units.",
            "The auditor knows candidate visits and outcomes and known membership for the supervised attack's fit patients.",
            "Patient IDs are for joining/reporting only, never model or attacker features.",
            "Two disjoint Auxiliary nonmember slices calibrate per-method ranks then the max across methods.",
            "The unsupervised envelope audits all training patients; the supervised envelope evaluates only patients held out from attacker training.",
            "The same Auxiliary ranking slice supplies self-inclusive ranks for its own patients; these are illustrative, not held-out p-values.",
            "p_envelope is a nonmember-tail p-value, NOT the probability of membership or patient-specific privacy loss.",
            "FPR validity assumes exchangeability of Auxiliary and future nonmembers for this target model.",
            "A subgroup's FPR is observed, not guaranteed equal to the global nominal FPR.",
            "This target Test has been examined in previous attacks; all results are exploratory, not a new independent test.",
            "CIs condition on one checkpoint, fitted attack, and calibration set; they exclude retraining variability.",
            "Shadow models use target-frozen preprocessing; preprocessing privacy is outside this audit.",
        ],
    }
    for name, values in {**candidate, "clinical_only_control": control,
                         "envelope": evidence}.items():
        if name == "clinical_only_control":
            # Control has the same fit/audit partition but is excluded from envelope.
            reference = control[parts["aux_cal"]]
        elif name == "envelope":
            reference = evidence[parts["aux_cal"]]
        else:
            reference = values[parts["aux_cal"]]
        result["evaluations"][name] = evaluate(frame, parts, values, reference,
                                               args.bootstrap, rng)
    result["evaluations"]["unsupervised_envelope_all_members"] = evaluate(
        frame, all_member_parts, unsupervised_evidence,
        unsupervised_evidence[parts["aux_cal"]], args.bootstrap, rng)

    rows = frame[["patient_nbr", "split", "n_records",
                                "positive_records", "late_records"]].copy()
    role = np.full(len(frame), "", dtype=object)
    for label in ("member_fit", "member_audit", "aux_fit", "aux_rank", "aux_cal", "test"):
        role[parts[label]] = label
    rows["audit_role"] = role
    for name, values in candidate.items():
        rows[f"score_{name}"] = values
        rows[f"p_rank_{name}"] = p_individual[name]
    rows["informed_envelope_stat"] = evidence
    rows["p_informed_envelope"] = final_p
    rows["unsupervised_envelope_stat"] = unsupervised_evidence
    rows["p_unsupervised_envelope"] = unsupervised_p
    for alpha in FPRS:
        rows[f"flag_informed_fpr_{alpha:.2f}"] = pd.array(
            np.where(np.isfinite(final_p), final_p <= alpha, pd.NA), dtype="boolean")
        rows[f"flag_unsupervised_fpr_{alpha:.2f}"] = (unsupervised_p <= alpha).astype(int)
    report_groups = {
        "1_record": frame.n_records.to_numpy() == 1,
        "2_to_4_records": frame.n_records.to_numpy().astype(int) >= 2,
        "5_plus_records": frame.n_records.to_numpy() >= 5,
    }
    report_groups["2_to_4_records"] &= frame.n_records.to_numpy() <= 4
    subgroups = []
    for method, selected, values, p_value in (
        ("informed_envelope", parts["member_audit"], evidence, final_p),
        ("unsupervised_envelope_all_members", all_member_parts["member_audit"],
         unsupervised_evidence, unsupervised_p),
    ):
        for name, group in report_groups.items():
            members = selected[group[selected]]
            tests = parts["test"][group[parts["test"]]]
            if not len(members) or not len(tests):
                continue
            auc = float(roc_auc_score(
                np.r_[np.ones(len(members)), np.zeros(len(tests))],
                np.r_[values[members], values[tests]]))
            for alpha in FPRS:
                tp = int(np.sum(p_value[members] <= alpha))
                fp = int(np.sum(p_value[tests] <= alpha))
                subgroups.append({"method": method, "subgroup": name,
                                  "nominal_global_fpr": alpha,
                                  "member_count": len(members), "test_count": len(tests),
                                  "auc": auc, "tp": tp, "fp": fp,
                                  "tpr": tp/len(members), "test_fpr": fp/len(tests),
                                  "tpr_95_wilson": wilson(tp, len(members)),
                                  "fpr_95_wilson": wilson(fp, len(tests))})
    output = args.output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") +
                                 "_" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "patient_audit_scores.csv", index=False)
    pd.DataFrame(subgroups).to_csv(output / "subgroup_metrics.csv", index=False)
    (output / "audit_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Mode: {result['mode']} | reference epoch: {manifest['target_epoch']}")
    print("Partitions:", result["partition_counts"])
    for name in [*candidate, "clinical_only_control", "envelope",
                 "unsupervised_envelope_all_members"]:
        summary = result["evaluations"][name]
        points = summary["operating_points"]
        print(f"{name}: AUC={summary['auc']:.4f}; " + "; ".join(
            f"TPR={p['tpr']:.2%} @ Test FPR={p['test_fpr']:.2%} "
            f"(nominal {p['nominal_fpr']:.0%})" for p in points))
    print(f"Exploratory audit saved: {output}")
    print("Scores and tail p-values are NOT individual privacy-risk probabilities.")


if __name__ == "__main__":
    main()
