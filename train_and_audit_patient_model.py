"""Train a NEW patient-level diabetes MLP and run a matched MIA audit.

Requires the preprocessed matrices from the original diabetes project, plus
patient_lira.py and patient_privacy_envelope.py in this script's directory.
Requires Python >=3.10, numpy, pandas, scipy, scikit-learn and torch.

Recommended first run, an explicitly labelled privacy STRESS TEST:
  python train_and_audit_patient_model.py --source-model-dir MODEL_DIR \
    --processed-dir PROCESSED_DIR --mode stress --train-patients 16000

Control run that preserves the original training-patient count:
  python train_and_audit_patient_model.py --source-model-dir MODEL_DIR \
    --processed-dir PROCESSED_DIR --mode full

The stress test changes the target model's training population and capacity;
it is NOT evidence that the previous epoch-4 checkpoint leaked strongly.
The train/test patient assignments and the original test records are never
used to select a checkpoint: only Validation AP selects the checkpoint. The
existing Test has already been examined, so all new Test numbers are still
exploratory. No experiment can guarantee strong membership inference without
looking at its results. A model below the stated Validation AP floor is a
technical attack positive control, NOT a clinically acceptable comparison.

The processed feature representation is frozen from the original training run.
In stress mode some originally training patients move to Auxiliary; learned
preprocessing therefore has seen them. The audit is conditional on that frozen
representation and DOES NOT measure preprocessing privacy. For a publishable
end-to-end claim, refit preprocessing within each model's own training subset.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(root: Path, split: str, dimension: int):
    folder = root / split
    x = sparse.load_npz(folder / "X.npz").tocsr().astype(np.float32)
    y = np.load(folder / "y.npy", allow_pickle=False)
    meta = pd.read_csv(folder / "metadata.csv", dtype={"patient_nbr": str,
                                                        "encounter_id": str})
    if (x.shape != (len(y), dimension) or len(meta) != len(y) or
        y.ndim != 1 or not np.isin(y, [0, 1]).all() or
        not np.isfinite(x.data).all() or
        meta.patient_nbr.isna().any() or meta.encounter_id.isna().any() or
        meta.encounter_id.duplicated().any() or
        not np.array_equal(meta.row_index.to_numpy(), np.arange(len(y))) or
        not np.array_equal(y, (meta.readmitted == "<30").to_numpy())):
        raise ValueError(f"Invalid X/y/metadata alignment in {folder}")
    return x, y.astype(np.float32), meta


def split_report(x, y, meta):
    counts = meta.patient_nbr.value_counts()
    return {"patients": int(len(counts)), "records": int(len(y)),
            "positive_records": int(y.sum()), "positive_rate": float(y.mean()),
            "patients_with_multiple_records": int((counts > 1).sum()),
            "max_records_per_patient": int(counts.max()),
            "features": int(x.shape[1])}


def write_split(folder: Path, x, y, meta):
    folder.mkdir(parents=True, exist_ok=False)
    assert x.shape[0] == len(y) == len(meta)
    ordered = meta.copy().reset_index(drop=True)
    ordered["row_index"] = np.arange(len(y))
    sparse.save_npz(folder / "X.npz", x.tocsr(), compressed=True)
    np.save(folder / "y.npy", y.astype(np.float32), allow_pickle=False)
    ordered.to_csv(folder / "metadata.csv", index=False)


def create_stress_view(original: Path, destination: Path, dimension: int,
                       n_train: int, seed: int):
    """Keep entire patients IN or OUT; moved patients become nonmembers."""
    train_x, train_y, train_meta = read_split(original, "train", dimension)
    aux_x, aux_y, aux_meta = read_split(original, "auxiliary", dimension)
    ids = np.sort(train_meta.patient_nbr.unique())
    if n_train < 1000 or n_train >= len(ids):
        raise ValueError(f"--train-patients must be in [1000, {len(ids)-1}]")
    picked = set(np.random.default_rng(seed).choice(ids, n_train, replace=False))
    train_rows = train_meta.patient_nbr.isin(picked).to_numpy()
    if (train_rows.sum() == 0 or not train_rows.any() or
        set(train_meta.loc[train_rows, "patient_nbr"]) != picked):
        raise ValueError("Patient-level target sampling failed")
    x_in, y_in, meta_in = train_x[train_rows], train_y[train_rows], train_meta[train_rows]
    x_out = sparse.vstack((aux_x, train_x[~train_rows]), format="csr")
    y_out = np.r_[aux_y, train_y[~train_rows]]
    meta_out = pd.concat((aux_meta, train_meta[~train_rows]), ignore_index=True)
    if set(meta_in.patient_nbr) & set(meta_out.patient_nbr):
        raise ValueError("A patient entered both training and Auxiliary")
    destination.mkdir(parents=True, exist_ok=False)
    write_split(destination / "train", x_in, y_in, meta_in)
    write_split(destination / "auxiliary", x_out, y_out, meta_out)
    for name in ("validation", "test"):
        source = original / name
        target = destination / name
        target.mkdir()
        for file in ("X.npz", "y.npy", "metadata.csv"):
            shutil.copy2(source / file, target / file)
    shutil.copy2(original / "feature_names.json", destination / "feature_names.json")
    assignments = pd.read_csv(original / "patient_splits.csv", dtype=str)
    if (assignments.patient_nbr.isna().any() or
        assignments.patient_nbr.duplicated().any()):
        raise ValueError("Invalid patient_splits.csv")
    originally_train = set(ids)
    if set(assignments.loc[assignments.split == "train", "patient_nbr"]) != originally_train:
        raise ValueError("Assignments and original training records disagree")
    assignments.loc[(assignments.split == "train") &
                    (~assignments.patient_nbr.isin(picked)), "split"] = "auxiliary"
    assignments.to_csv(destination / "patient_splits.csv", index=False)
    report = json.loads((original / "report.json").read_text(encoding="utf-8"))
    report = copy.deepcopy(report)
    report["splits"]["train"] = split_report(x_in, y_in, meta_in)
    report["splits"]["auxiliary"] = split_report(x_out, y_out, meta_out)
    report.setdefault("notes", []).append(
        "Experimental stress split. Original train-fitted preprocessing is FROZEN; "
        "some Auxiliary patients contributed to fitting that transform. Audit the "
        "neural-network checkpoint only, conditional on the representation.")
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report, len(ids)


def evaluate_ap(model, x, y, ids, device, batch_size):
    from patient_lira import patient_groups, validation_ap
    _, codes, counts, _ = patient_groups(ids.to_numpy())
    return validation_ap(model, (x, y, codes, counts), device, batch_size)


def train_target(data_dir: Path, output: Path, old_config: dict, dims: list[int],
                 seed: int, max_epochs: int, device_name: str, min_ap: float):
    import torch
    from patient_lira import patient_groups, split_data, train_shadow

    report = json.loads((data_dir / "report.json").read_text(encoding="utf-8"))
    feature_path = data_dir / "feature_names.json"
    if sha256(feature_path) != old_config["feature_names_sha256"]:
        raise ValueError("Target features differ from original config")
    assignments = pd.read_csv(data_dir / "patient_splits.csv", dtype=str)
    ids = {name: set(assignments.loc[assignments.split == name, "patient_nbr"])
           for name in ("train", "validation", "auxiliary", "test")}
    if any(not group for group in ids.values()) or sum(map(len, ids.values())) != len(assignments):
        raise ValueError("Invalid or empty patient assignment")
    train_x, train_y, train_meta = split_data(
        data_dir, "train", ids["train"], old_config["input_dim"])
    val_x, val_y, val_meta = split_data(
        data_dir, "validation", ids["validation"], old_config["input_dim"])
    _, _, _, groups = patient_groups(train_meta.patient_nbr.to_numpy())
    _, val_codes, val_counts, _ = patient_groups(val_meta.patient_nbr.to_numpy())
    validation = val_x, val_y, val_codes, val_counts

    config = dict(old_config)
    config.update({"seed": seed, "hidden_dims": dims, "max_epochs": max_epochs,
                   "preparation_report": report,
                   "data_dir": str(data_dir.resolve()),
                   "torch_version": str(torch.__version__),
                   "feature_names_sha256": sha256(feature_path),
                   "selection_metric": "validation patient-weighted average precision",
                   "experiment": "new model / explicit privacy stress test"})
    torch.set_num_threads(4)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    # train_shadow's seed is 43001+index. An all-True mask means precisely
    # this view's training patients. The routine is identical to the reference
    # models subsequently trained by patient_lira.py (patient batches, Adam,
    # validation AP checkpoint selection, early stopping).
    model, best_epoch, best_ap = train_shadow(
        seed - 43001, np.ones(len(groups), dtype=bool), groups, train_x,
        train_y, validation, config, device, batch_size=1024)
    train_ap = evaluate_ap(model, train_x, train_y, train_meta.patient_nbr,
                           device, 1024)
    model_dir = output / "target_model"
    model_dir.mkdir()
    config["device"] = str(device)
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "epoch": int(best_epoch),
                "feature_names_sha256": config["feature_names_sha256"],
                "input_dim": config["input_dim"], "hidden_dims": dims},
               model_dir / "best_model.pt")
    summary = {"train_patients": len(groups), "target_epoch": int(best_epoch),
               "train_ap": float(train_ap), "validation_ap": float(best_ap),
               "min_validation_ap": min_ap, "meets_utility_floor": bool(best_ap >= min_ap),
               "checkpoint_sha256": sha256(model_dir / "best_model.pt")}
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return model_dir, summary


def enough_shadows(n: int, n_train: int, pool: int, seed=2026):
    # Mirror patient_lira.py's fixed mask design before launching hours of
    # training, and ensure >=6 IN and >=6 OUT for EVERY candidate patient.
    from numpy.random import default_rng
    permutation = default_rng(seed).permutation(pool)
    counts = np.zeros(pool, dtype=np.int16)
    for j in range(n):
        start = (j*n_train) % pool
        selected = permutation[(start + np.arange(n_train)) % pool]
        counts[selected] += 1
    return bool(np.min(np.minimum(counts, n-counts)) >= 6)


def shadow_count(n_train: int, pool: int, requested: int | None) -> int:
    if requested is not None:
        if requested < 24 or not enough_shadows(requested, n_train, pool):
            raise ValueError("--n-shadows leaves some patients with <6 IN/OUT "
                             "reference models; increase it")
        return requested
    for n in range(24, 161, 4):
        if enough_shadows(n, n_train, pool):
            return n
    raise ValueError("Too few training patients for <=160 shadow models")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-model-dir", required=True, type=Path,
                        help="Original baseline folder, with config.json")
    parser.add_argument("--processed-dir", required=True, type=Path,
                        help="Original data/processed folder")
    parser.add_argument("--mode", choices=("stress", "full"), default="stress")
    parser.add_argument("--train-patients", type=int, default=16000,
                        help="Only used by stress mode; default 16000")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--min-validation-ap", type=float, default=.14,
                        help="Explicit clinical utility floor; below floor = positive control only")
    parser.add_argument("--n-shadows", type=int,
                        help="Automatically chosen when omitted; >=6 IN/OUT per patient")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip the time-consuming LiRA and envelope phase")
    parser.add_argument("--resume-run", type=Path,
                        help="Continue a prior output folder; reuse target and completed shadows")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent /
                        "results" / "new_model_audit")
    args = parser.parse_args()
    if (args.max_epochs < 2 or args.seed < 1 or not args.hidden_dims or
        any(width < 1 for width in args.hidden_dims) or
        not 0 < args.min_validation_ap <= 1):
        parser.error("Invalid model or utility settings")
    directory = Path(__file__).resolve().parent
    for dependency in ("patient_lira.py", "patient_privacy_envelope.py",
                       "shadow_patient_heterogeneity.py"):
        if not (directory / dependency).is_file():
            parser.error(f"Missing {dependency} next to this script")
    source = args.processed_dir.resolve()
    old_model = args.source_model_dir.resolve()
    original_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    old_config = json.loads((old_model / "config.json").read_text(encoding="utf-8"))
    if (old_config["preparation_report"] != original_report or
        old_config["feature_names_sha256"] != sha256(source / "feature_names.json")):
        raise ValueError("Original config and processed data do not match")
    if args.resume_run:
        output = args.resume_run.resolve()
        summary_path, model_dir = output / "training_summary.json", output / "target_model"
        if not summary_path.is_file() or not (model_dir / "best_model.pt").is_file():
            raise FileNotFoundError("--resume-run needs training_summary.json and target_model/")
        info = json.loads(summary_path.read_text(encoding="utf-8"))
        if sha256(model_dir / "best_model.pt") != info["checkpoint_sha256"]:
            raise ValueError("Saved target checkpoint changed; cannot resume")
        data_dir = output / "processed_stress" if (output / "processed_stress").is_dir() else source
        if args.mode == "stress" and data_dir == source:
            raise ValueError("Stress mode selected but --resume-run has no processed_stress/")
        if args.mode == "full" and data_dir != source:
            raise ValueError("Full mode selected for a stress --resume-run folder")
    else:
        output = args.output_root.resolve() / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        output.mkdir(parents=True, exist_ok=False)
        if args.mode == "stress":
            data_dir = output / "processed_stress"
            create_stress_view(source, data_dir, old_config["input_dim"],
                               args.train_patients, args.seed)
        else:
            data_dir = source
        model_dir, info = train_target(data_dir, output, old_config,
                                       args.hidden_dims, args.seed,
                                       args.max_epochs, args.device,
                                       args.min_validation_ap)
    print(f"New target: {model_dir} | epoch={info['target_epoch']} "
          f"| validation AP={info['validation_ap']:.4f} "
          f"| clinical utility floor={args.min_validation_ap:.4f} "
          f"| passed={info['meets_utility_floor']}", flush=True)
    if not info["meets_utility_floor"]:
        print("BELOW UTILITY FLOOR: positive control ONLY, not a valid clinical "
              "privacy/utility result.", flush=True)
    if args.train_only:
        print(f"Training result: {output}")
        return
    n_train = info["train_patients"]
    n_pool = (len(pd.read_csv(data_dir / "patient_splits.csv")) -
              json.loads((data_dir / "report.json").read_text(encoding="utf-8"))
              ["splits"]["validation"]["patients"])
    n_shadows = shadow_count(n_train, n_pool, args.n_shadows)
    print(f"Training matched LiRA references: {n_shadows} shadows; "
          f"{n_train} patients per shadow, {n_pool} candidate patients", flush=True)
    lira_root = output / "lira"
    existing = [p for p in lira_root.iterdir() if (p / "manifest.json").is_file()] \
        if lira_root.is_dir() else []
    if len(existing) > 1:
        raise ValueError("Several LiRA runs exist; choose one before resuming")
    if existing:
        prior = json.loads((existing[0] / "manifest.json").read_text(encoding="utf-8"))
        if prior["n_shadows"] != n_shadows:
            raise ValueError("Requested --n-shadows differs from previous LiRA run")
    command = [sys.executable, str(directory / "patient_lira.py"),
               "--model-dir", str(model_dir), "--data-dir", str(data_dir),
               "--output-root", str(lira_root), "--n-shadows", str(n_shadows),
               "--device", args.device]
    if existing:
        command += ["--run-dir", str(existing[0])]
    subprocess.run(command, check=True)
    lira_dirs = [p for p in lira_root.iterdir() if (p / "manifest.json").is_file()]
    if len(lira_dirs) != 1:
        raise RuntimeError("Could not identify the new LiRA run")
    subprocess.run([sys.executable, str(directory / "shadow_patient_heterogeneity.py"),
                    "--lira-run", str(lira_dirs[0]),
                    "--output-root", str(output / "shadow_patient")], check=True)
    command = [sys.executable, str(directory / "patient_privacy_envelope.py"),
               "--model-dir", str(model_dir), "--processed-dir", str(data_dir),
               "--lira-run", str(lira_dirs[0]),
               "--output-root", str(output / "envelope"), "--device", args.device]
    subprocess.run(command, check=True)
    print(f"Experiment folder: {output}")
    print("STRESS results are exploratory and assess a DIFFERENT model than the "
          "original epoch-4 checkpoint; strong leakage is not guaranteed.")


if __name__ == "__main__":
    main()
