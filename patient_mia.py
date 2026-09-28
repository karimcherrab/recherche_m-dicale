"""Audit MIA simple par patient: score = - moyenne des BCE de ses dossiers.
Python >= 3.10. Installation:
python -m pip install torch numpy pandas scipy scikit-learn matplotlib
Modifier les chemins ci-dessous puis lancer: python patient_mia.py

Auxiliary (non-membres) calibre les seuils. Train et Test evaluent l'attaque.
Validation n'est pas utilisee. Aucun entrainement ni modification du modele.
Le score n'est PAS une probabilite individuelle de divulgation.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import uuid

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================= CHEMINS A MODIFIER =================
DATA_DIR = Path(r"D:\Doctorat\prototype\data\processed")
MODEL_DIR = Path(r"D:\Doctorat\prototype\results\baseline\20260928_120542_756287")
OUTPUT_ROOT = Path(r"D:\Doctorat\prototype\results\mia")
# MODEL_DIR doit contenir best_model.pt ET config.json du meme entrainement.
TARGET_FPRS = (0.01, 0.05)
BOOTSTRAP_REPEATS = 300
SEED = 2026
BATCH_SIZE = 1024
CPU_THREADS = 4
DEVICE = "auto"
# =====================================================


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def score_patients(model, root, split, device, input_dim, expected_ids):
    folder = root / split
    X = sparse.load_npz(folder / "X.npz").tocsr().astype(np.float32)
    y = np.load(folder / "y.npy", allow_pickle=False)
    meta = pd.read_csv(folder / "metadata.csv",
                       dtype={"patient_nbr": str, "encounter_id": str})
    if X.shape != (len(y), input_dim) or len(meta) != len(y) or y.ndim != 1:
        raise ValueError(f"{split}: dimensions incompatibles")
    if not np.isfinite(X.data).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{split}: donnees invalides")
    if not np.array_equal(meta.row_index, np.arange(len(y))):
        raise ValueError(f"{split}: ordre des lignes invalide")
    if not np.array_equal(y, (meta.readmitted == "<30").to_numpy()):
        raise ValueError(f"{split}: labels desalignes")
    if meta.patient_nbr.isna().any() or meta.encounter_id.duplicated().any():
        raise ValueError(f"{split}: identifiants invalides")
    if set(meta.patient_nbr) != expected_ids:
        raise ValueError(f"{split}: patients incompatibles avec patient_splits.csv")
    losses = []
    for start in range(0, len(y), BATCH_SIZE):
        x = torch.from_numpy(X[start:start+BATCH_SIZE].toarray()).to(device)
        # Float64 stable BCE also retains precision for small losses.
        logits = model(x).cpu().numpy().astype(np.float64)
        loss = np.logaddexp(0, logits) - y[start:start+BATCH_SIZE] * logits
        losses.append(loss)
    losses = np.concatenate(losses)
    if not np.isfinite(losses).all():
        raise RuntimeError("Loss non finie")
    frame = pd.DataFrame({"patient_nbr": meta.patient_nbr, "loss": losses})
    patients = frame.groupby("patient_nbr", sort=True).agg(
        mean_loss=("loss", "mean"), n_records=("loss", "size")).reset_index()
    patients["attack_score"] = -patients.mean_loss
    patients["split"] = split
    patients["membership"] = int(split == "train")
    return patients


def calibrate(scores, target_fpr):
    # Higher score => member. Strict > handles ties conservatively.
    # At most floor(alpha*n) auxiliary scores exceed this threshold.
    values = np.sort(np.asarray(scores, dtype=float))
    allowed = int(math.floor(target_fpr * len(values)))
    index = len(values) - allowed - 1
    return float(values[index])


def wilson(successes, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-half), min(1., center+half)]


def auc_with_ci(members, nonmembers, repeats, rng):
    m, n = len(members), len(nonmembers)
    if m == 0 or n == 0:
        return {"auc": None, "bootstrap_95_ci": [None, None]}
    labels = np.r_[np.ones(m), np.zeros(n)]
    auc = float(roc_auc_score(labels, np.r_[members, nonmembers]))
    samples = []
    for _ in range(repeats):
        sample = np.r_[rng.choice(members, m, replace=True),
                        rng.choice(nonmembers, n, replace=True)]
        samples.append(roc_auc_score(labels, sample))
    return {"auc": auc, "bootstrap_95_ci": np.quantile(samples, [.025, .975]).tolist()}


def evaluate(members, nonmembers, threshold):
    tp = int((members > threshold).sum())
    fp = int((nonmembers > threshold).sum())
    m, n = len(members), len(nonmembers)
    tpr, fpr = (tp/m if m else None), (fp/n if n else None)
    return {"members": m, "nonmembers": n, "true_positives": tp,
            "false_positives": fp, "tpr": tpr, "observed_fpr": fpr,
            "tpr_95_ci": wilson(tp, m), "fpr_95_ci": wilson(fp, n),
            "advantage_tpr_minus_fpr": tpr-fpr if m and n else None,
            "balanced_accuracy": (tpr+1-fpr)/2 if m and n else None}


def main():
    if BOOTSTRAP_REPEATS < 100 or BATCH_SIZE < 1:
        raise ValueError("Utiliser au moins 100 bootstrap et un batch positif")
    if not all(0 < f < 1 for f in TARGET_FPRS):
        raise ValueError("Les FPR doivent etre entre 0 et 1")
    root, model_dir = Path(DATA_DIR), Path(MODEL_DIR)
    checkpoint_path = model_dir / "best_model.pt"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report != config["preparation_report"]:
        raise ValueError("Preparation differente de celle utilisee pour entrainer le modele.")
    feature_hash = sha256(root / "feature_names.json")
    if feature_hash != config["feature_names_sha256"]:
        raise ValueError("Encodage des caracteristiques incompatible")
    assignments = pd.read_csv(root / "patient_splits.csv", dtype=str)
    if assignments.patient_nbr.isna().any() or assignments.patient_nbr.duplicated().any():
        raise ValueError("Patients manquants/dupliques dans les groupes")
    names = ("auxiliary", "train", "test")
    ids = {s: set(assignments.loc[assignments.split == s, "patient_nbr"]) for s in names}
    if not all(ids.values()):
        raise ValueError("Groupe vide")
    torch.set_num_threads(CPU_THREADS)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if DEVICE == "auto" else DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["feature_names_sha256"] != feature_hash:
        raise ValueError("Checkpoint incompatible avec les caracteristiques")
    model = MLP(checkpoint["input_dim"], checkpoint["hidden_dims"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"Modele gele: epoch {checkpoint['epoch']} | Device: {device}", flush=True)
    frames = {}
    # Calibrate before loading/evaluating Train and Test.
    frames["auxiliary"] = score_patients(model, root, "auxiliary", device,
                                         checkpoint["input_dim"], ids["auxiliary"])
    auxiliary = frames["auxiliary"].attack_score.to_numpy()
    thresholds = {str(f): calibrate(auxiliary, f) for f in TARGET_FPRS}
    for name in ("train", "test"):
        frames[name] = score_patients(model, root, name, device,
                                      checkpoint["input_dim"], ids[name])
        print(f"Scores calcules: {name}, {len(frames[name])} patients", flush=True)
    members = frames["train"].attack_score.to_numpy()
    nonmembers = frames["test"].attack_score.to_numpy()
    rng = np.random.default_rng(SEED)
    print("Calcul des intervalles de confiance...", flush=True)
    global_auc = auc_with_ci(members, nonmembers, BOOTSTRAP_REPEATS, rng)
    operating = []
    subgroup_rows = []
    for target in TARGET_FPRS:
        threshold = thresholds[str(target)]
        result = evaluate(members, nonmembers, threshold)
        result.update({"nominal_calibration_fpr": target, "threshold": threshold,
                       "auxiliary_observed_fpr": float(np.mean(auxiliary > threshold))})
        operating.append(result)
    for label, low, high in [("1_record", 1, 1), ("2_to_4_records", 2, 4),
                              ("5_plus_records", 5, float("inf"))]:
        a = frames["train"].loc[frames["train"].n_records.between(low, high), "attack_score"].to_numpy()
        b = frames["test"].loc[frames["test"].n_records.between(low, high), "attack_score"].to_numpy()
        auc = auc_with_ci(a, b, BOOTSTRAP_REPEATS, rng)
        for target in TARGET_FPRS:
            row = evaluate(a, b, thresholds[str(target)])
            row.update({"group": label, "nominal_global_calibration_fpr": target,
                        "auc": auc["auc"], "auc_ci_low": auc["bootstrap_95_ci"][0],
                        "auc_ci_high": auc["bootstrap_95_ci"][1],
                        "small_group_warning": min(len(a), len(b)) < 100})
            subgroup_rows.append(row)
    folder = Path(OUTPUT_ROOT) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        for target in TARGET_FPRS:
            frame[f"predicted_member_nominal_fpr_{target}"] = (frame.attack_score > thresholds[str(target)]).astype(int)
        frame.to_csv(folder / f"{name}_patient_scores.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(folder / "subgroup_metrics.csv", index=False)
    result = {"attack": "negative_mean_patient_BCE", "checkpoint_sha256": sha256(checkpoint_path),
              "checkpoint_epoch": checkpoint["epoch"], "feature_names_sha256": feature_hash,
              "seed": SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS,
              "torch_version": str(torch.__version__), "device": str(device),
              "model_dir": str(model_dir.resolve()), "data_dir": str(root.resolve()),
              "patient_counts": {s: len(frames[s]) for s in names},
              "global": global_auc, "operating_points": operating,
              "notes": [
                  "Attacker knows all observed records and labels of each candidate patient and queries model probabilities.",
                  "Members: Train. Nonmember evaluation: Test. Nonmember calibration: Auxiliary. Validation unused.",
                  "Fixed score direction: high score means member. No direction/aggregation tuned on Test.",
                  "Nominal FPR refers to Auxiliary calibration. Actual Test FPR can differ; report both TPR and observed FPR.",
                  "Scores and binary attack decisions are NOT individual risk probabilities or clinical predictions.",
                  "AUC bootstrap resamples patients separately by membership, conditional on this fixed trained model.",
                  "Wilson intervals condition on the calibrated threshold; they exclude calibration/retraining uncertainty.",
                  "Subgroups use the GLOBAL threshold, so their FPR need not equal the nominal calibration FPR.",
                  "Low attack AUC is not a privacy guarantee; strong/individualized attacks may perform differently.",
                  "This is a single-model empirical audit, not an estimate of worst-case individual patient risk.",
                  "Test is now an audit set: do not repeatedly tune attacks/defenses on it and claim independent final evaluation."]}
    (folder / "mia_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    labels = np.r_[np.ones(len(members)), np.zeros(len(nonmembers))]
    fpr, tpr, _ = roc_curve(labels, np.r_[members, nonmembers])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Patient loss attack: AUC={global_auc['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
    for row in operating:
        ax.scatter(row["observed_fpr"], row["tpr"], label=f"Auxiliary FPR target {row['nominal_calibration_fpr']:.0%}")
    ax.set(xlabel="Observed false positive rate (Test)", ylabel="True positive rate (Train)",
           title="Membership inference per patient", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "mia_roc.png", dpi=160); plt.close(fig)
    print(f"\nMIA AUC = {global_auc['auc']:.4f} | IC95% = {global_auc['bootstrap_95_ci']}")
    for row in operating:
        print(f"FPR cible Auxiliary={row['nominal_calibration_fpr']:.1%} | "
              f"FPR observe Test={row['observed_fpr']:.2%} | TPR={row['tpr']:.2%}")
    print(f"Resultats: {folder.resolve()}")
    print("Scores d'attaque, PAS probabilites de risque individuel. Aucun modele reentraine.")


if __name__ == "__main__":
    main()
