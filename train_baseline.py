"""Baseline MLP, patients de poids egal. Python >= 3.10.
Installer: python -m pip install torch numpy pandas scipy scikit-learn matplotlib
Modifier DATA_DIR/OUTPUT_ROOT, puis lancer: python train_baseline.py
Les fichiers doivent provenir de prepare_data.py.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import random
import time
import uuid

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================= PARAMETRES A MODIFIER =================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_ROOT = BASE_DIR / "results" / "baseline"
# Exemple: DATA_DIR = Path(r"C:\Users\Karim\prototype\data\processed")
SEED = 42
HIDDEN_DIMS = (64, 32)
PATIENTS_PER_BATCH = 32
EVAL_ROWS_PER_BATCH = 1024
MAX_EPOCHS = 60
PATIENCE = 10
MIN_DELTA = 0.0001
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0  # baseline sans defense ni regularisation explicite
CPU_THREADS = 4
DEVICE = "auto"  # "auto", "cpu" ou "cuda"
# Activer uniquement une fois les choix de modele fixes.
EVALUATE_TEST = False
# =========================================================


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits, sans sigmoid ici


def load_split(root, name):
    folder = root / name
    X = sparse.load_npz(folder / "X.npz").tocsr().astype(np.float32)
    y = np.load(folder / "y.npy", allow_pickle=False)
    meta = pd.read_csv(folder / "metadata.csv",
                       dtype={"patient_nbr": str, "encounter_id": str})
    if len(y) != X.shape[0] or len(meta) != len(y) or y.ndim != 1:
        raise ValueError(f"{name}: dimensions X/y/metadata incompatibles")
    if not np.isfinite(X.data).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: caracteristiques ou labels invalides")
    if not np.array_equal(meta.row_index.to_numpy(), np.arange(len(y))):
        raise ValueError(f"{name}: ordre des lignes incorrect")
    if meta.patient_nbr.isna().any() or meta.encounter_id.duplicated().any():
        raise ValueError(f"{name}: identifiants invalides")
    if not np.array_equal(y, (meta.readmitted == "<30").to_numpy()):
        raise ValueError(f"{name}: labels desalignes")
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name}: deux classes requises pour les metriques")
    codes, ids = pd.factorize(meta.patient_nbr, sort=True)
    counts = np.bincount(codes)
    order = np.argsort(codes, kind="stable")
    groups = np.split(order, np.cumsum(counts)[:-1])
    weights = 1.0 / counts[codes]
    return {"X": X, "y": y.astype(np.float32), "meta": meta,
            "groups": groups, "weights": weights, "ids": set(ids)}


def check_disjoint(a, b):
    if a["ids"] & b["ids"]:
        raise ValueError("Fuite: patients partages entre deux groupes.")
    if a["X"].shape[1] != b["X"].shape[1]:
        raise ValueError("Nombre de caracteristiques different entre groupes.")


def train_epoch(model, data, optimizer, device, rng):
    model.train()
    permutation = rng.permutation(len(data["groups"]))
    total = 0.0
    for start in range(0, len(permutation), PATIENTS_PER_BATCH):
        selected = permutation[start:start + PATIENTS_PER_BATCH]
        groups = [data["groups"][i] for i in selected]
        rows = np.concatenate(groups)
        # Each patient's row losses sum to its mean loss.
        row_weights = np.concatenate([
            np.full(len(g), 1.0 / len(g), dtype=np.float32) for g in groups])
        x = torch.from_numpy(data["X"][rows].toarray()).to(device)
        y = torch.from_numpy(data["y"][rows]).to(device)
        w = torch.from_numpy(row_weights).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
        loss = (losses * w).sum() / len(groups)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss non finie: verifier les donnees/le learning rate.")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(groups)
    return total / len(permutation)


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    logits = []
    for start in range(0, len(data["y"]), EVAL_ROWS_PER_BATCH):
        x = torch.from_numpy(data["X"][start:start + EVAL_ROWS_PER_BATCH].toarray()).to(device)
        logits.append(model(x).cpu().numpy())
    logits = np.concatenate(logits).astype(np.float64)
    if not np.isfinite(logits).all():
        raise RuntimeError("Predictions non finies")
    # Stable sigmoid and BCE; useful later for the loss-based membership attack.
    probabilities = np.exp(-np.logaddexp(0, -logits))
    losses = np.logaddexp(0, logits) - data["y"] * logits
    return logits, probabilities, losses


def metrics(data, predictions, threshold=0.5):
    _, p, loss = predictions
    y, w = data["y"], data["weights"]
    binary = (p >= threshold).astype(int)
    return {
        "patient_weighted_loss": float(np.average(loss, weights=w)),
        "patient_weighted_roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "patient_weighted_average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "record_loss": float(loss.mean()),
        "record_roc_auc": float(roc_auc_score(y, p)),
        "record_average_precision": float(average_precision_score(y, p)),
        "patient_weighted_f1": float(f1_score(y, binary, sample_weight=w, zero_division=0)),
        "record_f1": float(f1_score(y, binary, zero_division=0)),
        "record_recall": float(recall_score(y, binary, zero_division=0)),
        "record_precision": float(precision_score(y, binary, zero_division=0)),
        "record_accuracy": float(accuracy_score(y, binary)),
        "record_confusion_matrix": confusion_matrix(y, binary, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }


def save_predictions(folder, name, data, predictions):
    logits, probabilities, losses = predictions
    frame = data["meta"][["row_index", "patient_nbr", "encounter_id"]].copy()
    frame["target"] = data["y"].astype(int)
    frame["logit"] = logits
    frame["probability"] = probabilities
    frame["loss"] = losses
    frame.to_csv(folder / f"{name}_predictions.csv", index=False)


def main():
    if min(MAX_EPOCHS, PATIENCE, PATIENTS_PER_BATCH, EVAL_ROWS_PER_BATCH, CPU_THREADS) < 1:
        raise ValueError("Les parametres de taille/duree doivent etre positifs.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if DEVICE == "auto" else DEVICE)
    root = Path(DATA_DIR).expanduser()
    train = load_split(root, "train")
    validation = load_split(root, "validation")
    check_disjoint(train, validation)
    feature_file = root / "feature_names.json"
    feature_names = json.loads(feature_file.read_text(encoding="utf-8"))
    if len(feature_names) != train["X"].shape[1]:
        raise ValueError("feature_names.json ne correspond pas a X.")
    folder = Path(OUTPUT_ROOT).expanduser() / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    config = {"seed": SEED, "hidden_dims": list(HIDDEN_DIMS),
              "input_dim": train["X"].shape[1], "patients_per_batch": PATIENTS_PER_BATCH,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
              "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "device": str(device), "data_dir": str(root.resolve()),
              "torch_version": str(torch.__version__),
              "selection_metric": "validation patient-weighted average precision",
              "feature_names_sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
              "preparation_report": json.loads((root / "report.json").read_text(encoding="utf-8"))}
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = MLP(config["input_dim"], HIDDEN_DIMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    history, best_score, patience_score, stale = [], -float("inf"), -float("inf"), 0
    best_path = folder / "best_model.pt"
    print(f"Device: {device} | Entrees: {config['input_dim']} | Patients train: {len(train['groups'])}")
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.perf_counter()
        online_loss = train_epoch(model, train, optimizer, device, rng)
        train_metrics = metrics(train, predict(model, train, device))
        val_metrics = metrics(validation, predict(model, validation, device))
        score = val_metrics["patient_weighted_average_precision"]
        history.append({"epoch": epoch, "online_train_loss": online_loss,
                        "train_loss": train_metrics["patient_weighted_loss"],
                        "validation_loss": val_metrics["patient_weighted_loss"],
                        "train_ap": train_metrics["patient_weighted_average_precision"],
                        "validation_ap": score,
                        "validation_roc_auc": val_metrics["patient_weighted_roc_auc"],
                        "seconds": time.perf_counter() - start})
        # Save every genuine best; patience uses a minimum meaningful improvement.
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(),
                        "input_dim": config["input_dim"], "hidden_dims": list(HIDDEN_DIMS),
                        "epoch": epoch, "validation_ap": score,
                        "feature_names_sha256": config["feature_names_sha256"]}, best_path)
        if score > patience_score + MIN_DELTA:
            patience_score, stale = score, 0
        else:
            stale += 1
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"Epoch {epoch:02d} | loss train={history[-1]['train_loss']:.4f} "
              f"val={history[-1]['validation_loss']:.4f} | AP val={score:.4f} "
              f"| AUC val={history[-1]['validation_roc_auc']:.4f}", flush=True)
        if stale >= PATIENCE:
            print("Arret anticipe: absence de progres suffisant sur Validation.")
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    vp = predict(model, validation, device)
    precision, recall, thresholds = precision_recall_curve(
        validation["y"], vp[1], sample_weight=validation["weights"])
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    threshold = float(thresholds[int(np.argmax(f1))])
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, best_path)
    results = {"best_epoch": checkpoint["epoch"], "threshold_selected_on_validation": threshold,
               "notes": ["AP = average precision, not trapezoidal PR area.",
                         "Patient-weighted metrics use per-record weight 1/n_records.",
                         "Predictions are per hospital encounter, not one clinical label per patient.",
                         "Validation is used for model and threshold selection; it is not a final estimate.",
                         "No privacy guarantee or membership-risk estimate is produced here."]}
    for name, data in [("train", train), ("validation", validation)]:
        predictions = vp if name == "validation" else predict(model, data, device)
        results[name] = metrics(data, predictions, threshold)
        save_predictions(folder, name, data, predictions)
    if EVALUATE_TEST:
        test = load_split(root, "test")
        check_disjoint(train, test)
        check_disjoint(validation, test)
        predictions = predict(model, test, device)
        results["test"] = metrics(test, predictions, threshold)
        save_predictions(folder, "test", test, predictions)
    (folder / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in [("train", "Train"), ("validation", "Validation")]:
        axes[0].plot(h.epoch, h[f"{name}_loss"], label=label)
        axes[1].plot(h.epoch, h[f"{name}_ap"], label=label)
    for axis, title in zip(axes, ["Patient-weighted BCE", "Patient-weighted average precision"]):
        axis.set(xlabel="Epoch", title=title)
        axis.axvline(checkpoint["epoch"], color="gray", linestyle="--", alpha=.6)
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(f"\nMeilleure epoch: {checkpoint['epoch']} | Seuil Validation: {threshold:.4f}")
    print(f"Resultats: {folder.resolve()}")
    if not EVALUATE_TEST:
        print("Test non charge. Conserver Test hors des reglages du modele.")


if __name__ == "__main__":
    main()
