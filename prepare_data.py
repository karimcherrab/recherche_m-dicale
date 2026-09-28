"""Preparation du dataset Diabetes 130-US Hospitals, sans fuite entre patients.

Installation: python -m pip install numpy pandas scipy scikit-learn joblib
Modifier INPUT_PATH et OUTPUT_PATH ci-dessous, puis executer: python prepare_data.py

Python >= 3.10, scikit-learn >= 1.2.
Les matrices sont creuses (CSR): convertir seulement les batches en tableaux
denses pour PyTorch, et non la matrice complete.
"""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# MODIFIER LES CHEMINS ICI. Par defaut, le CSV est a cote de ce script.
# Exemple Windows: INPUT_PATH = Path(r"C:\Users\Karim\Downloads\diabetic_data.csv")
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data/diabetic_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed"
SEED = 42
# Laisser vide sauf si les codes ont ete verifies dans le dictionnaire source.
EXCLUDE_DISCHARGE_CODES = []


NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
DROP = ["weight", "payer_code", "examide", "citoglipton"]
META = ["patient_nbr", "encounter_id", "readmitted", "target"]
SPLITS = ("train", "validation", "auxiliary", "test")


def clean_data(path, exclude_codes):
    # dtype=str preserves diagnosis codes and identifiers. In particular,
    # keep_default_na=False preserves literal 'None' (test not performed).
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = set(NUMERIC + ["patient_nbr", "encounter_id", "readmitted",
                              "discharge_disposition_id", "A1Cresult",
                              "max_glu_serum", "diag_1", "diag_2", "diag_3"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes obligatoires absentes: {missing}")
    for col in df:
        df[col] = df[col].str.strip()
    original_rows = len(df)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    for col in ("patient_nbr", "encounter_id"):
        if df[col].isin(["", "?"]).any():
            raise ValueError(f"Identifiant manquant dans {col}; corriger la source.")
    if df.encounter_id.duplicated().any():
        raise ValueError("Meme encounter_id avec des lignes differentes: verifier la source.")
    labels = set(df.readmitted.unique())
    if not labels <= {"<30", ">30", "NO"}:
        raise ValueError(f"Labels readmitted inattendus: {labels}")
    excluded = df.discharge_disposition_id.isin(exclude_codes)
    excluded_rows = int(excluded.sum())
    df = df.loc[~excluded].copy()
    if len(df) == 0:
        raise ValueError("Aucun enregistrement apres exclusions.")
    df["target"] = (df.readmitted == "<30").astype(np.int8)
    before_missing = {}
    for col in df.columns:
        if col != "target":
            before_missing[col] = int(df[col].isin(["", "?"]).sum())
            df[col] = df[col].replace({"?": np.nan, "": np.nan})
    # Unknown gender is kept as a category, not silently mapped to Male/Female.
    if "gender" in df:
        df["gender"] = df["gender"].replace({"Unknown/Invalid": "Unknown"})
    for col in NUMERIC:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & converted.isna()
        if bad.any():
            raise ValueError(f"Valeurs non numeriques dans {col}")
        if (converted.dropna() < 0).any() or np.isinf(converted.dropna()).any():
            raise ValueError(f"Valeurs numeriques invalides dans {col}")
        df[col] = converted
    df = df.drop(columns=DROP, errors="ignore")
    # Canonical ordering gives stable outputs for an identical set of rows.
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    audit = {
        "input_rows": original_rows, "exact_duplicates_removed": duplicates,
        "excluded_discharge_codes": sorted(exclude_codes),
        "excluded_rows": excluded_rows, "retained_rows": len(df),
        "retained_patients": int(df.patient_nbr.nunique()),
        "missing_before_imputation": before_missing,
    }
    return df, audit


def prepare(input_path, output_path, seed=42, exclude_codes=()):
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("Dossier de sortie non vide: choisir un nouveau dossier.")
    df, audit = clean_data(input_path, set(exclude_codes))
    patients = np.array(sorted(df.patient_nbr.unique()))
    if len(patients) < 20:
        raise ValueError("Au moins 20 patients requis pour ce decoupage.")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    # Fractions by patient, not by hospital encounter; remaining patients go to test.
    n = len(patients)
    cut = np.cumsum([int(n * .5), int(n * .1), int(n * .2)])
    groups = dict(zip(SPLITS, np.split(patients, cut)))
    lookup = {pid: split for split, ids in groups.items() for pid in ids}
    parts = {name: df.loc[df.patient_nbr.map(lookup) == name].copy()
             for name in SPLITS}
    for i, name in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if set(groups[name]) & set(groups[other]):
                raise RuntimeError("Chevauchement de patients.")

    features = [c for c in df if c not in META]
    train = parts["train"]
    # Learn additional constant/all-missing exclusions on TRAIN only.
    constant = [c for c in features if train[c].nunique(dropna=False) <= 1]
    features = [c for c in features if c not in constant]
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in features if c not in numeric]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                   dtype=np.float32)),
    ])
    processor = ColumnTransformer(
        [("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)],
        remainder="drop", sparse_threshold=1.0,
    )
    processor.fit(train[features])
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, output_path / "preprocessor.joblib")
    names = processor.get_feature_names_out().tolist()
    (output_path / "feature_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    assignments = pd.DataFrame({"patient_nbr": list(lookup), "split": list(lookup.values())})
    assignments.sort_values("patient_nbr").to_csv(output_path / "patient_splits.csv", index=False)
    df.to_csv(output_path / "cleaned_data.csv", index=False)
    summaries = {}
    for name, part in parts.items():
        folder = output_path / name
        folder.mkdir()
        X = sparse.csr_matrix(processor.transform(part[features]), dtype=np.float32)
        if not np.isfinite(X.data).all():
            raise RuntimeError(f"Valeurs NaN/inf dans X_{name}")
        y = part.target.to_numpy(dtype=np.int64)
        counts = part.groupby("patient_nbr").size()
        metadata = part[["patient_nbr", "encounter_id", "readmitted"]].copy()
        metadata.insert(0, "row_index", np.arange(len(part)))
        metadata["n_patient_records"] = part.patient_nbr.map(counts)
        metadata["patient_weight"] = 1.0 / metadata.n_patient_records
        # X, y and metadata have exactly the same row ordering.
        sparse.save_npz(folder / "X.npz", X)
        np.save(folder / "y.npy", y, allow_pickle=False)
        metadata.to_csv(folder / "metadata.csv", index=False)
        summaries[name] = {
            "patients": len(counts), "records": len(part),
            "positive_records": int(y.sum()), "positive_rate": float(y.mean()),
            "patients_with_multiple_records": int((counts > 1).sum()),
            "max_records_per_patient": int(counts.max()),
            "features": X.shape[1],
        }
    report = {
        "seed": seed, "target": "readmitted == '<30'",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "audit": audit, "train_only_constants_removed": constant,
        "input_features": features, "numeric_features": numeric,
        "categorical_features": categorical, "splits": summaries,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "notes": [
            "All learned preprocessing fitted on train only.",
            "None in lab results means test not performed and is preserved.",
            "No clinical discharge exclusions unless explicitly requested.",
            "Random patient split, not temporal or hospital-external validation.",
            "cleaned_data.csv is before imputation and includes all splits; do not train on it directly.",
            "metadata.csv is for grouping/auditing only, never model input.",
            "patient_weight=1/n_records; average losses within patient, then across patients.",
            "Unknown categories are encoded as zeros in their feature block.",
            "For shadow models, fit a separate preprocessor on their own training data.",
        ],
    }
    (output_path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).T.to_string())
    print(f"\nPreparation terminee: {output_path.resolve()}")
    return report


def main():
    input_path = Path(INPUT_PATH).expanduser()
    output_path = Path(OUTPUT_PATH).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"CSV introuvable: {input_path}\nModifier INPUT_PATH au debut du script."
        )
    prepare(input_path, output_path, SEED, EXCLUDE_DISCHARGE_CODES)


if __name__ == "__main__":
    main()
