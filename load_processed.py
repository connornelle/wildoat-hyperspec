from pathlib import Path
import numpy as np
import pandas as pd

VALID_DATASETS = [
    "06R_2024_airborne",
    "06R_2025_airborne",
    "06R_benchtop",
    "93R_2025_airborne",
    "93R_benchtop_A",
    "93R_benchtop_B",
]

VALID_PIXEL_DATASETS = [
    "06R_2024_airborne_pix",
    "06R_2025_airborne_pix",
    "93R_2025_airborne_pix",
    "06R_benchtop_pix",
    "93R_benchtop_A_pix",
    "93R_benchtop_B_pix",
]

processed_data = Path("processed_data")

def load_wavelengths() -> np.ndarray: #returns ndarray of wavelengths from the stored wavelengths file
    path = processed_data / "wavelengths.npy"
    wavelengths = np.load(path)
    return wavelengths

def get_dataset(dataset: str, target: str, dropna_target: bool): #dataset parquet, response target variable, drop obs if no record for target
    df = pd.read_parquet(processed_data / f"{dataset}.parquet")
    band_cols = [c for c in df.columns if c.startswith("b") and c[1:].isdigit()]
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"target {target} not in {dataset}")
        if dropna_target:
            df = df.dropna(subset=[target]).reset_index(drop=True)
        y = df[target].values.astype(np.float32)
    else:
        y = None
    X = df[band_cols].values.astype(np.float32)
    return df, X, y

def load(dataset: str, target: str = None, dropna_target: bool = True):
    df, X, y = get_dataset(dataset, target, dropna_target)
    RILs = df["RIL"].astype(str).values
    wl = load_wavelengths()
    print(f"Loaded {dataset}: {X.shape[0]:,} samples × {X.shape[1]} bands"
          + (f"  |  {target} range [{y.min():.2f}, {y.max():.2f}]"
             if y is not None else ""))
    return X, y, RILs, wl

def load_combined(datasets: list, target: str): #for use in UMAP aggregate analysis
    Xs, ys, Rs, Ss = [], [], [], []
    wl = None
    for d in datasets:
        X, y, R, wl = load(d, target=target)
        Xs.append(X); ys.append(y); Rs.append(R)
        Ss.append(np.full(len(X), d))
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(Rs), np.concatenate(Ss), wl)

def load_pixels(dataset: str, target: str = None, dropna_target: bool = True):
    df, X, y = get_dataset(dataset, target, dropna_target)
    RILs   = df["RIL"].astype(str).values
    plants = df["plant_id"].astype(str).values
    wl     = load_wavelengths()
    print(f"Loaded {dataset}: {X.shape[0]:,} pixels × {X.shape[1]} bands  "
          f"({len(np.unique(plants))} plants, {len(np.unique(RILs))} RILs)"
          + (f"  |  {target} range [{y.min():.2f}, {y.max():.2f}]"
             if y is not None else ""))
    return X, y, RILs, plants, wl