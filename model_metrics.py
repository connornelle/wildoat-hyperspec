import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from load_processed import load, load_pixels
from ril_prediction_functions import train_cnn, evaluate_predictions

#each mean dataset name and response columns
MEAN_DATASETS = {
    "06R_2024_airborne": ["FLU", "PIN", "TRA", "TRI"],
    "06R_2025_airborne": ["FLU", "PIN", "TRA", "TRI"],
    "06R_benchtop":      ["FLU", "PIN", "TRA", "TRI"],
    "93R_2025_airborne": ["TRI", "DIF"],
    "93R_benchtop_A":    ["TRI", "DIF"],
    "93R_benchtop_B":    ["TRI", "DIF"],
}

#each pixel dataset name and response columns
PIXEL_DATASETS = {
    "06R_2024_airborne_pix": ["FLU", "PIN", "TRA", "TRI"],
    "06R_2025_airborne_pix": ["FLU", "PIN", "TRA", "TRI"],
    "93R_2025_airborne_pix": ["TRI", "DIF"],
    "06R_benchtop_pix":      ["FLU", "PIN", "TRA", "TRI"],
    "93R_benchtop_A_pix":    ["TRI", "DIF"],
    "93R_benchtop_B_pix":    ["TRI", "DIF"],
}

#nested dict for assigning level to datasets
LEVEL_DATASETS = {"mean": MEAN_DATASETS, "pixel": PIXEL_DATASETS}

MODELS = ("plsr", "cnn")
LEVELS = ("mean", "pixel")

num_splits = 5 #number of folds
num_components = 10 #number plsr components
pixels_per_plant= 100 #subsample to 100 pixels perplant
epochs = 100 #cnn epochs
bottleneck = 16 #bottleneck dim
batch_mean = 32 #batch size for mean model
batch_pixel = 256 #batch size for pixel model
OUT_DIR = Path("outputs/metrics")


def subsample_pixels(X, y, RILs, plants, pixels_per_plant):
    if pixels_per_plant is None:
        return X, y, RILs, plants
    rng = np.random.default_rng(0)
    keep = [] #row indexes of retained pixels
    for p in np.unique(plants):
        idx = np.where(plants == p)[0] #find every row of unique plant, index it
        if len(idx) > pixels_per_plant: #if more pixels than pixels per plant, subset to n = pixels per plant
            idx = rng.choice(idx, size= pixels_per_plant, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep) #non-grouped
    return X[keep], y[keep], RILs[keep], plants[keep]


def fit_predict_plsr(X_tr, y_tr, X_val):
    pls = PLSRegression(n_components=min(num_components, X_tr.shape[1] - 1))
    pls.fit(X_tr, y_tr)
    prediction = pls.predict(X_val).ravel()
    return prediction

def fit_predict_cnn(X_tr, y_tr, X_val, batch_size):
    model, scaler, y_stats, _, device = train_cnn(
        X_tr, y_tr, X_tr, y_tr,
        bottleneck_dim=bottleneck  , n_epochs= epochs,
        batch_size=batch_size,
    )
    pred, _, _ = evaluate_predictions(
        model, X_val, np.zeros(len(X_val), dtype=np.float32), scaler, y_stats, device)
    prediction = np.asarray(pred).ravel()
    return prediction

def pooled_oof(X, y, RILs, model, batch_size):
    gkf = GroupKFold(n_splits=min(num_splits, len(np.unique(RILs))))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for tr, va in gkf.split(X, y, groups=RILs):
        if model == "plsr":
            oof[va] = fit_predict_plsr(X[tr], y[tr], X[va])
        elif model == "cnn":
            oof[va] = fit_predict_cnn(X[tr], y[tr], X[va], batch_size)
    return oof


def plant_level(y, oof, plants):
    g = (pd.DataFrame({"plant": plants, "y": y, "p": oof})
         .groupby("plant").agg(y=("y", "mean"), p=("p", "mean")).reset_index())
    return g["y"].values, g["p"].values


def metrics(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "n_plants": int(len(y_true)),
        "sd_y": float(np.std(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(rho),
    }


def save_scatter(y_true, y_pred, row, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, alpha=0.7, s=35, color="#2d6a9f",
               edgecolors="k", lw=0.3)
    ax.set_xlabel("Observed (plant mean)")
    ax.set_ylabel("Predicted (plant mean)")
    ax.set_title(
        f"{row['dataset']} | {row['herbicide']} | {row['model']}\n"
        f"rho={row['spearman']:+.2f}  R2={row['r2']:+.2f}  "
        f"RMSE={row['rmse']:.2f}  n={row['n_plants']}",
        fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_cell(level, dataset, herb, model, scatter_dir):
    if level == "mean":
        X, y, RILs, _ = load(dataset, target=herb)
        plants = None
    else:
        X, y, RILs, plants, _ = load_pixels(dataset, target=herb)
        X, y, RILs, plants = subsample_pixels(
            X, y, RILs, plants, pixels_per_plant)

    batch_size = batch_mean if level == "mean" else batch_pixel
    oof = pooled_oof(X, y, RILs, model, batch_size)
    if plants is None:
        y_pl, p_pl = y, oof
    else:
        y_pl, p_pl = plant_level(y, oof, plants)
    row = {"level": level, "dataset": dataset, "herbicide": herb,
           "model": model, "n_samples": int(len(y)),
           "n_pixels": int(len(y)) if level == "pixel" else np.nan,
           "n_rils": int(len(np.unique(RILs)))}
    row.update(metrics(y_pl, p_pl))
    print(f"  {model:5s} {level:5s} {dataset:22s} {herb:4s}  "
          f"R2={row['r2']:+.3f}  RMSE={row['rmse']:.3f}  "
          f"rho={row['spearman']:+.3f}")

    save_scatter(y_pl, p_pl, row,
                 scatter_dir / f"{level}_{dataset}_{herb}_{model}.png")
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scatter_dir = OUT_DIR / "scatters"
    scatter_dir.mkdir(exist_ok=True)

    rows = []
    for level in LEVELS:
        for dataset, herbs in LEVEL_DATASETS[level].items():
            for herb in herbs:
                for model in MODELS:
                    try:
                        row = run_cell(level, dataset, herb, model, scatter_dir)
                    except (KeyError, ValueError, FileNotFoundError) as e:
                        print(f"  skip {level}/{dataset}/{herb}/{model}: {e}")
                        continue
                    if row is not None:
                        rows.append(row)

    if not rows:
        print("\nNo cells had enough data.")
        return

    df = pd.DataFrame(rows)
    df = df[["level", "dataset", "herbicide", "model", "n_samples", "n_pixels",
             "n_rils", "n_plants", "sd_y", "r2", "rmse", "spearman"]]
    df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    print(f"\n-> {OUT_DIR / 'model_metrics.csv'}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()