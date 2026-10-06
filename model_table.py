from pathlib import Path
import pandas as pd

IN_CSV = Path("outputs/metrics/model_metrics.csv")
OUT_DIR = Path("outputs/metrics")

METRICS = ["spearman", "rmse"]
DASH = "\u2014"

DATASET_ORDER = [
    "06R_2024_airborne", "06R_2025_airborne", "06R_benchtop",
    "93R_2025_airborne", "93R_benchtop_A", "93R_benchtop_B",
]
HERB_ORDER  = ["FLU", "PIN", "TRA", "TRI", "DIF"]
LEVEL_ORDER = ["mean", "pixel"]
MODEL_ORDER = ["plsr", "cnn"]
METRIC_LABEL = {"spearman": "rho", "rmse": "RMSE", "r2": "R2", "sd_y": "SD"}
DECIMALS = 2

def base_dataset(name):
    return name[:-4] if name.endswith("_pix") else name

def build(df):
    df = df.copy()
    df["dataset_base"] = df["dataset"].map(base_dataset)
    long = df.melt(
        id_vars=["dataset_base", "herbicide", "level", "model"],
        value_vars=METRICS, var_name="metric", value_name="val",
    )
    long["metric"] = long["metric"].map(METRIC_LABEL)

    wide = long.pivot_table(
        index=["dataset_base", "herbicide"],
        columns=["level", "model", "metric"],
        values="val", aggfunc="first",
    ).round(DECIMALS)

    pairs = [(d, h) for d in DATASET_ORDER for h in HERB_ORDER
             if (d, h) in wide.index]
    wide = wide.reindex(index=pd.MultiIndex.from_tuples(pairs))
    metric_names = [METRIC_LABEL[m] for m in METRICS]
    order = [(lv, m, met) for lv in LEVEL_ORDER for m in MODEL_ORDER
             for met in metric_names if (lv, m, met) in wide.columns]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(order))
    wide.index = wide.index.set_names(["dataset", "herbicide"])
    wide.columns = wide.columns.set_names(["level", "model", "metric"])
    return wide.fillna(DASH)

def main():
    df = pd.read_csv(IN_CSV)
    wide = build(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "metrics_table.csv"
    wide.to_csv(out)
    print(f"-> {out}\n")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(wide)


if __name__ == "__main__":
    main()