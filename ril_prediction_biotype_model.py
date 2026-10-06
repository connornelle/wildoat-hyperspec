import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

from load_processed import load

PROC_DIR     = Path("processed_data")
N_EXTREME    = 25
N_ESTIMATORS = 2000
OUT_DIR      = Path("outputs/biotype_tail_transfer")

MODELS = {
    "93R": {"classes": ["93R", "S2"],  "pos": "93R",  "pop": "93R",
            "herbs": ["TRI", "DIF"]},
    "06R": {"classes": ["06R4", "S2"], "pos": "06R4", "pop": "06R",
            "herbs": ["FLU", "PIN", "TRA", "TRI"]},
}

def load_biotypes(classes):
    df = pd.read_parquet(PROC_DIR / "biotypes.parquet")
    df = df[df["biotype"].isin(classes)].reset_index(drop=True)
    band = [c for c in df.columns if c.startswith("b") and c[1:].isdigit()]
    X = df[band].values.astype(np.float32)
    return X, df["biotype"].values

def load_ril(pop, herb):
    if pop == "93R":
        XA, yA, RA, _ = load("93R_benchtop_A", target=herb)
        XB, yB, RB, _ = load("93R_benchtop_B", target=herb)
        return (np.concatenate([XA, XB]),
                np.concatenate([yA, yB]),
                np.concatenate([RA, RB]))
    X, y, R, _ = load("06R_benchtop", target=herb)
    return X, y, R


def train_rf(X, biotype, pos):
    y = (biotype == pos).astype(int)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced",
        random_state=551, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf

def predict_tails(clf, X, y, RILs, n):
    ril_means = (pd.DataFrame({"y": y, "RIL": RILs})
                 .groupby("RIL")["y"].mean().sort_values())
    if len(ril_means) < 2 * n:
        return None
    s_rils = set(ril_means.index[:n])
    r_rils = set(ril_means.index[-n:])

    keep = np.array([r in s_rils or r in r_rils for r in RILs])
    Xe, Re = X[keep], RILs[keep]
    true = np.array([1 if r in r_rils else 0 for r in Re], dtype=int)
    proba = clf.predict_proba(Xe)[:, 1]

    g = (pd.DataFrame({"RIL": Re, "true": true, "proba": proba})
         .groupby("RIL").agg(true=("true", "first"), proba=("proba", "mean"))
         .reset_index())
    g["pred"] = (g["proba"] >= 0.5).astype(int)
    g["correct"] = g["true"] == g["pred"]
    return g


def plot_confusion(g, title, path):
    cm = pd.crosstab(
        np.where(g["true"] == 0, "S", "R"),
        np.where(g["pred"] == 0, "S", "R"),
        rownames=["true"], colnames=["predicted"],
    ).reindex(index=["S", "R"], columns=["S", "R"], fill_value=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm.values, cmap="viridis", aspect="equal")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["S", "R"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["S", "R"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color="black")
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conf_dir = OUT_DIR / "confusion"
    conf_dir.mkdir(exist_ok=True)

    rows = []
    for name, cfg in MODELS.items():
        neg = [c for c in cfg["classes"] if c != cfg["pos"]][0]
        Xb, bt = load_biotypes(cfg["classes"])
        clf = train_rf(Xb, bt, cfg["pos"])
        print(f"\n== model {name}: {cfg['pos']} vs {neg} | "
              f"train n={len(Xb)} ({(bt == cfg['pos']).sum()} pos) ==")

        for herb in cfg["herbs"]:
            try:
                X, y, R = load_ril(cfg["pop"], herb)
            except (KeyError, ValueError):
                continue

            g = predict_tails(clf, X, y, R, N_EXTREME)
            if g is None:
                print(f"  skip {name}/{herb}: < {2 * N_EXTREME} RILs")
                continue

            acc   = accuracy_score(g["true"], g["pred"])
            kappa = cohen_kappa_score(g["true"], g["pred"])
            auc = (roc_auc_score(g["true"], g["proba"])
                   if g["true"].nunique() == 2 else np.nan)
            n_correct, n_total = int(g["correct"].sum()), len(g)
            print(f"  {cfg['pop']} tails ({herb}): acc={acc:.3f}  "
                  f"AUC={auc:.3f}  kappa={kappa:.3f}  "
                  f"({n_correct}/{n_total})")

            rows.append({
                "model": name, "pos_class": cfg["pos"], "neg_class": neg,
                "population": cfg["pop"], "herbicide": herb,
                "n_train": int(len(Xb)), "n_rils": n_total,
                "ril_acc": acc, "ril_auc": auc, "ril_kappa": kappa,
                "ril_correct": n_correct,
            })
            plot_confusion(
                g,
                f"{cfg['pos']} vs {neg} -> {cfg['pop']} tails ({herb})\n"
                f"acc={acc:.2f}  AUC={auc:.2f}  kappa={kappa:.2f}  "
                f"({n_correct}/{n_total})",
                conf_dir / f"{name}_{herb}.png",
            )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "transfer_summary.csv", index=False)
        print(f"\n-> {OUT_DIR / 'transfer_summary.csv'}")
        print("\n" + df[["model", "population", "herbicide", "n_rils",
                         "ril_acc", "ril_auc", "ril_kappa"]].to_string(index=False))


if __name__ == "__main__":
    main()