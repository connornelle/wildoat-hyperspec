import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, make_scorer
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data/bench/sg_test"
OUT_DIR = "outputs/sg_filter"
os.makedirs(OUT_DIR, exist_ok=True)
imp_dir = os.path.join(OUT_DIR, "importance")
os.makedirs(imp_dir, exist_ok=True)

sg_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "_pix" not in f]
print(sg_files)

RELABEL = {"20R": "20R2", "R3": "06R3", "R4": "06R4"}

results = []

for csv_file in sg_files:

    parts = os.path.splitext(csv_file)[0].split("_")
    window = parts[2].replace("w", "")
    poly = parts[3].replace("p", "")
    deriv = parts[4].replace("d", "") if len(parts) > 4 else "1"
    tag = f"w{window}_p{poly}_d{deriv}"

    print(f"\nProcessing: {csv_file}")
    print(f"Window: {window}, Poly: {poly}, Deriv: {deriv}")

    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    spec_cols = [c for c in df.columns if c.startswith("band_")]
    X = df[spec_cols].to_numpy(dtype=np.float32)
    if "biotype" in df.columns:
        y = df["biotype"].to_numpy(dtype=str)
    else:
        y = df["plant_id"].str.split("-").str[1].replace(RELABEL).to_numpy(dtype=str)

    classes = np.sort(np.unique(y))
    unique, counts = np.unique(y, return_counts=True)
    print(f"Classes: {dict(zip(unique, counts))}")

    spec_cols = [c for c in df.columns if c.startswith("band_")]
    wl = np.array([float(c.replace("band_", "")) for c in spec_cols])

    cv_outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    fold_kappas = []
    fold_accs = []
    all_y_true = []
    all_y_pred = []
    all_importances = []

    for train_ix, test_ix in cv_outer.split(X, y):

        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        cv_inner = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=1
        )

        search = GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=500, random_state=1),
            param_grid={
                "max_features": [2, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3]
            },
            scoring=make_scorer(cohen_kappa_score),
            cv=cv_inner,
            refit=True,
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        yhat = search.best_estimator_.predict(X_test)

        fold_kappas.append(cohen_kappa_score(y_test, yhat))
        fold_accs.append(accuracy_score(y_test, yhat))
        all_y_true.append(y_test)
        all_y_pred.append(yhat)
        all_importances.append(search.best_estimator_.feature_importances_)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    mean_kappa = np.mean(fold_kappas)
    std_kappa = np.std(fold_kappas)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"Kappa: {mean_kappa:.3f} +/- {std_kappa:.3f}  "
          f"Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}  (n={len(fold_kappas)} folds)")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        [f"Counts ({tag})", f"Normalized ({tag})"]
    ):
        im = ax.imshow(mat, cmap="viridis", aspect="equal")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="black")
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"SG window={window} poly={poly} deriv={deriv}  |  kappa={mean_kappa:.3f} +/- {std_kappa:.3f}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"confusion/confusion_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    mean_imp = np.mean(all_importances, axis=0)
    pd.DataFrame({"wavelength": wl, "importance": mean_imp}).to_csv(
        os.path.join(imp_dir, f"importance_{tag}.csv"), index=False)
    fig, ax = plt.subplots()
    ax.plot(wl, mean_imp)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Importance")
    ax.set_title(f"w={window} p={poly} d={deriv}")
    fig.savefig(os.path.join(imp_dir, f"importance_{tag}.png"))
    plt.close(fig)

    results.append({
        "file": csv_file,
        "window": int(window),
        "poly": int(poly),
        "deriv": int(deriv),
        "mean_kappa": mean_kappa,
        "std_kappa": std_kappa,
        "mean_acc": mean_acc,
        "std_acc": std_acc
    })

results_df = pd.DataFrame(results).sort_values("mean_kappa", ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, "sg_filter_comparison.csv"), index=False)
print("\n", results_df.to_string(index=False))
print(f"\n-> {OUT_DIR}/sg_filter_comparison.csv")

