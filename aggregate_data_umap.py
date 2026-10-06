import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from load_processed import load_combined
from ril_prediction_functions import train_cnn, extract_bottleneck, run_umap

POPULATIONS = {
    "06R": {
        "datasets": ["06R_2024_airborne", "06R_2025_airborne", "06R_benchtop"],
        "herbicides": ["FLU", "PIN", "TRA", "TRI"],
    },
    "93R": {
        "datasets": ["93R_2025_airborne", "93R_benchtop_A", "93R_benchtop_B"],
        "herbicides": ["TRI", "DIF"],
    },
}
HERB_FULL_NAMES = {
    "FLU": "flucarbazone",
    "PIN": "pinoxaden",
    "TRA": "tralkoxydim",
    "TRI": "triallate",
    "DIF": "difenzoquat",
}
bottleneck = 16
epochs = 100
lr = 1e-3
batch= 32
N_COLS = 2
CMAPS= {"06R": plt.cm.plasma, "93R": plt.cm.viridis}
MARKERS= ["o", "s", "^", "D", "v", "P", "*"]
OUT_DIR = Path("outputs/umap")


def run_cell(pop, datasets, herb):

    X, y, RILs, source, wl = load_combined(datasets, target=herb)
    print(f"Combined: {X.shape[0]} samples x {X.shape[1]} bands  "
          f"({len(np.unique(RILs))} RILs)")
    for src in datasets:
        print(f"  {src}: {(source == src).sum()}")

    print("train on full dataset")
    model, scaler, y_stats, history, device = train_cnn(
        X, y, X, y,
        bottleneck_dim = bottleneck,
        n_epochs = epochs,
        lr = lr,
        batch_size = batch,
    )

    bottleneck_features = extract_bottleneck(model, X, scaler, device)
    embedding = run_umap(bottleneck_features)

    feature_df = pd.DataFrame(
        bottleneck_features,
        columns=[f"bn_feat_{i}" for i in range(bottleneck_features.shape[1])],
    )
    feature_df.insert(0, herb, y)
    feature_df.insert(1, "source", source)
    feature_df.insert(2, "RIL", RILs)
    feature_df.insert(3, "umap_bn_1", embedding[:, 0])
    feature_df.insert(4, "umap_bn_2", embedding[:, 1])
    csv_path = OUT_DIR / f"{pop}_{herb}_features.csv"
    feature_df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")

    return {"emb": embedding, "y": y, "source": source}


def plot_grid(cells, path):
    import math

    layout = []
    for pop, cfg in POPULATIONS.items():
        herbs = cfg["herbicides"]
        n_pop_rows = math.ceil(len(herbs) / N_COLS)
        for i, herb in enumerate(herbs):
            layout.append((pop, herb, i // N_COLS, i % N_COLS, n_pop_rows))
    row_offsets, off = {}, 0
    for pop, cfg in POPULATIONS.items():
        row_offsets[pop] = off
        off += math.ceil(len(cfg["herbicides"]) / N_COLS)
    n_rows = off

    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(5.5 * N_COLS, 5.0 * n_rows),
                             squeeze=False)
    used = set()
    pop_axes = {}

    for pop, herb, r_in, c, _ in layout:
        r = row_offsets[pop] + r_in
        used.add((r, c))
        ax = axes[r, c]
        pop_axes.setdefault(pop, []).append(ax)
        cell = cells[(pop, herb)]
        emb, y, source = cell["emb"], cell["y"], cell["source"]
        groups = list(dict.fromkeys(source))
        cmap = CMAPS.get(pop, plt.cm.plasma)
        full_name = HERB_FULL_NAMES.get(herb, herb)

        for grp, marker in zip(groups, MARKERS):
            mask = source == grp
            sc = ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=y[mask], cmap=cmap,
                vmin=np.nanmin(y), vmax=np.nanmax(y),
                s=40, alpha=0.85, edgecolors="k", lw=0.25,
                marker=marker,
            )
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(f"{pop} {full_name}", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    for r in range(n_rows):
        for c in range(N_COLS):
            if (r, c) not in used:
                axes[r, c].axis("off")

    fig.tight_layout()

    pop_legends = []
    for pop, axs in pop_axes.items():
        groups = list(dict.fromkeys(cells[(pop, POPULATIONS[pop]["herbicides"][0])]["source"]))
        handles = [Line2D([0], [0], marker=m, color="w",
                          markerfacecolor="0.6", markeredgecolor="k",
                          markersize=8, lw=0, label=g)
                   for g, m in zip(groups, MARKERS)]
        boxes = [a.get_position() for a in axs]
        y_center = (min(b.y0 for b in boxes) + max(b.y1 for b in boxes)) / 2
        leg = fig.legend(handles=handles, title=f"{pop} source",
                         fontsize=11, title_fontsize=12, frameon=True,
                         loc="center left", bbox_to_anchor=(1.0, y_center))
        pop_legends.append(leg)

    fig.savefig(path, dpi=300, bbox_inches="tight",
                bbox_extra_artists=pop_legends)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = {}
    for pop, cfg in POPULATIONS.items():
        for herb in cfg["herbicides"]:
            cells[(pop, herb)] = run_cell(pop, cfg["datasets"], herb)

    grid_path = OUT_DIR / "umap_bottleneck_grid.png"
    plot_grid(cells, grid_path)
    print(f"saved {grid_path}")


if __name__ == "__main__":
    main()