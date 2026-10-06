import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data/bench/sg_test"
OUT_DIR = "outputs/sg_filter"
RESULTS_CSV = os.path.join(OUT_DIR, "sg_filter_comparison.csv")
os.makedirs(OUT_DIR, exist_ok=True)

RELABEL = {"20R": "20R2", "R3": "06R3", "R4": "06R4"}
CLASSES = ["20R2", "93R", "06R3", "06R4", "S1", "S2"]
COLORS = {"20R2": "#e41a1c", "93R": "#377eb8", "06R3": "#4daf4a",
           "06R4": "#984ea3", "S1": "#ff7f00", "S2": "#a65628"}

sg_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "_pix" not in f)
results_df = pd.read_csv(RESULTS_CSV)

first_csv = pd.read_csv(os.path.join(DATA_DIR, sg_files[0]))
if "biotype" not in first_csv.columns:
    first_csv["biotype"] = first_csv["plant_id"].str.split("-").str[1].replace(RELABEL)
rep_plants = {}
for cls in CLASSES:
    sub = first_csv[first_csv["biotype"] == cls]
    if not sub.empty:
        rep_plants[cls] = sub["plant_id"].iloc[0]

spectra = []
for csv_file in sg_files:
    parts = os.path.splitext(csv_file)[0].split("_")
    w = int(parts[2].replace("w", ""))
    p = int(parts[3].replace("p", ""))
    d = int(parts[4].replace("d", "")) if len(parts) > 4 else 1
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    spec_cols = [c for c in df.columns if c.startswith("band_")]
    wl = np.array([float(c.replace("band_", "")) for c in spec_cols])
    for cls, pid in rep_plants.items():
        row = df[df["plant_id"] == pid]
        if row.empty:
            continue
        spectra.append({"w": w, "p": p, "d": d, "cls": cls,
                        "wl": wl, "vals": row[spec_cols].values.ravel()})

polys = sorted(set(s["p"] for s in spectra))
derivs = sorted(set(s["d"] for s in spectra))

best_window = {}
for (p, d), grp in results_df.groupby(["poly", "deriv"]):
    best_window[(int(p), int(d))] = int(grp.loc[grp["mean_kappa"].idxmax(), "window"])

IMP_DIR = os.path.join(OUT_DIR, "importance")

fig, axes = plt.subplots(len(polys), len(derivs), figsize=(5 * len(derivs), 3 * len(polys)),
                         sharex=True)
if len(polys) == 1:
    axes = [axes]
if len(derivs) == 1:
    axes = [[ax] for ax in axes]

for i, p in enumerate(polys):
    for j, d in enumerate(derivs):
        ax = axes[i][j]
        bw = best_window.get((p, d))
        sub = [s for s in spectra if s["p"] == p and s["d"] == d]
        for s in sub:
            is_best = s["w"] == bw
            ax.plot(s["wl"], s["vals"],
                    alpha=1.0 if is_best else 0.1,
                    lw=1.5 if is_best else 0.5,
                    color=COLORS[s["cls"]],
                    label=s["cls"] if is_best else None)
        imp_path = os.path.join(IMP_DIR, f"importance_w{bw}_p{p}_d{d}.csv")
        if bw and os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path)
            ax2 = ax.twinx()
            ax2.fill_between(imp_df["wavelength"], imp_df["importance"],
                             alpha=0.2, color="gray")
            ax2.set_ylabel("Importance", fontsize=7, color="gray")
            ax2.tick_params(axis="y", labelsize=6, colors="gray")
        ax.set_title(f"poly={p}  deriv={d}  (best w={bw})")
        if i == len(polys) - 1:
            ax.set_xlabel("Wavelength (nm)")
        if j == 0:
            ax.set_ylabel("Value")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_handles = [Line2D([0], [0], color=COLORS[c], lw=1.5, label=c) for c in CLASSES]
legend_handles.append(Patch(facecolor="gray", alpha=0.2, label="Importance"))
fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sg_spectra_comparison.png"),dpi = 300, bbox_inches="tight")
plt.close(fig)
print(f"-> {OUT_DIR}/sg_spectra_comparison.png")

derivs = sorted(results_df["deriv"].unique())
fig, axes = plt.subplots(1, len(derivs), figsize=(5 * len(derivs), 4), sharey=True)
if len(derivs) == 1:
    axes = [axes]
for ax, d in zip(axes, derivs):
    sub = results_df[results_df["deriv"] == d]
    for poly, grp in sub.groupby("poly"):
        grp = grp.sort_values("window")
        ax.plot(grp["window"], grp["mean_kappa"], marker="o", label=f"poly={poly}")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title(f"Derivative = {d}")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.savefig(os.path.join(OUT_DIR, "sg_filter_lineplot.png"),dpi = 300, bbox_inches="tight")
plt.close(fig)