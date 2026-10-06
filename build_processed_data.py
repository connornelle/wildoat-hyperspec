from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

WL_MIN, WL_MAX = 430.0, 950.0
BUILD_PIXELS = True #pixel datasets take longer
RAW_DIR = Path("data")
BENCH_DIR = Path("bench/final/d1_p2_w11") #edited to use tagged file
OUT_DIR = Path("processed_data")

#log relevant columns for each dataset
META_06R_2024 = ["poly_id", "RIL", "FLU", "PIN", "TRA", "TRI", "source_cube"]
META_06R_2025 = ["poly_id", "RIL", "FLU", "PIN", "TRA", "TRI", "source_cube"]
META_93R_2025 = ["poly_id", "RIL", "TRI", "DIF", "source_cube"]
META_06R_2024_PIX = ["ID", "poly_id", "RIL", "FLU", "PIN", "TRA", "TRI", "source_cube"]
META_06R_2025_PIX = ["ID", "poly_id", "RIL", "FLU", "PIN", "TRA", "TRI", "source_cube"]
META_93R_2025_PIX = ["ID", "poly_id", "RIL", "TRI", "DIF", "source_cube"]
HERBICIDES = ["FLU", "PIN", "TRA", "TRI", "DIF"]
BIOTYPE_RELABEL = {"20R": "20R2", "R3": "06R3", "R4": "06R4"}
BENCH_PIX_06R = "bench_pix_06R_RIL.csv"
BENCH_PIX_93R = "bench_pix_93R_RIL.csv"
BENCH_PIX_BIOTYPES = "bench_pix_biotypes.csv"

def load_herb_lookups():
    herb_06r = pd.read_excel(RAW_DIR / "herb" / "06R_herb_inj_data.xlsx")
    herb_06r = herb_06r.rename(columns={"flu.inj": "FLU", "axial.inj": "PIN",
                                        "achieve.inj": "TRA", "triallate.inj": "TRI"})
    herb_06r["RIL"] = herb_06r["45RIL"].astype(int).astype(str)
    herb_06r = herb_06r[["RIL", "FLU", "PIN", "TRA", "TRI"]]

    herb_93r = pd.read_excel(RAW_DIR / "herb" / "93R_bench_with_ave.xlsx")
    herb_93r = herb_93r.rename(columns={"tri.inj.05": "TRI", "dif.ave.05": "DIF"})
    herb_93r["RIL"] = herb_93r["93RIL"].astype(int).astype(str)
    herb_93r = herb_93r[["RIL", "TRI", "DIF"]]
    return herb_06r, herb_93r

def normalize(X: np.ndarray) -> np.ndarray: #SNV (mean 0, sd 1)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / sd

def align(X: np.ndarray, src_wl: np.ndarray, ref_wl: np.ndarray) -> np.ndarray: #align to 2024 airborne data
    f = interp1d(src_wl, X, axis=1, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    if ref_wl.min() < src_wl.min() or ref_wl.max() > src_wl.max(): #resampling onto ref grid is cropping to 430-950 nm
        print(f"    WARN extrapolating: src [{src_wl.min():.1f},{src_wl.max():.1f}] "
              f"vs ref [{ref_wl.min():.1f},{ref_wl.max():.1f}]")
    return f(ref_wl).astype(np.float32)

def get_ref_wavelengths() -> np.ndarray:
    df = pd.read_csv(RAW_DIR / "2024" / "06r_2024_spec_deriv.csv", nrows=0)
    wl = np.array([float(c.replace(" nanometers", "")) for c in df.columns if c not in META_06R_2024])
    return wl[(wl >= WL_MIN) & (wl <= WL_MAX)]

def to_parquet(X: np.ndarray, meta: pd.DataFrame, path: Path) -> None:
    band_cols = [f"b{i:03d}" for i in range(X.shape[1])]
    df_spec = pd.DataFrame(X, columns=band_cols, index=meta.index)
    out = pd.concat([meta.reset_index(drop=True),
                     df_spec.reset_index(drop=True)], axis=1)
    for h in HERBICIDES:
        if h not in out.columns:
            out[h] = np.nan
    front = ["source", "plant_id", "RIL", "biotype"] + HERBICIDES
    out = out[[c for c in front if c in out.columns] + band_cols]
    out.to_parquet(path, index=False)
    out.to_csv(path.with_suffix('.csv'), index= False)
    print(f"  → {path.name}: {len(out):,} rows × {X.shape[1]} bands")

def parse_wl(spec_cols):
    return np.array([float(c.replace(" nanometers", "")) for c in spec_cols])

def build_biotypes(ref_wl):
    df = pd.read_csv(RAW_DIR / BENCH_DIR / "bench_mean_biotypes.csv").copy()
    spec_cols = [c for c in df.columns if c.startswith("band_")]
    src_wl = np.array([float(c.replace("band_", "")) for c in spec_cols])
    X = df[spec_cols].values.astype(np.float32)
    X = align(X, src_wl, ref_wl)
    X = normalize(X)
    biotype = df["plant_id"].str.split("-").str[1].replace(BIOTYPE_RELABEL).values
    meta = pd.DataFrame({
        "source": "biotype",
        "plant_id": df["plant_id"].astype(str).values,
        "biotype": biotype,
    })
    return X, meta

def build_06r_bench(ref_wl, herb_lookup_06r):
    df = pd.read_csv(RAW_DIR / BENCH_DIR / "bench_mean_06R_RIL.csv").copy()
    df["RIL"] = df["plant_id"].str.extract(r"B-45-(\d+)").astype(str)

    spec_cols = [c for c in df.columns if c.startswith("band_")]
    src_wl = np.array([float(c.replace("band_", "")) for c in spec_cols])
    X = df[spec_cols].values.astype(np.float32)

    df = df[["plant_id", "RIL"]].merge(herb_lookup_06r, on="RIL", how="left")
    keep = df[["FLU", "PIN", "TRA", "TRI"]].notna().any(axis=1).values
    X, df = X[keep], df[keep].reset_index(drop=True)

    X = align(X, src_wl, ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "06R_benchtop",
        "plant_id": df["plant_id"].astype(str).values,
        "RIL": df["RIL"].values,
        "FLU": df["FLU"].values, "PIN": df["PIN"].values,
        "TRA": df["TRA"].values, "TRI": df["TRI"].values,
    })
    return X, meta


def build_93r_bench(ref_wl, herb_lookup_93r):
    df = pd.read_csv(RAW_DIR / BENCH_DIR / "bench_mean_93R_RIL.csv").copy()
    df["RIL"] = df["plant_id"].str.extract(r"[AB]-93R-(\d+)").astype(str)
    df["rep"] = df["plant_id"].str.extract(r"([AB])-93R-\d+")

    spec_cols = [c for c in df.columns if c.startswith("band_")]
    src_wl = np.array([float(c.replace("band_", "")) for c in spec_cols])
    X_all = df[spec_cols].values.astype(np.float32)

    df = df[["plant_id", "RIL", "rep"]].merge(herb_lookup_93r, on="RIL", how="left")
    keep = df[["TRI", "DIF"]].notna().any(axis=1).values
    X_all, df = X_all[keep], df[keep].reset_index(drop=True)

    X_all = align(X_all, src_wl, ref_wl)
    X_all = normalize(X_all)

    out = {}
    for rep in ["A", "B"]:
        m = df["rep"].values == rep
        meta = pd.DataFrame({
            "source": f"93R_benchtop_{rep}",
            "plant_id": df.loc[m, "plant_id"].astype(str).values,
            "RIL": df.loc[m, "RIL"].values,
            "TRI": df.loc[m, "TRI"].values, "DIF": df.loc[m, "DIF"].values,
        })
        out[rep] = (X_all[m], meta)
    return out

def build_06r_2024(ref_wl):
    df = pd.read_csv(RAW_DIR / "2024" / "06r_2024_spec_deriv.csv")
    spec_cols = [c for c in df.columns if c not in META_06R_2024]
    df = df.dropna(subset=spec_cols + ["FLU"])

    X = align(df[spec_cols].values.astype(np.float32), parse_wl(spec_cols), ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "06R_2024_airborne",
        "plant_id": df["poly_id"].astype(str).values,
        "RIL": df["RIL"].astype(str).values,
        "FLU": df["FLU"].values, "PIN": df["PIN"].values,
        "TRA": df["TRA"].values, "TRI": df["TRI"].values,
    })
    return X, meta

def build_06r_2025(ref_wl):
    df = pd.read_csv(RAW_DIR / "2025" / "06r_2025_spec_deriv.csv")
    spec_cols = [c for c in df.columns if c not in META_06R_2025]
    df = df.dropna(subset=spec_cols)

    X = align(df[spec_cols].values.astype(np.float32),
              parse_wl(spec_cols), ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "06R_2025_airborne",
        "plant_id": df["poly_id"].astype(str).values,
        "RIL": df["RIL"].astype(str).values,
        "FLU": df["FLU"].values, "PIN": df["PIN"].values,
        "TRA": df["TRA"].values, "TRI": df["TRI"].values,
    })
    return X, meta

def build_93r_2025(ref_wl):
    df = pd.read_csv(RAW_DIR / "2025" / "93r_2025_spec_deriv.csv")
    spec_cols = [c for c in df.columns if c not in META_93R_2025]
    df = df.dropna(subset=spec_cols)

    X = align(df[spec_cols].values.astype(np.float32),
              parse_wl(spec_cols), ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "93R_2025_airborne",
        "plant_id": df["poly_id"].astype(str).values,
        "RIL": df["RIL"].astype(str).values,
        "TRI": df["TRI"].values, "DIF": df["DIF"].values,
    })
    return X, meta, df

def build_06r_2024_pix(ref_wl):
    df = pd.read_csv(RAW_DIR / "2024" / "06r_2024_pix_spec_deriv.csv")
    spec_cols = [c for c in df.columns if c not in META_06R_2024_PIX]
    df = df.dropna(subset=spec_cols)

    X = align(df[spec_cols].values.astype(np.float32), parse_wl(spec_cols), ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source":   "06R_2024_airborne_pix",
        "plant_id": df["poly_id"].astype(str).values,
        "RIL":      df["RIL"].astype(str).values,
        "FLU": df["FLU"].values, "PIN": df["PIN"].values,
        "TRA": df["TRA"].values, "TRI": df["TRI"].values,
    })
    return X, meta

def build_06r_2025_pix(ref_wl):
    df = pd.read_csv(RAW_DIR / "2025" / "06r_2025_pix_spec_deriv.csv")
    spec_cols = [c for c in df.columns if c not in META_06R_2025_PIX]
    df = df.dropna(subset=spec_cols)

    X = align(df[spec_cols].values.astype(np.float32),
              parse_wl(spec_cols), ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "06R_2025_airborne_pix",
        "plant_id": df["poly_id"].astype(str).values,
        "RIL": df["RIL"].astype(str).values,
        "FLU": df["FLU"].values, "PIN": df["PIN"].values,
        "TRA": df["TRA"].values, "TRI": df["TRI"].values,
    })
    return X, meta

def build_93r_2025_pix(ref_wl):
    df = pd.read_csv(RAW_DIR / "2025" / "93r_2025_pix_spec_deriv.csv")
    spec_cols = [c for c in df.columns if c not in META_93R_2025_PIX]
    df = df.dropna(subset=spec_cols)

    X = align(df[spec_cols].values.astype(np.float32),
              parse_wl(spec_cols), ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "93R_2025_airborne_pix",
        "plant_id": df["poly_id"].astype(str).values,
        "RIL": df["RIL"].astype(str).values,
        "TRI": df["TRI"].values, "DIF": df["DIF"].values,
    })
    return X, meta

def parse_bench_pix(csv_path):
    df = pd.read_csv(csv_path)
    spec_cols = [c for c in df.columns if c.startswith("band_")]
    src_wl = np.array([float(c.replace("band_", "")) for c in spec_cols])
    X = df[spec_cols].values.astype(np.float32)
    return df, X, src_wl

def build_biotypes_pix(ref_wl):
    df, X, src_wl = parse_bench_pix(RAW_DIR / BENCH_DIR / BENCH_PIX_BIOTYPES)
    X = align(X, src_wl, ref_wl)
    X = normalize(X)
    biotype = df["plant_id"].str.split("-").str[1].replace(BIOTYPE_RELABEL).values
    meta = pd.DataFrame({
        "source": "biotype_pix",
        "plant_id": df["plant_id"].astype(str).values,
        "biotype": biotype,
    })
    return X, meta

def build_06r_bench_pix(ref_wl, herb_lookup_06r):
    df, X, src_wl = parse_bench_pix(RAW_DIR / BENCH_DIR / BENCH_PIX_06R)
    df["RIL"] = df["plant_id"].str.extract(r"B-45-(\d+)").astype(str)
    df = df[["pixel_id", "plant_id", "RIL"]].merge(herb_lookup_06r, on="RIL", how="left")
    keep = df[["FLU", "PIN", "TRA", "TRI"]].notna().any(axis=1).values
    X, df = X[keep], df[keep].reset_index(drop=True)
    X = align(X, src_wl, ref_wl)
    X = normalize(X)
    meta = pd.DataFrame({
        "source": "06R_benchtop_pix",
        "plant_id": df["plant_id"].astype(str).values,
        "RIL": df["RIL"].values,
        "FLU": df["FLU"].values, "PIN": df["PIN"].values,
        "TRA": df["TRA"].values, "TRI": df["TRI"].values,
    })
    return X, meta

def build_93r_bench_pix(ref_wl, herb_lookup_93r):
    df, X, src_wl = parse_bench_pix(RAW_DIR / BENCH_DIR / BENCH_PIX_93R)
    df["RIL"] = df["plant_id"].str.extract(r"[AB]-93R-(\d+)").astype(str)
    df["rep"] = df["plant_id"].str.extract(r"([AB])-93R-\d+")
    df = df[["pixel_id", "plant_id", "RIL", "rep"]].merge(herb_lookup_93r, on="RIL", how="left")
    keep = df[["TRI", "DIF"]].notna().any(axis=1).values
    X, df = X[keep], df[keep].reset_index(drop=True)
    X = align(X, src_wl, ref_wl)
    X = normalize(X)
    out = {}
    for rep in ["A", "B"]:
        m = df["rep"].values == rep
        meta = pd.DataFrame({
            "source": f"93R_benchtop_{rep}_pix",
            "plant_id": df.loc[m, "plant_id"].astype(str).values,
            "RIL": df.loc[m, "RIL"].values,
            "TRI": df.loc[m, "TRI"].values, "DIF": df.loc[m, "DIF"].values,
        })
        out[rep] = (X[m], meta)
    return out

def main():
    OUT_DIR.mkdir(exist_ok=True)
    ref_wl = get_ref_wavelengths()
    herb_06r, herb_93r = load_herb_lookups()
    print(f"Reference grid: {len(ref_wl)} bands [{ref_wl.min():.1f}–{ref_wl.max():.1f} nm]")
    print(f"Build pixels: {BUILD_PIXELS}\n")

    print("06R 2024 airborne")
    X, meta = build_06r_2024(ref_wl)
    to_parquet(X, meta, OUT_DIR / "06R_2024_airborne.parquet")

    print("06R 2025 airborne")
    X, meta = build_06r_2025(ref_wl)
    to_parquet(X, meta, OUT_DIR / "06R_2025_airborne.parquet")

    print("93R 2025 airborne")
    X, meta, df_raw = build_93r_2025(ref_wl)
    to_parquet(X, meta, OUT_DIR / "93R_2025_airborne.parquet")

    print("Biotypes benchtop")
    X, meta = build_biotypes(ref_wl)
    to_parquet(X, meta, OUT_DIR / "biotypes.parquet")

    print("06R benchtop")
    X, meta = build_06r_bench(ref_wl, herb_06r)
    to_parquet(X, meta, OUT_DIR / "06R_benchtop.parquet")

    print("93R benchtop")
    bench = build_93r_bench(ref_wl, herb_93r)
    for rep, (X, meta) in bench.items():
        to_parquet(X, meta, OUT_DIR / f"93R_benchtop_{rep}.parquet")

    if BUILD_PIXELS:
        print("\n06R 2024 airborne PIXELS")
        X, meta = build_06r_2024_pix(ref_wl)
        to_parquet(X, meta, OUT_DIR / "06R_2024_airborne_pix.parquet")

        print("06R 2025 airborne PIXELS")
        X, meta = build_06r_2025_pix(ref_wl)
        to_parquet(X, meta, OUT_DIR / "06R_2025_airborne_pix.parquet")

        print("93R 2025 airborne PIXELS")
        X, meta = build_93r_2025_pix(ref_wl)
        to_parquet(X, meta, OUT_DIR / "93R_2025_airborne_pix.parquet")

        print("\nBiotypes benchtop PIXELS")
        X, meta = build_biotypes_pix(ref_wl)
        to_parquet(X, meta, OUT_DIR / "biotypes_pix.parquet")

        print("06R benchtop PIXELS")
        X, meta = build_06r_bench_pix(ref_wl, herb_06r)
        to_parquet(X, meta, OUT_DIR / "06R_benchtop_pix.parquet")

        print("93R benchtop PIXELS")
        bench_pix = build_93r_bench_pix(ref_wl, herb_93r)
        for rep, (X, meta) in bench_pix.items():
            to_parquet(X, meta, OUT_DIR / f"93R_benchtop_{rep}_pix.parquet")

    np.save(OUT_DIR / "wavelengths.npy", ref_wl)
    pd.DataFrame({
        "band": [f"b{i:03d}" for i in range(len(ref_wl))],
        "wavelength_nm": np.round(ref_wl, 1),
    }).to_csv(OUT_DIR / "band_wavelengths.csv", index=False)
    print(f"\nWavelengths saved to {OUT_DIR}")

if __name__ == "__main__":
    main()