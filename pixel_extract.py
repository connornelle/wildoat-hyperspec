import os
import numpy as np
import pandas as pd
from spectral import envi
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, remove_small_holes, disk


def load_hsi(filepath_hdr):
    img = envi.open(filepath_hdr)
    cube = img.load().astype(np.float32)

    metadata = img.metadata
    if 'wavelength' in metadata:
        wavelengths = np.array(metadata['wavelength'], dtype=np.float32)
        print("Using metadata wavelengths")
    else:
        wavelengths = np.arange(cube.shape[2])
    return cube, wavelengths

def find_band_index(wavelengths, target_wv, tol=20):
    idx = int(np.argmin(np.abs(wavelengths - target_wv))) #closest index to target wv
    return idx

def crop_spectra(cube, wavelengths, wv_min=430, wv_max=950):
    band_mask = (wavelengths >= wv_min) & (wavelengths <= wv_max)
    return cube[:, :, band_mask], wavelengths[band_mask]

def apply_sg_derivative(cube, window_length=11, polyorder=2, deriv=1, delta=1.0):
    return savgol_filter(
        cube,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=2
    ).astype(np.float32)

def ndvi_mask(cube_raw, wavelengths, red_wv=670, nir_wv=780,
              ndvi_thresh=0.55, intensity_thresh=0.05,
              min_size=5000, hole_size=200, morph_radius=15):

    red_idx = find_band_index(wavelengths, red_wv)
    nir_idx = find_band_index(wavelengths, nir_wv)

    red = cube_raw[:, :, red_idx]
    nir = cube_raw[:, :, nir_idx]

    ndvi = (nir - red) / (nir + red + 1e-6) #add 1e-6 to avoid / 0
    intensity = cube_raw.mean(axis=2) #average reflectance across all wavelengths, necessitates this fnc after spectral cropping

    mask = (ndvi > ndvi_thresh) & (intensity > intensity_thresh)

    mask = binary_opening(mask, disk(morph_radius))
    mask = remove_small_objects(mask, min_size=min_size) #removed closing as there are enough pixels per scan, deprecated since submission

    return mask

def extract_mean_spectra(cube, mask, wavelengths, plant_id):
    pixels = cube[mask, :]
    mean_spectrum = pixels.mean(axis=0)

    band_names = [f"band_{round(w, 2)}" for w in wavelengths]
    df = pd.DataFrame([mean_spectrum], columns=band_names)
    df.insert(0, 'pixel_count', pixels.shape[0])
    df.insert(0, 'plant_id', plant_id)

    return df

def extract_rand_spectra(cube, mask, wavelengths, plant_id, n_samples=1000, random_state=578): #for sg_filter testing, decrease load when creating datasets
    random_num = np.random.default_rng(random_state)
    coords = np.argwhere(mask) #get index of masked pixels

    min_n = min(n_samples, coords.shape[0]) #if masked pixels are less than n_samples
    sample_idx = random_num.choice(coords.shape[0], size=min_n, replace=False)
    sampled_coords = coords[sample_idx]

    pixels = cube[sampled_coords[:, 0], sampled_coords[:, 1], :]
    band_names = [f"band_{round(w, 2)}" for w in wavelengths]

    mean_spectrum = pixels.mean(axis=0)
    df_mean = pd.DataFrame([mean_spectrum], columns=band_names)
    df_mean.insert(0, "pixel_count", min_n)
    df_mean.insert(0, "plant_id", plant_id)

    df_pix = pd.DataFrame(pixels, columns=band_names)
    df_pix.insert(0, "pixel_id", [f"{plant_id}_px{i}" for i in range(min_n)])
    df_pix.insert(1, "plant_id", plant_id)

    return df_mean, df_pix


def save_masked_rgb(cube_raw, mask, wavelengths, plant_id, output_folder, red_wv=660, green_wv=550, blue_wv=450):
    red_idx = find_band_index(wavelengths, red_wv)
    green_idx = find_band_index(wavelengths, green_wv)
    blue_idx = find_band_index(wavelengths, blue_wv)

    rgb = np.stack([
        cube_raw[:, :, red_idx],
        cube_raw[:, :, green_idx],
        cube_raw[:, :, blue_idx]
    ], axis=-1)

    rgb = rgb-rgb.min()
    rgb = rgb/(rgb.max() + 1e-6)

    masked_rgb = rgb.copy()
    for i in range(3):
        channel = masked_rgb[:, :, i]
        channel[~mask] = 0
        masked_rgb[:, :, i] = channel

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{plant_id}_masked_rgb.png")
    plt.imsave(out_path, masked_rgb)
    print(f"Saved RGB image: {out_path}")


def save_ndvi_image(cube_raw, wavelengths, plant_id, output_folder, red_wv=670, nir_wv=780, cmap='viridis'):
    red_idx = find_band_index(wavelengths, red_wv)
    nir_idx = find_band_index(wavelengths, nir_wv)

    red  = cube_raw[:, :, red_idx]
    nir  = cube_raw[:, :, nir_idx]
    ndvi = (nir - red) / (nir + red + 1e-6)

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{plant_id}_ndvi.png")
    plt.imsave(out_path, np.clip(ndvi, 0, 1), cmap=cmap)
    print(f"Saved NDVI image: {out_path}")

def process_directory(directory_path, output_csv, output_pix_csv=None,
                      wv_min=430, wv_max=950,
                      red_wv=670, nir_wv=780,
                      sg_window=11, sg_polyorder=2, sg_deriv=1,
                      n_pixel_samples=1000):
    all_mean_dfs = []
    all_pix_dfs = []
    hdr_files = [f for f in os.listdir(directory_path) if f.endswith(".hdr")]

    for filename in sorted(hdr_files):
        hdr_path = os.path.join(directory_path, filename)
        plant_id = filename.replace(
            "-Crop Spatially.bil", ""
        ).replace(".hdr", "")

        print(f"Processing: {plant_id}")
        try:
            cube_raw, wavelengths = load_hsi(hdr_path)
            cube_raw, wavelengths = crop_spectra(cube_raw, wavelengths,
                                                 wv_min=wv_min, wv_max=wv_max)
            print(f"    Crop:{wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm "
                  f"({len(wavelengths)} bands)")

            delta = float(np.mean(np.diff(wavelengths)))
            print(f"    delta = {delta}")

            mask = ndvi_mask(cube_raw, wavelengths, red_wv=red_wv, nir_wv=nir_wv)
            pixel_count = mask.sum()
            print(f"    Masked pixels: {pixel_count}")

            save_masked_rgb(cube_raw, mask, wavelengths, plant_id,
                            output_folder=os.path.join(directory_path, "masked_pngs"))

            cube_deriv = apply_sg_derivative( #it would be a lot faster to mask then sg filt, this is much less efficient
                cube_raw,
                window_length=sg_window,
                polyorder=sg_polyorder,
                deriv=sg_deriv,
                delta=delta
            )

            df_mean, df_pix = extract_rand_spectra(
                cube_deriv, mask, wavelengths, plant_id,
                n_samples=n_pixel_samples
            )
            all_mean_dfs.append(df_mean)
            if df_pix is not None:
                all_pix_dfs.append(df_pix)
        except Exception as e:
            print(f"    ERROR {filename}: {e}")

    if all_mean_dfs:
        full_df = pd.concat(all_mean_dfs, ignore_index=True)
        full_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(all_mean_dfs)} plants to {output_csv}")

    if all_pix_dfs and output_pix_csv:
        pix_df = pd.concat(all_pix_dfs, ignore_index=True)
        pix_df.to_csv(output_pix_csv, index=False)
        print(f"Saved {len(pix_df)} pixels to {output_pix_csv}")

def process_directory_multi(directory_path, output_dir,wv_min=430, wv_max=950,red_wv=670, nir_wv=780,sg_params=None,prefix="bench_biotypes"): #for processing sg filters
    hdr_files = sorted(f for f in os.listdir(directory_path) if f.endswith(".hdr"))

    sg_filt_data = {f"w{w}_p{p}_d{d}": [] for w, p, d in sg_params} #every combination of w, p, d gets a data repo in dict

    for filename in hdr_files:
        hdr_path = os.path.join(directory_path, filename)
        plant_id = filename.replace("-Crop Spatially.bil", "").replace(".hdr", "")
        print(f"{plant_id}")

        cube_raw, wavelengths = load_hsi(hdr_path)
        cube_raw, wavelengths = crop_spectra(cube_raw, wavelengths,
                                             wv_min=wv_min, wv_max=wv_max)
        delta = float(np.mean(np.diff(wavelengths))) if len(wavelengths) > 1 else 1.0

        mask = ndvi_mask(cube_raw, wavelengths, red_wv=red_wv, nir_wv=nir_wv)

        save_masked_rgb(cube_raw, mask, wavelengths, plant_id,
                        output_folder=os.path.join(directory_path, "masked_pngs"))

        for window, polyorder, deriv in sg_params:
            combo_tag = f"w{window}_p{polyorder}_d{deriv}"
            cube_deriv = apply_sg_derivative(
                cube_raw, window_length=window,
                polyorder=polyorder, deriv=deriv, delta=delta
            )
            df_mean = extract_mean_spectra(cube_deriv, mask, wavelengths, plant_id)
            sg_filt_data[combo_tag].append(df_mean)

    for combo_tag, dfs in sg_filt_data.items():
        if dfs:
            out_df = pd.concat(dfs, ignore_index=True)
            out_df.to_csv(os.path.join(output_dir, f"{prefix}_{combo_tag}.csv"), index=False)
            print(f"Saved {len(out_df)} plants -> {prefix}_{combo_tag}.csv")


red_wv = 670
nir_wv = 780
wv_min = 427
wv_max = 953

sg_params = [
    (w, p, d)
    for w in range(5, 27, 2) #5, 27, 2, window range 5-27
    for p in [2, 3, 4] #2, 3, 4, polynomial degree
    for d in [0, 1, 2] #0, 1, 2, derivative
    if p < w #p cannot be > w
]

#uncomment to build sg-filt datasets
# process_directory_multi(
#     r"D:/bench_parents/preproc",
#     "data/bench/sg_test",
#     wv_min=wv_min, wv_max=wv_max,
#     red_wv=red_wv, nir_wv=nir_wv,
#     sg_params=sg_params,
#     prefix="bench_biotypes"
# )


final_w, final_p, final_d = 11, 2, 1

#data stored on external D: drive, edit for local run

#biotypes
process_directory(
    r"D:/bench_parents/preproc",
    f"data/bench/final/bench_mean_biotypes.csv",
    output_pix_csv=f"data/bench/bench_pix_biotypes.csv",
    wv_min=wv_min, wv_max=wv_max,
    red_wv=red_wv, nir_wv=nir_wv,
    sg_window=final_w , sg_polyorder=final_p, sg_deriv=final_d
)
#06R4
process_directory(
    r"D:/bench_rils/06R/preproc/",
    f"data/bench/final/bench_mean_06R_RIL.csv",
    output_pix_csv=f"data/bench/bench_pix_06R_RIL.csv",
    wv_min=wv_min, wv_max=wv_max,
    red_wv=red_wv, nir_wv=nir_wv,
    sg_window=final_w, sg_polyorder=final_p, sg_deriv=final_d
)
#93R
process_directory(
    r"D:/bench_rils/93R/preproc/",
    f"data/bench/final/bench_mean_93R_RIL.csv",
    output_pix_csv=f"data/bench/bench_pix_93R_RIL.csv",
    wv_min=wv_min, wv_max=wv_max,
    red_wv=red_wv, nir_wv=nir_wv,
    sg_window=final_w, sg_polyorder=final_p, sg_deriv=final_d
)