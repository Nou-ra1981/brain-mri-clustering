from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
LABELS_CSV = OUTPUTS / "cluster_labels_final_pca2d.csv"
META_NPZ = OUTPUTS / "all_subjects_meta_firstocc.npz"
DATA_ROOT = ROOT / "fwp_brain_volume_clustering_test_data" / "DLDiReCT_results" / "DL+DiReCT_results"
EXPORT_ROOT = ROOT / "spectral_cluster_visual_inspection"
N_PER_CLUSTER = 10
KEPT_CLUSTER_COLORS = ["blue", "brown", "cyan", "gray"]
OUTLIER_COLOR_NAME = "green"


def build_subject_source_maps() -> tuple[dict[str, Path], dict[str, Path]]:
    if not META_NPZ.exists():
        raise FileNotFoundError(f"Missing metadata NPZ: {META_NPZ}")

    with np.load(META_NPZ, allow_pickle=True) as meta:
        if "subject_id" not in meta or "source_path" not in meta:
            raise ValueError("Metadata NPZ must contain 'subject_id' and 'source_path' arrays.")
        subject_ids = [str(x) for x in meta["subject_id"]]
        source_paths = [Path(str(x)) for x in meta["source_path"]]

    subject_to_source: dict[str, Path] = {}
    subject_to_preview: dict[str, Path] = {}

    for sid, src in zip(subject_ids, source_paths):
        subject_to_source[sid] = src
        session_name = src.parent.name
        preview_name = f"{session_name}_T1w_norm_seg_preview.png"
        subject_to_preview[sid] = OUTPUTS / preview_name

    return subject_to_source, subject_to_preview


def save_center_slice_png(nifti_path: Path, png_path: Path) -> None:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape}")

    z_idx = data.shape[2] // 2
    slice_2d = np.asarray(data[:, :, z_idx], dtype=np.float32)
    slice_2d = np.nan_to_num(slice_2d, nan=0.0, posinf=0.0, neginf=0.0)

    p_low, p_high = np.percentile(slice_2d, [1, 99])
    if p_high > p_low:
        norm = np.clip((slice_2d - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        mn = float(slice_2d.min())
        mx = float(slice_2d.max())
        if mx > mn:
            norm = (slice_2d - mn) / (mx - mn)
        else:
            norm = np.zeros_like(slice_2d, dtype=np.float32)

    # Rotate for more natural visual orientation.
    norm = np.rot90(norm)
    plt.imsave(str(png_path), norm, cmap="gray")


def main() -> None:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")

    subject_to_source, subject_to_preview = build_subject_source_maps()

    df = pd.read_csv(LABELS_CSV)
    required_cols = {"subject_id", "pc1", "pc2", "spectral_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {sorted(missing)}")

    work = df[["subject_id", "pc1", "pc2", "spectral_label"]].copy()
    work["spectral_label"] = work["spectral_label"].astype(int)

    # Infer the outward cluster as the centroid farthest from the global centroid.
    global_center = work[["pc1", "pc2"]].mean().to_numpy(dtype=float)
    centroid_rows = []
    for lbl, grp in work.groupby("spectral_label"):
        center = grp[["pc1", "pc2"]].mean().to_numpy(dtype=float)
        dist_from_global = float(np.linalg.norm(center - global_center))
        centroid_rows.append(
            {
                "spectral_label": int(lbl),
                "centroid_pc1": float(center[0]),
                "centroid_pc2": float(center[1]),
                "dist_from_global": dist_from_global,
                "n_subjects": int(len(grp)),
            }
        )

    centroid_df = pd.DataFrame(centroid_rows).sort_values("spectral_label").reset_index(drop=True)
    outlier_label = int(centroid_df.loc[centroid_df["dist_from_global"].idxmax(), "spectral_label"])
    keep_labels = [int(x) for x in sorted(centroid_df[centroid_df["spectral_label"] != outlier_label]["spectral_label"].tolist())]

    if len(keep_labels) != 4:
        raise ValueError(
            f"Expected 4 kept clusters after removing one outlier, got {len(keep_labels)} kept labels: {keep_labels}."
        )

    label_to_color: dict[int, str] = {
        lbl: color for lbl, color in zip(keep_labels, KEPT_CLUSTER_COLORS)
    }

    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, object]] = []

    for lbl in keep_labels:
        color_name = label_to_color[lbl]
        cluster_df = work[work["spectral_label"] == lbl].copy()
        center = cluster_df[["pc1", "pc2"]].mean().to_numpy(dtype=float)
        points = cluster_df[["pc1", "pc2"]].to_numpy(dtype=float)
        dists = np.linalg.norm(points - center, axis=1)
        cluster_df["dist_to_centroid"] = dists
        cluster_df = cluster_df.sort_values("dist_to_centroid", ascending=True).head(N_PER_CLUSTER).reset_index(drop=True)

        cluster_folder = EXPORT_ROOT / f"cluster_label_{lbl}_{color_name}"
        preview_folder = cluster_folder / "preview_png"
        nii_folder = cluster_folder / "original_nifti"
        preview_folder.mkdir(parents=True, exist_ok=True)
        nii_folder.mkdir(parents=True, exist_ok=True)

        for rank, row in cluster_df.iterrows():
            rank1 = int(rank) + 1
            subject_id = str(row["subject_id"])
            dist = float(row["dist_to_centroid"])

            src_nifti_seg = subject_to_source.get(subject_id)
            if src_nifti_seg is None:
                src_nifti_seg = Path("")

            # Prefer original intensity image if available in same session folder.
            src_nifti = src_nifti_seg
            if src_nifti_seg.exists():
                t1w_norm = src_nifti_seg.parent / "T1w_norm.nii.gz"
                if t1w_norm.exists():
                    src_nifti = t1w_norm

            src_preview = subject_to_preview.get(subject_id, OUTPUTS / f"{subject_id}_preview.png")

            session_id = src_nifti_seg.parent.name if src_nifti_seg.exists() else subject_id

            dst_preview = preview_folder / f"{rank1:02d}_{subject_id}_preview.png"
            dst_nifti = nii_folder / f"{rank1:02d}_{session_id}_{src_nifti.name}"

            preview_status = "missing"
            png_status = "missing"
            nifti_status = "missing"

            if src_preview.exists():
                shutil.copy2(src_preview, dst_preview)
                preview_status = "copied"
                png_status = "copied_source_preview"

            if src_nifti.exists():
                shutil.copy2(src_nifti, dst_nifti)
                nifti_status = "copied"

            if png_status == "missing" and src_nifti.exists():
                try:
                    save_center_slice_png(src_nifti, dst_preview)
                    png_status = "generated_from_nifti"
                except Exception as ex:
                    png_status = f"generate_failed:{type(ex).__name__}"

            selected_rows.append(
                {
                    "cluster_label": lbl,
                    "cluster_color": color_name,
                    "rank": rank1,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "dist_to_centroid": dist,
                    "preview_status": preview_status,
                    "png_status": png_status,
                    "png_path": str(dst_preview),
                    "preview_source": str(src_preview),
                    "nifti_status": nifti_status,
                    "nifti_source": str(src_nifti),
                }
            )

    selected_df = pd.DataFrame(selected_rows).sort_values(["cluster_label", "rank"]).reset_index(drop=True)
    selected_df.to_csv(EXPORT_ROOT / "selection_manifest.csv", index=False)
    centroid_df.to_csv(EXPORT_ROOT / "cluster_centroids.csv", index=False)

    with (EXPORT_ROOT / "README.txt").open("w", encoding="ascii") as f:
        f.write("Spectral cluster sample export\n")
        f.write("==============================\n\n")
        f.write(f"Outlier cluster removed: label {outlier_label} ({OUTLIER_COLOR_NAME})\n")
        f.write(f"Kept clusters: {keep_labels}\n")
        f.write("Kept cluster color map:\n")
        for lbl in keep_labels:
            f.write(f"- label {lbl}: {label_to_color[lbl]}\n")
        f.write("\n")
        f.write(f"Subjects per kept cluster: {N_PER_CLUSTER}\n\n")
        f.write("For each cluster folder:\n")
        f.write("- preview_png/: 10 PNG previews (copied if available, otherwise generated from NIfTI).\n")
        f.write("- original_nifti/: corresponding original T1w NIfTI files.\n")

    copied_preview = int((selected_df["preview_status"] == "copied").sum())
    generated_png = int((selected_df["png_status"] == "generated_from_nifti").sum())
    ready_png = int(selected_df["png_status"].isin(["copied_source_preview", "generated_from_nifti"]).sum())
    copied_nifti = int((selected_df["nifti_status"] == "copied").sum())
    print(f"Export folder: {EXPORT_ROOT}")
    print(f"Outlier spectral label removed: {outlier_label}")
    print(f"Kept labels: {keep_labels}")
    print(f"Requested samples per kept cluster: {N_PER_CLUSTER}")
    print(f"Copied source preview PNG files: {copied_preview}/{len(selected_df)}")
    print(f"Generated preview PNG files from NIfTI: {generated_png}/{len(selected_df)}")
    print(f"Total usable preview PNG files: {ready_png}/{len(selected_df)}")
    print(f"Copied NIfTI files: {copied_nifti}/{len(selected_df)}")


if __name__ == "__main__":
    main()
