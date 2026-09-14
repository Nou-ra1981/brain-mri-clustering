from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

# =========================
# SETTINGS
# =========================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
LABELS_CSV = OUTPUT_DIR / "cluster_labels_final_pca2d.csv"
META_NPZ = OUTPUT_DIR / "all_subjects_meta_firstocc.npz"
METADATA_CSV = ROOT / "metadata.csv"
RESULTS_DIR = ROOT / "cluster_analysis_results"

# These names are kept consistent with your visual cluster convention.
KEPT_CLUSTER_COLORS = ["blue", "brown", "cyan", "gray"]
OUTLIER_COLOR_NAME = "green"


def load_cluster_labels() -> tuple[pd.DataFrame, int, list[int], dict[int, str]]:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    required_cols = {"subject_id", "pc1", "pc2", "spectral_label"}
    missing = required_cols - set(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {sorted(missing)}")

    labels = labels.copy()
    labels["spectral_label"] = labels["spectral_label"].astype(int)

    # Same rule as export: outlier cluster is the centroid farthest from global centroid.
    global_center = labels[["pc1", "pc2"]].mean().to_numpy(dtype=float)
    centroid_rows = []
    for lbl, grp in labels.groupby("spectral_label"):
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
    centroid_df.to_csv(RESULTS_DIR / "cluster_centroids.csv", index=False)

    outlier_label = int(centroid_df.loc[centroid_df["dist_from_global"].idxmax(), "spectral_label"])
    keep_labels = [
        int(x)
        for x in sorted(
            centroid_df.loc[centroid_df["spectral_label"] != outlier_label, "spectral_label"].tolist()
        )
    ]

    if len(keep_labels) != 4:
        raise ValueError(
            f"Expected 4 kept clusters after removing outlier, got {len(keep_labels)} labels: {keep_labels}"
        )

    label_to_color = {lbl: color for lbl, color in zip(keep_labels, KEPT_CLUSTER_COLORS)}
    return labels, outlier_label, keep_labels, label_to_color


def load_subject_sources() -> pd.DataFrame:
    if not META_NPZ.exists():
        raise FileNotFoundError(f"Missing NPZ file: {META_NPZ}")

    with np.load(META_NPZ, allow_pickle=True) as npz:
        required = {"subject_id", "source_path"}
        missing = required - set(npz.files)
        if missing:
            raise ValueError(f"NPZ missing arrays: {sorted(missing)}")

        source_df = pd.DataFrame(
            {
                "subject_id": [str(x) for x in npz["subject_id"]],
                "source_path": [str(x) for x in npz["source_path"]],
            }
        )

    # Keep first occurrence in case of duplicates.
    source_df = source_df.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    return source_df


def extract_2d_density_from_volume(data_3d: np.ndarray) -> float:
    z_idx = data_3d.shape[2] // 2
    slice_2d = np.asarray(data_3d[:, :, z_idx], dtype=np.float32)
    slice_2d = np.nan_to_num(slice_2d, nan=0.0, posinf=0.0, neginf=0.0)

    p_low, p_high = np.percentile(slice_2d, [1, 99])
    if p_high > p_low:
        norm = np.clip((slice_2d - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        mn = float(slice_2d.min())
        mx = float(slice_2d.max())
        norm = (slice_2d - mn) / (mx - mn) if mx > mn else np.zeros_like(slice_2d)

    # Threshold chosen to estimate foreground tissue occupancy in the normalized slice.
    return float((norm > 0.15).sum() / norm.size)


def analyze_all_subjects(
    labels_df: pd.DataFrame,
    keep_labels: list[int],
    label_to_color: dict[int, str],
) -> pd.DataFrame:
    source_df = load_subject_sources()

    work = labels_df.merge(source_df, on="subject_id", how="left")
    work = work[work["spectral_label"].isin(keep_labels)].copy()
    if work.empty:
        raise ValueError("No rows found for kept cluster labels.")

    rows: list[dict[str, object]] = []

    for _, row in work.iterrows():
        label = int(row["spectral_label"])
        color = label_to_color[label]
        cluster_name = f"cluster_label_{label}_{color}"

        source_path = Path(str(row["source_path"])) if pd.notna(row["source_path"]) else Path("")
        session_id = source_path.parent.name if source_path else None
        source_subject = session_id.split("_ses-")[0] if session_id and "_ses-" in session_id else str(row["subject_id"])

        nifti_seg = source_path if source_path.exists() else Path("")
        nifti_t1 = nifti_seg.parent / "T1w_norm.nii.gz" if nifti_seg else Path("")
        nifti_path = nifti_t1 if nifti_t1.exists() else nifti_seg

        if not nifti_path or not nifti_path.exists():
            rows.append(
                {
                    "cluster_label": label,
                    "cluster_color": color,
                    "cluster_name": cluster_name,
                    "subject_id": str(row["subject_id"]),
                    "SOURCE_SUBJECT": source_subject,
                    "SUBJECT_ID": session_id,
                    "file": None,
                    "nifti_path": str(nifti_path),
                    "density_ratio": np.nan,
                    "density_ratio_2d": np.nan,
                    "nonzero_voxels": np.nan,
                    "shape_x": np.nan,
                    "shape_y": np.nan,
                    "shape_z": np.nan,
                }
            )
            continue

        try:
            img = nib.load(str(nifti_path))
            data = np.asarray(img.get_fdata(), dtype=np.float32)

            if data.ndim == 4:
                data = data[..., 0]
            if data.ndim != 3:
                raise ValueError(f"Unexpected data ndim {data.ndim} for {nifti_path.name}")

            nonzero = int((data > 0).sum())
            total = int(data.size)
            density_3d = float(nonzero / total)
            density_2d = extract_2d_density_from_volume(data)

            rows.append(
                {
                    "cluster_label": label,
                    "cluster_color": color,
                    "cluster_name": cluster_name,
                    "subject_id": str(row["subject_id"]),
                    "SOURCE_SUBJECT": source_subject,
                    "SUBJECT_ID": session_id,
                    "file": nifti_path.name,
                    "nifti_path": str(nifti_path),
                    "density_ratio": density_3d,
                    "density_ratio_2d": density_2d,
                    "nonzero_voxels": nonzero,
                    "shape_x": int(data.shape[0]),
                    "shape_y": int(data.shape[1]),
                    "shape_z": int(data.shape[2]),
                }
            )
        except Exception as ex:
            print(f"Warning: failed to read {nifti_path}: {ex}")
            rows.append(
                {
                    "cluster_label": label,
                    "cluster_color": color,
                    "cluster_name": cluster_name,
                    "subject_id": str(row["subject_id"]),
                    "SOURCE_SUBJECT": source_subject,
                    "SUBJECT_ID": session_id,
                    "file": nifti_path.name,
                    "nifti_path": str(nifti_path),
                    "density_ratio": np.nan,
                    "density_ratio_2d": np.nan,
                    "nonzero_voxels": np.nan,
                    "shape_x": np.nan,
                    "shape_y": np.nan,
                    "shape_z": np.nan,
                }
            )

    nifti_df = pd.DataFrame(rows)
    return nifti_df


def merge_metadata(nifti_df: pd.DataFrame) -> pd.DataFrame:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {METADATA_CSV}")

    metadata = pd.read_csv(METADATA_CSV, sep=";")
    metadata["SUBJECT_ID"] = metadata["SUBJECT_ID"].astype(str).str.strip()
    metadata["SOURCE_SUBJECT"] = metadata["SOURCE_SUBJECT"].astype(str).str.strip()

    keep_cols = ["SUBJECT_ID", "SOURCE_SUBJECT", "SEX", "AGE", "DIAGNOSIS_SUBTYPE"]
    missing = set(keep_cols) - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")

    metadata = metadata[keep_cols].copy()

    merged = nifti_df.merge(metadata, on="SUBJECT_ID", how="left", suffixes=("", "_meta"))

    # Keep SOURCE_SUBJECT from image-derived path; if absent, use metadata-derived value.
    if "SOURCE_SUBJECT" not in merged.columns and "SOURCE_SUBJECT_meta" in merged.columns:
        merged["SOURCE_SUBJECT"] = merged["SOURCE_SUBJECT_meta"]
    elif "SOURCE_SUBJECT_meta" in merged.columns:
        merged["SOURCE_SUBJECT"] = merged["SOURCE_SUBJECT"].fillna(merged["SOURCE_SUBJECT_meta"])

    # Fallback for rows where exact session-level SUBJECT_ID wasn't found.
    fallback_idx = merged["AGE"].isna()
    if fallback_idx.any():
        meta_by_source = (
            metadata.sort_values("SUBJECT_ID")
            .drop_duplicates(subset=["SOURCE_SUBJECT"], keep="first")
            .reset_index(drop=True)
        )
        fb = merged.loc[fallback_idx, ["SOURCE_SUBJECT"]].merge(
            meta_by_source[["SOURCE_SUBJECT", "SEX", "AGE", "DIAGNOSIS_SUBTYPE"]],
            on="SOURCE_SUBJECT",
            how="left",
        )
        merged.loc[fallback_idx, "SEX"] = fb["SEX"].values
        merged.loc[fallback_idx, "AGE"] = fb["AGE"].values
        merged.loc[fallback_idx, "DIAGNOSIS_SUBTYPE"] = fb["DIAGNOSIS_SUBTYPE"].values

    # SOURCE_SUBJECT_meta is only an intermediate merge artifact.
    if "SOURCE_SUBJECT_meta" in merged.columns:
        merged = merged.drop(columns=["SOURCE_SUBJECT_meta"])

    merged["AGE"] = pd.to_numeric(merged["AGE"], errors="coerce")
    merged["SEX"] = merged["SEX"].fillna("Unknown").astype(str)
    merged["DIAGNOSIS_SUBTYPE"] = merged["DIAGNOSIS_SUBTYPE"].fillna("Unknown").astype(str)
    return merged


def save_plots(merged: pd.DataFrame, cluster_order: list[str]) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.boxplot(x="cluster_name", y="AGE", data=merged, order=cluster_order)
    plt.title("Age Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Age")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "age_by_cluster.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.countplot(x="cluster_name", hue="SEX", data=merged, order=cluster_order)
    plt.title("Sex Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sex_by_cluster.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.countplot(x="cluster_name", hue="DIAGNOSIS_SUBTYPE", data=merged, order=cluster_order)
    plt.title("CDR Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cdr_by_cluster.png", dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== FULL-CLUSTER ANALYSIS (ALL SUBJECTS) ===")

    labels_df, outlier_label, keep_labels, label_to_color = load_cluster_labels()

    print(f"Outlier cluster removed: label {outlier_label} ({OUTLIER_COLOR_NAME})")
    print(f"Kept labels: {keep_labels}")
    print(f"Label->color: {label_to_color}")

    nifti_df = analyze_all_subjects(labels_df, keep_labels, label_to_color)

    redundant_csv = RESULTS_DIR / "nifti_metadata_full.csv"
    if redundant_csv.exists():
        redundant_csv.unlink()

    print("\n=== NIfTI METRICS (ALL SUBJECTS IN KEPT CLUSTERS) ===")
    numeric_cols = ["density_ratio", "density_ratio_2d", "nonzero_voxels", "shape_x", "shape_y", "shape_z"]
    print(nifti_df.groupby("cluster_name")[numeric_cols].mean(numeric_only=True))

    merged = merge_metadata(nifti_df)
    merged.to_csv(RESULTS_DIR / "nifti_with_metadata.csv", index=False)

    print(f"\nMerged rows: {len(merged)}")
    print(f"Rows with AGE available: {int(merged['AGE'].notna().sum())}/{len(merged)}")

    print("\n=== AGE ===")
    print(merged.groupby("cluster_name")["AGE"].describe())

    print("\n=== SEX ===")
    print(pd.crosstab(merged["cluster_name"], merged["SEX"]))

    print("\n=== CDR ===")
    print(pd.crosstab(merged["cluster_name"], merged["DIAGNOSIS_SUBTYPE"]))

    cluster_order = [f"cluster_label_{lbl}_{label_to_color[lbl]}" for lbl in keep_labels]
    save_plots(merged, cluster_order)

    summary = (
        merged.groupby(["cluster_label", "cluster_color", "cluster_name"], as_index=False)
        .agg(
            n_subjects=("subject_id", "count"),
            age_mean=("AGE", "mean"),
            density_ratio_2d_mean=("density_ratio_2d", "mean"),
            density_ratio_3d_mean=("density_ratio", "mean"),
            nonzero_voxels_mean=("nonzero_voxels", "mean"),
        )
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )
    summary.to_csv(RESULTS_DIR / "cluster_final_summary.csv", index=False)

    print("\n=== FINAL SUMMARY (WITH CLUSTER NAME + COLOR) ===")
    print(summary.to_string(index=False))

    print("\n✅ ALL ANALYSIS COMPLETE ✅")


if __name__ == "__main__":
    main()
