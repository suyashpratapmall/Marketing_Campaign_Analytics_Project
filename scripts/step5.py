# step5.py
# Efficient Step 5: Customer Segmentation (scalable for large data)
# Uses sampling for silhouette and MiniBatchKMeans for fast clustering on full data.

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)

# ---- Config ----
SILHOUETTE_SAMPLE = 10000   # sample size to compute silhouette score (keeps it fast)
MBK_BATCH_SIZE = 1024       # minibatch size for MiniBatchKMeans
MBK_MAX_ITER = 200
RANDOM_STATE = 42

# ---- Load data (same as before) ----
candidates = [
    Path("data") / "data.csv",
    Path("data") / "marketing_campaign_dataset.csv",
    Path("marketing_campaign_dataset.csv"),
    Path("data.csv"),
]
df = None
for p in candidates:
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        print("Loaded:", p)
        break
if df is None:
    raise FileNotFoundError("No CSV found. Put your CSV in data/data.csv or project root.")

# ---- Normalize column names ----
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"__+", "_", regex=True)
    .str.strip("_")
    .str.lower()
)

def find_col(*opts):
    for o in opts:
        if o in df.columns:
            return o
    return None

eng_col = find_col("engagement_score", "engagement")
conv_col = find_col("conversion_rate", "conversion_rate", "conversion")
roi_col = find_col("roi",)

print("Using columns:", eng_col, conv_col, roi_col)
if not eng_col or not conv_col or not roi_col:
    raise RuntimeError("Missing required columns: engagement_score, conversion_rate, roi")

# ---- Clean numeric fields ----
df[eng_col] = pd.to_numeric(df[eng_col], errors="coerce")
df[conv_col] = df[conv_col].astype(str).str.replace("%","", regex=False)
df[conv_col] = pd.to_numeric(df[conv_col], errors="coerce")
if pd.notna(df[conv_col].max()) and df[conv_col].max() > 1.5:
    df[conv_col] = df[conv_col] / 100.0
df[roi_col] = pd.to_numeric(df[roi_col], errors="coerce")

# Keep identifiers if present
id_cols = [c for c in ("campaign_id", "company", "customer_segment") if c in df.columns]
seg_df = df[id_cols + [eng_col, conv_col, roi_col]].dropna(subset=[eng_col, conv_col, roi_col]).copy()
print("Rows used for segmentation:", len(seg_df))

# ---- Standardize features ----
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
except Exception as e:
    raise RuntimeError("sklearn is required for this efficient script. Install with: pip install scikit-learn") from e

X = seg_df[[eng_col, conv_col, roi_col]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_samples = X_scaled.shape[0]
print(f"Scaled feature matrix: rows={n_samples}, cols={X_scaled.shape[1]}")

# ---- Choose best k with silhouette on a sample ----
k_candidates = range(2, 7)
best_k = None
best_score = -1.0

# sample indices for silhouette (use stratified random sample)
samp_n = min(SILHOUETTE_SAMPLE, n_samples)
if n_samples > samp_n:
    samp_idx = np.random.RandomState(RANDOM_STATE).choice(n_samples, size=samp_n, replace=False)
    X_samp = X_scaled[samp_idx]
else:
    X_samp = X_scaled

print(f"Computing silhouette on sample of size {len(X_samp)} (this is fast)")

for k in k_candidates:
    print(f" -> Trying k={k} ...", end="", flush=True)
    try:
        # use MiniBatchKMeans for speed during selection too
        mbk = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=MBK_BATCH_SIZE, max_iter=100, n_init=3)
        labels_samp = mbk.fit_predict(X_samp)
        # silhouette on the sample (fast)
        score = silhouette_score(X_samp, labels_samp)
        print(f" silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    except Exception as ex:
        print(" failed:", ex)

if best_k is None:
    best_k = 3
    print("Could not determine best_k by silhouette: defaulting to k=3")
else:
    print(f"Selected best_k={best_k} (silhouette={best_score:.4f})")

# ---- Fit final MiniBatchKMeans on full data ----
print("Fitting MiniBatchKMeans on full dataset (fast)...")
mbk_final = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, batch_size=MBK_BATCH_SIZE, max_iter=MBK_MAX_ITER, n_init=5)
labels_full = mbk_final.fit_predict(X_scaled)
seg_df["cluster"] = labels_full
print("Clustering complete. Cluster counts:")
print(seg_df["cluster"].value_counts().sort_index())

# ---- Save clusters and summaries ----
out_cols = id_cols + [eng_col, conv_col, roi_col, "cluster"]
seg_df[out_cols].to_csv(VIS_DIR / "clusters.csv", index=False)
print("Saved clusters.csv")

cluster_summary = seg_df.groupby("cluster").agg(
    n=(eng_col, "count"),
    engagement_mean=(eng_col, "mean"),
    conversion_mean=(conv_col, "mean"),
    roi_mean=(roi_col, "mean")
).reset_index().sort_values("cluster")
cluster_summary.to_csv(VIS_DIR / "cluster_summary.csv", index=False)
print("Saved cluster_summary.csv")

# ---- Centroids (in original feature space) ----
centroids_scaled = mbk_final.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_orig, columns=[eng_col, conv_col, roi_col])
centroids_df["cluster"] = centroids_df.index
centroids_df.to_csv(VIS_DIR / "kmeans_centroids.csv", index=False)
print("Saved kmeans_centroids.csv")

# ---- Simple PCA 2D visualization (sample to avoid overplotting) ----
try:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X_scaled)
    seg_df["_pca1"] = Xp[:,0]
    seg_df["_pca2"] = Xp[:,1]
    sample = seg_df.sample(n=min(5000, len(seg_df)), random_state=RANDOM_STATE)
    plt.figure(figsize=(8,6))
    import seaborn as sns
    sns.scatterplot(x="_pca1", y="_pca2", hue="cluster", data=sample, palette="tab10", s=20, alpha=0.7, legend="brief")
    plt.title(f"KMeans (k={best_k}) clusters in PCA space")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "pca_clusters.png")
    plt.close()
    print("Saved pca_clusters.png")
except Exception as e:
    print("PCA failed/skipped:", e)

# ---- Cluster counts plot ----
plt.figure(figsize=(8,4))
cluster_summary.set_index("cluster")["n"].plot(kind="bar")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Cluster sizes")
plt.tight_layout()
plt.savefig(VIS_DIR / "cluster_counts.png")
plt.close()
print("Saved cluster_counts.png")

# ---- Interpret clusters into human-friendly insights ----
# Use dataset-wide tertiles to map cluster means to Low/Med/High
eng_bins = pd.qcut(seg_df[eng_col], q=3, retbins=True)[1]
conv_bins = pd.qcut(seg_df[conv_col], q=3, retbins=True)[1]
roi_bins = pd.qcut(seg_df[roi_col], q=3, retbins=True)[1]

def tertile_label(value, bins):
    if value <= bins[1]:
        return "Low"
    elif value <= bins[2]:
        return "Medium"
    else:
        return "High"

insights = []
for _, row in cluster_summary.iterrows():
    cl = int(row["cluster"])
    e = row["engagement_mean"]
    c = row["conversion_mean"]
    r = row["roi_mean"]
    e_lab = tertile_label(e, eng_bins)
    c_lab = tertile_label(c, conv_bins)
    r_lab = tertile_label(r, roi_bins)
    label = f"{e_lab}Eng/{c_lab}Conv/{r_lab}ROI"
    if r_lab=="High" and e_lab=="Low":
        action = "High ROI, low engagement — nurture with email/remarketing & personalized offers."
    elif r_lab=="High" and e_lab=="High":
        action = "High ROI & high engagement — scale with lookalike audiences and increase budget cautiously."
    elif r_lab=="Low" and e_lab=="High":
        action = "High engagement but low ROI — improve funnel & landing page tests."
    elif r_lab=="Low" and e_lab=="Low":
        action = "Low engagement & ROI — deprioritize or redesign creatives."
    else:
        action = "Balanced segment — targeted experiments."
    insights.append((cl, int(row["n"]), e, c, r, label, action))

ins_df = pd.DataFrame(insights, columns=["cluster","n","engagement_mean","conversion_mean","roi_mean","label","action"])
ins_df.to_csv(VIS_DIR / "cluster_insights.csv", index=False)

# Write human-readable text file
with open(VIS_DIR / "cluster_insights.txt", "w", encoding="utf-8") as fh:
    fh.write(f"KMeans (MiniBatch) segmentation k={best_k}\n")
    for _, r in ins_df.iterrows():
        fh.write(f"Cluster {int(r['cluster'])}: n={int(r['n'])} | {r['label']} | action: {r['action']} | means: eng={r['engagement_mean']:.2f}, conv={r['conversion_mean']:.4f}, roi={r['roi_mean']:.2f}\n")
print("Saved cluster_insights.txt and cluster_insights.csv")

print("Segmentation complete. Files in:", VIS_DIR.resolve())
