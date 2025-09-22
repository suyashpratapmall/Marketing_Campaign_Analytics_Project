# step3.py
# EDA for Marketing Campaign Analytics — robust, handles large data & messy column names
# Updated: fallback Spearman via rank-correlation so scipy is not required.

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", rc={"figure.dpi": 120})

# ----- Config -----
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)
SAMPLE_SIZE = 5000   # sample size for scatter to avoid overplotting

# ----- Load dataset (try common locations) -----
candidates = [
    Path("data") / "data.csv",
    Path("data") / "marketing_campaign_dataset.csv",
    Path("marketing_campaign_dataset.csv"),
    Path("data.csv"),
    Path("marketing_campaign_dataset.csv"),
]
df = None
for p in candidates:
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        print("Loaded:", p)
        break
if df is None:
    raise FileNotFoundError("No CSV found. Put your CSV in data/data.csv or project root.")

print("Initial shape:", df.shape)

# ----- Normalize column names to safe snake_case lowercase -----
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"__+", "_", regex=True)
    .str.strip("_")
    .str.lower()
)
print("Columns:", df.columns.tolist())

# ----- Common column names (after normalization) -----
def find_col(*options):
    for o in options:
        if o in df.columns:
            return o
    return None

campaign_col = find_col("campaign_type", "campaign")
channel_col = find_col("channel_used", "channel")
cost_col = find_col("acquisition_cost", "cost", "spend")
conv_col = find_col("conversion_rate", "conversion_rate_pct", "conversion", "conversionrate")
roi_col = find_col("roi",)
date_col = find_col("date", "start_date")

print("Using columns:", {
    "campaign": campaign_col,
    "channel": channel_col,
    "cost": cost_col,
    "conversion": conv_col,
    "roi": roi_col,
    "date": date_col,
})

# ----- Parse / clean numeric fields -----
if cost_col:
    df[cost_col] = (
        df[cost_col]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
    )
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

if conv_col:
    df[conv_col] = pd.to_numeric(df[conv_col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    # if values look like percentages (e.g., max > 1.5), divide
    try:
        max_conv = df[conv_col].max(skipna=True)
        if pd.notna(max_conv) and max_conv > 1.5:
            df[conv_col] = df[conv_col] / 100.0
    except Exception:
        pass

if roi_col:
    df[roi_col] = pd.to_numeric(df[roi_col], errors="coerce")

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

# Quick sanity print (ASCII-safe)
print("After cleaning - non-null counts:")
count_cols = [c for c in [campaign_col, channel_col, cost_col, conv_col, roi_col, date_col] if c]
print(df[count_cols].notnull().sum())

# ----- 1) ROI by Campaign Type (aggregate) -----
if campaign_col and roi_col:
    roi_by_type = df.groupby(campaign_col)[roi_col].mean().reset_index().sort_values(by=roi_col, ascending=False)
    roi_by_type.to_csv(VIS_DIR / "roi_by_campaign_type.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=campaign_col, y=roi_col, data=roi_by_type, dodge=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average ROI by Campaign Type")
    plt.ylabel("Average ROI")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "roi_by_campaign_type.png")
    plt.close()
    print("Saved:", VIS_DIR / "roi_by_campaign_type.png")
else:
    print("Skipping ROI by campaign (missing columns).")

# ----- 2) Conversion Rate vs Acquisition Cost (sample + aggregate) -----
corr = np.nan
if cost_col and conv_col:
    valid_for_scatter = df.dropna(subset=[cost_col, conv_col])
    n = min(SAMPLE_SIZE, len(valid_for_scatter))
    if n == 0:
        print("Not enough data for conversion vs cost scatter.")
    else:
        sample_df = valid_for_scatter.sample(n, random_state=42)
        plt.figure(figsize=(8, 6))
        if campaign_col:
            sns.scatterplot(x=cost_col, y=conv_col, hue=campaign_col, data=sample_df, alpha=0.6, legend="brief")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            sns.scatterplot(x=cost_col, y=conv_col, data=sample_df, alpha=0.6)
        plt.xlabel("Acquisition Cost")
        plt.ylabel("Conversion Rate")
        plt.title(f"Conversion Rate vs Acquisition Cost (sample n={n})")
        plt.tight_layout()
        plt.savefig(VIS_DIR / "conversion_vs_cost_sample.png")
        plt.close()
        print("Saved:", VIS_DIR / "conversion_vs_cost_sample.png")

    # compute Spearman without scipy: correlation of ranks
    corr_df = valid_for_scatter[[cost_col, conv_col]].dropna()
    if len(corr_df) >= 10:
        try:
            # Spearman via Pearson of ranks
            corr = corr_df[cost_col].rank().corr(corr_df[conv_col].rank(), method="pearson")
        except Exception:
            corr = np.nan
else:
    print("Skipping conversion vs cost (missing columns).")

# ----- 3) ROI trend by Channel over time (aggregate by month) -----
if date_col and channel_col and roi_col:
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    top_channels = df[channel_col].value_counts().nlargest(6).index.tolist()
    roi_time = df[df[channel_col].isin(top_channels)].groupby(["month", channel_col])[roi_col].mean().reset_index()
    roi_time["month_dt"] = pd.to_datetime(roi_time["month"] + "-01", errors="coerce")
    roi_time = roi_time.sort_values("month_dt")
    roi_time.to_csv(VIS_DIR / "roi_time_by_channel_top.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="month_dt", y=roi_col, hue=channel_col, data=roi_time, marker="o")
    plt.gca().xaxis.set_tick_params(rotation=45)
    plt.title("ROI Trend Over Time - Top Channels")
    plt.xlabel("Month")
    plt.ylabel("Average ROI")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "roi_trend_by_channel_top.png")
    plt.close()
    print("Saved:", VIS_DIR / "roi_trend_by_channel_top.png")

    trend = (
        roi_time.groupby(channel_col)
        .apply(lambda g: float(g.sort_values("month_dt")[roi_col].iloc[-1] - g.sort_values("month_dt")[roi_col].iloc[0]))
        .reset_index(name="roi_change")
        .sort_values("roi_change", ascending=False)
    )
    trend.to_csv(VIS_DIR / "channel_roi_trend_change.csv", index=False)
else:
    print("Skipping ROI trend by channel (missing columns).")

# ----- 4) Channel summary pivot -----
if channel_col:
    agg_funcs = {}
    if roi_col:
        agg_funcs[roi_col] = "mean"
    if conv_col:
        agg_funcs[conv_col] = "mean"
    if cost_col:
        agg_funcs[cost_col] = "mean"
    if agg_funcs:
        channel_summary = df.groupby(channel_col).agg(agg_funcs).reset_index()
        rename_map = {}
        if roi_col: rename_map[roi_col] = "avg_roi"
        if conv_col: rename_map[conv_col] = "avg_conversion"
        if cost_col: rename_map[cost_col] = "avg_cost"
        channel_summary = channel_summary.rename(columns=rename_map)
        channel_summary.to_csv(VIS_DIR / "channel_summary.csv", index=False)
        print("Saved:", VIS_DIR / "channel_summary.csv")

# ----- 5) Generate short insights (text) -----
insights = []
try:
    if campaign_col and roi_col:
        top_campaign = roi_by_type.iloc[0][campaign_col]
        top_campaign_roi = float(roi_by_type.iloc[0][roi_col])
        insights.append(f"Top campaign type by average ROI: {top_campaign} (avg ROI = {top_campaign_roi:.2f}).")
    if channel_col and roi_col and (VIS_DIR / "channel_summary.csv").exists():
        ch_summ = pd.read_csv(VIS_DIR / "channel_summary.csv")
        if not ch_summ.empty:
            best_channel = ch_summ.sort_values("avg_roi", ascending=False).iloc[0]
            insights.append(f"Best channel by ROI: {best_channel[channel_col]} (avg ROI = {best_channel['avg_roi']:.2f}).")
    if not np.isnan(corr):
        if corr > 0.2:
            insights.append(f"There is a positive relationship between Acquisition Cost and Conversion Rate (Spearman rho = {corr:.2f}). Consider testing higher spend on efficient creatives.")
        elif corr < -0.2:
            insights.append(f"There is a negative relationship between Acquisition Cost and Conversion Rate (Spearman rho = {corr:.2f}). High spend may not guarantee conversions.")
        else:
            insights.append(f"Little or no monotonic relationship between Acquisition Cost and Conversion Rate (Spearman rho = {corr:.2f}).")
    if (VIS_DIR / "channel_roi_trend_change.csv").exists():
        tr = pd.read_csv(VIS_DIR / "channel_roi_trend_change.csv")
        up = tr.sort_values("roi_change", ascending=False).iloc[0]
        down = tr.sort_values("roi_change", ascending=True).iloc[0]
        insights.append(f"Channel with largest ROI increase over time: {up[channel_col]} (change = {up['roi_change']:.2f}).")
        insights.append(f"Channel with largest ROI decrease over time: {down[channel_col]} (change = {down['roi_change']:.2f}).")
except Exception as e:
    insights.append("Could not compute some insights due to data issues: " + str(e))

with open(VIS_DIR / "insights.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(insights))
print("Insights written to", VIS_DIR / "insights.txt")
print("---- Quick insights ----")
for s in insights:
    print("-", s)

print("EDA completed. Visuals & csvs in:", VIS_DIR.resolve())
