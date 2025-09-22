# step4.py
# Multi-Touch Attribution (MTA) — robust script that works with
# - event-level touchpoint journeys (if available), OR
# - aggregated campaign-level data (fallback)
#
# Outputs:
# - visualizations/attribution_summary.csv
# - visualizations/attribution_by_channel.png
# - visualizations/time_decay_attribution.png (if applicable)
# - visualizations/channel_summary.csv
# - visualizations/recommendations.txt

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid", rc={"figure.dpi": 120})
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)

# ----- Load data (same approach as step3) -----
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

# normalize columns (snake_case lowercase)
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"__+", "_", regex=True)
    .str.strip("_")
    .str.lower()
)

print("Columns:", df.columns.tolist())

# detect event-level journey columns
has_user = any(c in df.columns for c in ["user_id", "customer_id", "client_id", "session_id"])
has_time = any(c in df.columns for c in ["timestamp", "touchpoint_time", "time", "date"])
has_touch_channel = any(c in df.columns for c in ["touchpoint", "touchpoint_channel", "channel", "channel_used"])

# Helper finds
def find_col(*opts):
    for o in opts:
        if o in df.columns:
            return o
    return None

channel_col = find_col("touchpoint_channel", "channel_used", "channel")
date_col = find_col("timestamp", "touchpoint_time", "date")
user_col = find_col("user_id", "customer_id", "client_id", "session_id")
clicks_col = find_col("clicks",)
impr_col = find_col("impressions",)
conv_rate_col = find_col("conversion_rate", "conversion")
acq_cost_col = find_col("acquisition_cost", "cost", "spend")
roi_col = find_col("roi",)
revenue_col = find_col("revenue",)

print("Detected columns:", {"user": user_col, "time": date_col, "channel": channel_col})

# -----------------------
# Prepare base metrics (aggregated fallback)
# -----------------------
# Ensure numeric parsing consistent with step3 assumptions
if acq_cost_col:
    df[acq_cost_col] = (
        df[acq_cost_col].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
    )
    df[acq_cost_col] = pd.to_numeric(df[acq_cost_col], errors="coerce")

if conv_rate_col:
    df[conv_rate_col] = pd.to_numeric(df[conv_rate_col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    try:
        if pd.notna(df[conv_rate_col].max()) and df[conv_rate_col].max() > 1.5:
            df[conv_rate_col] = df[conv_rate_col] / 100.0
    except Exception:
        pass

if roi_col and acq_cost_col and revenue_col is None:
    # revenue approximation: revenue = roi * acquisition_cost
    df["_revenue_est"] = df[roi_col] * df[acq_cost_col]
    revenue_col = "_revenue_est"

# conversions estimate: prefer clicks * conv_rate, else impressions * conv_rate
if conv_rate_col:
    if clicks_col:
        df["_conversions_est"] = df[clicks_col] * df[conv_rate_col]
    elif impr_col:
        df["_conversions_est"] = df[impr_col] * df[conv_rate_col]
    else:
        # fallback: conversions = conversion_rate * 1 (rare)
        df["_conversions_est"] = df[conv_rate_col]

# if conversions column already exists, prefer it
explicit_conv_col = find_col("conversions", "conversion_count")
if explicit_conv_col:
    df["_conversions_est"] = pd.to_numeric(df[explicit_conv_col], errors="coerce")

# if clicks missing for all, create proxy using impressions
if clicks_col is None and impr_col is not None:
    df["_clicks_proxy"] = df[impr_col] * 0.01  # assume 1% CTR proxy
    clicks_col_eff = "_clicks_proxy"
else:
    clicks_col_eff = clicks_col

# -----------------------
# Attribution functions
# -----------------------
def fractional_by_clicks(df_in, channel_col, clicks_col_eff, conversions_col="_conversions_est", revenue_col=revenue_col, cost_col=acq_cost_col):
    """
    Fractional attribution proportional to clicks per channel.
    """
    df_valid = df_in.dropna(subset=[clicks_col_eff, conversions_col]) if clicks_col_eff else df_in.dropna(subset=[conversions_col])
    # aggregate per channel
    agg = df_valid.groupby(channel_col).agg(
        clicks_sum=(clicks_col_eff, "sum") if clicks_col_eff else (conversions_col, "sum"),
        convs_sum=(conversions_col, "sum"),
        revenue_sum=(revenue_col, "sum") if revenue_col else (conversions_col, "sum"),
        cost_sum=(cost_col, "sum") if cost_col else (clicks_col_eff, "sum")
    ).reset_index()
    # if clicks present allocate conversion credit proportionally to clicks
    total_clicks = agg["clicks_sum"].sum() if "clicks_sum" in agg.columns else 0
    if total_clicks > 0:
        agg["click_share"] = agg["clicks_sum"] / total_clicks
        total_convs = agg["convs_sum"].sum()
        agg["attributed_conversions"] = agg["click_share"] * total_convs
    else:
        # fallback: proportional to conversions themselves (no fractional)
        agg["attributed_conversions"] = agg["convs_sum"]
        agg["click_share"] = np.nan
    # attributed revenue = revenue_sum * click_share (or revenue_sum if no clicks)
    agg["attributed_revenue"] = (agg["click_share"] * agg["revenue_sum"]).fillna(agg["revenue_sum"])
    # attributed cost = cost_sum * click_share (or cost_sum)
    agg["attributed_cost"] = (agg["click_share"] * agg["cost_sum"]).fillna(agg["cost_sum"])
    agg["attributed_roi"] = agg["attributed_revenue"] / agg["attributed_cost"].replace({0: np.nan})
    return agg.sort_values("attributed_roi", ascending=False)

def time_decay_by_clicks(df_in, channel_col, date_col, clicks_col_eff, conversions_col="_conversions_est", revenue_col=revenue_col, cost_col=acq_cost_col, lambda_month=0.5):
    """
    Time-decay fractional attribution: recent touchpoints get higher weight.
    lambda_month controls decay aggressiveness.
    """
    df_tmp = df_in.copy()
    if date_col is None:
        return None
    df_tmp[date_col] = pd.to_datetime(df_tmp[date_col], errors="coerce")
    max_date = df_tmp[date_col].max()
    df_tmp["_age_months"] = ((max_date - df_tmp[date_col]).dt.days / 30.0).clip(lower=0)
    df_tmp["_decay_weight"] = np.exp(-lambda_month * df_tmp["_age_months"])
    if clicks_col_eff:
        df_tmp["_weighted_clicks"] = df_tmp[clicks_col_eff] * df_tmp["_decay_weight"]
    else:
        df_tmp["_weighted_clicks"] = df_tmp["_decay_weight"]
    df_tmp["_weighted_revenue"] = df_tmp.get(revenue_col, 0) * df_tmp["_decay_weight"] if revenue_col else 0
    agg = df_tmp.groupby(channel_col).agg(
        weighted_clicks=("_weighted_clicks", "sum"),
        convs_sum=(conversions_col, "sum"),
        weighted_revenue=("_weighted_revenue", "sum"),
        cost_sum=(cost_col, "sum") if cost_col else ("_weighted_clicks", "sum")
    ).reset_index()
    total_wclicks = agg["weighted_clicks"].sum()
    if total_wclicks > 0:
        agg["wclick_share"] = agg["weighted_clicks"] / total_wclicks
        total_convs = agg["convs_sum"].sum()
        agg["attributed_conversions"] = agg["wclick_share"] * total_convs
    else:
        agg["attributed_conversions"] = agg["convs_sum"]
        agg["wclick_share"] = np.nan
    agg["attributed_revenue"] = (agg["wclick_share"] * agg["weighted_revenue"]).fillna(agg["weighted_revenue"])
    agg["attributed_cost"] = (agg["wclick_share"] * agg["cost_sum"]).fillna(agg["cost_sum"])
    agg["attributed_roi"] = agg["attributed_revenue"] / agg["attributed_cost"].replace({0: np.nan})
    return agg.sort_values("attributed_roi", ascending=False)

def last_touch_approx(df_in, channel_col, conversions_col="_conversions_est", revenue_col=revenue_col, cost_col=acq_cost_col):
    """
    Last-touch approximation on aggregated data: assume conversions belong to channel of record.
    (Works if each row corresponds to a conversion/touch pairing.)
    """
    agg = df_in.groupby(channel_col).agg(
        convs_sum=(conversions_col, "sum"),
        revenue_sum=(revenue_col, "sum") if revenue_col else (conversions_col, "sum"),
        cost_sum=(cost_col, "sum") if cost_col else (conversions_col, "sum")
    ).reset_index()
    agg["attributed_conversions"] = agg["convs_sum"]
    agg["attributed_revenue"] = agg["revenue_sum"]
    agg["attributed_cost"] = agg["cost_sum"]
    agg["attributed_roi"] = agg["attributed_revenue"] / agg["attributed_cost"].replace({0: np.nan})
    return agg.sort_values("attributed_roi", ascending=False)

# -----------------------
# Choose method depending on data
# -----------------------
if has_user and has_time and channel_col:
    # Placeholder: event-level path attribution (first/last/linear) would go here
    # But since your dataset may not be event-level journeys, we default to aggregated methods.
    print("User-level journeys detected; but script will use aggregated methods by default unless touchpoint events present.")
    method = "aggregated_fractional"
else:
    method = "aggregated_fractional"

print("Running attribution method:", method)

# Run fractional_by_clicks (baseline)
if channel_col:
    baseline = fractional_by_clicks(df, channel_col, clicks_col_eff)
    baseline.to_csv(VIS_DIR / "attribution_fractional_by_clicks.csv", index=False)
    print("Saved fractional_by_clicks results:", VIS_DIR / "attribution_fractional_by_clicks.csv")
else:
    raise RuntimeError("No channel column detected. Cannot perform channel attribution.")

# Run time-decay attribution if date_col present
time_decay_result = None
if date_col:
    time_decay_result = time_decay_by_clicks(df, channel_col, date_col, clicks_col_eff)
    if time_decay_result is not None:
        time_decay_result.to_csv(VIS_DIR / "attribution_time_decay.csv", index=False)
        print("Saved time-decay results:", VIS_DIR / "attribution_time_decay.csv")

# Run last-touch approx
last_touch = last_touch_approx(df, channel_col)
last_touch.to_csv(VIS_DIR / "attribution_last_touch_approx.csv", index=False)
print("Saved last-touch approx:", VIS_DIR / "attribution_last_touch_approx.csv")

# -----------------------
# Build consolidated summary (choose baseline fractional unless you want other)
# -----------------------
summary = baseline.copy()
summary = summary.rename(columns={
    channel_col: "channel",
    "clicks_sum": "clicks_sum",
    "convs_sum": "conversions_sum",
    "revenue_sum": "revenue_sum",
    "cost_sum": "cost_sum",
    "click_share": "click_share",
    "attributed_conversions": "attributed_conversions",
    "attributed_revenue": "attributed_revenue",
    "attributed_cost": "attributed_cost",
    "attributed_roi": "attributed_roi"
})
summary = summary[["channel", "clicks_sum", "conversions_sum", "revenue_sum", "cost_sum", "click_share", "attributed_conversions", "attributed_revenue", "attributed_cost", "attributed_roi"]]

summary.to_csv(VIS_DIR / "attribution_summary.csv", index=False)
print("Saved attribution_summary.csv")

# -----------------------
# Budget reallocation recommendation (simple, conservative)
# Approach:
# - Compute current spend share per channel (cost_sum / total_cost)
# - Compute relative performance = attributed_roi / mean_attributed_roi
# - Proposed new share = current_share * (1 + alpha * (relative_perf - 1))
# - Normalize to 1 and clamp changes to +/- 30% to avoid radical swings
# -----------------------
rec = summary.copy()
if "cost_sum" not in rec.columns:
    rec["cost_sum"] = rec["attributed_cost"].fillna(0)

total_cost = rec["cost_sum"].sum() if rec["cost_sum"].sum() > 0 else 1.0
rec["current_share"] = rec["cost_sum"] / total_cost
mean_roi = rec["attributed_roi"].dropna()
mean_roi = mean_roi.mean() if not mean_roi.empty else rec["attributed_roi"].replace([np.inf, -np.inf], np.nan).mean()

rec["relative_perf"] = rec["attributed_roi"] / (mean_roi if mean_roi and not np.isnan(mean_roi) else 1.0)
alpha = 0.35  # aggressiveness
rec["proposed_share_raw"] = rec["current_share"] * (1 + alpha * (rec["relative_perf"] - 1))
# clamp change to +/- 30% of current_share
def clamp_proposal(row):
    cur = row["current_share"]
    raw = row["proposed_share_raw"]
    minv = max(0, cur * 0.7)
    maxv = cur * 1.3
    return min(max(raw, minv), maxv)
rec["proposed_share_clamped"] = rec.apply(clamp_proposal, axis=1)
# normalize to sum to 1
rec["proposed_share"] = rec["proposed_share_clamped"] / rec["proposed_share_clamped"].sum()

rec_out = rec[["channel", "cost_sum", "current_share", "attributed_roi", "relative_perf", "proposed_share"]].sort_values("proposed_share", ascending=False)
rec_out.to_csv(VIS_DIR / "attribution_recommendations.csv", index=False)
print("Saved recommendations:", VIS_DIR / "attribution_recommendations.csv")

# save short human-friendly recommendations
lines = []
lines.append("Multi-touch attribution results (fractional by clicks) saved to visualizations/attribution_summary.csv")
lines.append("Recommendations (conservative reallocation) saved to visualizations/attribution_recommendations.csv")
lines.append("")
lines.append("Top channels by attributed ROI (baseline):")
top5 = summary.sort_values("attributed_roi", ascending=False).head(5)
for _, r in top5.iterrows():
    lines.append(f"- {r['channel']}: attributed ROI = {r['attributed_roi']:.2f}, attributed revenue = {r['attributed_revenue']:.2f}")

lines.append("")
lines.append("Recommendation rationale:")
lines.append("- Increase budget toward channels with above-average attributed ROI, but limit any single-channel change to +/-30% of its current spend in this first reallocation step.")
lines.append("- Run 4-week A/B tests after reallocating to validate impact on conversions and ROI.")
with open(VIS_DIR / "recommendations.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("Wrote human-readable recommendations to", VIS_DIR / "recommendations.txt")

# Visualization: bar chart of attributed ROI by channel
plt.figure(figsize=(8,5))
plot_df = summary.sort_values("attributed_roi", ascending=False).head(12)
sns.barplot(x="attributed_roi", y="channel", data=plot_df)
plt.title("Attributed ROI by Channel (fractional by clicks)")
plt.xlabel("Attributed ROI")
plt.tight_layout()
plt.savefig(VIS_DIR / "attributed_roi_by_channel.png")
plt.close()
print("Saved:", VIS_DIR / "attributed_roi_by_channel.png")

# If time-decay ran, visualize its top channels
if time_decay_result is not None:
    plt.figure(figsize=(8,5))
    td = time_decay_result.sort_values("attributed_roi", ascending=False).head(12)
    sns.barplot(x="attributed_roi", y=channel_col, data=td)
    plt.title("Time-decay Attributed ROI by Channel")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "time_decay_attributed_roi.png")
    plt.close()
    print("Saved:", VIS_DIR / "time_decay_attributed_roi.png")

print("Done. All outputs in:", VIS_DIR.resolve())
