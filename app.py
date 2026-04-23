import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="TX Roadhouse Expense Analyzer",
    page_icon="🤠",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FDFAF7; }
    .block-container { padding-top: 2rem; }
    h1 { color: #B85C20; }
    h2, h3 { color: #3A1A0A; }
    .stMetric { background: #F9F0EA; border-radius: 8px; padding: 0.5rem; }
    .flag-box { background: #FDECEA; border-left: 4px solid #E24B4A;
                padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────────
CATEGORY_RULES = {
    "Food & Beverage": [
        "sysco", "us foods", "produce", "meat", "dairy",
        "beverage", "food", "grocery", "fresh", "farm", "local",
    ],
    "Labor": [
        "adp", "payroll", "paychex", "labor", "staffing",
        "shift", "workforce", "hr", "human resources",
    ],
    "Utilities": [
        "energy", "electric", "gas", "water", "txu", "atmos",
        "utility", "power", "oncor",
    ],
    "Maintenance & Repairs": [
        "hobart", "repair", "service", "hvac", "plumbing",
        "ecolab", "maintenance", "equipment", "cleaning",
    ],
    "Supplies": [
        "staples", "unifirst", "cintas", "uniform", "linen",
        "supply", "supplies", "janitorial", "office",
    ],
    "Technology": [
        "square", "opentable", "pos", "software", "tech",
        "cloud", "subscription", "saas",
    ],
    "Waste & Recycling": [
        "republic", "waste", "recycl", "trash", "disposal",
    ],
}

REQUIRED_COLUMNS = {"store_id", "city", "date", "vendor", "amount"}
ROLLING_WINDOW   = 14
Z_SCORE_THRESH   = 2.5
BRAND_RED        = "#B85C20"
BRAND_DARK       = "#3A1A0A"
FLAG_RED         = "#E24B4A"
GRAY             = "#888780"


# ── Helper functions ─────────────────────────────────────────
def clean_transactions(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df.dropna(subset=["date", "amount", "store_id", "vendor"], inplace=True)
    df = df[df["amount"] > 0]
    for col in ["vendor", "store_id", "city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    df["week"]    = df["date"].dt.isocalendar().week.astype(int)
    df["month"]   = df["date"].dt.month
    df["weekday"] = df["date"].dt.day_name()
    df.reset_index(drop=True, inplace=True)
    return df


def categorize(vendor):
    v = str(vendor).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in v for kw in keywords):
            return category
    return "Uncategorized"


def detect_anomalies(df, window=ROLLING_WINDOW, threshold=Z_SCORE_THRESH):
    daily = (
        df.groupby(["store_id", "category", "date"])["amount"]
        .sum().reset_index()
        .rename(columns={"amount": "daily_total"})
        .sort_values(["store_id", "category", "date"])
    )
    results = []
    for (store, cat), grp in daily.groupby(["store_id", "category"]):
        grp = grp.copy().set_index("date").sort_index()
        roll = grp["daily_total"].rolling(window=window, min_periods=3)
        grp["rolling_mean"] = roll.mean()
        grp["rolling_std"]  = roll.std().fillna(0)
        grp["z_score"] = np.where(
            grp["rolling_std"] > 0,
            (grp["daily_total"] - grp["rolling_mean"]) / grp["rolling_std"],
            0.0,
        )
        grp["is_anomaly"] = grp["z_score"].abs() >= threshold
        grp["pct_vs_avg"] = np.where(
            grp["rolling_mean"] > 0,
            ((grp["daily_total"] - grp["rolling_mean"]) / grp["rolling_mean"] * 100).round(1),
            0.0,
        )
        grp["store_id"] = store
        grp["category"] = cat
        results.append(grp.reset_index())
    return pd.concat(results, ignore_index=True)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(report_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in report_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def build_charts(clean_df, anomaly_df, flagged):
    plt.rcParams.update({
        "font.family":      "serif",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Texas Roadhouse — Expense Dashboard",
        fontsize=14, fontweight="bold", color=BRAND_DARK, y=1.01,
    )

    # Plot 1: Spend by category
    ax1 = axes[0, 0]
    cat_totals = clean_df.groupby("category")["amount"].sum().sort_values()
    bar_colors = [BRAND_RED if c == cat_totals.idxmax() else GRAY for c in cat_totals.index]
    ax1.barh(cat_totals.index, cat_totals / 1_000, color=bar_colors, edgecolor="white")
    ax1.set_xlabel("Total Spend ($K)")
    ax1.set_title("Spend by Category", fontweight="bold")
    for patch, val in zip(ax1.patches, cat_totals):
        ax1.text(patch.get_width() + 0.3, patch.get_y() + patch.get_height() / 2,
                 f"${val/1_000:,.0f}K", va="center", fontsize=8, color=BRAND_DARK)

    # Plot 2: Monthly trend
    ax2 = axes[0, 1]
    top3 = cat_totals.nlargest(3).index.tolist()
    monthly_trend = (
        clean_df[clean_df["category"].isin(top3)]
        .groupby(["month", "category"])["amount"].sum().reset_index()
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    months = sorted(clean_df["month"].unique())
    for cat, color in zip(top3, [BRAND_RED, BRAND_DARK, GRAY]):
        sub = monthly_trend[monthly_trend["category"] == cat].sort_values("month")
        ax2.plot(sub["month"], sub["amount"] / 1_000,
                 marker="o", label=cat, color=color, linewidth=2, markersize=5)
    ax2.set_xticks(months)
    ax2.set_xticklabels([month_names[m] for m in months])
    ax2.set_ylabel("Spend ($K)")
    ax2.set_title("Monthly Trend — Top 3 Categories", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.5)

    # Plot 3: Spend by store
    ax3 = axes[1, 0]
    store_totals = clean_df.groupby(["store_id", "city"])["amount"].sum().reset_index()
    store_totals["label"] = store_totals["store_id"] + "\n" + store_totals["city"]
    bar_colors3 = [BRAND_RED if i == store_totals["amount"].idxmax() else GRAY
                   for i in store_totals.index]
    bars = ax3.bar(store_totals["label"], store_totals["amount"] / 1_000,
                   color=bar_colors3, edgecolor="white", width=0.55)
    ax3.set_ylabel("Total Spend ($K)")
    ax3.set_title("Total Spend by Store", fontweight="bold")
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${bar.get_height():,.0f}K", ha="center", fontsize=8, color=BRAND_DARK)

    # Plot 4: Anomaly timeline
    ax4 = axes[1, 1]
    if len(flagged):
        top_flag   = flagged.sort_values("z_score", ascending=False).iloc[0]
        f_store    = top_flag["store_id"]
        f_category = top_flag["category"]
        subset = anomaly_df[
            (anomaly_df["store_id"] == f_store) &
            (anomaly_df["category"] == f_category)
        ].copy()
        normal_pts  = subset[~subset["is_anomaly"]]
        flagged_pts = subset[subset["is_anomaly"]]
        ax4.plot(subset["date"], subset["rolling_mean"],
                 color=GRAY, linewidth=1.5, linestyle="--", label="14-Day Avg", alpha=0.9)
        ax4.fill_between(subset["date"],
                         subset["rolling_mean"] - 2 * subset["rolling_std"],
                         subset["rolling_mean"] + 2 * subset["rolling_std"],
                         alpha=0.1, color=BRAND_RED, label="±2σ Band")
        ax4.scatter(normal_pts["date"], normal_pts["daily_total"],
                    s=20, color=BRAND_DARK, alpha=0.6, zorder=3, label="Normal")
        ax4.scatter(flagged_pts["date"], flagged_pts["daily_total"],
                    s=80, color=FLAG_RED, zorder=5, marker="^", label="⚑ Flagged")
        ax4.set_title(f"Anomaly Timeline — {f_store} | {f_category}", fontweight="bold")
        ax4.set_ylabel("Daily Spend ($)")
        ax4.tick_params(axis="x", rotation=30)
        ax4.legend(fontsize=9, framealpha=0.5)
    else:
        ax4.text(0.5, 0.5, "No anomalies detected.",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=11, color=GRAY)
        ax4.set_title("Anomaly Timeline", fontweight="bold")

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════

st.title("🤠 Texas Roadhouse Expense Analyzer")
st.markdown("Upload any store expense CSV to get instant categorization, anomaly detection, and downloadable reports.")

# ── Sidebar settings ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    rolling_window  = st.slider("Rolling average window (days)", 7, 30, ROLLING_WINDOW)
    z_thresh        = st.slider("Anomaly sensitivity (z-score)", 1.5, 4.0, Z_SCORE_THRESH, step=0.1)
    st.markdown("---")
    st.markdown("**Required CSV columns:**")
    st.code("store_id\ncity\ndate\nvendor\namount")
    st.markdown("Optional: `transaction_id`, `invoice_number`, `notes`")
    st.markdown("---")
    st.download_button(
        label="⬇️ Download sample CSV",
        data=open("texas_roadhouse_mock_expenses.csv", "rb").read() if os.path.exists("texas_roadhouse_mock_expenses.csv") else b"",
        file_name="texas_roadhouse_mock_expenses.csv",
        mime="text/csv",
        disabled=not os.path.exists("texas_roadhouse_mock_expenses.csv"),
    )

# ── File upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your expense CSV",
    type=["csv"],
    help="CSV must include: store_id, city, date, vendor, amount",
)

if uploaded_file is None:
    st.info("👆 Upload a CSV file above to get started. Download the sample file from the sidebar to try it out.")
    st.stop()

# ── Load & validate ───────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

raw_df.columns = raw_df.columns.str.lower().str.strip()
missing = REQUIRED_COLUMNS - set(raw_df.columns)
if missing:
    st.error(f"❌ CSV is missing required columns: `{missing}`. Found: `{list(raw_df.columns)}`")
    st.stop()

# ── Process ───────────────────────────────────────────────────
with st.spinner("Cleaning data..."):
    clean_df = clean_transactions(raw_df)

with st.spinner("Categorizing transactions..."):
    clean_df["category"] = clean_df["vendor"].apply(categorize)

with st.spinner("Running anomaly detection..."):
    anomaly_df = detect_anomalies(clean_df, window=rolling_window, threshold=z_thresh)
    flagged    = anomaly_df[anomaly_df["is_anomaly"]].copy()

# ── Summary metrics ───────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Summary")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{len(clean_df):,}")
col2.metric("Total Spend",        f"${clean_df['amount'].sum():,.0f}")
col3.metric("Stores",             clean_df["store_id"].nunique())
col4.metric("Anomalies Flagged",  len(flagged))
col5.metric("Uncategorized",      (clean_df["category"] == "Uncategorized").sum())

# ── Charts ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Dashboard")
with st.spinner("Building charts..."):
    fig = build_charts(clean_df, anomaly_df, flagged)
    st.pyplot(fig)

# ── Anomaly detail ────────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Anomaly Flags")

if len(flagged):
    st.warning(f"{len(flagged)} anomalous spending days detected at z-score ≥ {z_thresh}. Adjust sensitivity in the sidebar.")
    display = flagged[["store_id","category","date","daily_total","rolling_mean","z_score","pct_vs_avg"]].copy()
    display.columns = ["Store","Category","Date","Daily Total ($)","14-Day Avg ($)","Z-Score","% vs Avg"]
    display = display.sort_values("Z-Score", ascending=False).round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.success(f"No anomalies detected at current threshold (z ≥ {z_thresh}). Try lowering sensitivity in the sidebar.")

# ── Category breakdown ────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Category Breakdown by Store")

cat_report = (
    clean_df.groupby(["store_id", "city", "category"])["amount"]
    .agg(total_spend="sum", transactions="count", avg_transaction="mean")
    .reset_index().round(2)
    .sort_values(["store_id", "total_spend"], ascending=[True, False])
)
st.dataframe(cat_report, use_container_width=True, hide_index=True)

# ── Downloads ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("⬇️ Download Reports")

dl1, dl2, dl3, dl4 = st.columns(4)

dl1.download_button(
    label="📄 Cleaned Transactions",
    data=to_csv_bytes(clean_df),
    file_name="cleaned_transactions.csv",
    mime="text/csv",
)
dl2.download_button(
    label="📂 Category Summary",
    data=to_csv_bytes(cat_report),
    file_name="category_summary.csv",
    mime="text/csv",
)
dl3.download_button(
    label="🚨 Anomaly Flags",
    data=to_csv_bytes(flagged[["store_id","category","date","daily_total","rolling_mean","z_score","pct_vs_avg"]].round(2)),
    file_name="anomaly_flags.csv",
    mime="text/csv",
)
if EXCEL_AVAILABLE:
    excel_data = to_excel_bytes({
        "Category Summary": cat_report,
        "Anomaly Flags":    flagged[["store_id","category","date","daily_total","rolling_mean","z_score","pct_vs_avg"]].round(2),
        "Cleaned Data":     clean_df,
    })
    dl4.download_button(
        label="📊 Full Excel Report",
        data=excel_data,
        file_name="texas_roadhouse_expense_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.caption("Texas Roadhouse Expense Automation MVP · Built with Streamlit")
