from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import re, calendar

# ───────────────────────────
# CONFIG: Google Sheets (Secondary GSOM + Primary Auction)
# ───────────────────────────
SHEET_TBILL_ID = "1rD831MnVWUGlitw1jUmdwt5jQPrkKMwrQTWBy6P9tAs"
SHEET_TBILL_GID = "1446111990"
SHEET_TBOND_ID = "1ma25T-_yMlzdrzOYxAr2P6eu1gsbjPzq3jxF4PK-xtk"
SHEET_TBOND_GID = "632609507"

# Primary Auction → Filtered_Data tab (gid=0)
SHEET_PRIMARY_ID = "1O5seVugWVYfCo7M7Zkn4VW6GltC77G1w0EsmhZEwNkk"
SHEET_PRIMARY_GID = "0"

def csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

URL_TBILL = csv_url(SHEET_TBILL_ID, SHEET_TBILL_GID)
URL_TBOND = csv_url(SHEET_TBOND_ID, SHEET_TBOND_GID)
URL_PRIMARY = csv_url(SHEET_PRIMARY_ID, SHEET_PRIMARY_GID)

MIN_CAL_DATE = date(2000, 1, 1)
MAX_CAL_DATE = date.today()

# ───────────────────────────
# LOADERS (cached)
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def load_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ───────────────────────────
# SECONDARY (GSOM)
# ───────────────────────────
def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    COLMAP = {
        "Date": ["Date"],
        "ISIN": ["ISIN"],
        "InstrumentText": ["Securities", "Securities "],
        "RemainingMaturity": ["RemainingMaturity", "Remaining Maturity"],
        "MarketYield": ["MarketYield", "Market Yield"],
        "MarketPrice": ["MarketPrice", "Market Price"],
        "Outstanding": ["Outstanding"],
    }
    out = pd.DataFrame()
    for new, candidates in COLMAP.items():
        col = next((c for c in candidates if c in df.columns), None)
        out[new] = df[col] if col else np.nan

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["RemainingMaturity", "MarketYield", "MarketPrice", "Outstanding"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["InstrumentText"] = out["InstrumentText"].astype(str).str.strip()
    return out

def add_helpers(df: pd.DataFrame, inst_type: str) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = inst_type  # Bill / Bond
    # Bills: RemainingMaturity in days → years; Bonds already years
    df["MaturityYears"] = np.where(
        df["Type"].str.lower().eq("bill"),
        df["RemainingMaturity"] / 365.0,
        df["RemainingMaturity"],
    )
    df = df[(df["MaturityYears"] > 0) & (df["MarketYield"].notna())]
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Year"] = df["Date"].dt.year
    df["MonthNum"] = df["Date"].dt.month
    return df

@st.cache_data(ttl=60 * 30)
def get_secondary() -> pd.DataFrame:
    tbill = add_helpers(coerce_cols(load_csv(URL_TBILL)), "Bill")
    tbond = add_helpers(coerce_cols(load_csv(URL_TBOND)), "Bond")
    combined = pd.concat([tbill, tbond], ignore_index=True)
    subset_cols = ["Date", "ISIN"] if "ISIN" in combined.columns else ["Date", "InstrumentText"]
    combined = combined.sort_values(["Date"] + subset_cols[1:]).drop_duplicates(subset_cols, keep="last")
    return combined

# ───────────────────────────
# PRIMARY (Auction) — reading Filtered_Data (3yr FRT already removed)
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def get_primary() -> pd.DataFrame:
    df = load_csv(URL_PRIMARY)
    if df.empty:
        return df

    # Flexible column detection
    c_date = next((c for c in ["Issue Date","IssueDate","Date"] if c in df.columns), None)
    c_instr = next((c for c in ["Instrument","Security","Securities"] if c in df.columns), None)
    c_yld = next((c for c in ["Cut-off Yield (%)","Cutoff Yield (%)","Cut Off Yield (%)","Cut-off Yield","Cutoff Yield"] if c in df.columns), None)
    if not all([c_date, c_instr, c_yld]):
        return pd.DataFrame()

    df = df.rename(columns={c_date:"IssueDate", c_instr:"Instrument", c_yld:"CutoffYield"})
    df["IssueDate"] = pd.to_datetime(df["IssueDate"], errors="coerce", dayfirst=True)
    df["CutoffYield"] = pd.to_numeric(df["CutoffYield"], errors="coerce")

    # derive tenor in years from the Instrument text
    def parse_tenor(txt):
        if not isinstance(txt, str):
            return np.nan
        s = txt.lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*(day|days|month|months|year|years|yr|yrs|y)\b', s)
        if not m:
            return np.nan
        val, unit = float(m.group(1)), m.group(2)
        if "day" in unit:
            return val / 365.0
        if "month" in unit:
            return (val * 30.0) / 365.0
        return val  # years

    df["TenorYears"] = df["Instrument"].apply(parse_tenor)
    df["Month"] = df["IssueDate"].dt.to_period("M").astype(str)
    df["Year"] = df["IssueDate"].dt.year
    df["MonthNum"] = df["IssueDate"].dt.month
    df = df.dropna(subset=["TenorYears","CutoffYield","IssueDate"])
    return df

# ───────────────────────────
# PLOT HELPERS
# ───────────────────────────
def plot_secondary_single(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for typ, g in df.groupby("Type"):
        g = g.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=g["MaturityYears"], y=g["MarketYield"],
                mode="lines+markers", name=typ,
                line=dict(shape="spline"),
                hovertemplate=("Type: %{text}<br>Maturity: %{x:.3f} yrs<br>"
                               "Yield: %{y:.3f}%<extra></extra>"),
                text=g["Type"],
            )
        )
    fig.update_yaxes(range=[0, 16], title="Yield (%)")
    fig.update_xaxes(title="Remaining Maturity (years)")
    fig.update_layout(title=title, height=520)
    return fig

def plot_secondary_compare(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for label, g in df.groupby("Label"):
        g = g.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=g["MaturityYears"], y=g["MarketYield"],
                mode="lines+markers", name=str(label),
                line=dict(shape="spline"),
                connectgaps=True,
                marker=dict(size=5),
                text=g["Type"],
                hovertemplate=("Date: " + str(label) +
                               "<br>Type: %{text}"
                               "<br>Maturity: %{x:.3f} yrs"
                               "<br>Yield: %{y:.3f}%<extra></extra>"),
            )
        )
    fig.update_yaxes(range=[0, 16], title="Yield (%)")
    fig.update_xaxes(title="Remaining Maturity (years)")
    fig.update_layout(title=title, height=560)
    return fig

def plot_primary(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for label, g in df.groupby("Label"):
        g = g.sort_values("TenorYears")
        fig.add_trace(
            go.Scatter(
                x=g["TenorYears"], y=g["CutoffYield"],
                mode="lines+markers", name=label,
                line=dict(shape="spline"),
                text=g["Instrument"],
                hovertemplate=("Instrument: %{text}<br>"
                               "Tenor: %{x:.2f} yrs<br>"
                               "Yield: %{y:.2f}%<extra></extra>")
            )
        )
    fig.update_yaxes(range=[0, 16], title="Cut-off Yield (%)")
    fig.update_xaxes(title="Tenor (years)")
    fig.update_layout(title=title, height=520)
    return fig

# ───────────────────────────
# TABLE HELPER
# ───────────────────────────
def show_sheet_table(df: pd.DataFrame, columns_wanted: list[str]):
    cols_available = [c for c in columns_wanted if c in df.columns]
    missing = [c for c in columns_wanted if c not in df.columns]
    st.dataframe(df[cols_available], use_container_width=True)
    if missing:
        st.caption("Note: missing column(s) in sheet → " + ", ".join(missing))

# ───────────────────────────
# APP UI
# ───────────────────────────
st.set_page_config(page_title="UCB AML – Yield Curve Dashboard", layout="wide")
st.title("Bangladesh Govt Securities – Yield Curve Dashboard")

view_mode = st.sidebar.radio(
    "Select View Mode:",
    [
        "Secondary: Single Date",
        "Secondary: Compare Two Dates",
        "Secondary: Compare Two Months",
        "Primary Auction: Compare Two Months",
    ],
    index=0,
)

# ─────────────── SECONDARY VIEWS ───────────────
if view_mode.startswith("Secondary"):
    sec = get_secondary()
    if sec.empty:
        st.error("No secondary data found.")
        st.stop()

    all_dates = sorted(sec["Date"].dropna().unique())

    def nearest(target):
        if not len(all_dates):
            return None
        dist = pd.Series([abs((pd.Timestamp(d) - target).days) for d in all_dates], index=all_dates)
        return dist.idxmin()

    def curve(d):
        return sec[sec["Date"] == d].copy()

    # Single Date
    if "Single" in view_mode:
        d = st.date_input("Pick Date", value=MAX_CAL_DATE)
        n = nearest(pd.Timestamp(d))
        df_used = curve(n).sort_values("MaturityYears")
        st.plotly_chart(plot_secondary_single(df_used, f"Secondary Yield Curve — {n.date()}"), use_container_width=True)
        st.subheader("Sheet table view (rows used)")
        show_sheet_table(
            df_used[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            ["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"],
        )

    # Compare Two Dates
    elif "Two Dates" in view_mode:
        c1, c2 = st.columns(2)
        with c1: d1 = st.date_input("Date A", value=MAX_CAL_DATE)
        with c2: d2 = st.date_input("Date B", value=MAX_CAL_DATE)
        n1, n2 = nearest(pd.Timestamp(d1)), nearest(pd.Timestamp(d2))
        df1 = curve(n1).assign(Label=str(n1.date()))
        df2 = curve(n2).assign(Label=str(n2.date()))
        merged = pd.concat([df1, df2], ignore_index=True).sort_values(["Label","MaturityYears"])
        st.plotly_chart(plot_secondary_compare(merged, f"Secondary Comparison — {n1.date()} vs {n2.date()}"), use_container_width=True)
        st.subheader("Sheet table view (rows used)")
        show_sheet_table(
            merged[["Label","Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            ["Label","Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"],
        )

    # Compare Two Months (Year → Month pickers)
    else:
        years = sorted(sec["Year"].dropna().unique().tolist())
        months_all = sorted(sec["Month"].unique())
        latest_m, prev_m = months_all[-1], (months_all[-2] if len(months_all) > 1 else months_all[-1])

        def split_month(mstr): y, m = mstr.split("-"); return int(y), int(m)
        def month_label(y, m): return f"{y}-{calendar.month_name[m]}"

        def_yearA, def_monA = split_month(prev_m)
        def_yearB, def_monB = split_month(latest_m)

        c1, c2 = st.columns(2)
        with c1:
            yearA = st.selectbox("Year A", years, index=years.index(def_yearA))
            monA_opts = sorted(sec.loc[sec["Year"] == yearA, "MonthNum"].unique())
            monA = st.selectbox("Month A", monA_opts, index=monA_opts.index(def_monA), format_func=lambda m: calendar.month_name[m])
        with c2:
            yearB = st.selectbox("Year B", years, index=years.index(def_yearB))
            monB_opts = sorted(sec.loc[sec["Year"] == yearB, "MonthNum"].unique())
            monB = st.selectbox("Month B", monB_opts, index=monB_opts.index(def_monB), format_func=lambda m: calendar.month_name[m])

        m1, m2 = f"{yearA}-{monA:02d}", f"{yearB}-{monB:02d}"

        def latest_in_month(m):
            s = sec.loc[sec["Month"] == m, "Date"]
            return s.max() if not s.empty else None

        d1, d2 = latest_in_month(m1), latest_in_month(m2)
        if not d1 or not d2:
            st.warning("No data found for one (or both) months.")
            st.stop()

        df1 = curve(d1).assign(Label=month_label(yearA, monA))
        df2 = curve(d2).assign(Label=month_label(yearB, monB))
        merged = pd.concat([df1, df2], ignore_index=True).sort_values(["Label","MaturityYears"])

        st.plotly_chart(
            plot_secondary_compare(merged, f"Secondary Comparison — {month_label(yearA,monA)} vs {month_label(yearB,monB)}"),
            use_container_width=True
        )
        st.subheader("Sheet table view (rows used)")
        show_sheet_table(
            merged[["Label","Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            ["Label","Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"],
        )

# ─────────────── PRIMARY VIEW ───────────────
else:
    pri = get_primary()
    if pri.empty:
        st.error("Primary auction data unavailable or empty.")
        st.stop()

    years = sorted(pri["Year"].dropna().unique().tolist())
    months_all = sorted(pri["Month"].unique())
    if not years or not months_all:
        st.warning("No valid months in primary auction data.")
        st.stop()

    latest_m = months_all[-1]
    prev_m = months_all[-2] if len(months_all) > 1 else months_all[-1]

    def split_month(mstr): y, m = mstr.split("-"); return int(y), int(m)
    def month_label(y, m): return f"{y}-{calendar.month_name[m]}"

    def_yearA, def_monA = split_month(prev_m)
    def_yearB, def_monB = split_month(latest_m)

    c1, c2 = st.columns(2)
    with c1:
        yearA = st.selectbox("Year A", years, index=years.index(def_yearA) if def_yearA in years else len(years)-1)
        monA_opts = sorted(pri.loc[pri["Year"] == yearA, "MonthNum"].unique().tolist())
        monA = st.selectbox("Month A", monA_opts, index=monA_opts.index(def_monA) if def_monA in monA_opts else len(monA_opts)-1,
                            format_func=lambda m: calendar.month_name[m])
    with c2:
        yearB = st.selectbox("Year B", years, index=years.index(def_yearB) if def_yearB in years else len(years)-1)
        monB_opts = sorted(pri.loc[pri["Year"] == yearB, "MonthNum"].unique().tolist())
        monB = st.selectbox("Month B", monB_opts, index=monB_opts.index(def_monB) if def_monB in monB_opts else len(monB_opts)-1,
                            format_func=lambda m: calendar.month_name[m])

    m1, m2 = f"{yearA}-{monA:02d}", f"{yearB}-{monB:02d}"

    def month_df(m: str) -> pd.DataFrame:
        sub = pri[pri["Month"] == m].copy()
        if sub.empty:
            return sub
        # keep only plottable rows
        sub = sub[sub["TenorYears"].notna() & sub["CutoffYield"].notna()]
        if sub.empty:
            return sub
        # latest auction per tenor within the month
        idx = sub.sort_values("IssueDate").groupby("TenorYears")["IssueDate"].idxmax()
        return sub.loc[idx].sort_values("TenorYears")

    dfA, dfB = month_df(m1), month_df(m2)
    if dfA.empty or dfB.empty:
        st.warning("No plottable primary-auction data for one (or both) selected months.")
        st.stop()

    dfA["Label"], dfB["Label"] = month_label(yearA, monA), month_label(yearB, monB)
    merged = pd.concat([dfA, dfB], ignore_index=True)

    st.plotly_chart(
        plot_primary(merged, f"Primary Auction Comparison — {month_label(yearA, monthA)} vs {month_label(yearB, monthB)}"),
        use_container_width=True,
    )
    st.dataframe(
        merged[["IssueDate","Instrument","TenorYears","CutoffYield","Label"]]
        .rename(columns={"IssueDate":"Issue Date","TenorYears":"Tenor (years)","CutoffYield":"Cut-off Yield (%)","Label":"Month"})
        .sort_values(["Month","Tenor (years)"]),
        use_container_width=True,
    )

st.markdown("---")
st.caption("Secondary = GSOM market yields. Primary = auction cut-off yields (reading Filtered_Data). Year → Month pickers for both primary and secondary monthly comparisons; Y-axis fixed at 0–16%.")
