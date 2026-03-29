import  streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer - By Srikrushna",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #090e1a; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] label {
    color: #64ffda !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1424 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #64ffda !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stButton > button {
    background: linear-gradient(135deg, #0a3d62, #1a6b9a);
    color: #64ffda;
    border: 1px solid #64ffda44;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    transition: all 0.25s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a6b9a, #2196f3);
    border-color: #64ffda;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(100,255,218,0.15);
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #64ffda;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e3a5f;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #64ffda, #2196f3, #64ffda);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
@keyframes shimmer {
    0%   { background-position: 0% }
    100% { background-position: 200% }
}

.sub-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

.weight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    margin: 0.3rem 0;
    background: #0d1424;
    border: 1px solid #1e2d4a;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1424;
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #64748b;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #64ffda !important;
    border-radius: 6px;
}

hr { border-color: #1e2d4a; }

.stAlert {
    background: #0d1424 !important;
    border-left-color: #64ffda !important;
    color: #94a3b8 !important;
}

/* Native chart area dark fix */
[data-testid="stArrowVegaLiteChart"] canvas,
[data-testid="stArrowVegaLiteChart"] { background: transparent !important; }

.chart-container {
    background: #0d1424;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1rem 1.2rem 0.5rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* Heatmap cell */
.hm-cell {
    display: inline-block;
    text-align: center;
    vertical-align: middle;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    border-radius: 4px;
    padding: 2px 0;
}

/* Bar chart custom */
.bar-wrap { margin: 2px 0; display: flex; align-items: center; gap: 8px; }
.bar-label { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #94a3b8; width: 120px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; background: #1e2d4a; border-radius: 4px; height: 18px; position: relative; }
.bar-fill  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #64ffda, #2196f3); }
.bar-val   { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #64ffda; width: 52px; }
</style>
""", unsafe_allow_html=True)


# ── Stock universe ───────────────────────────────────────────────────────────
NIFTY_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS":                 "TCS.NS",
    "Infosys":             "INFY.NS",
    "HDFC Bank":           "HDFCBANK.NS",
    "ICICI Bank":          "ICICIBANK.NS",
    "Wipro":               "WIPRO.NS",
    "HCL Technologies":    "HCLTECH.NS",
    "Axis Bank":           "AXISBANK.NS",
    "Bajaj Finance":       "BAJFINANCE.NS",
    "Kotak Mahindra":      "KOTAKBANK.NS",
    "SBI":                 "SBIN.NS",
    "Maruti Suzuki":       "MARUTI.NS",
    "Sun Pharma":          "SUNPHARMA.NS",
    "Asian Paints":        "ASIANPAINT.NS",
    "Titan":               "TITAN.NS",
}


# ── Core functions ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    raw.dropna(how="all", inplace=True)
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_stats(weights, mean_ret, cov, rf, trading_days=252):
    w   = np.array(weights)
    ret = float(np.dot(w, mean_ret) * trading_days)
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov * trading_days, w))))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def run_optimization(mean_ret, cov, rf, n_stocks, method="max_sharpe"):
    w0     = np.ones(n_stocks) / n_stocks
    bounds = tuple((0.0, 1.0) for _ in range(n_stocks))
    cons   = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, mean_ret, cov, rf)
        return -s

    def min_vol(w):
        _, v, _ = portfolio_stats(w, mean_ret, cov, rf)
        return v

    def max_ret(w):
        r, _, _ = portfolio_stats(w, mean_ret, cov, rf)
        return -r

    obj = {"max_sharpe": neg_sharpe, "min_vol": min_vol, "max_return": max_ret}[method]
    res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    return res.x if res.success else w0


def monte_carlo_frontier(mean_ret, cov, rf, n=4000, trading_days=252):
    n_assets = len(mean_ret)
    rets, vols, sharpes = [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = portfolio_stats(w, mean_ret, cov, rf, trading_days)
        rets.append(r * 100)
        vols.append(v * 100)
        sharpes.append(s)
    return np.array(rets), np.array(vols), np.array(sharpes)


def compute_var_cvar(returns_series, confidence=0.95):
    mu  = returns_series.mean()
    std = returns_series.std()
    var  = norm.ppf(1 - confidence, mu, std) * np.sqrt(252)
    cvar = (mu - std * norm.pdf(norm.ppf(confidence)) / (1 - confidence)) * np.sqrt(252)
    return var, cvar


# ── HTML chart helpers ───────────────────────────────────────────────────────

def render_bar_chart(labels, values, title, color_fn=None, suffix=""):
    """Horizontal bar chart rendered as pure HTML."""
    max_v = max(abs(v) for v in values) or 1
    bars_html = ""
    for lbl, val in zip(labels, values):
        pct    = abs(val) / max_v * 100
        color  = color_fn(val) if color_fn else "linear-gradient(90deg,#64ffda,#2196f3)"
        bars_html += f"""
        <div class="bar-wrap">
          <div class="bar-label" title="{lbl}">{lbl}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
          </div>
          <div class="bar-val">{val:.2f}{suffix}</div>
        </div>"""
    st.markdown(f"""
    <div class="chart-container">
      <div class="chart-title">{title}</div>
      {bars_html}
    </div>""", unsafe_allow_html=True)


def gradient_color(val, vmin, vmax):
    """Map value → red-white-blue hex (for correlation heatmap)."""
    t   = (val - vmin) / (vmax - vmin + 1e-9)     # 0 → 1
    t   = max(0.0, min(1.0, t))
    if t < 0.5:
        r = int(200 + (255 - 200) * (1 - t * 2))
        g = int(t * 2 * 100)
        b = int(t * 2 * 200)
    else:
        r = int((1 - t) * 2 * 200)
        g = int((1 - t) * 2 * 100)
        b = int(200 + (255 - 200) * ((t - 0.5) * 2))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_heatmap(corr_df, title="Correlation Matrix"):
    """Render correlation matrix as an HTML table."""
    labels   = corr_df.columns.tolist()
    n        = len(labels)
    cell_w   = max(60, min(90, 700 // n))

    header = "".join(
        f'<th style="font-family:DM Mono,monospace;font-size:0.65rem;'
        f'color:#64ffda;padding:4px;text-align:center;width:{cell_w}px">{lbl[:8]}</th>'
        for lbl in labels
    )
    rows_html = ""
    for i, row_lbl in enumerate(labels):
        cells = f'<td style="font-family:DM Mono,monospace;font-size:0.65rem;color:#94a3b8;padding:4px 6px;white-space:nowrap">{row_lbl}</td>'
        for j, col_lbl in enumerate(labels):
            val  = corr_df.iloc[i, j]
            bg   = gradient_color(val, -1, 1)
            txt  = "#090e1a" if abs(val) > 0.5 else "#e2e8f0"
            cells += (
                f'<td style="background:{bg};color:{txt};font-family:DM Mono,monospace;'
                f'font-size:0.7rem;text-align:center;padding:4px;border-radius:3px;'
                f'width:{cell_w}px">{val:.2f}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div class="chart-container">
      <div class="chart-title">{title}</div>
      <div style="overflow-x:auto">
        <table style="border-collapse:separate;border-spacing:3px">
          <thead><tr><th></th>{header}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>""", unsafe_allow_html=True)


def render_scatter_frontier(vols, rets, sharpes, opt_vol, opt_ret, eq_vol, eq_ret, title):
    """
    Render an efficient-frontier scatter using st.scatter_chart (via DataFrame).
    We pass a tidy DataFrame with 3 series: MC cloud, Optimal star, EW diamond.
    """
    # Build the DataFrame streamlit scatter_chart expects
    df_mc   = pd.DataFrame({"Volatility (%)": vols, "Return (%)": rets})
    df_mc["Series"] = "Simulated"

    df_opt  = pd.DataFrame({
        "Volatility (%)": [opt_vol * 100],
        "Return (%)":     [opt_ret * 100],
        "Series":         ["Optimal"],
    })
    df_eq   = pd.DataFrame({
        "Volatility (%)": [eq_vol * 100],
        "Return (%)":     [eq_ret * 100],
        "Series":         ["Equal Weight"],
    })
    df_all  = pd.concat([df_mc, df_opt, df_eq], ignore_index=True)

    st.markdown(f'<div class="chart-container"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.scatter_chart(
        df_all,
        x="Volatility (%)",
        y="Return (%)",
        color="Series",
        height=400,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_line_chart(df_lines: dict, title: str, y_label="Value", height=350):
    """Render a line chart from a dict of {name: pd.Series}."""
    df = pd.DataFrame(df_lines)
    st.markdown(f'<div class="chart-container"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.line_chart(df, height=height)
    st.markdown("</div>", unsafe_allow_html=True)


def render_donut_table(names, weights, title="Allocation"):
    """Render allocation as a styled HTML bar table (no Plotly)."""
    max_w = max(weights) or 1
    rows  = sorted(zip(names, weights), key=lambda x: -x[1])
    rows_html = ""
    for lbl, w in rows:
        pct   = w / max_w * 100
        rows_html += f"""
        <div class="weight-row">
          <span style="color:#e2e8f0;font-weight:500">{lbl}</span>
          <span style="color:#64ffda;font-weight:700">{w*100:.1f}%</span>
        </div>
        <div style="height:4px;background:#1e2d4a;border-radius:2px;margin:-4px 0 6px">
          <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#64ffda,#2196f3);border-radius:2px"></div>
        </div>"""
    st.markdown(f"""
    <div class="chart-container">
      <div class="chart-title">{title}</div>
      {rows_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="section-label">Portfolio Setup</p>', unsafe_allow_html=True)

    selected_names = st.multiselect(
        "Select Stocks",
        options=list(NIFTY_STOCKS.keys()),
        default=["Reliance Industries", "TCS", "Infosys", "HDFC Bank"],
        help="Choose 2–10 stocks",
    )

    st.markdown('<p class="section-label">Date Range</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start", value=pd.to_datetime("2020-01-01"))
    with c2:
        end_date = st.date_input("End",   value=pd.to_datetime("2024-01-01"))

    st.markdown('<p class="section-label">Parameters</p>', unsafe_allow_html=True)
    rf_rate    = st.slider("Risk-Free Rate (%)", 0.0, 12.0, 6.5, 0.25) / 100
    mc_sims    = st.slider("Monte Carlo Simulations", 1000, 8000, 4000, 500)
    confidence = st.slider("VaR Confidence Level (%)", 90, 99, 95) / 100

    st.markdown('<p class="section-label">Optimization Goal</p>', unsafe_allow_html=True)
    opt_goal = st.selectbox(
        "Objective",
        ["Max Sharpe Ratio", "Min Volatility", "Max Return"],
        index=0,
    )
    goal_map = {
        "Max Sharpe Ratio": "max_sharpe",
        "Min Volatility":   "min_vol",
        "Max Return":       "max_return",
    }

    run_btn = st.button("🚀 Optimize Portfolio", use_container_width=True)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Portfolio Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Scipy-Powered · Efficient Frontier · Indian Equities</div>', unsafe_allow_html=True)
st.markdown("---")

if len(selected_names) < 2:
    st.warning("Please select at least 2 stocks from the sidebar.")
    st.stop()

tickers = [NIFTY_STOCKS[n] for n in selected_names]

# auto-run on load
if run_btn or True:

    with st.spinner("Fetching market data…"):
        prices = fetch_data(tickers, str(start_date), str(end_date))

    if prices.empty or prices.shape[1] != len(tickers):
        st.error("Could not fetch data. Check ticker symbols or date range.")
        st.stop()

    prices.columns = selected_names
    returns    = compute_returns(prices)
    mean_ret   = returns.mean()
    cov_matrix = returns.cov()
    rf_daily   = rf_rate / 252

    # Scipy optimisation
    opt_weights = run_optimization(
        mean_ret, cov_matrix, rf_daily, len(tickers),
        method=goal_map[opt_goal]
    )
    opt_ret, opt_vol, opt_sharpe = portfolio_stats(
        opt_weights, mean_ret, cov_matrix, rf_daily
    )

    # Monte Carlo
    with st.spinner("Running Monte Carlo simulation…"):
        mc_rets, mc_vols, mc_sharpes = monte_carlo_frontier(
            mean_ret, cov_matrix, rf_daily, n=mc_sims
        )

    # Portfolio daily returns
    port_daily = returns.dot(opt_weights)
    ann_var, ann_cvar = compute_var_cvar(port_daily, confidence)

    # Equal-weight benchmark
    eq_w = np.ones(len(tickers)) / len(tickers)
    eq_ret, eq_vol, eq_sharpe = portfolio_stats(eq_w, mean_ret, cov_matrix, rf_daily)

    # ── KPI metrics ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Optimized Portfolio — Key Metrics</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Annual Return",         f"{opt_ret*100:.2f}%",  f"{(opt_ret-eq_ret)*100:+.2f}% vs EW")
    k2.metric("Annual Volatility",     f"{opt_vol*100:.2f}%",  f"{(opt_vol-eq_vol)*100:+.2f}% vs EW")
    k3.metric("Sharpe Ratio",          f"{opt_sharpe:.3f}",    f"{opt_sharpe-eq_sharpe:+.3f} vs EW")
    k4.metric(f"VaR ({int(confidence*100)}%)",  f"{ann_var*100:.2f}%",  "Annual")
    k5.metric(f"CVaR ({int(confidence*100)}%)", f"{ann_cvar*100:.2f}%", "Expected Shortfall")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Efficient Frontier",
        "🥧 Allocation",
        "📈 Performance",
        "🔥 Correlation",
    ])

    # ── TAB 1 : Efficient Frontier ───────────────────────────────────────────
    with tab1:
        render_scatter_frontier(
            mc_vols, mc_rets, mc_sharpes,
            opt_vol, opt_ret,
            eq_vol,  eq_ret,
            title="Efficient Frontier — Monte Carlo + Scipy Optimal"
        )

        # Legend note (no plotly hover → add a static info box)
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"""
            <div style="background:#0d1424;border:1px solid #1e2d4a;border-radius:8px;padding:0.75rem 1rem;font-family:'DM Mono',monospace;font-size:0.78rem">
              <span style="color:#64ffda">★ Optimal Portfolio</span><br>
              Return: <b>{opt_ret*100:.2f}%</b> &nbsp;|&nbsp;
              Volatility: <b>{opt_vol*100:.2f}%</b> &nbsp;|&nbsp;
              Sharpe: <b>{opt_sharpe:.3f}</b>
            </div>""", unsafe_allow_html=True)
        with col_r:
            st.markdown(f"""
            <div style="background:#0d1424;border:1px solid #1e2d4a;border-radius:8px;padding:0.75rem 1rem;font-family:'DM Mono',monospace;font-size:0.78rem">
              <span style="color:#f59e0b">◆ Equal-Weight Benchmark</span><br>
              Return: <b>{eq_ret*100:.2f}%</b> &nbsp;|&nbsp;
              Volatility: <b>{eq_vol*100:.2f}%</b> &nbsp;|&nbsp;
              Sharpe: <b>{eq_sharpe:.3f}</b>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Distribution of Sharpe ratios from MC
        sharpe_series = pd.Series(mc_sharpes, name="Sharpe Ratio")
        bins          = pd.cut(sharpe_series, bins=30)
        hist          = sharpe_series.groupby(bins, observed=True).count().rename("Count")
        hist.index    = [f"{b.mid:.2f}" for b in hist.index]
        st.markdown('<p class="section-label">Distribution of Sharpe Ratios (Monte Carlo)</p>', unsafe_allow_html=True)
        st.bar_chart(hist, height=220)

    # ── TAB 2 : Allocation ───────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            render_donut_table(selected_names, opt_weights, title="Optimal Weight Allocation")

        with col_b:
            st.markdown('<p class="section-label">Weight Breakdown Table</p>', unsafe_allow_html=True)
            weight_df = pd.DataFrame({
                "Stock":                  selected_names,
                "Weight (%)":             (opt_weights * 100).round(2),
                "Exp. Annual Return (%)": (mean_ret.values * 252 * 100).round(2),
                "Ann. Volatility (%)":    (returns.std().values * np.sqrt(252) * 100).round(2),
            }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

            st.dataframe(
                weight_df.style
                    .format({
                        "Weight (%)":             "{:.2f}",
                        "Exp. Annual Return (%)": "{:.2f}",
                        "Ann. Volatility (%)":    "{:.2f}",
                    })
                    .background_gradient(subset=["Weight (%)"], cmap="YlGnBu"),
                use_container_width=True,
                hide_index=True,
            )

            # Individual stock expected annual returns — horizontal bar
            ind_rets   = (mean_ret * 252 * 100).values
            def ret_color(v):
                return "linear-gradient(90deg,#64ffda,#2196f3)" if v >= 0 else "linear-gradient(90deg,#ef4444,#f59e0b)"
            render_bar_chart(
                selected_names, ind_rets,
                title="Expected Annual Return per Stock",
                color_fn=ret_color, suffix="%"
            )

    # ── TAB 3 : Performance ──────────────────────────────────────────────────
    with tab3:
        # Cumulative returns
        port_cumret  = ((1 + port_daily).cumprod() - 1) * 100
        eq_daily_r   = returns.dot(eq_w)
        eq_cumret    = ((1 + eq_daily_r).cumprod() - 1) * 100
        stock_cumret = ((1 + returns).cumprod() - 1) * 100

        lines = {"Optimal Portfolio": port_cumret, "Equal Weight": eq_cumret}
        for col in selected_names:
            lines[col] = stock_cumret[col]

        render_line_chart(
            lines,
            title="Cumulative Returns — Optimal vs Equal Weight vs Individual Stocks (%)",
            height=400
        )

        # Rolling 60-day Sharpe
        st.markdown('<p class="section-label">Rolling 60-Day Sharpe Ratio</p>', unsafe_allow_html=True)
        rolling_sharpe = (
            port_daily.rolling(60).mean() /
            port_daily.rolling(60).std()
        ) * np.sqrt(252)
        rolling_df = rolling_sharpe.dropna().to_frame("Rolling Sharpe (60d)")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.line_chart(rolling_df, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

        # Drawdown
        st.markdown('<p class="section-label">Portfolio Drawdown</p>', unsafe_allow_html=True)
        cumulative   = (1 + port_daily).cumprod()
        rolling_max  = cumulative.cummax()
        drawdown     = ((cumulative - rolling_max) / rolling_max) * 100
        dd_df        = drawdown.to_frame("Drawdown (%)")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.line_chart(dd_df, height=240)
        st.markdown("</div>", unsafe_allow_html=True)

        # Summary stats table
        st.markdown('<p class="section-label">Descriptive Statistics — Daily Returns</p>', unsafe_allow_html=True)
        stats_df = returns.describe().T.round(5)
        stats_df.insert(0, "Stock", stats_df.index)
        stats_df.reset_index(drop=True, inplace=True)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── TAB 4 : Correlation ──────────────────────────────────────────────────
    with tab4:
        corr = returns.corr().round(3)
        render_heatmap(corr, title="Stock Return Correlation Matrix")

        # Average correlation / diversification score
        n          = len(corr)
        avg_corr   = (corr.values.sum() - n) / (n * n - n)
        div_score  = (1 - avg_corr) * 100

        d1, d2, d3 = st.columns(3)
        d1.metric("Diversification Score", f"{div_score:.1f} / 100",
                  help="Higher = lower avg correlation = better diversification")
        d2.metric("Avg Pairwise Correlation", f"{avg_corr:.3f}")
        d3.metric("Number of Assets", str(n))

        # Correlation sorted list
        st.markdown('<p class="section-label">Pairwise Correlations (sorted)</p>', unsafe_allow_html=True)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "Pair":        f"{corr.index[i]} – {corr.columns[j]}",
                    "Correlation": corr.iloc[i, j],
                })
        pairs_df = pd.DataFrame(pairs).sort_values("Correlation").reset_index(drop=True)
        st.dataframe(
            pairs_df.style.background_gradient(subset=["Correlation"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.7rem;color:#334155;text-align:center">'
        'Built with Scipy · yFinance · Streamlit · Pandas · NumPy &nbsp;|&nbsp; '
        'For educational purposes only — not financial advice.'
        '</p>',
        unsafe_allow_html=True,
    )