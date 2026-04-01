"""
Insurance Charges Predictor — All-in-One Streamlit App
=======================================================
Run: streamlit run insurance_app.py

Everything is in this single file:
  • Data loading / synthetic fallback
  • Feature engineering
  • Model training (6 models, auto-selects best)
  • EDA charts
  • Interactive prediction UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    border-radius: 18px; padding: 2rem 2.5rem; color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
.hero p  { font-size: 1rem; opacity: 0.85; margin: 0.3rem 0 0; }

.result-card {
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(21,101,192,0.3); margin-top: 1rem;
}
.result-card .amount  { font-size: 3rem; font-weight: 900; }
.result-card .monthly { font-size: 0.95rem; opacity: 0.75; margin-top: 0.3rem; }

.stButton > button {
    background: #1565C0 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-size: 1.05rem !important;
    font-weight: 700 !important; width: 100% !important;
}
.stButton > button:hover { background: #0D47A1 !important; }

.section-title {
    font-size: 1.2rem; font-weight: 700; color: #1565C0;
    border-bottom: 2px solid #E3EAFF; padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    if os.path.exists("insurance.csv"):
        df = pd.read_csv("insurance.csv")
        source = "Loaded from **insurance.csv**"
    else:
        np.random.seed(42)
        n        = 1338
        ages     = np.random.randint(18, 65, n)
        bmis     = np.round(np.random.normal(30.5, 6.1, n), 2)
        children = np.random.randint(0, 6, n)
        smoker   = np.random.choice(["yes","no"], n, p=[0.205, 0.795])
        sex      = np.random.choice(["male","female"], n)
        region   = np.random.choice(["southwest","southeast","northwest","northeast"], n)
        charges  = (ages * 256 + bmis * 340 + children * 475
                    + (smoker == "yes") * 23500
                    + np.random.normal(0, 2500, n))
        charges  = np.abs(charges) + 1200
        df = pd.DataFrame({
            "age": ages, "sex": sex, "bmi": bmis,
            "children": children, "smoker": smoker,
            "region": region, "charges": np.round(charges, 2)
        })
        source = "**insurance.csv not found** — using synthetic demo data. Place your CSV in the same folder and rerun."
    return df, source

df, data_source = load_data()

# ══════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING + TRAIN MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df):
    df_m = df.copy()

    le_sex    = LabelEncoder().fit(df_m["sex"])
    le_smoker = LabelEncoder().fit(df_m["smoker"])
    le_region = LabelEncoder().fit(df_m["region"])

    df_m["sex"]    = le_sex.transform(df_m["sex"])
    df_m["smoker"] = le_smoker.transform(df_m["smoker"])
    df_m["region"] = le_region.transform(df_m["region"])

    df_m["age_bmi"]      = df_m["age"]    * df_m["bmi"]
    df_m["smoker_bmi"]   = df_m["smoker"] * df_m["bmi"]
    df_m["smoker_age"]   = df_m["smoker"] * df_m["age"]
    df_m["age_squared"]  = df_m["age"]    ** 2
    df_m["bmi_gt30"]     = (df_m["bmi"] > 30).astype(int)
    df_m["smoker_obese"] = df_m["smoker"] * df_m["bmi_gt30"]

    feature_cols = ["age","sex","bmi","children","smoker","region",
                    "age_bmi","smoker_bmi","smoker_age",
                    "age_squared","bmi_gt30","smoker_obese"]

    X = df_m[feature_cols]
    y = df_m["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler         = StandardScaler()
    X_train_sc     = scaler.fit_transform(X_train)
    X_test_sc      = scaler.transform(X_test)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10),
        "Lasso Regression":  Lasso(alpha=10),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=4, random_state=42),
    }

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, mdl in candidates.items():
        linear = "Regression" in name
        Xtr = X_train_sc if linear else X_train
        Xte = X_test_sc  if linear else X_test
        mdl.fit(Xtr, y_train)
        y_pred = mdl.predict(Xte)
        cv     = cross_val_score(mdl, Xtr, y_train, cv=kf, scoring="r2")
        results[name] = {
            "model":  mdl,
            "r2":     r2_score(y_test, y_pred),
            "mae":    mean_absolute_error(y_test, y_pred),
            "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
            "cv_r2":  cv.mean(),
            "y_pred": y_pred,
            "linear": linear,
        }

    best_name = max(results, key=lambda k: results[k]["r2"])
    encoders  = {
        "le_sex": le_sex, "le_smoker": le_smoker, "le_region": le_region,
        "feature_cols": feature_cols, "best_model_name": best_name,
    }
    return results, best_name, scaler, encoders, X_test, y_test, feature_cols

results, best_name, scaler, encoders, X_test, y_test, feature_cols = train_models(df)
best = results[best_name]

# ══════════════════════════════════════════════════════════════
# STEP 3 — PREDICTION HELPER
# ══════════════════════════════════════════════════════════════
def predict_charges(age, sex, bmi, children, smoker, region):
    sex_enc    = encoders["le_sex"].transform([sex])[0]
    smoker_enc = encoders["le_smoker"].transform([smoker])[0]
    region_enc = encoders["le_region"].transform([region])[0]
    bmi_gt30   = int(bmi > 30)
    row        = [age, sex_enc, bmi, children, smoker_enc, region_enc,
                  age * bmi, smoker_enc * bmi, smoker_enc * age,
                  age**2, bmi_gt30, smoker_enc * bmi_gt30]
    X = pd.DataFrame([row], columns=feature_cols)
    if best["linear"]:
        return max(best["model"].predict(scaler.transform(X))[0], 0)
    return max(best["model"].predict(X)[0], 0)

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🏥 Insurance Charges Predictor</h1>
  <p>Machine-learning powered cost estimation · trains automatically on every run</p>
</div>
""", unsafe_allow_html=True)

st.caption(data_source)

tab_predict, tab_eda, tab_models = st.tabs(["🔮 Predict", "📊 Data Analysis", "🤖 Model Performance"])

# ─────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────
with tab_predict:
    col_form, col_out = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Patient Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age      = st.slider("Age", 18, 65, 35)
            bmi      = st.slider("BMI", 10.0, 55.0, 28.5, step=0.1)
            children = st.selectbox("Children", [0,1,2,3,4,5])
        with c2:
            sex    = st.radio("Sex",    ["male","female"])
            smoker = st.radio("Smoker", ["no","yes"])
            region = st.selectbox("Region", ["southwest","southeast","northwest","northeast"])

        bmi_cat   = ("Underweight" if bmi < 18.5 else
                     "Normal"      if bmi < 25   else
                     "Overweight"  if bmi < 30   else "Obese")
        bmi_color = "#2E7D32" if bmi < 25 else "#EF6C00" if bmi < 30 else "#C62828"
        st.markdown(f"BMI category: <b style='color:{bmi_color}'>{bmi_cat}</b>", unsafe_allow_html=True)
        st.markdown("")
        clicked = st.button("🔮 Predict Charges")

    with col_out:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        if clicked:
            pred = predict_charges(age, sex, bmi, children, smoker, region)
            risk, r_color = (("🔴 High Risk",  "#C62828") if pred > 20000 else
                             ("🟡 Medium Risk", "#EF6C00") if pred >  8000 else
                             ("🟢 Low Risk",    "#2E7D32"))
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.9rem;opacity:0.8;margin-bottom:0.3rem">Estimated Annual Premium</div>
                <div class="amount">${pred:,.0f}</div>
                <div class="monthly">≈ ${pred/12:,.0f} / month</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Risk Level:** <span style='color:{r_color};font-weight:700'>{risk}</span>",
                        unsafe_allow_html=True)
            st.markdown("**Key risk factors:**")
            if smoker == "yes":  st.markdown("- 🚬 **Smoking** — biggest single driver")
            if bmi >= 30:        st.markdown("- ⚖️ **Obesity (BMI ≥ 30)** — elevated risk")
            if age >= 50:        st.markdown("- 🎂 **Age 50+** — costs rise with age")
            if children >= 3:   st.markdown("- 👨‍👩‍👦 **3+ dependents** — slight increase")
            if smoker == "no" and bmi < 30 and age < 50:
                st.markdown("- ✅ **No major risk factors detected**")
            st.info("💡 Toggle **Smoker** to see the biggest impact on cost!")
        else:
            st.markdown("""
            <div style='border:2px dashed #CBD5E1;border-radius:14px;padding:3rem;
                        text-align:center;color:#94A3B8;margin-top:0.5rem'>
                <div style='font-size:3rem'>🔮</div>
                <div style='margin-top:0.8rem'>Fill in details and click<br><b>Predict Charges</b></div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────
with tab_eda:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",         f"{len(df):,}")
    c2.metric("Features",     "6")
    c3.metric("Avg Charges",  f"${df['charges'].mean():,.0f}")
    c4.metric("Max Charges",  f"${df['charges'].max():,.0f}")

    with st.expander("📋 Raw Data Sample"):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Insurance Dataset — EDA", fontsize=16, fontweight="bold", y=0.99)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    plt.rcParams.update({"font.size": 9})

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["charges"], bins=40, color="#1565C0", edgecolor="white", alpha=0.85)
    ax.set_title("Charges Distribution", fontweight="bold")
    ax.set_xlabel("Charges ($)"); ax.set_ylabel("Count")

    ax = fig.add_subplot(gs[0, 1])
    for i, (grp, d) in enumerate(df.groupby("smoker")["charges"]):
        ax.boxplot(d, positions=[i], widths=0.5,
                   boxprops=dict(color="#1565C0"),
                   medianprops=dict(color="red", linewidth=2),
                   whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
    ax.set_xticks([0,1]); ax.set_xticklabels(["No","Yes"])
    ax.set_title("Charges by Smoker", fontweight="bold"); ax.set_ylabel("Charges ($)")

    ax = fig.add_subplot(gs[0, 2])
    colors = df["smoker"].map({"yes":"#E53935","no":"#43A047"})
    ax.scatter(df["age"], df["charges"], c=colors, alpha=0.45, s=12)
    ax.set_title("Age vs Charges (red=smoker)", fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Charges ($)")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(df["bmi"], df["charges"], c="#7B1FA2", alpha=0.35, s=12)
    ax.set_title("BMI vs Charges", fontweight="bold")
    ax.set_xlabel("BMI"); ax.set_ylabel("Charges ($)")

    ax = fig.add_subplot(gs[1, 1])
    cm = df.groupby("children")["charges"].mean()
    ax.bar(cm.index, cm.values, color="#00897B", edgecolor="white")
    ax.set_title("Avg Charges by Children", fontweight="bold")
    ax.set_xlabel("Children"); ax.set_ylabel("Avg Charges ($)")

    ax = fig.add_subplot(gs[1, 2])
    sorted_regions = sorted(df["region"].unique())
    for i, reg in enumerate(sorted_regions):
        d = df[df["region"] == reg]["charges"]
        ax.boxplot(d, positions=[i], widths=0.5,
                   boxprops=dict(color="#F57C00"),
                   medianprops=dict(color="red", linewidth=2),
                   whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
    ax.set_xticks(range(len(sorted_regions)))
    ax.set_xticklabels(sorted_regions, rotation=25, ha="right")
    ax.set_title("Charges by Region", fontweight="bold"); ax.set_ylabel("Charges ($)")

    ax = fig.add_subplot(gs[2, :2])
    df_enc = df.copy()
    for col in ["sex","smoker","region"]:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
    sns.heatmap(df_enc.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.4, square=True, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap", fontweight="bold")

    ax = fig.add_subplot(gs[2, 2])
    for i, (grp, d) in enumerate(df.groupby("sex")["charges"]):
        ax.boxplot(d, positions=[i], widths=0.5,
                   boxprops=dict(color="#C62828"),
                   medianprops=dict(color="navy", linewidth=2),
                   whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
    ax.set_xticks([0,1]); ax.set_xticklabels(["Female","Male"])
    ax.set_title("Charges by Sex", fontweight="bold"); ax.set_ylabel("Charges ($)")

    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────
# TAB 3 — MODELS
# ─────────────────────────────
with tab_models:
    st.markdown('<div class="section-title">All Models Compared</div>', unsafe_allow_html=True)

    rows = []
    for name, r in results.items():
        rows.append({
            "Model":    ("🏆 " if name == best_name else "   ") + name,
            "R²":       round(r["r2"],    4),
            "MAE ($)":  round(r["mae"],   2),
            "RMSE ($)": round(r["rmse"],  2),
            "CV R²":    round(r["cv_r2"], 4),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.success(f"🏆 **Best:** {best_name}  |  R² = {best['r2']:.4f}  |  MAE = ${best['mae']:,.2f}  |  RMSE = ${best['rmse']:,.2f}")

    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
    names   = list(results.keys())
    palette = ["#1565C0" if n == best_name else "#90CAF9" for n in names]

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    for ax, key, title, fmt in zip(
            axes,
            ["r2", "mae", "rmse"],
            ["R² (higher = better)", "MAE ($) lower = better", "RMSE ($) lower = better"],
            [".4f", ",.0f", ",.0f"]):
        vals = [results[n][key] for n in names]
        bars = ax.bar(names, vals, color=palette, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    format(v, fmt), ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.markdown(f'<div class="section-title">Best Model — {best_name}</div>', unsafe_allow_html=True)
    y_pred_best = best["y_pred"]
    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle(f"{best_name} — Diagnostics", fontsize=12, fontweight="bold")

    ax_a.scatter(y_test, y_pred_best, alpha=0.5, color="#1976D2", s=18)
    mn, mx = y_test.min(), y_test.max()
    ax_a.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
    ax_a.set_xlabel("Actual ($)"); ax_a.set_ylabel("Predicted ($)")
    ax_a.set_title("Actual vs Predicted"); ax_a.legend()

    residuals = y_test.values - y_pred_best
    ax_b.scatter(y_pred_best, residuals, alpha=0.5, color="#7B1FA2", s=18)
    ax_b.axhline(0, color="red", linestyle="--", lw=2)
    ax_b.set_xlabel("Predicted ($)"); ax_b.set_ylabel("Residuals ($)")
    ax_b.set_title("Residual Plot")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    if best_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
        st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
        importances = best["model"].feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]
        fig4, ax4   = plt.subplots(figsize=(10, 4))
        ax4.bar(range(len(feature_cols)), importances[sorted_idx],
                color="#1565C0", edgecolor="white")
        ax4.set_xticks(range(len(feature_cols)))
        ax4.set_xticklabels([feature_cols[i] for i in sorted_idx], rotation=38, ha="right")
        ax4.set_title(f"Feature Importances — {best_name}", fontweight="bold")
        ax4.set_ylabel("Importance")
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

# ── FOOTER
st.markdown("---")
st.caption("⚠️ For educational / demo purposes only.")