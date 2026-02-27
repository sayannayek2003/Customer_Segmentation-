import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# Load trained model & scaler
# -----------------------------
kmeans_model = pickle.load(open("models/kmeans.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Initialize session state for persistent prediction
if "pred_result" not in st.session_state:
    st.session_state["pred_result"] = None

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS — Dark Premium Theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3f 0%, #16213e 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}
[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

/* All input fields in sidebar — force white text + visible background */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] [data-baseweb="input"] input {
    background: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid rgba(139, 92, 246, 0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    caret-color: #a78bfa !important;
}

/* Placeholder text */
[data-testid="stSidebar"] input::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* Input wrapper background */
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
}

/* Step +/- buttons */
[data-testid="stSidebar"] [data-baseweb="input"] button,
[data-testid="stSidebar"] .stNumberInput button {
    background: rgba(139, 92, 246, 0.25) !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stSidebar"] .stNumberInput button:hover {
    background: rgba(139, 92, 246, 0.5) !important;
}

/* Input labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #c4b5fd !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Focus glow */
[data-testid="stSidebar"] input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
    outline: none !important;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
    color: white !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 12px !important;
}

/* Divider */
hr {
    border-color: rgba(139, 92, 246, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/marketing_campaign.csv").dropna()
    CURRENT_YEAR = 2026
    df["Age"] = CURRENT_YEAR - df["Year_Birth"]
    spending_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Spending"] = df[spending_cols].sum(axis=1)
    return df

customer_data = load_data()

# -----------------------------
# PCA constants & cached helper (top-level so caching is stable)
# -----------------------------
TRAINING_FEATURES = [
    "Age", "Income", "Recency", "Total_Spending",
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
]
CLUSTER_COLORS = {
    0: "#3b82f6",   # Budget  — blue
    1: "#f59e0b",   # Premium — amber
    2: "#ef4444",   # Inactive — red
    3: "#10b981",   # Regular — green
}
CLUSTER_LABELS = {
    0: "💸 Budget Customer",
    1: "💎 Premium Customer",
    2: "😴 Inactive Customer",
    3: "🛒 Regular Customer",
}

@st.cache_data
def compute_pca(df: pd.DataFrame):
    """Scale dataset, assign clusters, fit 2-D PCA. Returns (components, labels, pca_model)."""
    X = df[TRAINING_FEATURES].dropna()
    X_scaled = scaler.transform(X)
    labels = kmeans_model.predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    return components, labels, pca

@st.cache_data
def segment_kpis(df: pd.DataFrame):
    """Return per-cluster summary: count, avg income, avg spending, avg recency."""
    X = df[TRAINING_FEATURES].dropna()
    labels = kmeans_model.predict(scaler.transform(X))
    df2 = df[TRAINING_FEATURES].dropna().copy()
    df2["Cluster"] = labels
    df2["Income"]   = df.loc[df2.index, "Income"]
    df2["Total_Spending"] = df.loc[df2.index, "Total_Spending"]
    df2["Recency"]  = df.loc[df2.index, "Recency"]
    stats = df2.groupby("Cluster").agg(
        Count=("Income", "count"),
        Avg_Income=("Income", "mean"),
        Avg_Spending=("Total_Spending", "mean"),
        Avg_Recency=("Recency", "mean"),
    ).reset_index()
    return stats

# Precompute once
pca_components, pca_labels, pca_model = compute_pca(customer_data)
seg_stats = segment_kpis(customer_data)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(109,40,217,0.1) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    backdrop-filter: blur(10px);
">
    <h1 style="
        margin: 0 0 8px 0;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #7c3aed, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    ">🛍️ Customer Segmentation System</h1>
    <p style="margin: 0; color: #94a3b8; font-size: 1rem; font-weight: 400;">
        Machine Learning powered analytics to understand and group your customers intelligently.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# KPI Metric Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 Total Customers", f"{len(customer_data):,}")
with col2:
    st.metric("💰 Avg. Income", f"${customer_data['Income'].mean():,.0f}")
with col3:
    st.metric("🛒 Avg. Spending", f"${customer_data['Total_Spending'].mean():,.0f}")
with col4:
    st.metric("📅 Avg. Recency", f"{customer_data['Recency'].mean():.0f} days")

st.markdown("<div style='margin: 24px 0;'></div>", unsafe_allow_html=True)

# -----------------------------
# Per-Segment KPI Cards
# -----------------------------
st.markdown("<p style='color:#94a3b8; font-size:0.8rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>📊 Segment Breakdown</p>", unsafe_allow_html=True)
seg_cols = st.columns(4)
for _, row in seg_stats.iterrows():
    cid   = int(row["Cluster"])
    color = CLUSTER_COLORS[cid]
    label = CLUSTER_LABELS[cid]
    with seg_cols[cid]:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid {color}55;
            border-top: 3px solid {color};
            border-radius: 14px;
            padding: 16px 18px;
            backdrop-filter: blur(8px);
        ">
            <div style="font-size:1.05rem; font-weight:700; color:{color}; margin-bottom:10px;">{label}</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.82rem; color:#e2e8f0;">
                <tr><td style="color:#94a3b8; padding:3px 0;">👥 Customers</td><td style="text-align:right; font-weight:600;">{int(row['Count']):,}</td></tr>
                <tr><td style="color:#94a3b8; padding:3px 0;">💰 Avg Income</td><td style="text-align:right; font-weight:600;">${row['Avg_Income']:,.0f}</td></tr>
                <tr><td style="color:#94a3b8; padding:3px 0;">🛒 Avg Spending</td><td style="text-align:right; font-weight:600;">${row['Avg_Spending']:,.0f}</td></tr>
                <tr><td style="color:#94a3b8; padding:3px 0;">📅 Avg Recency</td><td style="text-align:right; font-weight:600;">{row['Avg_Recency']:.0f}d</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar — Customer Input
# -----------------------------
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(109,40,217,0.1));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 16px;
">
    <h3 style="margin:0; color:#a78bfa; font-size:1rem; font-weight:700; letter-spacing:0.5px;">
        📥 Customer Profiler
    </h3>
    <p style="margin:4px 0 0 0; font-size:0.75rem; color:#94a3b8;">
        Enter details to predict segment
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("👤 Demographics")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

st.sidebar.subheader("💰 Financials")
income = st.sidebar.number_input("Annual Income ($)", value=50000, step=1000)
total_spending = st.sidebar.number_input("Total Spending ($)", value=500, step=50)

st.sidebar.subheader("🛍️ Purchase Behaviour")
deals = st.sidebar.number_input("Deals Purchases", value=2, min_value=0)
web = st.sidebar.number_input("Web Purchases", value=5, min_value=0)
catalog = st.sidebar.number_input("Catalog Purchases", value=1, min_value=0)
store = st.sidebar.number_input("Store Purchases", value=4, min_value=0)

st.sidebar.subheader("📆 Engagement")
recency = st.sidebar.number_input("Recency (days since last purchase)", value=30, min_value=0)
visits = st.sidebar.number_input("Web Visits / Month", value=6, min_value=0)

st.sidebar.markdown("<div style='margin:16px 0 8px 0;'></div>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("🎯 Predict Segment", use_container_width=True)

# -----------------------------
# Prediction Result
# -----------------------------
SEGMENT_CONFIG = {
    0: {
        "name": "Budget Customer",
        "icon": "💸",
        "color": "#3b82f6",
        "bg": "rgba(59,130,246,0.12)",
        "border": "rgba(59,130,246,0.4)",
        "desc": "Price-sensitive shopper with low spending habits.",
        "tip": "📌 Strategy: Offer discounts, bundle deals, and loyalty reward programs."
    },
    1: {
        "name": "Premium Customer",
        "icon": "💎",
        "color": "#f59e0b",
        "bg": "rgba(245,158,11,0.12)",
        "border": "rgba(245,158,11,0.4)",
        "desc": "High-income, high-spending loyal customer.",
        "tip": "📌 Strategy: Exclusive offers, premium memberships, and VIP early access."
    },
    2: {
        "name": "Inactive Customer",
        "icon": "😴",
        "color": "#ef4444",
        "bg": "rgba(239,68,68,0.12)",
        "border": "rgba(239,68,68,0.4)",
        "desc": "High recency — hasn't purchased recently, needs re-engagement.",
        "tip": "📌 Strategy: Win-back campaigns, personalised re-engagement emails & special offers."
    },
    3: {
        "name": "Regular Customer",
        "icon": "🛒",
        "color": "#10b981",
        "bg": "rgba(16,185,129,0.12)",
        "border": "rgba(16,185,129,0.4)",
        "desc": "Moderate income with consistent, steady purchase behaviour.",
        "tip": "📌 Strategy: Cross-sell complementary products and introduce subscription plans."
    },
}

# Store prediction in session state when button is clicked
if predict_btn:
    input_data = np.array([[age, income, recency, total_spending,
                            deals, web, catalog, store, visits]])
    scaled_input = scaler.transform(input_data)
    predicted_cluster = int(kmeans_model.predict(scaled_input)[0])
    st.session_state["pred_result"] = {
        "cluster": predicted_cluster,
        "income": income,
        "total_spending": total_spending,
        "deals": deals,
        "web": web,
        "catalog": catalog,
        "store": store,
        "age": age,
        "recency": recency,
        "visits": visits,
    }

# Display prediction result from session state (persists across reruns)
if st.session_state["pred_result"] is not None:
    pr = st.session_state["pred_result"]
    predicted_cluster = pr["cluster"]
    seg = SEGMENT_CONFIG.get(predicted_cluster, SEGMENT_CONFIG[3])

    st.markdown(f"""
    <div style="
        background: {seg['bg']};
        border: 1px solid {seg['border']};
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    ">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:12px;">
            <span style="font-size:2.5rem; line-height:1;">{seg['icon']}</span>
            <div>
                <div style="font-size:0.8rem; color:#94a3b8; font-weight:500; letter-spacing:1px; text-transform:uppercase;">
                    Predicted Segment — Cluster {predicted_cluster}
                </div>
                <div style="font-size:1.8rem; font-weight:800; color:{seg['color']}; line-height:1.2;">
                    {seg['name']}
                </div>
            </div>
        </div>
        <p style="margin:0 0 10px 0; color:#cbd5e1; font-size:0.95rem;">{seg['desc']}</p>
        <div style="
            background: rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 12px 16px;
            color: #e2e8f0;
            font-size: 0.9rem;
            font-weight: 500;
        ">{seg['tip']}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Tabs — Analytics / Data
# -----------------------------
tab1, tab2 = st.tabs(["📊 Analytics", "📋 Dataset Preview"])

with tab1:

    pr = st.session_state["pred_result"]

    if pr is None:
        # ── No prediction yet → show a friendly prompt ──────────────────
        st.markdown("""
        <div style="
            text-align: center;
            padding: 70px 30px;
            background: rgba(139,92,246,0.05);
            border: 1px dashed rgba(139,92,246,0.35);
            border-radius: 20px;
            margin-top: 20px;
        ">
            <div style="font-size: 3.5rem; margin-bottom: 16px;">🎯</div>
            <h3 style="color: #a78bfa; font-size: 1.4rem; font-weight: 700; margin-bottom: 10px;">
                No Segment Predicted Yet
            </h3>
            <p style="color: #94a3b8; font-size: 1rem; max-width: 460px; margin: 0 auto;">
                Fill in the customer details in the <strong style="color:#c4b5fd;">sidebar</strong>
                and click <strong style="color:#c4b5fd;">🎯 Predict Segment</strong> to see
                segment-specific analytics and visualisations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Prediction exists → show all charts for that segment only ───

        # Build cluster-labelled dataframe (cached)
        @st.cache_data
        def get_clustered_df(df: pd.DataFrame) -> pd.DataFrame:
            X = df[TRAINING_FEATURES].dropna()
            labels = kmeans_model.predict(scaler.transform(X))
            out = df.loc[X.index].copy()
            out["Cluster"] = labels
            return out

        clustered_df  = get_clustered_df(customer_data)
        active_cluster = pr["cluster"]
        seg_color      = SEGMENT_CONFIG[active_cluster]["color"]
        seg_name       = SEGMENT_CONFIG[active_cluster]["name"]
        plot_df        = clustered_df[clustered_df["Cluster"] == active_cluster]

        st.markdown(
            f"<p style='color:#94a3b8;font-size:0.82rem;margin:-4px 0 16px 0;'>"
            f"📌 Showing analytics for <strong style='color:{seg_color};'>{seg_name}</strong> "
            f"customers &nbsp;({len(plot_df):,} customers in this segment)</p>",
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------------
        # 1. Scatter — Income vs Total Spending (segment only)
        # --------------------------------------------------------------
        st.markdown("### 📈 Income vs Total Spending")

        fig_scatter = px.scatter(
            plot_df,
            x="Income", y="Total_Spending",
            opacity=0.72,
            labels={"Income": "Annual Income ($)", "Total_Spending": "Total Spending ($)"},
            template="plotly_dark",
            hover_data={"Income": ":$,.0f", "Total_Spending": ":$,.0f"},
        )
        fig_scatter.update_traces(marker=dict(size=7, color=seg_color, line=dict(width=0)))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            height=420,
        )
        # Overlay the new customer
        fig_scatter.add_trace(go.Scatter(
            x=[pr["income"]], y=[pr["total_spending"]],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color="#ffffff",
                        line=dict(width=2, color=seg_color)),
            text=["  ⭐ You"],
            textposition="middle right",
            textfont=dict(color="white", size=12),
            name="Your Customer",
            showlegend=True,
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)

        # --------------------------------------------------------------
        # 2. Bar — Average Spend by Channel (segment only + your input)
        # --------------------------------------------------------------
        st.markdown("### 🛍️ Average Spend by Channel")

        channel_avg = {
            "Web Purchases":     plot_df["NumWebPurchases"].mean()     * 50,
            "Store Purchases":   plot_df["NumStorePurchases"].mean()   * 50,
            "Catalog Purchases": plot_df["NumCatalogPurchases"].mean() * 50,
            "Deal Purchases":    plot_df["NumDealsPurchases"].mean()   * 50,
        }
        channel_df = pd.DataFrame(list(channel_avg.items()), columns=["Channel", "Avg Spend ($)"])
        channel_df = channel_df.sort_values("Avg Spend ($)", ascending=True)

        fig_bar = px.bar(
            channel_df,
            x="Avg Spend ($)", y="Channel",
            orientation="h",
            color="Avg Spend ($)",
            color_continuous_scale=[seg_color] * 4,
            template="plotly_dark",
            text_auto=".0f",
        )
        fig_bar.update_traces(textfont_color="white", marker_line_width=0)

        customer_channels = {
            "Web Purchases":     pr["web"]     * 50,
            "Store Purchases":   pr["store"]   * 50,
            "Catalog Purchases": pr["catalog"] * 50,
            "Deal Purchases":    pr["deals"]   * 50,
        }
        cust_df = pd.DataFrame(list(customer_channels.items()), columns=["Channel", "Avg Spend ($)"])
        cust_df = cust_df.set_index("Channel").reindex(channel_df["Channel"]).reset_index()
        fig_bar.add_trace(go.Bar(
            x=cust_df["Avg Spend ($)"],
            y=cust_df["Channel"],
            orientation="h",
            name="Your Input",
            marker_color="#ffffff",
            opacity=0.9,
            text=cust_df["Avg Spend ($)"].astype(int).astype(str),
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            coloraxis_showscale=False,
            barmode="group",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(font=dict(color="#94a3b8")),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --------------------------------------------------------------
        # 3. Donut — Spending Category Breakdown (segment only)
        # --------------------------------------------------------------
        st.markdown("### 🥧 Spending Category Breakdown")

        spending_cols_map = {
            "Wines": "MntWines", "Meat": "MntMeatProducts",
            "Fish": "MntFishProducts", "Fruits": "MntFruits",
            "Sweets": "MntSweetProducts", "Gold": "MntGoldProds",
        }
        category_totals = {k: plot_df[v].sum() for k, v in spending_cols_map.items()}
        cat_df = pd.DataFrame(list(category_totals.items()), columns=["Category", "Total"])

        fig_donut = px.pie(
            cat_df, names="Category", values="Total",
            hole=0.55,
            color_discrete_sequence=[seg_color, "#a78bfa", "#c4b5fd", "#7c3aed", "#6d28d9", "#4c1d95"],
            template="plotly_dark",
        )
        fig_donut.update_traces(textposition="outside", textfont_size=13)
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(color="#94a3b8")),
            height=380,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # --------------------------------------------------------------
        # 4. PCA — Only the predicted cluster + new customer marker
        # --------------------------------------------------------------
        st.markdown("### 🔵 PCA Cluster Visualization")

        fig_pca, ax = plt.subplots(figsize=(9, 5), facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")

        # Draw ONLY the predicted segment's cloud
        mask = pca_labels == active_cluster
        ax.scatter(
            pca_components[mask, 0], pca_components[mask, 1],
            c=seg_color, s=35, alpha=0.80, linewidths=0,
            zorder=3, label=f"{CLUSTER_LABELS[active_cluster]}",
        )

        # New customer X marker
        new_input = np.array([[
            pr["age"], pr["income"], pr["recency"], pr["total_spending"],
            pr["deals"], pr["web"], pr["catalog"], pr["store"], pr["visits"],
        ]])
        new_pca = pca_model.transform(scaler.transform(new_input))
        ax.scatter(
            new_pca[0, 0], new_pca[0, 1],
            c="#ffffff", marker="X", s=300,
            linewidths=1.5, edgecolors="#ff00ff",
            zorder=10, label="⭐ Your Customer",
        )

        ax.legend(
            loc="upper right", fontsize=9, framealpha=0.25,
            facecolor="#1e1e3f", edgecolor=(0.545, 0.361, 0.965, 0.4),
            labelcolor="white",
        )
        ax.set_xlabel("Principal Component 1", color="#94a3b8", fontsize=10)
        ax.set_ylabel("Principal Component 2", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor((0.545, 0.361, 0.965, 0.25))

        plt.tight_layout()
        st.pyplot(fig_pca, use_container_width=True)
        plt.close(fig_pca)

with tab2:
    st.markdown("### 📋 Customer Dataset")
    st.markdown(f"<p style='color:#94a3b8;font-size:0.85rem;'>Showing first 50 rows of {len(customer_data):,} total customers.</p>", unsafe_allow_html=True)
    st.dataframe(
        customer_data.head(50),
        use_container_width=True,
        height=500,
    )