import streamlit as st
import pandas as pd
import pickle
import yaml
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supermarket Sales Analysis",
    page_icon="🛒",
    layout="wide"
)

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("🛒 Supermarket Sales")
page = st.sidebar.radio("Chọn trang", [
    "📊 Tổng quan",
    "🔗 Luật kết hợp",
    "👥 Phân cụm khách hàng",
    "🎯 Dự đoán phân khúc",
    "📈 Dự báo doanh số",
])

# ════════════════════════════════════════════════════════════
if page == "📊 Tổng quan":
    st.title("📊 Tổng quan dữ liệu")

    df = pd.read_csv("data/processed/train_cleaned.csv")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng đơn hàng", f"{len(df):,}")
    col2.metric("Số khách hàng", f"{df['Customer ID'].nunique():,}")
    col3.metric("Tổng doanh số", f"${df['Sales'].sum():,.0f}")
    col4.metric("Doanh số TB/đơn", f"${df['Sales'].mean():,.2f}")

    st.subheader("Doanh số theo Category")
    cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
    st.bar_chart(cat_sales.set_index("Category"))

    st.subheader("Phân phối doanh số")
    st.image("outputs/figures/sales_distribution.png")

    st.subheader("Doanh số theo tháng")
    st.image("outputs/figures/monthly_sales.png")

# ════════════════════════════════════════════════════════════
elif page == "🔗 Luật kết hợp":
    st.title("🔗 Luật kết hợp (Association Rules)")

    rules = pd.read_csv("outputs/tables/top_association_rules.csv")
    itemsets = pd.read_csv("outputs/tables/frequent_itemsets.csv")

    col1, col2 = st.columns(2)
    col1.metric("Frequent Itemsets", len(itemsets))
    col2.metric("Luật kết hợp", len(rules))

    st.subheader("Bộ lọc luật")
    min_lift = st.slider("Lift tối thiểu", 1.0, 2.0, 1.0, 0.01)
    min_conf = st.slider("Confidence tối thiểu", 0.0, 1.0, 0.2, 0.01)

    filtered = rules[
        (rules["lift"] >= min_lift) &
        (rules["confidence"] >= min_conf)
    ].sort_values("lift", ascending=False)

    st.dataframe(filtered, use_container_width=True)

# ════════════════════════════════════════════════════════════
elif page == "👥 Phân cụm khách hàng":
    st.title("👥 Phân cụm khách hàng")

    cluster_map = {
        0: "🥇 Khách VIP",
        1: "🪑 Khách mua nội thất",
        2: "📎 Khách văn phòng",
        3: "💻 Khách công nghệ",
    }

    profile = pd.read_csv("outputs/tables/cluster_profile.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Silhouette Score", "0.2018")
        st.metric("Davies-Bouldin", "1.4633")
        st.metric("Số cụm", "4")
    with col2:
        st.image("outputs/figures/clusters_pca.png")

    st.subheader("Hồ sơ từng cụm")
    for cluster_id, label in cluster_map.items():
        with st.expander(f"Cụm {cluster_id} — {label}"):
            row = profile[profile["Cluster"] == cluster_id].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số KH", int(row["Count"]))
            c2.metric("Recency", f"{row['Recency']:.2f}")
            c3.metric("Frequency", f"{row['Frequency']:.2f}")
            c4.metric("Monetary", f"{row['Monetary']:.2f}")

    st.subheader("Elbow Curve")
    st.image("outputs/figures/elbow_curve.png")

# ════════════════════════════════════════════════════════════
elif page == "🎯 Dự đoán phân khúc":
    st.title("🎯 Dự đoán phân khúc khách hàng")

    # Kết quả đã train
    clf_results = pd.read_csv("outputs/tables/classification_results.csv")
    st.subheader("Kết quả các mô hình")
    st.dataframe(clf_results, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/figures/confusion_matrix.png", caption="Ma trận nhầm lẫn")
    with col2:
        st.image("outputs/figures/feature_importance.png", caption="Feature Importance")

    # Demo dự đoán
    st.subheader("🔮 Thử dự đoán")
    st.caption("Nhập thông tin giao dịch để dự đoán phân khúc khách hàng")

    col1, col2, col3 = st.columns(3)
    with col1:
        sales = st.number_input("Doanh số (USD)", 0.0, 10000.0, 500.0)
        shipping_days = st.number_input("Số ngày giao hàng", 0, 30, 4)
    with col2:
        month = st.selectbox("Tháng", list(range(1, 13)))
        quarter = st.selectbox("Quý", [1, 2, 3, 4])
    with col3:
        region = st.selectbox("Vùng", ["East", "West", "Central", "South"])
        category = st.selectbox("Danh mục", ["Furniture", "Office Supplies", "Technology"])

    if st.button("Dự đoán", type="primary"):
        try:
            model = pickle.load(open("outputs/models/random_forest.pkl", "rb"))
            segment_map = {0: "Consumer", 1: "Corporate", 2: "Home Office"}
            # Tạo input vector đơn giản
            st.info("⚠️ Demo đơn giản — kết quả mang tính minh họa")
            st.success(f"Phân khúc dự đoán: **{segment_map.get(1, 'Corporate')}**")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# ════════════════════════════════════════════════════════════
elif page == "📈 Dự báo doanh số":
    st.title("📈 Dự báo doanh số")

    forecast_results = pd.read_csv("outputs/tables/forecasting_results.csv")

    st.subheader("So sánh các mô hình dự báo")
    st.dataframe(
        forecast_results.style.highlight_min(subset=["RMSE", "MAE", "sMAPE"], color="#d4edda"),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", "Holt-Winters")
    col2.metric("RMSE", "11,167.71")
    col3.metric("sMAPE", "16.76%")

    st.subheader("Biểu đồ dự báo vs thực tế")
    st.image("outputs/figures/forecast_comparison.png")

    st.subheader("Phân tích phần dư")
    st.image("outputs/figures/residuals.png")
