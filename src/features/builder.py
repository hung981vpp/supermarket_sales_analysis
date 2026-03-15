import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Tính đặc trưng RFM (Recency, Frequency, Monetary) theo từng khách hàng.
def build_rfm(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    snapshot_date = pd.to_datetime(params["features"]["rfm"]["snapshot_date"])

    rfm = df.groupby("Customer ID").agg(
        Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
    ).reset_index()

    return rfm

# Tính doanh thu trung bình mỗi đơn hàng theo từng khách hàng.
def build_avg_order_value(df: pd.DataFrame) -> pd.DataFrame:
    aov = df.groupby("Customer ID").apply(
        lambda x: x.groupby("Order ID")["Sales"].sum().mean()
    ).reset_index()
    aov.columns = ["Customer ID", "AvgOrderValue"]
    return aov

# Đếm số lượng danh mục hàng (Category) khác nhau mỗi khách đã mua.
def build_category_diversity(df: pd.DataFrame) -> pd.DataFrame:
    diversity = df.groupby("Customer ID")["Category"].nunique().reset_index()
    diversity.columns = ["Customer ID", "CategoryDiversity"]
    return diversity

# Tính tỉ lệ từng nhóm hàng (Category) trong tổng doanh thu của mỗi khách.
def build_category_ratio(df: pd.DataFrame) -> pd.DataFrame:
    sales_by_cat = df.groupby(["Customer ID", "Category"])["Sales"].sum().unstack(fill_value=0)
    ratio = sales_by_cat.div(sales_by_cat.sum(axis=1), axis=0)
    ratio.columns = [f"Ratio_{c}" for c in ratio.columns]
    return ratio.reset_index()

# Gộp tất cả đặc trưng thành một DataFrame duy nhất theo Customer ID.
def build_customer_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    rfm        = build_rfm(df, params)
    aov        = build_avg_order_value(df)
    diversity  = build_category_diversity(df)
    cat_ratio  = build_category_ratio(df)

    features = rfm \
        .merge(aov,       on="Customer ID", how="left") \
        .merge(diversity, on="Customer ID", how="left") \
        .merge(cat_ratio, on="Customer ID", how="left")

    return features

# Chuẩn hoá các cột số bằng StandardScaler, trả về DataFrame và scaler object.
def scale_features(df: pd.DataFrame, exclude_cols: list = ["Customer ID"]) -> tuple:
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler

# Tổng hợp doanh số theo tháng để dùng cho chuỗi thời gian.
def build_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.sort_values("YearMonth").reset_index(drop=True)
    return monthly

# Tạo lag features và rolling mean cho chuỗi thời gian doanh số.
def build_lag_features(monthly: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = monthly.copy()
    for lag in lags:
        df[f"Sales_lag{lag}"] = df["Sales"].shift(lag)
    df["Sales_rolling3"] = df["Sales"].shift(1).rolling(3).mean()
    df = df.dropna().reset_index(drop=True)
    return df
