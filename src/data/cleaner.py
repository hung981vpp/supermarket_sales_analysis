import pandas as pd
import numpy as np
from pathlib import Path

# Xoá các cột không cần thiết theo danh sách trong params.
def drop_unnecessary_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])

# Xoá các dòng trùng lặp hoàn toàn.
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[drop_duplicates] Removed {before - len(df)} duplicate rows.")
    return df

# Xử lý giá trị thiếu: Sales điền median, categorical điền 'Unknown'.
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["str", "object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")
    return df

# Loại bỏ outlier cột Sales bằng Z-score (ngưỡng mặc định 3.0).
def remove_outliers_sales(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    before = len(df)
    z_scores = np.abs((df["Sales"] - df["Sales"].mean()) / df["Sales"].std())
    df = df[z_scores < z_thresh].copy()
    print(f"[remove_outliers_sales] Removed {before - len(df)} outlier rows.")
    return df

# Label-encode các cột categorical để dùng cho clustering và modeling.
def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df

# Trích xuất Year, Month, DayOfWeek, Quarter từ cột Order Date.
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"]      = df["Order Date"].dt.year
    df["Month"]     = df["Order Date"].dt.month
    df["Quarter"]   = df["Order Date"].dt.quarter
    df["DayOfWeek"] = df["Order Date"].dt.dayofweek
    return df

# Tính số ngày giao hàng = Ship Date - Order Date.
def add_shipping_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df

# Lưu DataFrame đã xử lý ra data/processed/.
def save_processed(df: pd.DataFrame, params: dict) -> None:
    out_path = Path(params["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[save_processed] Saved to {out_path} — shape: {df.shape}")

# Chạy toàn bộ pipeline làm sạch theo thứ tự chuẩn.
def run_cleaning_pipeline(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = drop_unnecessary_columns(df, params["preprocessing"]["drop_columns"])
    df = drop_duplicates(df)
    df = handle_missing(df)
    df = remove_outliers_sales(df)
    df = add_time_features(df)
    df = add_shipping_days(df)
    save_processed(df, params)
    return df
