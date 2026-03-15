import pandas as pd
import yaml
from pathlib import Path

# Đọc file cấu hình params.yaml, trả về dict tham số pipeline.
def load_params(config_path: str = "configs/params.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Đọc CSV gốc, parse ngày dd/mm/yyyy, validate đủ 18 cột schema.
def load_raw_data(params: dict) -> pd.DataFrame:
    raw_path = Path(params["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {raw_path}")
    df = pd.read_csv(
        raw_path,
        parse_dates=["Order Date", "Ship Date"],
        dayfirst=True,
    )
    _validate_schema(df)
    return df

# Kiểm tra DataFrame có đủ 18 cột theo data dictionary Superstore.
def _validate_schema(df: pd.DataFrame) -> None:
    expected_columns = [
        "Row ID", "Order ID", "Order Date", "Ship Date",
        "Ship Mode", "Customer ID", "Customer Name", "Segment",
        "Country", "City", "State", "Postal Code", "Region",
        "Product ID", "Category", "Sub-Category", "Product Name", "Sales",
    ]
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thiếu các cột: {missing}")
    
# Đọc data đã tiền xử lý từ data/processed/, hỗ trợ .parquet và .csv.
def load_processed_data(params: dict) -> pd.DataFrame:

    processed_path = Path(params["data"]["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at: {processed_path}. "
            "Hãy chạy notebook 02_preprocess_feature.ipynb trước."
        )
    if str(processed_path).endswith(".parquet"):
        return pd.read_parquet(processed_path)
    return pd.read_csv(processed_path, parse_dates=["Order Date", "Ship Date"])

# Trả về dict thống kê nhanh: shape, missing, date range, số orders/customers/products.
def get_data_info(df: pd.DataFrame) -> dict:
    
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "date_range": (df["Order Date"].min(), df["Order Date"].max()),
        "n_orders": df["Order ID"].nunique(),
        "n_customers": df["Customer ID"].nunique(),
        "n_products": df["Product ID"].nunique(),
    }
