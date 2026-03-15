import pandas as pd
from pathlib import Path

# Lưu DataFrame kết quả ra file CSV trong outputs/tables/.
def save_table(df: pd.DataFrame, filename: str, params: dict) -> None:
    out_dir = Path(params["outputs"]["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"[save_table] Saved to {out_path}")

# Tổng hợp kết quả phân lớp nhiều model thành 1 bảng so sánh.
def summarize_classification(results: list) -> pd.DataFrame:
    return pd.DataFrame(results)[["model", "accuracy", "f1_macro", "cv_f1", "roc_auc"]] \
             .sort_values("f1_macro", ascending=False) \
             .reset_index(drop=True)

# Tổng hợp kết quả forecasting nhiều model thành 1 bảng so sánh.
def summarize_forecasting(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values("RMSE").reset_index(drop=True)

# Tổng hợp cluster profile thành bảng insight dễ đọc.
def summarize_clusters(profile: pd.DataFrame) -> pd.DataFrame:
    return profile.sort_values("Cluster").reset_index(drop=True)

# Tổng hợp top luật kết hợp thành bảng báo cáo.
def summarize_association_rules(rules: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return rules[cols].head(n).reset_index(drop=True)

# In bảng tổng kết toàn bộ pipeline ra stdout.
def print_summary(clf_results: pd.DataFrame, forecast_results: pd.DataFrame,
                  cluster_profile: pd.DataFrame) -> None:
    print("=" * 60)
    print("CLASSIFICATION RESULTS")
    print(clf_results.to_string(index=False))
    print("\nFORECASTING RESULTS")
    print(forecast_results.to_string(index=False))
    print("\nCLUSTER PROFILE")
    print(cluster_profile.to_string(index=False))
    print("=" * 60)
