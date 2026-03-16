import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

# Hỗ trợ hiển thị tiếng Việt trong matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# Lưu figure ra outputs/figures/.
def save_fig(fig, filename: str, params: dict) -> None:
    out_dir = Path(params["outputs"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches="tight", dpi=150)
    plt.close(fig)


# Vẽ phân phối doanh số (histogram + KDE).
def plot_sales_distribution(df: pd.DataFrame, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Sales"], kde=True, ax=ax, color="steelblue")
    ax.set_title("Phân phối doanh số (Sales)")
    ax.set_xlabel("Doanh số (USD)")
    ax.set_ylabel("Số lượng")
    save_fig(fig, "sales_distribution.png", params)


# Vẽ doanh số theo tháng (line chart chuỗi thời gian).
def plot_monthly_sales(monthly: pd.DataFrame, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["YearMonth"], monthly["Sales"], marker="o", color="steelblue")
    ax.set_title("Doanh số theo tháng")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Doanh số (USD)")
    plt.xticks(rotation=45)
    save_fig(fig, "monthly_sales.png", params)


# Vẽ Elbow curve để chọn số cụm tối ưu.
def plot_elbow(inertias: dict, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(inertias.keys()), list(inertias.values()), marker="o", color="coral")
    ax.set_title("Phương pháp Elbow — Chọn số cụm tối ưu")
    ax.set_xlabel("Số cụm (k)")
    ax.set_ylabel("Inertia (tổng bình phương khoảng cách)")
    save_fig(fig, "elbow_curve.png", params)


# Vẽ scatter 2D các cụm sau khi giảm chiều bằng PCA.
def plot_clusters_pca(X_2d: np.ndarray, labels: np.ndarray, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Cụm")
    ax.set_title("Phân cụm khách hàng (PCA 2 chiều)")
    ax.set_xlabel("Thành phần chính 1")
    ax.set_ylabel("Thành phần chính 2")
    save_fig(fig, "clusters_pca.png", params)


# Vẽ confusion matrix dạng heatmap.
def plot_confusion_matrix(cm: pd.DataFrame, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Ma trận nhầm lẫn (Confusion Matrix)")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_xlabel("Nhãn dự đoán")
    save_fig(fig, "confusion_matrix.png", params)


# Vẽ forecast vs actual cho chuỗi thời gian.
def plot_forecast(train: pd.DataFrame, test: pd.DataFrame,
                  forecasts: dict, params: dict) -> None:
    label_map = {
        "naive":        "Naive",
        "moving_avg":   "Trung bình động",
        "arima":        "ARIMA",
        "holt_winters": "Holt-Winters",
        "prophet":      "Prophet",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["YearMonth"], train["Sales"], label="Tập train", color="steelblue")
    ax.plot(test["YearMonth"],  test["Sales"],  label="Thực tế",   color="black", linewidth=2)
    colors = ["coral", "green", "purple", "orange", "brown"]
    for (name, y_pred), color in zip(forecasts.items(), colors):
        ax.plot(test["YearMonth"], y_pred,
                label=label_map.get(name, name), linestyle="--", color=color)
    ax.set_title("Dự báo doanh số — So sánh các mô hình")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Doanh số (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    save_fig(fig, "forecast_comparison.png", params)


# Vẽ feature importance của Random Forest (top N).
def plot_feature_importance(fi: pd.DataFrame, params: dict, top_n: int = 15) -> None:
    vi_map = {
        "Sales":             "Doanh số",
        "Shipping Days":     "Số ngày giao hàng",
        "Month":             "Tháng",
        "Year":              "Năm",
        "Quarter":           "Quý",
        "DayOfWeek":         "Ngày trong tuần",
        "Ship Mode":         "Hình thức vận chuyển",
        "Region":            "Vùng",
        "Category":          "Danh mục",
        "Sub-Category":      "Danh mục con",
        "Segment":           "Phân khúc",
    }
    fi_top = fi.head(top_n).copy()
    fi_top["feature"] = fi_top["feature"].map(lambda x: vi_map.get(x, x))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_top, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} đặc trưng quan trọng nhất (Random Forest)")
    ax.set_xlabel("Mức độ quan trọng")
    ax.set_ylabel("Đặc trưng")
    save_fig(fig, "feature_importance.png", params)


# Vẽ residual plot cho forecasting model tốt nhất.
def plot_residuals(residuals: np.ndarray, params: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, marker="o", color="coral")
    axes[0].axhline(0, linestyle="--", color="black")
    axes[0].set_title("Phần dư theo thời gian")
    axes[0].set_xlabel("Quan sát")
    axes[0].set_ylabel("Phần dư (USD)")
    sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Phân phối phần dư")
    axes[1].set_xlabel("Phần dư (USD)")
    axes[1].set_ylabel("Số lượng")
    save_fig(fig, "residuals.png", params)
