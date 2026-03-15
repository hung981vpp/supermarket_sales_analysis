import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    ax.set_xlabel("Sales (USD)")
    save_fig(fig, "sales_distribution.png", params)

# Vẽ doanh số theo tháng (line chart chuỗi thời gian).
def plot_monthly_sales(monthly: pd.DataFrame, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["YearMonth"], monthly["Sales"], marker="o", color="steelblue")
    ax.set_title("Doanh số theo tháng")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Sales (USD)")
    plt.xticks(rotation=45)
    save_fig(fig, "monthly_sales.png", params)

# Vẽ Elbow curve để chọn số cụm tối ưu.
def plot_elbow(inertias: dict, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(inertias.keys()), list(inertias.values()), marker="o", color="coral")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Số cụm (k)")
    ax.set_ylabel("Inertia")
    save_fig(fig, "elbow_curve.png", params)

# Vẽ scatter 2D các cụm sau khi giảm chiều bằng PCA.
def plot_clusters_pca(X_2d: np.ndarray, labels: np.ndarray, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title("Phân cụm khách hàng (PCA 2D)")
    save_fig(fig, "clusters_pca.png", params)

# Vẽ confusion matrix dạng heatmap.
def plot_confusion_matrix(cm: pd.DataFrame, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    save_fig(fig, "confusion_matrix.png", params)

# Vẽ forecast vs actual cho chuỗi thời gian.
def plot_forecast(train: pd.DataFrame, test: pd.DataFrame,
                  forecasts: dict, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["YearMonth"], train["Sales"], label="Train", color="steelblue")
    ax.plot(test["YearMonth"],  test["Sales"],  label="Actual", color="black", linewidth=2)
    colors = ["coral", "green", "purple", "orange", "brown"]
    for (name, y_pred), color in zip(forecasts.items(), colors):
        ax.plot(test["YearMonth"], y_pred, label=name, linestyle="--", color=color)
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Sales (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    save_fig(fig, "forecast_comparison.png", params)

# Vẽ feature importance của Random Forest (top N).
def plot_feature_importance(fi: pd.DataFrame, params: dict, top_n: int = 15) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fi_top = fi.head(top_n)
    sns.barplot(data=fi_top, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importance (Random Forest)")
    save_fig(fig, "feature_importance.png", params)

# Vẽ residual plot cho forecasting model tốt nhất.
def plot_residuals(residuals: np.ndarray, params: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, marker="o", color="coral")
    axes[0].axhline(0, linestyle="--", color="black")
    axes[0].set_title("Residuals theo thời gian")
    sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Phân phối Residuals")
    save_fig(fig, "residuals.png", params)
