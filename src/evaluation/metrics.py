import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error,
)

# Tính đầy đủ metrics phân lớp: accuracy, f1_macro, roc_auc.
def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_proba is not None:
        metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
    return metrics

# Tính MAE, RMSE, sMAPE cho bài toán hồi quy / dự báo chuỗi thời gian.
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2 * np.abs(y_true - y_pred) /
                    (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}

# Tính Silhouette Score và Davies-Bouldin Index cho clustering.
def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    return {
        "silhouette":     silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }

# Trả về confusion matrix dạng DataFrame có tên lớp rõ ràng.
def get_confusion_matrix(y_true, y_pred, class_names: list) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=class_names, columns=class_names)

# In classification report ra stdout.
def print_classification_report(y_true, y_pred, class_names: list) -> None:
    print(classification_report(y_true, y_pred, target_names=class_names))
