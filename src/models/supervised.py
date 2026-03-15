import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Tách features và target (Segment), encode nhãn sang số.
def prepare_xy(df: pd.DataFrame, params: dict) -> tuple:
    target = params["preprocessing"]["target_column"]
    drop   = ["Customer ID", "Order ID", "Customer Name", "Product ID",
              "Product Name", "Order Date", "Ship Date", target]
    feature_cols = [c for c in df.columns if c not in drop]

    le = LabelEncoder()
    X  = df[feature_cols].select_dtypes(include=[np.number])
    y  = le.fit_transform(df[target])
    return X, y, le

# Chia train/test theo tỉ lệ trong params, giữ nguyên tỉ lệ lớp (stratify).
def split_data(X: pd.DataFrame, y: np.ndarray, params: dict) -> tuple:
    test_size    = params["preprocessing"]["test_size"]
    random_state = params["general"]["seed"]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)

# Khởi tạo các model theo danh sách models_to_train trong params.
def build_models(params: dict) -> dict:
    cfg  = params["models"]["supervised"]
    seed = params["general"]["seed"]
    model_map = {
        "logistic_regression": LogisticRegression(
            max_iter=cfg["logistic_regression"]["max_iter"],
            random_state=seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=cfg["decision_tree"]["max_depth"],
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg["random_forest"]["n_estimators"],
            max_depth=cfg["random_forest"]["max_depth"],
            random_state=seed,
        ),
    }
    return {name: model_map[name] for name in cfg["models_to_train"]}

# Train và đánh giá từng model, trả về dict kết quả metrics.
def train_evaluate(models: dict, X_train, X_test, y_train, y_test, params: dict) -> pd.DataFrame:
    cv      = StratifiedKFold(n_splits=params["models"]["supervised"]["cv_folds"], shuffle=True,
                              random_state=params["general"]["seed"])
    records = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        cv_f1   = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro").mean()
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else None

        records.append({
            "model":     name,
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1_macro":  f1_score(y_test, y_pred, average="macro"),
            "cv_f1":     cv_f1,
            "roc_auc":   roc_auc,
        })
        print(f"[{name}] f1_macro={records[-1]['f1_macro']:.4f} | roc_auc={roc_auc:.4f}")

    return pd.DataFrame(records).sort_values("f1_macro", ascending=False)

# In confusion matrix và classification report cho model chỉ định.
def error_analysis(model, X_test, y_test, label_encoder: LabelEncoder) -> None:
    y_pred  = model.predict(X_test)
    classes = label_encoder.classes_
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

# Lấy feature importance từ Random Forest, trả về DataFrame đã sort.
def get_feature_importance(model: RandomForestClassifier, feature_names: list) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance

# Lưu model đã train ra outputs/models/.
def save_model(model, model_name: str, params: dict) -> None:
    out_dir = Path(params["outputs"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.pkl"
    joblib.dump(model, out_path)
    print(f"[save_model] Saved {model_name} to {out_path}")

# Chạy toàn bộ pipeline phân lớp, trả về (models, results_df, label_encoder).
def run_supervised_pipeline(df: pd.DataFrame, params: dict) -> tuple:
    X, y, le                        = prepare_xy(df, params)
    X_train, X_test, y_train, y_test = split_data(X, y, params)
    models                           = build_models(params)
    results                          = train_evaluate(models, X_train, X_test, y_train, y_test, params)

    for name, model in models.items():
        save_model(model, name, params)

    return models, results, le, (X_train, X_test, y_train, y_test)
