import sys
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "4"

# Thêm root project vào sys.path để import src.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_params, load_raw_data, load_processed_data
from src.data.cleaner import run_cleaning_pipeline
from src.features.builder import build_customer_features, scale_features, build_monthly_sales
from src.mining.association import run_association_pipeline, format_rules, get_top_rules
from src.mining.clustering import run_clustering_pipeline, elbow_method, reduce_pca
from src.models.supervised import run_supervised_pipeline, error_analysis, get_feature_importance
from src.models.forecasting import run_forecasting_pipeline
from src.evaluation.report import (
    save_table, summarize_classification, summarize_forecasting,
    summarize_clusters, summarize_association_rules, print_summary,
)
from src.visualization.plots import (
    plot_sales_distribution, plot_monthly_sales, plot_elbow,
    plot_clusters_pca, plot_confusion_matrix, plot_forecast,
    plot_feature_importance, plot_residuals,
)

def main():
    print("=" * 60)
    print("SUPERMARKET SALES ANALYSIS — FULL PIPELINE")
    print("=" * 60)

    # ── 1. Load & Clean ──────────────────────────────────────────
    print("\n[1/6] Loading & Cleaning data...")
    params  = load_params()
    df_raw  = load_raw_data(params)
    df      = run_cleaning_pipeline(df_raw, params)

    # ── 2. EDA Plots ─────────────────────────────────────────────
    print("\n[2/6] Generating EDA plots...")
    plot_sales_distribution(df, params)
    monthly = build_monthly_sales(df)
    plot_monthly_sales(monthly, params)

    # ── 3. Association Rules ──────────────────────────────────────
    print("\n[3/6] Running Association Rules (Apriori)...")
    frequent_itemsets, rules = run_association_pipeline(df, params)
    top_rules = format_rules(get_top_rules(rules, n=10))
    save_table(top_rules, "top_association_rules.csv", params)
    save_table(frequent_itemsets.head(50), "frequent_itemsets.csv", params)

    # ── 4. Clustering ─────────────────────────────────────────────
    print("\n[4/6] Running Clustering (KMeans)...")
    customer_features        = build_customer_features(df, params)
    customer_scaled, scaler  = scale_features(customer_features)
    X = customer_scaled.drop(columns=["Customer ID"]).values

    inertias = elbow_method(X)
    plot_elbow(inertias, params)

    df_clustered, profile, km_model = run_clustering_pipeline(customer_scaled, params)
    X_2d = reduce_pca(X)
    plot_clusters_pca(X_2d, df_clustered["Cluster"].values, params)
    save_table(profile, "cluster_profile.csv", params)

    # ── 5. Classification ─────────────────────────────────────────
    print("\n[5/6] Running Classification (Segment prediction)...")
    df_processed = load_processed_data(params)
    models, clf_results, le, (X_train, X_test, y_train, y_test) = \
        run_supervised_pipeline(df_processed, params)

    clf_summary = summarize_classification(clf_results.to_dict("records"))
    save_table(clf_summary, "classification_results.csv", params)

    plot_confusion_matrix(
        __import__("src.evaluation.metrics", fromlist=["get_confusion_matrix"])
        .get_confusion_matrix(y_test, models["random_forest"].predict(X_test), list(le.classes_)),
        params,
    )

    fi = get_feature_importance(models["random_forest"], list(X_train.columns))
    plot_feature_importance(fi, params)
    save_table(fi, "feature_importance.csv", params)

    # ── 6. Forecasting ────────────────────────────────────────────
    print("\n[6/6] Running Forecasting (Time Series)...")
    fc_results, forecasts, train, test, residual = \
        run_forecasting_pipeline(monthly, params)

    save_table(fc_results, "forecasting_results.csv", params)
    plot_forecast(train, test, forecasts, params)
    plot_residuals(residual["residuals"], params)

    # ── Summary ───────────────────────────────────────────────────
    print_summary(clf_summary, fc_results, profile)
    print("\n✅ Pipeline completed. Check outputs/ for results.")

if __name__ == "__main__":
    main()
