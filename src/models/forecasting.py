import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Kiểm định tính dừng của chuỗi thời gian bằng ADF test.
def adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna())
    return {
        "adf_statistic": result[0],
        "p_value":       result[1],
        "is_stationary": result[1] < 0.05,
    }

# Tính MAE, RMSE, sMAPE giữa giá trị thực và dự báo.
def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2 * np.abs(y_true - y_pred) /
                    (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}

# Split chuỗi thời gian theo thứ tự (không shuffle), test = n_test tháng cuối.
def time_split(monthly: pd.DataFrame, n_test: int = 6) -> tuple:
    train = monthly.iloc[:-n_test]
    test  = monthly.iloc[-n_test:]
    return train, test

# Baseline Naive: dự báo = giá trị tháng liền trước.
def baseline_naive(train: pd.DataFrame, n_test: int) -> np.ndarray:
    last_value = train["Sales"].iloc[-1]
    return np.full(n_test, last_value)

# Baseline Moving Average: dự báo = trung bình window tháng cuối.
def baseline_moving_average(train: pd.DataFrame, n_test: int, window: int = 3) -> np.ndarray:
    ma_value = train["Sales"].iloc[-window:].mean()
    return np.full(n_test, ma_value)

# Chạy mô hình ARIMA với order lấy từ params.
def run_arima(train: pd.DataFrame, n_test: int, params: dict) -> np.ndarray:
    order = tuple(params["models"]["forecasting"]["arima"]["order"])
    model = ARIMA(train["Sales"], order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=n_test)
    return forecast.values

# Chạy mô hình Holt-Winters (additive trend + seasonal).
def run_holt_winters(train: pd.DataFrame, n_test: int, params: dict) -> np.ndarray:
    cfg = params["models"]["forecasting"]["holt_winters"]
    model = ExponentialSmoothing(
        train["Sales"],
        trend=cfg["trend"],
        seasonal=cfg["seasonal"],
        seasonal_periods=cfg["seasonal_periods"],
    )
    fitted   = model.fit()
    forecast = fitted.forecast(n_test)
    return forecast.values

# Chạy mô hình Prophet (Facebook).
def run_prophet(train: pd.DataFrame, n_test: int) -> np.ndarray:
    from prophet import Prophet
    df_prophet = train.rename(columns={"YearMonth": "ds", "Sales": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False)
    model.fit(df_prophet)
    future   = model.make_future_dataframe(periods=n_test, freq="MS")
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-n_test:].values

# So sánh tất cả models, trả về DataFrame tổng hợp MAE/RMSE/sMAPE.
def compare_forecasts(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.DataFrame:
    n_test   = len(test)
    y_true   = test["Sales"].values
    records  = []

    forecasts = {
        "naive":         baseline_naive(train, n_test),
        "moving_avg":    baseline_moving_average(train, n_test),
        "arima":         run_arima(train, n_test, params),
        "holt_winters":  run_holt_winters(train, n_test, params),
        "prophet":       run_prophet(train, n_test),
    }

    for name, y_pred in forecasts.items():
        metrics = forecast_metrics(y_true, y_pred)
        metrics["model"] = name
        records.append(metrics)
        print(f"[{name}] MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | sMAPE={metrics['sMAPE']:.2f}%")

    df_results = pd.DataFrame(records)[["model", "MAE", "RMSE", "sMAPE"]]
    return df_results.sort_values("RMSE").reset_index(drop=True), forecasts

# Phân tích residual của model tốt nhất: mean, std, kiểm định dừng.
def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    return {
        "mean":          residuals.mean(),
        "std":           residuals.std(),
        "adf":           adf_test(pd.Series(residuals)),
        "residuals":     residuals,
    }

# Chạy toàn bộ pipeline forecasting, trả về (results_df, forecasts, residual_info).
def run_forecasting_pipeline(monthly: pd.DataFrame, params: dict) -> tuple:
    n_test          = params["models"]["forecasting"]["forecast_periods"]
    train, test     = time_split(monthly, n_test)

    print(f"[forecasting] Train: {len(train)} months | Test: {n_test} months")
    print(f"[forecasting] ADF test: {adf_test(train['Sales'])}")

    results, forecasts = compare_forecasts(train, test, params)

    best_model  = results.iloc[0]["model"]
    y_true      = test["Sales"].values
    residual    = residual_analysis(y_true, forecasts[best_model])
    print(f"[forecasting] Best model: {best_model}")

    return results, forecasts, train, test, residual
