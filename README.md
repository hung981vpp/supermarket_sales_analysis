# 🛒 Supermarket Sales Analysis

Bài tập lớn môn **Dữ liệu lớn, Khai phá dữ liệu** — Học kỳ II năm học 2025–2026.
Đề tài 1: Phân tích doanh số siêu thị (Kaggle Superstore Sales Dataset).

---

## 👥 Thành viên nhóm

| MSV | Họ tên | Lớp |
|-----|--------|-----|
| 1771020333 | Đàm Vĩnh Hưng | CNTT 17-10 |
| 1771020066 | Vương Thị Ngọc Ánh | CNTT 17-10 |
| 1771020060 | Phạm Thị Yến Anh | CNTT 17-10 |

---

## 📋 Mô tả đề tài

Xây dựng pipeline khai phá dữ liệu doanh số siêu thị bao gồm:

- **Luật kết hợp** — Apriori trên giỏ hàng theo hoá đơn, gợi ý combo/cross-sell
- **Phân cụm** — KMeans + HAC trên đặc trưng RFM, hồ sơ cụm khách hàng
- **Phân lớp** — Dự đoán phân khúc khách hàng (LogReg / DT / RF)
- **Chuỗi thời gian** — Dự báo doanh số theo tháng (Naive / MA / ARIMA / Holt-Winters / Prophet)

---

## 📦 Dataset

- **Nguồn:** [Kaggle — Superstore Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
- **File:** `data/raw/train.csv`
- **Kích thước:** ~1906 dòng × 18 cột
- **Thời gian:** 2015 – 2018

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| Order ID | string | Mã đơn hàng |
| Order Date | date | Ngày đặt hàng (dd/mm/yyyy) |
| Ship Mode | categorical | Hình thức vận chuyển |
| Customer ID | string | Mã khách hàng |
| Segment | categorical | Phân khúc: Consumer / Corporate / Home Office |
| Region | categorical | Vùng: South / West / Central / East |
| Category | categorical | Furniture / Office Supplies / Technology |
| Sub-Category | categorical | 17 sub-category |
| Sales | float | Doanh thu đơn hàng (USD) — target forecasting |

> ⚠️ Dataset không có cột Profit, Discount, Quantity.

---

## 🗂️ Cấu trúc repo

```
supermarket_sales_analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml          # seed, paths, hyperparams
├── data/
│   ├── raw/                 # train.csv (Kaggle)
│   └── processed/           # train_cleaned.csv (sau tiền xử lý)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation_report.ipynb
│   └── runs/                # notebook output (papermill)
├── src/
│   ├── data/                # loader.py, cleaner.py
│   ├── features/            # builder.py (RFM, time features)
│   ├── mining/              # association.py, clustering.py
│   ├── models/              # supervised.py, forecasting.py
│   ├── evaluation/          # metrics.py, report.py
│   └── visualization/       # plots.py
├── scripts/
│   ├── run_pipeline.py      # chạy toàn bộ pipeline 1 lệnh
│   └── run_papermill.py     # chạy notebook tự động
└── outputs/
    ├── figures/             # biểu đồ .png
    ├── tables/              # kết quả .csv
    └── models/              # model .pkl
```

---

## ⚙️ Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/hung981vpp/supermarket_sales_analysis.git
cd supermarket_sales_analysis

# 2. Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Cài thư viện
pip install -r requirements.txt
```

---

## 🚀 Chạy pipeline

### Cách 1 — Chạy toàn bộ bằng 1 lệnh
```bash
python scripts/run_pipeline.py
```

### Cách 2 — Chạy từng notebook theo thứ tự
```bash
python scripts/run_papermill.py
```

### Cách 3 — Chạy thủ công từng notebook
```bash
jupyter lab
# Mở và chạy lần lượt notebooks/01 → 05
```

---

## 📊 Kết quả

| Task | Model tốt nhất | Metric |
|------|---------------|--------|
| Phân lớp Segment | Random Forest | F1-macro, ROC-AUC |
| Dự báo doanh số | Prophet | MAE, RMSE, sMAPE |
| Phân cụm | KMeans (k=4) | Silhouette, DBI |
| Luật kết hợp | Apriori | Support, Confidence, Lift |

> Kết quả chi tiết xem trong `outputs/tables/` và `reports/final_report.pdf`.

---

## 🔧 Cấu hình tham số

Toàn bộ tham số pipeline được khai báo trong `configs/params.yaml`:

```yaml
general:
  seed: 42
mining:
  association:
    min_support: 0.02
    min_confidence: 0.3
  clustering:
    n_clusters: 4
models:
  forecasting:
    model: prophet
    forecast_periods: 6
```

---

## 📝 Báo cáo

Báo cáo đầy đủ tại `reports/final_report.pdf`.
