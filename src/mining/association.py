import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Chuẩn hoá giỏ hàng: mỗi Order ID là 1 giao dịch, item là Sub-Category.
def build_basket(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    basket_by  = params["mining"]["association"]["basket_by"]
    item_col   = params["mining"]["association"]["item_column"]

    basket = df.groupby([basket_by, item_col])["Sales"] \
               .sum().unstack(fill_value=0)
    basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)
    return basket

# Chuyển danh sách giao dịch dạng list-of-lists sang ma trận one-hot.
def encode_transactions(transactions: list) -> pd.DataFrame:
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)

# Chạy thuật toán Apriori, trả về frequent itemsets thoả min_support.
def run_apriori(basket: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_support = params["mining"]["association"]["min_support"]
    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    return frequent_itemsets.sort_values("support", ascending=False).reset_index(drop=True)

# Sinh luật kết hợp từ frequent itemsets, lọc theo min_confidence và min_lift.
def run_association_rules(frequent_itemsets: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_confidence = params["mining"]["association"]["min_confidence"]
    min_lift       = params["mining"]["association"]["min_lift"]
    metric         = params["mining"]["association"]["metric"]

    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_lift,
    )
    rules = rules[rules["confidence"] >= min_confidence]
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules

# Lấy top N luật kết hợp mạnh nhất theo lift.
def get_top_rules(rules: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return rules[cols].head(n)

# Chuyển frozenset sang string để dễ đọc khi in báo cáo.
def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules

# Lọc các luật gợi ý cross-sell: antecedent thuộc category A, consequent thuộc category B.
def filter_crosssell_rules(rules: pd.DataFrame, cat_a: str, cat_b: str) -> pd.DataFrame:
    mask = (
        rules["antecedents"].apply(lambda x: cat_a in x) &
        rules["consequents"].apply(lambda x: cat_b in x)
    )
    return rules[mask].reset_index(drop=True)

# Chạy toàn bộ pipeline luật kết hợp, trả về (frequent_itemsets, rules).
def run_association_pipeline(df: pd.DataFrame, params: dict) -> tuple:
    basket            = build_basket(df, params)
    frequent_itemsets = run_apriori(basket, params)
    rules             = run_association_rules(frequent_itemsets, params)
    print(f"[association] Frequent itemsets: {len(frequent_itemsets)} | Rules: {len(rules)}")
    return frequent_itemsets, rules
