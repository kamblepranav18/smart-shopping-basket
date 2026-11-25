from sklearn.model_selection import train_test_split
from .recommender import get_rules, recommend_from_cart

def evaluate_rules(df):
    invoices = df["InvoiceNo"].dropna().astype(str).unique()
    if len(invoices) < 4:
        raise ValueError("At least four transactions are needed for evaluation.")
    train_ids, test_ids = train_test_split(invoices, test_size=0.30, random_state=42)
    train = df[df["InvoiceNo"].astype(str).isin(train_ids)]
    test = df[df["InvoiceNo"].astype(str).isin(test_ids)]
    rules = get_rules(train, min_support=0.01, min_confidence=0.10)
    precisions, recalls = [], []
    for _, group in test.groupby("InvoiceNo"):
        actual = set(group["Description"].astype(str))
        if len(actual) < 2:
            continue
        # Use all but one item as the basket and the remaining item as a held-out target.
        target = sorted(actual)[-1]
        basket = sorted(actual - {target})
        recs = recommend_from_cart(basket, rules)[:5]
        predicted = {name for name, _ in recs}
        hit = 1 if target in predicted else 0
        precisions.append(hit / 5)
        recalls.append(hit)
    if not precisions:
        raise ValueError("Not enough multi-item test baskets to calculate metrics.")
    return {
        "precision_at_5": sum(precisions) / len(precisions),
        "recall_at_5": sum(recalls) / len(recalls),
        "test_baskets": len(precisions)
    }
