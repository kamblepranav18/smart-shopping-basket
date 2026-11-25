import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from .data_utils import prepare_basket_data

def get_rules(df, min_support=0.05, min_confidence=0.3, algorithm="apriori"):
    basket = prepare_basket_data(df)
    if basket.empty or basket.shape[1] < 2:
        return pd.DataFrame()
    miner = fpgrowth if algorithm == "fpgrowth" else apriori
    frequent = miner(basket, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame()
    try:
        rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    except TypeError:
        rules = association_rules(frequent, num_itemsets=len(basket), metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules
    rules["antecedents_text"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_text"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules.sort_values(["lift","confidence"], ascending=False)

def recommend_from_cart(selected, rules):
    if rules.empty:
        return []
    selected = set(selected)
    scores = {}
    for _, row in rules.iterrows():
        if set(row["antecedents"]).issubset(selected):
            for item in row["consequents"]:
                if item not in selected:
                    scores[item] = max(scores.get(item, 0), float(row["confidence"] * row["lift"]))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def item_similarity_recommendations(df, selected):
    basket = prepare_basket_data(df)
    if basket.empty:
        return []
    scores = {}
    for item in selected:
        if item not in basket.columns:
            continue
        similarity = basket.corrwith(basket[item]).drop(labels=[item], errors="ignore").dropna()
        for other, score in similarity.items():
            if other not in selected and score > 0:
                scores[other] = max(scores.get(other, 0), float(score))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
