import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"

def make_sample_data():
    baskets = [
        ["Bread","Milk","Butter"], ["Bread","Milk"], ["Milk","Butter","Eggs"],
        ["Bread","Butter","Jam"], ["Milk","Eggs"], ["Bread","Milk","Eggs"],
        ["Coffee","Sugar","Milk"], ["Coffee","Sugar"], ["Bread","Jam"],
        ["Milk","Butter"], ["Tea","Sugar","Biscuits"], ["Tea","Biscuits"],
        ["Coffee","Milk","Biscuits"], ["Bread","Milk","Butter"],
        ["Eggs","Bread"], ["Tea","Sugar"], ["Coffee","Sugar","Biscuits"],
        ["Milk","Bread"], ["Butter","Jam"], ["Bread","Eggs","Milk"],
        ["Mobile Phone","Phone Case"], ["Mobile Phone","Screen Guard"],
        ["Laptop","Laptop Bag"], ["Laptop","Mouse"], ["Mobile Phone","Phone Case","Screen Guard"],
        ["Headphones","Mobile Phone"], ["Laptop","Mouse","Laptop Bag"],
        ["Phone Case","Screen Guard"], ["Headphones","Laptop"], ["Mobile Phone","Headphones"]
    ]
    rows = []
    for i, basket in enumerate(baskets, 1):
        for item in basket:
            rows.append({"InvoiceNo": f"T{i:03d}", "Description": item, "Quantity": 1})
    return pd.DataFrame(rows)

def load_transactions():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return make_sample_data()

def prepare_basket_data(df):
    clean = df.dropna(subset=["InvoiceNo", "Description"]).copy()
    clean["InvoiceNo"] = clean["InvoiceNo"].astype(str)
    clean["Description"] = clean["Description"].astype(str).str.strip()
    clean = clean[clean["Description"] != ""]
    basket = clean.pivot_table(index="InvoiceNo", columns="Description",
                               values="Quantity", aggfunc="sum", fill_value=0)
    return basket.gt(0).astype(int)
