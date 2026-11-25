import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_utils import load_transactions, prepare_basket_data, make_sample_data
from src.recommender import get_rules, recommend_from_cart, item_similarity_recommendations
from src.evaluation import evaluate_rules

st.set_page_config(page_title="Smart Basket AI", page_icon="🛒", layout="wide")
st.markdown("""
<style>
.main {background:#f7f9fc}
[data-testid="stSidebar"] {background:#10233f}
[data-testid="stSidebar"] * {color:white}
.metric-card {background:white;padding:18px;border-radius:12px;border:1px solid #e7edf5}
</style>
""", unsafe_allow_html=True)

if "transactions" not in st.session_state:
    st.session_state.transactions = load_transactions()

df = st.session_state.transactions
st.sidebar.title("🛒 Smart Basket AI")
st.sidebar.caption("Product recommendation mini-project")
page = st.sidebar.radio("Navigation", [
    "Home Dashboard", "Dataset Explorer", "Purchase Pattern Analysis",
    "Smart Shopping Basket", "Algorithm Comparison", "Model Evaluation", "About Project"
])

if page == "Home Dashboard":
    st.title("Smart Shopping Basket Recommendation")
    st.write("Discover products that customers often purchase together and get useful basket recommendations.")
    baskets = df.groupby("InvoiceNo")["Description"].nunique()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions", f"{df.InvoiceNo.nunique():,}")
    c2.metric("Unique products", f"{df.Description.nunique():,}")
    c3.metric("Rows", f"{len(df):,}")
    c4.metric("Avg. items / basket", f"{baskets.mean():.1f}")
    st.subheader("Most purchased products")
    top = df.groupby("Description")["Quantity"].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top, x="Quantity", y="Description", orientation="h",
                           title="Top 10 products by quantity", color="Quantity"), use_container_width=True)
    st.info("Use the sidebar to explore purchase patterns and try the interactive recommendation demo.")

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.write("The app starts with a clearly labelled sample dataset. You can upload a transaction CSV with InvoiceNo, Description and Quantity columns.")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            required = {"InvoiceNo", "Description", "Quantity"}
            if not required.issubset(new_df.columns):
                st.error("CSV must contain: InvoiceNo, Description, Quantity.")
            else:
                st.session_state.transactions = new_df.dropna(subset=["InvoiceNo","Description"]).copy()
                df = st.session_state.transactions
                st.success("Dataset loaded for this session.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing up to 100 rows from {len(df)} total rows.")
    st.download_button("Download current dataset", df.to_csv(index=False), "transactions.csv", "text/csv")

elif page == "Purchase Pattern Analysis":
    st.title("Purchase Pattern Analysis")
    st.write("Association rules identify products that frequently appear in the same shopping basket.")
    min_support = st.slider("Minimum support", 0.01, 0.50, 0.05, 0.01)
    min_conf = st.slider("Minimum confidence", 0.10, 1.00, 0.30, 0.05)
    with st.spinner("Mining association rules..."):
        try:
            rules = get_rules(df, min_support=min_support, min_confidence=min_conf)
            if rules.empty:
                st.warning("No rules found. Try reducing minimum support or confidence.")
            else:
                view = rules[["antecedents_text","consequents_text","support","confidence","lift"]].head(30)
                st.dataframe(view, use_container_width=True)
                st.plotly_chart(px.scatter(rules, x="support", y="confidence", size="lift",
                    hover_data=["antecedents_text","consequents_text"], title="Association rules"), use_container_width=True)
        except Exception as e:
            st.error(str(e))

elif page == "Smart Shopping Basket":
    st.title("Smart Shopping Basket")
    st.write("Select products already in a customer's basket to generate recommendations.")
    products = sorted(df["Description"].dropna().astype(str).unique())
    selected = st.multiselect("Products in current basket", products, default=products[:1])
    method = st.radio("Recommendation method", ["Association Rules (Apriori)", "Item-based similarity"], horizontal=True)
    if st.button("Recommend products", type="primary"):
        if not selected:
            st.warning("Select at least one product.")
        elif method.startswith("Association"):
            rules = get_rules(df, min_support=0.01, min_confidence=0.15)
            recs = recommend_from_cart(selected, rules)
            if recs:
                st.subheader("Recommended products")
                for name, score in recs[:8]:
                    st.write(f"**{name}** — recommendation score: {score:.2f}")
            else:
                st.info("No matching association rule found. Try another basket.")
        else:
            recs = item_similarity_recommendations(df, selected)
            if recs:
                for name, score in recs[:8]:
                    st.write(f"**{name}** — similarity score: {score:.2f}")
            else:
                st.info("No similar products found for this selection.")

elif page == "Algorithm Comparison":
    st.title("Algorithm Comparison")
    st.write("Apriori and FP-Growth are association-rule mining methods. Item-based collaborative filtering recommends items similar to those in the basket.")
    st.markdown("""
| Algorithm | Main idea | Output |
|---|---|---|
| Apriori | Builds frequent itemsets level by level | Association rules |
| FP-Growth | Uses an FP-tree to find frequent itemsets | Association rules |
| Item-based similarity | Finds products bought in similar baskets | Similar products |
""")
    if st.button("Run comparison"):
        results = []
        for algo in ["apriori", "fpgrowth"]:
            try:
                rules = get_rules(df, min_support=0.01, min_confidence=0.10, algorithm=algo)
                results.append({"Algorithm": algo.upper(), "Rules found": len(rules),
                                "Average confidence": rules.confidence.mean() if len(rules) else 0,
                                "Average lift": rules.lift.mean() if len(rules) else 0})
            except Exception as e:
                results.append({"Algorithm": algo.upper(), "Rules found": 0, "Average confidence": 0, "Average lift": 0})
                st.warning(f"{algo}: {e}")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.caption("The number of rules depends on the selected dataset and thresholds.")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("A simple holdout evaluation checks whether rules learned from earlier transactions can suggest items in later transactions.")
    if st.button("Evaluate recommendation rules"):
        with st.spinner("Evaluating..."):
            try:
                result = evaluate_rules(df)
                a,b,c = st.columns(3)
                a.metric("Precision@5", f"{result['precision_at_5']:.3f}")
                b.metric("Recall@5", f"{result['recall_at_5']:.3f}")
                c.metric("Test baskets", result["test_baskets"])
                st.caption("This is a lightweight demonstration metric; results vary with the sample data and split.")
            except Exception as e:
                st.error(str(e))

else:
    st.title("About Project")
    st.subheader("Problem statement")
    st.write("Online stores contain large amounts of transaction data. Finding products commonly purchased together can help recommend relevant items and improve the shopping experience.")
    st.subheader("Objectives")
    st.markdown("- Explore transaction data and identify popular products.\\n- Discover product associations using Apriori and FP-Growth.\\n- Recommend products using association rules and item similarity.\\n- Compare algorithms and report basic evaluation metrics.")
    st.subheader("Technology stack")
    st.write("Python • Streamlit • Pandas • Mlxtend • Scikit-learn • Plotly")
    st.subheader("Important note")
    st.warning("The included dataset is synthetic demonstration data, not real customer data. Replace it with a properly licensed dataset for formal experiments.")
