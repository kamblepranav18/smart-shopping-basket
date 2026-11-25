# Smart Shopping Basket Recommendation

A beginner-friendly machine-learning mini-project that discovers products frequently purchased together and recommends products for a shopping basket.

## Features
- Dashboard with transaction and product KPIs
- Dataset preview and CSV upload
- Association rule mining with Apriori and FP-Growth
- Basket recommendations using association rules
- Item-based product similarity recommendations
- Algorithm comparison and basic Precision@5 / Recall@5 evaluation

## Tech stack
Python, Streamlit, Pandas, Scikit-learn, Mlxtend and Plotly.

## Project structure
```text
Smart_Shopping_Basket_Recommendation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_transactions.csv
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── recommender.py
    └── evaluation.py
```

## Run locally (Windows / VS Code)
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal and create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate it:
   ```bash
   venv\\Scripts\\activate
   ```
5. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the application:
   ```bash
   streamlit run app.py
   ```

The browser should open the dashboard automatically. If it does not, use the local URL printed in the terminal.

## Dataset format
If uploading your own data, use a CSV with these columns:
- `InvoiceNo`: transaction or basket ID
- `Description`: product name
- `Quantity`: quantity purchased (positive values represent purchases)

The included CSV is synthetic sample data created for demonstration. It is not a real customer dataset.

## GitHub upload
1. Create a new repository on GitHub, for example `smart-shopping-basket-recommendation`.
2. Extract the ZIP file.
3. Open the extracted folder.
4. Upload the project files and folders to the repository, or use Git:
   ```bash
   git init
   git add .
   git commit -m "Initial smart shopping basket project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/smart-shopping-basket-recommendation.git
   git push -u origin main
   ```
5. Replace `YOUR-USERNAME` with your GitHub username.

## Notes
- The app uses a small synthetic dataset so it runs immediately without downloading a dataset.
- Association rules require products to co-occur in enough transactions. If no rules appear, lower the minimum support/confidence sliders or upload a larger dataset.
- Evaluation metrics are demonstration metrics, not a formal benchmark.
