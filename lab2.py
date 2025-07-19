import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load and clean dataset
dataset = []
with open('lab2-dataset/store_data.csv', 'r') as file:
    for line in file:
        # Remove empty items and strip whitespace
        items = [item.strip() for item in line.strip().split(',') if item.strip() != '']
        if items:
            dataset.append(items)

print("Cleaned Transactions (first 5):\n", dataset[:5])

# Step 2: One-hot encode the transactions
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print("\nOne-Hot Encoded Data (first 5):\n", df.head())

# Step 3: Find frequent itemsets with min support = 0.03 (3%)
frequent_itemsets = apriori(df, min_support=0.03, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# Step 4: Generate association rules with confidence ≥ 0.3
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
print("\nAssociation Rules:\n", rules)

# Step 5: Display rules in readable format
print("\nFormatted Rules:")
for _, row in rules.iterrows():
    print(f"\nRule: {set(row['antecedents'])} -> {set(row['consequents'])}")
    print(f"Support: {row['support']:.2f}")
    print(f"Confidence: {row['confidence']:.2f}")
    print(f"Lift: {row['lift']:.2f}")
