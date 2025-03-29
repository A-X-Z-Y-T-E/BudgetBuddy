import pandas as pd
import random
from datetime import datetime, timedelta

# Configuration
START_BALANCE = 10000.00
START_DATE = datetime(2023, 1, 1)
END_DATE = START_DATE + timedelta(days=365)

# Categories and their transaction types
CATEGORIES = {
    1: {"description": "Food", "range": (5, 50), "daily_prob": 0.8, "type": "debit"},
    2: {"description": "Transport", "range": (2, 30), "daily_prob": 0.6, "type": "debit"},
    3: {"description": "Bills", "range": (50, 300), "fixed_days": [1, 15], "type": "debit"},
    4: {"description": "Entertainment", "range": (10, 150), "daily_prob": 0.3, "type": "debit"},
    5: {"description": "Shopping", "range": (20, 500), "daily_prob": 0.2, "type": "debit"},
    6: {"description": "Salary", "range": (2500, 3500), "fixed_days": [1], "type": "credit"},
    7: {"description": "Refund", "range": (20, 200), "daily_prob": 0.05, "type": "credit"},
    8: {"description": "Investment", "range": (100, 1000), "daily_prob": 0.02, "type": "credit"}
}

# Generate transactions
transactions = []
balance = START_BALANCE
current_date = START_DATE

while current_date <= END_DATE:
    # Process all categories for each day
    for category_id, params in CATEGORIES.items():
        # Handle fixed day transactions (bills/salary)
        if params.get("fixed_days") and current_date.day in params["fixed_days"]:
            amount = round(random.uniform(*params["range"]), 2)
            
            if params["type"] == "credit":
                balance += amount
            else:  # debit
                if amount > balance:
                    amount = round(balance * 0.9, 2)
                balance -= amount
            
            transactions.append({
                "transaction_id": "",
                "date": current_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "type": params["type"],
                "category_description": params["description"],
                "balance_left": round(balance, 2)
            })
            continue
            
        # Handle probabilistic transactions
        if random.random() < params.get("daily_prob", 0):
            amount = round(random.uniform(*params["range"]), 2)
            
            if params["type"] == "credit":
                balance += amount
            else:  # debit
                if amount > balance:
                    amount = round(balance * random.uniform(0.1, 0.9), 2)
                if amount >= 1:  # Skip tiny transactions
                    balance -= amount
                else:
                    continue
            
            transactions.append({
                "transaction_id": "",
                "date": current_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "type": params["type"],
                "category_description": params["description"],
                "balance_left": round(balance, 2)
            })
    
    current_date += timedelta(days=1)

# Create DataFrame and sort by date
df = pd.DataFrame(transactions)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Assign transaction IDs
debit_count = 1
credit_count = 1
for i, row in df.iterrows():
    if row['type'] == "debit":
        df.at[i, 'transaction_id'] = f"DBT-{debit_count}"
        debit_count += 1
    else:
        df.at[i, 'transaction_id'] = f"CRD-{credit_count}"
        credit_count += 1

# Reorder columns
df = df[['transaction_id', 'date', 'amount', 'type', 'category_description', 'balance_left']]

# Save to CSV
csv_path = "transactions_with_types.csv"
df.to_csv(csv_path, index=False)

print(f"Generated {len(df)} transactions")
print(f"Final balance: ${balance:.2f}")
print("\nSample data:")
print(df.sample(10))