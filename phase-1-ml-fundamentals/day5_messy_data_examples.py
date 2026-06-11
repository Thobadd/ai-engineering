import pandas as pd
import numpy as np

# Example 1 - Missing values
print("=== MESSY DATA EXAMPLE 1: Missing Values ===")
messy_df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'age': [25, None, 35, 40, 28],
    'email': ['alice@email.com', 'bob@email.com', 'carol@email.com', None, 'eve@email.com'],
    'salary': [50000, 60000, 75000, None, 55000]
})
print(messy_df)
print("\nMissing values per column:")
print(messy_df.isnull().sum())

# Example 2 - Wrong data types
print("\n=== MESSY DATA EXAMPLE 2: Wrong Data Types ===")
messy_df2 = pd.DataFrame({
    'user_id': ['1', '2', '3', '4', '5'],  # Should be integers
    'purchase_date': ['01/15/2024', '2024-01-16', '15-01-2024', '01/17/2024', '2024/01/18'],  # Inconsistent formats
    'amount': ['$100', '$250', '75', '$300', '120'],  # Mixed currency and numbers
    'active': ['yes', 'no', 'Y', 'false', '1']  # All different ways to say yes/no
})
print(messy_df2)

# Example 3 - Duplicates
print("\n=== MESSY DATA EXAMPLE 3: Duplicates ===")
messy_df3 = pd.DataFrame({
    'customer_id': [1, 2, 1, 3, 2, 4, 1],
    'name': ['John', 'Jane', 'John', 'Bob', 'Jane', 'Alice', 'John'],
    'purchase': [100, 200, 100, 150, 200, 300, 100]
})
print(messy_df3)
print("\nDuplicate rows:")
print(messy_df3[messy_df3.duplicated()])

# Example 4 - Inconsistent spacing/capitalization
print("\n=== MESSY DATA EXAMPLE 4: Inconsistent Text ===")
messy_df4 = pd.DataFrame({
    'category': ['Electronics', 'ELECTRONICS', ' electronics', 'Electronic', 'ELECTRO NICS'],
    'status': [' active ', 'Active', 'ACTIVE', ' active', 'active ']
})
print(messy_df4)

# Example 5 - Outliers (extreme values)
print("\n=== MESSY DATA EXAMPLE 5: Outliers ===")
messy_df5 = pd.DataFrame({
    'age': [25, 28, 150, 32, -5, 27, 999, 23],  # 150, -5, 999 are unrealistic
    'salary': [50000, 60000, 75000, 55000, 9999999, 62000, 48000, 71000]  # 9999999 is unrealistic
})
print(messy_df5)