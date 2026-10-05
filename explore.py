# Look at the California housing data before training

from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()

# put everything in one table
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target   # price is in units of $100,000

print("Shape:", df.shape)
print()
print(df.head())
print()
print(df.describe())
