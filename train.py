# Train a random forest on the California housing data and save it

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

print("Loading data...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print("Total rows:", len(X))

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# check how good it is on data it has not seen
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Average error: ${mae * 100000:,.0f}")
print(f"R2 score: {r2:.2f}")

# save the model and the column names for the API
joblib.dump(model, "house_model.joblib")
joblib.dump(list(X.columns), "house_features.joblib")
print("Model saved.")
