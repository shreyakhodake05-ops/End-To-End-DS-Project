import pandas as pd
import numpy as np

# STEP 1: LOAD DATA

df = pd.read_csv("archive/listings.csv", on_bad_lines='skip', low_memory=False)

print("Original Shape:", df.shape)

# STEP 2: DROP USELESS COLUMNS

drop_cols = [
    "listing_id",
    "cover_photo_url",
    "host_id",
    "amenities",
    "city",
    "state",
    "country",
    "currency"
    "registration", 
    "instant_book",   
    "professional_management"
]

df = df.drop(columns=drop_cols, errors='ignore')

# STEP 3: HANDLE MISSING VALUES

# Target missing → remove
df = df[df["ttm_avg_rate"].notna()]

# Numeric fill
num_cols = df.select_dtypes(include=['float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical fill
cat_cols_all = df.select_dtypes(include=['object', 'string']).columns

# Only keep important categorical
keep_cats = ["listing_type", "room_type", "cancellation_policy", "superhost"]

for col in cat_cols_all:
    if col not in keep_cats:
        df = df.drop(col, axis=1)

# Fill remaining
df[keep_cats] = df[keep_cats].fillna("missing")

print("After Cleaning Shape:", df.shape)

# STEP 4: TARGET

y = df["ttm_avg_rate"]
X = df.drop("ttm_avg_rate", axis=1)

# STEP 5: ENCODE ONLY FEW COLUMNS

cat_cols = ["listing_type", "room_type", "cancellation_policy"]

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Convert true/false → 0/1

bool_cols = ["superhost"]

for col in bool_cols:
    if col in X.columns:
        X[col] = X[col].map({
            "true": 1,
            "false": 0,
            "missing": 0
        })

# STEP 6: ONLY CONVERT NUMERIC COLUMNS

#  (no full conversion)
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

# 👉 DO NOT drop all rows
X = X.fillna(0)

print("Final Shape:", X.shape)

# STEP 7: TRAIN MODEL

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)

# STEP 8: EVALUATE

from sklearn.metrics import mean_squared_error

pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, pred))

# STEP 9: SAVE MODEL
import pickle
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model saved")