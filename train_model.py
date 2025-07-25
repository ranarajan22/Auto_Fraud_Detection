import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import zscore
from preprocess import preprocess_data

# --- Step 1: Load and combine datasets ---
file_paths = [
    "insurance.csv",
    "data/Auto_Insurance_Fraud_Claims_File02.csv",
    "data/Auto_Insurance_Fraud_Claims_File03.csv",
]

dfs = []
for path in file_paths:
    try:
        df = pd.read_csv(path)
        print(f"📂 Loaded '{path}' with shape: {df.shape}")
        dfs.append(df)
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")

# Combine all datasets
df = pd.concat(dfs, ignore_index=True)
print(f"\n🔗 Combined dataset shape: {df.shape}")

# --- Step 2: Clean the data ---

# Missing values
missing_values = df.isnull().sum().sum()
print(f"🚫 Total missing values: {missing_values}")
df.dropna(inplace=True)

# Duplicates
duplicates = df.duplicated().sum()
print(f"📦 Duplicate rows: {duplicates}")
df.drop_duplicates(inplace=True)

# Outliers using Z-score
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
z_scores = df[numeric_cols].apply(zscore)
outliers = (abs(z_scores) > 3).any(axis=1).sum()
print(f"📊 Rows with outliers (|z| > 3): {outliers}")
df = df[(abs(z_scores) <= 3).all(axis=1)]

print(f"✅ Final cleaned dataset shape: {df.shape}")

# --- Step 3: Preprocess ---
df_processed = preprocess_data(df, for_training=True)

X = df_processed.drop("Fraud_Ind", axis=1)
y = df_processed["Fraud_Ind"]

# Save feature names
joblib.dump(X.columns.tolist(), "model_features.pkl")

# --- Step 4: Split and train ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=15, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# --- Step 5: Evaluate ---
y_pred = model.predict(X_test)
print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("🧾 Classification Report:\n", classification_report(y_test, y_pred))

# --- Step 6: Save model ---
joblib.dump(model, "fraud_model.pkl")
print("✅ Model and features saved successfully.")
