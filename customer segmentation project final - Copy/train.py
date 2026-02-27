print("🚀 Customer Segmentation Training Started...")

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
print("📊 Loading dataset...")

customer_data = pd.read_csv("data/marketing_campaign.csv")

# -----------------------------
# 2️⃣ Data Cleaning
# -----------------------------
print("🧹 Cleaning missing values...")

customer_data = customer_data.dropna()

# -----------------------------
# 3️⃣ Feature Engineering
# -----------------------------
print("⚙ Creating new features...")

CURRENT_YEAR = 2026

# Calculate Age
customer_data["Age"] = CURRENT_YEAR - customer_data["Year_Birth"]

# Calculate Total Spending
spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

customer_data["Total_Spending"] = customer_data[spending_columns].sum(axis=1)

# -----------------------------
# 4️⃣ Select Features
# -----------------------------
print("🎯 Selecting important features...")

selected_features = [
    "Age",
    "Income",
    "Recency",
    "Total_Spending",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth"
]

X = customer_data[selected_features]

# -----------------------------
# 5️⃣ Feature Scaling
# -----------------------------
print("⚖ Scaling features...")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# -----------------------------
# 6️⃣ Train KMeans Model
# -----------------------------
print("🤖 Training KMeans clustering model...")

kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(scaled_features)

# -----------------------------
# 7️⃣ Save Model & Scaler
# -----------------------------
print("💾 Saving trained model...")

pickle.dump(kmeans_model, open("models/kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("✅ Training Completed Successfully!")