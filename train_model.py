import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load the data
data = pd.read_csv("employee_data.csv")

# 2️⃣ Split into Features (X) & Target (y)
X = data.drop(columns=["Attrition"])  # Everything except the target
y = data["Attrition"]  # Target column

# 3️⃣ Split into Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Standardize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Save the Model & Scaler
with open("attrition_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved!")
