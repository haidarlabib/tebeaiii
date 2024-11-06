import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import joblib

# Load dataset
file_path = "nutrisi_minuman_dataset.xlsx"  # Pastikan file ini ada di direktori yang sama dengan skrip
df = pd.read_excel(file_path)

# Encoding 'Jenis Kelamin' to numerical values (Pria = 1, Wanita = 0)
df["Jenis Kelamin"] = df["Jenis Kelamin"].map({"Pria": 1, "Wanita": 0})

# Prepare features and target
X = df[["Usia", "Jenis Kelamin", "Takaran Saji (ml)", "Jumlah Sajian per Kemasan", 
        "Energi Total (kcal)", "Energi dari Lemak (kcal)", "Lemak Total (g)", 
        "Lemak Jenuh (g)", "Kolesterol (mg)", "Karbohidrat Total (g)", "Gula (g)", 
        "Garam (g)", "Protein (g)"]]

y = df["Label Kesehatan"].map({"Sehat": 1, "Tidak Sehat": 0})  # Encoding target for binary classification

# Split data with adjustable test size
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Using adjustable data split
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix (Manual Display)
conf_matrix = confusion_matrix(y_test, y_pred)
TP, FN, FP, TN = conf_matrix.ravel()

# Accuracy and Precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(f"True Positive (TP): {TP}")
print(f"False Negative (FN): {FN}")
print(f"False Positive (FP): {FP}")
print(f"True Negative (TN): {TN}")
print("\nAccuracy:", accuracy)
print("Precision:", precision)

# Save the model to a .pkl file
model_path = "random_forest_model.pkl"
joblib.dump(model, model_path)
print(f"Model has been saved to {model_path}")
