import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import cast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, mean_absolute_error, mean_squared_error,confusion_matrix

# --- 1. SETUP PATHS ---
PROCESSED_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\data\processed\train_clean.csv"
OUTPUTS_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs"
FIGURES_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\figures"
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

TARGET_COLUMN = 'INCOME-IS->50K'

# --- 2. LOAD CLEANED DATA ---
print("Loading cleaned data...")
df = cast(pd.DataFrame, pd.read_csv(PROCESSED_PATH)) # type: ignore

# --- 3. DEFINE FEATURES (X) AND TARGET (y) ---
print("\n---ALL COLUMNS IN YOUR DATASET---")
print(df.columns.tolist())
print("----------------------------------\n")
print("Unique values in target:", df[TARGET_COLUMN].unique())
print("Data type of target:", df[TARGET_COLUMN].dtype)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)

# --- 4. ENCODE CATEGORICAL VARIABLES ---
# Text columns (WORK-CLASS, EDUCATION, GENDER, ...) must become numbers
X = pd.get_dummies(X, drop_first=True, dtype=int)
print(f"Data shape after encoding: {X.shape}")

# --- 5. TRAIN / TEST SPLIT ---
# stratify keeps the same >50K / <=50K ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. TRAIN MODEL ---
print("\nTraining Random Forest Model...")
model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1, min_samples_leaf=5)
model.fit(X_train, y_train)
print("Model training complete!")

# --- 7. MAKE PREDICTIONS ---
predictions = model.predict(X_test)

# --- 8. GENERATE AND SAVE OUTPUTS ---

# A. Save Predictions CSV
PREDICTIONS_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\test_predictions.csv"
results_df = pd.DataFrame({'ACTUAL': y_test.values, 'PREDICTED': predictions})
results_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"[OK] Saved predictions to: {PREDICTIONS_PATH}")

# B. Save Metrics Text File
METRICS_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\model_metrics.txt"
auc = float(roc_auc_score(y_test, predictions))
mae = float(mean_absolute_error(y_test, predictions)) #type:ignore
rmse = np.sqrt(mean_squared_error(y_test, predictions)).item()

with open(METRICS_PATH, "w") as f:
    f.write("--- Model Evaluation ---\n")
    f.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(str(classification_report(y_test, predictions)))
print(f"[OK] Saved metrics to: {METRICS_PATH}")

# C. Save Confusion Matrix Figure
CONFUSION_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\figures\confusion_matrix.png"
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Income is >50K Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(CONFUSION_PATH)
plt.close()
print(f"[OK] Saved confusion matrix figure to: {CONFUSION_PATH}")

# D. Save Feature Importance Figure
FEATURE_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\figures\feature_importance.png"
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df,
            hue='Feature', palette='viridis', legend=False)
plt.title('Top 10 Feature Importances for Income is >50K Prediction')
plt.tight_layout()
plt.savefig(FEATURE_PATH)
plt.close()
print(f"[OK] Saved feature importance figure to: {FEATURE_PATH}")

# E. Save the trained model
MODEL_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\income_model.pkl"
joblib.dump(model, MODEL_PATH, compress=3)
print(f"[OK] Saved trained model to: {MODEL_PATH}")

print("\n--- ALL DONE! Check your outputs folder. ---")