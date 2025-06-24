# 🔹 Required imports
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Load the saved model, features, and scaler
model_top = joblib.load('xgboost_top_model.joblib')
top_features = joblib.load('selected_features.joblib')
scaler = joblib.load('scaler.joblib')

# 🔹 Load the data again
df = pd.read_csv(r"C:\Users\momin\OneDrive\Desktop\Git_Projects\Credit_card_default\working_credit_card_instances_data.csv")
df = df.drop(columns=['ID'])
X = df.drop(columns=['default payment next month'])
y = df['default payment next month']

# 🔹 Transform test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
X_test_top = X_test_scaled[top_features]

# 🔹 Make predictions
y_pred = model_top.predict(X_test_top)

# 🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
cm_df = pd.DataFrame([[TN, FP], [FN, TP]],
                     columns=['Predicted 0', 'Predicted 1'],
                     index=['Actual 0', 'Actual 1'])
cm_df.to_csv("confusion_matrix_table.csv")
print("📁 Confusion matrix saved to 'confusion_matrix_table.csv'")

# 🔹 Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("model_classification_report.csv")
print("📁 Classification report saved to 'model_classification_report.csv'")

# 🔹 Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# 🔹 Visualize Selected Classification Metrics
filtered_report = report_df.loc[['0', '1', 'weighted avg'], ['precision', 'recall', 'f1-score']]
filtered_report.plot(kind='bar', figsize=(8, 5))
plt.title("Model Metrics: Class 0, Class 1, Weighted Avg")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 🔹 Save Top Features with Importance (reload and calculate)
from xgboost import XGBClassifier
model_all = XGBClassifier()
model_all._Booster = model_top.get_booster()  # clone booster

importance_dict = model_all.get_booster().get_score(importance_type='weight')

# Build importance DataFrame
importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Importance': list(importance_dict.values())
}).sort_values(by='Importance', ascending=False)

importance_df['Rank'] = range(1, len(importance_df) + 1)
importance_df.to_csv("xgboost_top_feature_importance.csv", index=False)
print("📁 Top features saved to 'xgboost_top_feature_importance.csv'")
