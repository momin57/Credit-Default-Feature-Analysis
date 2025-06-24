# Step 1: Import libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Load and prepare the dataset
df = pd.read_csv(r"C:\Users\momin\OneDrive\Desktop\Git_Projects\Credit_card_default\working_credit_card_instances_data.csv")
df = df.drop(columns=['ID'])

# Step 3: Split features and target
X = df.drop(columns=['default payment next month'])
y = df['default payment next month']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Step 6: Train initial XGBoost model with all features
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
model_all = XGBClassifier(random_state=42, scale_pos_weight=pos_weight,
                          use_label_encoder=False, eval_metric='logloss',
                          n_estimators=100, max_depth=4, learning_rate=0.1)
# model_all = XGBClassifier(random_state=42, scale_pos_weight=pos_weight,
#                           use_label_encoder=False, eval_metric='logloss')
model_all.fit(X_train_scaled, y_train)

# Step 7: Get and sort feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_all.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Add Rank column
importance_df['Rank'] = range(1, len(importance_df) + 1)

# Save full feature importance list
importance_df.to_csv("xgboost_feature_importance_list.csv", index=False)
print("📁 Feature importance saved to 'xgboost_feature_importance.csv'")
print(importance_df)

# Optional: Visualize top 15 features
plt.figure(figsize=(10, 6))
plot_importance(model_all, max_num_features=15)
plt.title("Top 15 Important Features - XGBoost")
plt.tight_layout()
plt.show()

# Step 8: Select top N features
N = 9
top_features = importance_df['Feature'].head(N).tolist()

# Subset scaled data with top features
X_train_top = X_train_scaled[top_features]
X_test_top = X_test_scaled[top_features]

# Step 9: Retrain XGBoost using only top features
model_top = XGBClassifier(random_state=42, scale_pos_weight=pos_weight,
                          use_label_encoder=False, eval_metric='logloss',
                          n_estimators=100, max_depth=4, learning_rate=0.1)
model_top.fit(X_train_top, y_train)

# Step 10: Evaluate model
y_pred = model_top.predict(X_test_top)
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Save model, features, and scaler
joblib.dump(model_top, 'xgboost_top_model.joblib')
joblib.dump(top_features, 'selected_features.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("✅ Model, top features, and scaler saved successfully.")

# 🔹 Get importance used by plot_importance (weight-based)
booster = model_all.get_booster()
importance_weight = booster.get_score(importance_type='weight')

# Convert to DataFrame and sort
importance_weight_df = pd.DataFrame(importance_weight.items(), columns=['Feature', 'Importance'])
importance_weight_df = importance_weight_df.sort_values(by='Importance', ascending=False)
importance_weight_df['Rank'] = range(1, len(importance_weight_df) + 1)

# Save to CSV
importance_weight_df.to_csv("xgboost_plot_importance_weight.csv", index=False)
print("📁 Feature importance used in bar chart saved to 'xgboost_plot_importance_weight.csv'")
