import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("fusion_training_data.csv")

print("Total Samples:", len(df))

X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nLogistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)

# Cross-validation
rf_cv_scores = cross_val_score(rf, X, y, cv=5)
print("\nRandom Forest 5-Fold CV Scores:", rf_cv_scores)
print("Mean CV Accuracy:", rf_cv_scores.mean())

# Classification report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))

# Confusion matrix
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Feature importance
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Save best model
best_model = rf if rf_acc >= lr_acc else lr
joblib.dump(best_model, "fusion_model.pkl")

print("\nModel saved as fusion_model.pkl")
print("Confusion matrix saved as confusion_matrix.png")
print("Feature importance saved as feature_importance.png")
