# ==========================================
# Breast Cancer Prediction
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ==========================================
# 1. Load Dataset
# ==========================================

data = load_breast_cancer()

print("Dataset Loaded Successfully!")

print("\nNumber of Records:", data.data.shape[0])
print("Number of Features:", data.data.shape[1])

print("\nFeature Names:")
print(data.feature_names)

print("\nTarget Names:")
print(data.target_names)


# ==========================================
# 2. Create DataFrame
# ==========================================

df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["Target"] = data.target

print("\nFirst 5 Records:")
print(df.head())


# ==========================================
# 3. Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("\nMissing values handled using median.")
else:
    print("\nNo missing values found.")


# ==========================================
# 4. Summary Statistics - EDA
# ==========================================

print("\nSummary Statistics:")
print(df.describe())


# ==========================================
# 5. Feature Correlation
# ==========================================

plt.figure(figsize=(14, 10))

sns.heatmap(
    df.drop("Target", axis=1).corr(),
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()


# ==========================================
# 6. Separate Features and Target
# ==========================================

X = df.drop("Target", axis=1)
y = df["Target"]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


# ==========================================
# 7. Split Dataset into Training and Testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 8. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed.")


# ==========================================
# 9. Build Machine Learning Model
# ==========================================

model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully.")


# ==========================================
# 10. Prediction
# ==========================================

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)


# ==========================================
# 11. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# ==========================================
# 12. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
).plot()

plt.title("Confusion Matrix")
plt.show()


# ==========================================
# 13. Precision, Recall and F1-Score
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ==========================================
# 14. Final Conclusion
# ==========================================

print("\n==========================================")
print("Observations and Conclusion")
print("==========================================")

print("1. Breast Cancer dataset was loaded successfully.")
print("2. Dataset contains 569 records and 30 features.")
print("3. Missing values were checked and handled if present.")
print("4. Feature scaling was performed using StandardScaler.")
print("5. Dataset was divided into training and testing sets.")
print("6. Logistic Regression was used as the classification model.")
print("7. Model was evaluated using Accuracy, Confusion Matrix,")
print("   Precision, Recall and F1-Score.")
print("8. The model can classify tumors as Malignant or Benign.")