# ============================================================
# Customer Loan Approval Using Voting Classification
# ============================================================

# Step 1: Import Required Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# Step 2: Load the Dataset
# ============================================================

df = pd.read_csv("Customer_Loan_Approval.csv")

print("First 5 Records:")
print(df.head())


# ============================================================
# Step 3: Check Dataset Information
# ============================================================

print("\nDataset Information:")
print(df.info())


# ============================================================
# Step 4: Check for Missing Values
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# Remove missing values if present
df = df.dropna()


# ============================================================
# Step 5: Separate Input and Output Variables
# ============================================================

X = df[
    [
        "Age",
        "Income",
        "CreditScore",
        "ExistingLoan",
        "EmploymentExperience",
        "LoanAmount"
    ]
]

Y = df["LoanApproved"]


print("\nInput Variables:")
print(X.head())

print("\nOutput Variable:")
print(Y.head())


# ============================================================
# Step 6: Split Dataset into Training and Testing Data
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# Step 7: Apply Standard Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# Step 8: Train Logistic Regression
# ============================================================

lr = LogisticRegression()

lr.fit(X_train_scaled, Y_train)

Y_pred_lr = lr.predict(X_test_scaled)

accuracy_lr = accuracy_score(Y_test, Y_pred_lr)

print("\nLogistic Regression Accuracy:",
      accuracy_lr * 100, "%")


# ============================================================
# Step 9: Train Decision Tree
# ============================================================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, Y_train)

Y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(Y_test, Y_pred_dt)

print("Decision Tree Accuracy:",
      accuracy_dt * 100, "%")


# ============================================================
# Step 10: Train K-Nearest Neighbors
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, Y_train)

Y_pred_knn = knn.predict(X_test_scaled)

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)

print("KNN Accuracy:",
      accuracy_knn * 100, "%")


# ============================================================
# Step 11: Create Hard Voting Classifier
# ============================================================

hard_voting = VotingClassifier(
    estimators=[
        ("Logistic Regression", lr),
        ("Decision Tree", dt),
        ("KNN", knn)
    ],
    voting="hard"
)

# For voting, use the same scaled data
hard_voting.fit(X_train_scaled, Y_train)

Y_pred_hard = hard_voting.predict(X_test_scaled)

accuracy_hard = accuracy_score(Y_test, Y_pred_hard)

print("Hard Voting Accuracy:",
      accuracy_hard * 100, "%")


# ============================================================
# Step 12: Create Soft Voting Classifier
# ============================================================

soft_voting = VotingClassifier(
    estimators=[
        ("Logistic Regression", lr),
        ("Decision Tree", dt),
        ("KNN", knn)
    ],
    voting="soft"
)

soft_voting.fit(X_train_scaled, Y_train)

Y_pred_soft = soft_voting.predict(X_test_scaled)

accuracy_soft = accuracy_score(Y_test, Y_pred_soft)

print("Soft Voting Accuracy:",
      accuracy_soft * 100, "%")


# ============================================================
# Step 13: Compare All Models
# ============================================================

print("\n==========================================")
print("           MODEL COMPARISON")
print("==========================================")

print("Logistic Regression :", accuracy_lr * 100, "%")
print("Decision Tree       :", accuracy_dt * 100, "%")
print("KNN                 :", accuracy_knn * 100, "%")
print("Hard Voting         :", accuracy_hard * 100, "%")
print("Soft Voting         :", accuracy_soft * 100, "%")


# ============================================================
# Step 14: Display Comparison in Table Format
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Hard Voting",
        "Soft Voting"
    ],
    "Accuracy": [
        accuracy_lr * 100,
        accuracy_dt * 100,
        accuracy_knn * 100,
        accuracy_hard * 100,
        accuracy_soft * 100
    ]
})

print("\nFinal Comparison Table:")
print(comparison.to_string(index=False))


# ============================================================
# Step 15: Find Best Model
# ============================================================

accuracies = {
    "Logistic Regression": accuracy_lr,
    "Decision Tree": accuracy_dt,
    "KNN": accuracy_knn,
    "Hard Voting": accuracy_hard,
    "Soft Voting": accuracy_soft
}

best_model = max(accuracies, key=accuracies.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", accuracies[best_model] * 100, "%")