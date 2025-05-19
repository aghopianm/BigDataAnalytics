import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load and prepare the dataset
weather_data = pd.read_csv('weather.nominal.csv', delimiter=',', encoding='utf-8')
features = weather_data.iloc[:, :4]
output = weather_data.iloc[:, 4]
features_encoded = pd.get_dummies(features)

# ========== 1. Full training set (overfitting check) ==========
print("\n=== Training on Full Dataset (Overfitting Scenario) ===")
dt_full = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
dt_full.fit(features_encoded, output)
output_pred_full = dt_full.predict(features_encoded)

print("Confusion Matrix:")
print(confusion_matrix(output, output_pred_full))
print("Classification Report:")
print(classification_report(output, output_pred_full))

# Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(dt_full, feature_names=features_encoded.columns, class_names=dt_full.classes_, filled=True)
plt.title("Decision Tree Trained on Full Dataset")
plt.show()

# ========== 2. Different train-test splits ==========
print("\n=== Train-Test Splits (50%, 60%, 70% Training) ===")
split_sizes = [0.5, 0.4, 0.3]  # Corresponding to 50%, 60%, 70% training

for test_size in split_sizes:
    train_size = 1 - test_size
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, output, test_size=test_size, random_state=0)

    dt_split = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
    dt_split.fit(X_train, y_train)
    y_pred = dt_split.predict(X_test)

    print(f"\n--- Train size: {int(train_size*100)}% ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# ========== 3. Cross-validation with different folds ==========
print("\n=== Cross-Validation (5, 7, 10 Folds) ===")
for k in [5, 7]:
    dt_cv = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
    y_pred_cv = cross_val_predict(dt_cv, features_encoded, output, cv=k)

    print(f"\n--- {k}-Fold Cross-Validation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(output, y_pred_cv))
    print("Classification Report:")
    print(classification_report(output, y_pred_cv))
