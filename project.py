import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE

variables = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
            ]
 
data = pd.read_csv("adult.data", header = None, names = variables)

# Column education-num represents education
data = data.drop("education", axis = 1)
# fnlwgt column not needed
data = data.drop("fnlwgt", axis = 1)

# Removing spaces in dataset
data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
# Dropping rows with missing values and duplicate rows
data = data.replace("?", np.nan)
data = data.dropna()
data = data.drop_duplicates(keep = False)

# Converting categorical features to dummy features
data = pd.get_dummies(data, columns=["marital-status", "race", "relationship"], dtype = "int")
# Male is 1, Female is 0
data["sex"] = data["sex"].apply(lambda x: 1 if x == "Male" else 0)
# >50K is 1, <=50K is 0
data["income"] = data["income"].apply(lambda x: 1 if x == ">50K" else 0)

for col in ["native-country", "occupation", "workclass"]:
    freq_encoding = data[col].value_counts(normalize = True)
    data[col] = data[col].map(freq_encoding)

# Separating features and target classes
X = data.drop("income", axis = 1)
y = data["income"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Scaling data for regression, SVM, KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
scale_models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

tree_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}

# Train and evaluate models
print("---- Model Performance (All Features) ----")
results = {}
for name, model in scale_models.items():
    accuracy = cross_val_score(model, X_train_scaled, y_train, cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42), scoring = "accuracy")
    mean_accuracy = np.mean(accuracy)
    results[name] = mean_accuracy
    print(f"{name}: {mean_accuracy}")

for name, model in tree_models.items():
    accuracy = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42), scoring = "accuracy")
    mean_accuracy = np.mean(accuracy)
    results[name] = mean_accuracy
    print(f"{name}: {mean_accuracy}")

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X.columns)

# Feature Selection
selector = RFE(RandomForestClassifier(), n_features_to_select = 7)

# For regression, SVM, KNN
X_train_selected = selector.fit_transform(X_train_scaled_df, y_train)
X_test_selected = selector.transform(X_test_scaled_df)

#selected_features = X_train_scaled_df.columns[selector.support_]
selected_features = X.columns[selector.support_]
feature_importances = pd.Series(selector.estimator_.feature_importances_, index=selected_features)
print("\nTop Selected Features:")
print(feature_importances.sort_values(ascending=False))

# For trees
X_train_selected_unscaled = X_train[selected_features]
X_test_selected_unscaled = X_test[selected_features]

# Compare models with selected features
print("\n---- Model Performance (After Feature Selection) ----")
feature_results = {}
for name, model in scale_models.items():
    accuracy = cross_val_score(model, X_train_selected, y_train, cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42), scoring = "accuracy")
    mean_accuracy = np.mean(accuracy)
    feature_results[name] = mean_accuracy
    print(f"{name} (with feature selection): {mean_accuracy}")

for name, model in tree_models.items():
    accuracy = cross_val_score(model, X_train_selected_unscaled, y_train, cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42), scoring = "accuracy")
    mean_accuracy = np.mean(accuracy)
    feature_results[name] = mean_accuracy
    print(f"{name} (with feature selection): {mean_accuracy}")

# Final model evaluation on test set
print("\n---- Final Model Evaluation on Test Set ----")
for name, model in scale_models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

for name, model in tree_models.items():
    model.fit(X_train_selected_unscaled, y_train)
    y_pred = model.predict(X_test_selected_unscaled)
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

# Compare accuracy before and after feature selection
print("\n---- Feature Selection Impact ----")
for name in results:
    before = results[name]
    after = feature_results[name]
    print(f"{name}: Before = {before}, After = {after}, Difference = {after - before}")
