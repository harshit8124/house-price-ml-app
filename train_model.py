import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os

# ensure folder exists
# os.makedirs("models", exist_ok=True)

df = pd.read_csv("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/house_price_dataset_1000_rows.csv")

# dataset (area, bedrooms, bathrooms)
X = df.drop(["age", "price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LinearRegression()
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

y_pred_linear = log_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)
y_pred_forest = forest_model.predict(X_test)

scores = {
    "linear": r2_score(y_test, y_pred_linear),
    "tree": r2_score(y_test, y_pred_tree),
    "forest": r2_score(y_test, y_pred_forest)
}

pickle.dump(log_model, open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/forest.pkl", "wb"))
pickle.dump(tree_model, open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/logistic.pkl", "wb"))
pickle.dump(forest_model, open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/tree.pkl", "wb"))
pickle.dump(scores, open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/scores.pkl", "wb"))

print("Models saved correctly")
