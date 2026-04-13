import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/house_price_dataset_1000_rows.csv")

# dataset (area, bedrooms, bathrooms)
X = df.drop(["age", "price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe_lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipe_tree_model = Pipeline([
    ("model", DecisionTreeRegressor(random_state=42))
])

pipe_random_model = Pipeline([
    ("model", RandomForestRegressor(random_state=42))
])

models = {
    "linear_model" : pipe_lr_model,
    "tree_model" : pipe_tree_model,
    "random_model" : pipe_random_model
}

for name, model in models.items():
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # print(name, " ", cv_score)
    print(name, "Mean R2:", cv_score.mean())


params_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

random_pipeline = Pipeline([
    ("model", RandomForestRegressor(random_state=42))
])

grid = GridSearchCV(
    estimator=random_pipeline,
    param_grid=params_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
feature_names = X.columns
importance = best_model.named_steps["model"].feature_importances_
print(importance)
print(r2_score(y_test, y_pred))

def fun_new_data(data):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(data)
    importance = best_model.named_steps["model"].feature_importances_
    print(importance)
    print(y_pred)

fun_new_data(X_test)