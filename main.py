from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from sklearn.metrics import r2_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

linear = pickle.load(open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/logistic.pkl", "rb"))
tree = pickle.load(open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/forest.pkl", "rb"))
forest = pickle.load(open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/tree.pkl", "rb"))
scores = pickle.load(open("E:/data science/python for data science/ml/ml_from_basic/basic/supervised/project/backend/models/scores.pkl", "rb"))

@app.get("/scores")
def get_scores():
    return scores

@app.post("/predict/{model_name}")
def predict(model_name: str, data: dict):
    input_data = np.array([[
        data["area_sqft"],
        data["bedrooms"],
        data["bathrooms"],
        data["floors"],
        data["garage"],
        data["location_score"]
    ]])

    if model_name == "linear":
        result = linear.predict(input_data)
    elif model_name == "tree":
        result = tree.predict(input_data)
    elif model_name == "forest":
        result = forest.predict(input_data)
    else:
        return {"error": "Invalid model"}
    

    return {
        "model": model_name,
        "price": float(result[0]),
        "score": round(scores[model_name], 2)
    }