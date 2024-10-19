from predict import predict
from fastapi import FastAPI
from pydantic import BaseModel
import json


app = FastAPI()

@app.get("/api/predict/{serialized_observation}")
def get_prediction(serialized_observation:str):
    observation = json.loads(serialized_observation)
    action = predict(observation)
    return {'action': action}