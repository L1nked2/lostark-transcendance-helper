from predict import predict
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json


app = FastAPI()

@app.get("/api/predict/{serialized_observation}")
def get_prediction(serialized_observation:str):
    try:
      obserbation_dict = json.loads(serialized_observation)
      observation = np.concatenate([np.atleast_1d(v).flatten() for v in obserbation_dict.values()])
      predict(observation)
    except Exception as e:
      return {'Error': e} 