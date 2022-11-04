import bentoml
from bentoml.io import JSON, NumpyNdarray, PandasDataFrame
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import numpy as np
numerical=["hour","temp","feelslike","dew","humidity","cloudcover","visibility"]
categorical=["WorkDay","Holiday","conditions"]
col_names=numerical+categorical
arr=pd.DataFrame([[False, False, 'Partially cloudy', 0, 7.3, 7.3, 5.1, 85.95, 62.6,12.2]],columns=col_names)
input_spec = PandasDataFrame.from_sample(arr)

model=bentoml.sklearn.get("demand_prediction_service:tvopn7c3t6pohqr2")

model_runner = model.to_runner()
svc = bentoml.Service("demand_prediction_model",runners=[model_runner],)





# class predictionRecord(BaseModel):    
#     hour: int = 0
#     temp: float = 7.3
#     feelslike: float = 7.3
#     dew: float = 5.1
#     humidity: float = 85.95
#     cloudcover: float = 62.6
#     visibility: float = 12.2
#     WorkDay: bool = False
#     Holiday: bool = False
#     conditions: str = "Partially cloudy"

# class DemandApp(BaseModel):
#     records: List[predictionRecord]

 

# @svc.api(input=JSON(pydantic_model=DemandApp), output=JSON())
# @svc.api(input=JSON(), output=JSON())
#@svc.api(input=NumpyNdarray(shape=(-1,10),enforce_shape=True), output=JSON())
#@svc.api(input=input_spec, output=JSON())
#Here we need to use pandas dataframe since the predict function takes a 2D array as input
#
@svc.api(input=PandasDataFrame(
    columns=col_names,
    orient="records",
    enforce_shape=True,
    dtype={"hour":int,"temp":float,"feelslike":float,"dew":float,"humidity":float,"cloudcover":float,"visibility":float,"WorkDay":bool,"Holiday":bool,"conditions":str},
    enforce_dtype=False,
    shape=(-1,10),
), output=JSON())
def predict(data: pd.DataFrame):
    #print(data,type(data))
    prediction = model_runner.predict.run(data)
    return {"prediction":prediction}

