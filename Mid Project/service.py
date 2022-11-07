import bentoml
from bentoml.io import JSON, NumpyNdarray, PandasDataFrame
import pandas as pd


# Here we define the variables to use in the prediction
numerical=["hour","temp","feelslike","dew","humidity","cloudcover","visibility"]
categorical=["WorkDay","Holiday","conditions"]
col_names=numerical+categorical
arr=pd.DataFrame([[False, False, 'Partially cloudy', 0, 7.3, 7.3, 5.1, 85.95, 62.6,12.2]],columns=col_names)
input_spec = PandasDataFrame.from_sample(arr)

model=bentoml.sklearn.get("demand_prediction_service:tvopn7c3t6pohqr2")

model_runner = model.to_runner()
svc = bentoml.Service("demand_prediction_model",runners=[model_runner],)

#Here we need to use pandas dataframe since the predict function takes a 2D array as input
#so we provide the type data that it accepts
@svc.api(input=PandasDataFrame(
    columns=col_names,
    orient="records",
    enforce_shape=True,
    #dtype={"hour":int,"temp":float,"feelslike":float,"dew":float,"humidity":float,"cloudcover":float,"visibility":float,"WorkDay":bool,"Holiday":bool,"conditions":str},
    enforce_dtype=False,
    shape=(-1,10),
), output=JSON())
async def predict(data: pd.DataFrame):
    #print(data,type(data))
    prediction = await model_runner.predict.async_run(data)
    return {"prediction":prediction}

