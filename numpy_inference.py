import numpy as np
import json 
import mlflow
# Setting a tracking uri to log the mlflow logs in a particular location tracked by 
from mlflow.tracking import MlflowClient
tracking_uri = "http://localhost:5005"
client = MlflowClient(tracking_uri=tracking_uri)
mlflow.set_tracking_uri(tracking_uri)

model = mlflow.pyfunc.load_model('file:///mnt/cyverse/mlruns/0/ab8f1cce2c964e7c8e6d77d864a9da64/artifacts/np_model')


np_array = np.random.rand(6)

output = model.predict(np_array)

print(output)
