import os
import sys
import mlflow
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from mlflow import pyfunc

# Setting a tracking uri to log the mlflow logs in a particular location tracked by 
from mlflow.tracking import MlflowClient
tracking_uri = "http://localhost:5005"
client = MlflowClient(tracking_uri=tracking_uri)
mlflow.set_tracking_uri(tracking_uri)


## Model Wrapper that takes 
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self,context):
        import numpy as np
        self.model = np.load(context.artifacts['model_path'], allow_pickle=True).tolist()
        print("Model initialized")
    
    def predict(self, context, inputs):
        import numpy as np
        if len(inputs.shape) == 2:
            print('batch inference')
            predictions = []
            for idx in range(inputs.shape[0]):
                prediction = np.matmul(inputs[idx,:],self.model['weights'].T) + self.model['bias']
                predictions.append(prediction.tolist())
        elif len(inputs.shape) == 1:
            print('single inference')
            predictions = np.matmul(inputs,self.model['weights'].T) + self.model['bias']
            predictions = predictions.tolist()
        else:
            raise ValueError('invalid input shape')
        return predictions


# instantiate the python inference model wrapper for the server
model_wrapper = ModelWrapper()


# define the model weights randomly
np_weights = np.random.rand(3,6)
np_bias = np.random.rand(3)

# checkpointing and logging the model in mlflow
artifact_path = './np_model'
np.save(artifact_path, {'weights':np_weights, 'bias':np_bias})
model_artifacts = {"model_path" : artifact_path+'.npy'}

#Conda environment
env = mlflow.sklearn.get_default_conda_env()
with mlflow.start_run():
    mlflow.log_param("features",6)
    mlflow.log_param("labels",3)
    mlflow.pyfunc.log_model("np_model", python_model=model_wrapper, artifacts=model_artifacts, conda_env=env)