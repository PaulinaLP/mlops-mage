from typing import Tuple
import mlflow
import mlflow.sklearn

import pandas as pd

from mlops.utils.models.regression import training

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    prev_inputs: Tuple, **kwargs
):    
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        model_path = "models/" + name
        vect_path = "vect/" + name        
        dfs=prev_inputs[0]    
        vectorizer=prev_inputs[1]
        model= training (dfs[0], dfs[1])
        mlflow.sklearn.log_model('model', artifact_path=model_path)
        mlflow.log_artifact('vect', artifact_path=vect_path)
        artifacts=[vectorizer, model]


    return artifacts