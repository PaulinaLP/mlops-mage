import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    response = requests.get(
                'https://github.com/PaulinaLP/mlops-mage/raw/main/yellow_tripdata_2023_03.parquet'
            )

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))       
    print (df.columns)     

    return df
