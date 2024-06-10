from typing import Tuple

import pandas as pd

from mlops.utils.data_preparation.calculate_duration import clean
from mlops.utils.data_preparation.vectorize_chunk import vectorize_train

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:    

    df_clean = clean(df)    
    print (df_clean.shape)
    df_target=df_clean[['duration']] 
    result= vectorize_train(df_clean)
    dfs=[df_clean, result[0]]
    vect= result[1]
    result_fin=[dfs, vect]

    return result_fin


    