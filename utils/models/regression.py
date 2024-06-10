import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def training(
    df_clean: pd.DataFrame,
    df_vect: pd.DataFrame
):
 # Initialize and fit the linear regression model
    regressor = LinearRegression()
    print("starting fit")
    regressor.fit(df_vect, df_clean['duration'])

    # Predict durations on training data
    print("starting predict")
    predictions_train = regressor.predict(df_vect)
    rmse_train = np.sqrt(mean_squared_error(df_clean['duration'], predictions_train))
    print("RMSE on training data:", rmse_train)
    print("Intercept:", regressor.intercept_)
    return regressor