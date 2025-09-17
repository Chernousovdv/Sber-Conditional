import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Type
from statsmodels.tsa.api import VAR


class DummyConditionalModel:
    """
    A placeholder model for demonstration.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.params = kwargs

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):
        conditioning_col = columns[0]

        # This model can predict all columns except the one it's conditioned on
        predictable_cols = [col for col in self.all_columns if col != conditioning_col]

        predicted_series_list = []
        for col in predictable_cols:
            last_val = self.df[col].iloc[-1]
            prediction = (np.full(horizon, last_val)).tolist()
            predicted_series_list.append(prediction)

        return predictable_cols, predicted_series_list


class VARModel:
    """
    A Vector Autoregression (VAR) model.

    This model captures the linear interdependencies among multiple time series.
    It's a more sophisticated, standard forecasting model. Note that this
    implementation provides an unconditional forecast and does not use the
    future `values` passed to make_prediction, per the user request.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.params = kwargs
        self.model_fit = None
        try:
            model = VAR(df)
            self.model_fit = model.fit(maxlags=self.params.get("maxlags", 15), ic="aic")
        except Exception as e:
            print(f"Warning: VAR model could not be fitted. Error: {e}")

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):
        if not self.model_fit:
            return [], []

        lag_order = self.model_fit.k_ar

        forecast_input = self.df.values[-lag_order:]

        forecast = self.model_fit.forecast(y=forecast_input, steps=horizon)


        predicted_series_list = forecast.T.tolist()

        predictable_cols = self.all_columns

        return predictable_cols, predicted_series_list
