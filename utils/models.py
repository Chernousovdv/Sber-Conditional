import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Type


class DummyConditionalModel:
    """
    A placeholder model for demonstration.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.params = kwargs

    def make_prediction(
        self, horizon: int, columns: List[str], values: List[float]
    ):
        conditioning_col = columns[0]

        # This model can predict all columns except the one it's conditioned on
        predictable_cols = [col for col in self.all_columns if col != conditioning_col]

        predicted_series_list = []
        for col in predictable_cols:
            last_val = self.df[col].iloc[-1]
            prediction = (np.full(horizon, last_val)).tolist()
            predicted_series_list.append(prediction)

        return predictable_cols, predicted_series_list
