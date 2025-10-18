import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Type

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pmdarima as pm
import torch
from prophet import Prophet
from pytorch_forecasting import RecurrentNetwork, TimeSeriesDataSet
from statsmodels.tsa.api import VAR
import pandas as pd
from typing import List, Dict, Any
from prophet import Prophet
import pmdarima as pm
import statsmodels.api as sm
from torch.utils.data import DataLoader
from sktime.forecasting.vecm import VECM


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


class SARIMAModel:
    """
    A model that fits an individual SARIMA model for each time series.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.sarima_params = {
            "m": kwargs.get("m", 12),
            "seasonal": kwargs.get("seasonal", True),
            "stepwise": kwargs.get("stepwise", True),
            "suppress_warnings": kwargs.get("suppress_warnings", True),
            "error_action": kwargs.get("error_action", "ignore"),
        }
        self.fitted_models = {}

        # Fit a separate auto_arima model for each column
        for col in self.all_columns:
            model = pm.auto_arima(df[col], **self.sarima_params)
            self.fitted_models[col] = model

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):
        """
        Unconditional forecast.
        """
        predictable_cols = []
        predicted_series_list = []

        for col, model in self.fitted_models.items():
            forecast = model.predict(n_periods=horizon)
            predicted_series_list.append(forecast.tolist())
            predictable_cols.append(col)

        return predictable_cols, predicted_series_list


class ProphetModel:
    """
    It fits an independent Prophet model for each time series variable in the df
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.params = kwargs
        self.fitted_models = {}

        for col in self.all_columns:
            prophet_df = pd.DataFrame({"ds": df.index, "y": df[col]})

            model = Prophet(**self.params)
            model.fit(prophet_df)
            self.fitted_models[col] = model

    def make_prediction(
        self, horizon: int, columns: List[str], values: List[float]
    ) -> Tuple[List[str], List[List[float]]]:
        predictable_cols = []
        predicted_series_list = []

        for col, model in self.fitted_models.items():
            future = model.make_future_dataframe(
                periods=horizon, freq=self.df.index.freq
            )

            forecast = model.predict(future)
            predicted_values = forecast["yhat"].iloc[-horizon:].tolist()
            predictable_cols.append(col)
            predicted_series_list.append(predicted_values)

        return predictable_cols, predicted_series_list


class LinearExtrapolator:
    """
    A simple conditional model that linearly extrapolates each conditioned series
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df.copy()
        self.all_columns = list(df.columns)
        self.params = kwargs

    def make_prediction(
        self, horizon: int, columns: List[str], values: List[float]
    ) -> Tuple[List[str], List[List[float]]]:

        predicted_cols: List[str] = []
        predicted_series_list: List[List[float]] = []

        for col, target in zip(columns, values):
            last_val = float(self.df[col].iloc[-1])
            if horizon == 1:
                preds = [float(target)]
            else:
                preds = [
                    (last_val + (float(target) - last_val) * (t / horizon))
                    for t in range(1, horizon + 1)
                ]
                preds = [float(x) for x in preds]

            predicted_cols.append(col)
            predicted_series_list.append(preds)

        return predicted_cols, predicted_series_list


def tsds(df):
    df_long = df.reset_index().melt(id_vars="index", value_vars=df.columns.tolist())
    df_long.columns = ["time", "group", "value"]
    df_long["time_idx"] = pd.factorize(df_long["time"])[0]

    return TimeSeriesDataSet(
        df_long[
            lambda x: x.time_idx
            <= df_long["time_idx"].max() - self.max_prediction_length
        ],
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=self.max_encoder_length,
        max_prediction_length=self.max_prediction_length,
        static_categoricals=["group"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["value"],
        allow_missing_timesteps=True,
    )


class torchForecastingRNN:
    def __init__(self, df, **kwargs):
        self.df = df
        self.params = kwargs
        self.max_prediction_length = kwargs.get("horizon", 12)
        self.dataset = tsds(df)

        train_dataloader = self.dataset.to_dataloader(
            train=True, batch_size=kwargs.get("batch_size", 64), num_workers=0
        )

        self.model = RecurrentNetwork.from_dataset(
            self.dataset,
            hidden_size=kwargs.get("hidden_size", 20),
            rnn_layers=kwargs.get("rnn_layers", 2),
            learning_rate=kwargs.get("learning_rate", 0.01),
        )

        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 30),
        )

        trainer.fit(self.model, train_dataloaders=train_dataloader)

    def make_prediction(self, horizon, columns, values):
        pred_dataloader = self.dataset.to_dataloader(
            train=False,
            batch_size=len(self.df.columns.tolist()),
        )

        raw_predictions = self.model.predict(pred_dataloader, return_index=False)
        predicted_series_list = raw_predictions.numpy().tolist()

        return self.df.columns.tolist(), predicted_series_list


class MixModel:
    def __init__(
        self,
        df: pd.DataFrame,
        **kwargs,
    ):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.sarima_columns = {
            "Огурцы тепличные, руб./кг",
            "Томаты тепличные, руб./кг",
        }
        self.linextr_model_cls = LinearExtrapolator
        self.sarima_model_cls = SARIMAModel
        self.params = kwargs

    def make_prediction(
        self, horizon: int, columns: List[str], values: List[float]
    ) -> Tuple[List[str], List[List[float]]]:
        conditioning_col = columns[0]
        predictions: Dict[str, List[float]] = {}

        for col in self.all_columns:
            if col == conditioning_col:
                # pass only conditioning column
                model = self.linextr_model_cls(self.df[[col]], **self.params)
                preds = model.make_prediction(horizon, [col], values)[1][0]
            elif col in self.sarima_columns:
                # pass only SARIMA column
                model = self.sarima_model_cls(self.df[[col]], **self.params)
                preds = model.make_prediction(horizon, [col], values)[1][0]
            else:
                preds = [self.df[col].iloc[-1]] * horizon
            predictions[col] = preds

        return list(predictions.keys()), list(predictions.values())


class DynamicFactorModel:  # TODO check the right order
    """
    DFM
    - k_factors (int): The number of unobserved latent factors. Defaults to 1.
    - factor_order (int): The order of the vector autoregression (VAR) for the factors. Defaults to 1.
    - error_order (int): The order of the autoregression for the error term. Defaults to 1.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.fitted_model = None

        k_factors = kwargs.get("k_factors", 1)
        factor_order = kwargs.get("factor_order", 1)
        error_order = kwargs.get("error_order", 1)
        maxiter = kwargs.get("maxiter", 50)

        try:

            model = sm.tsa.DynamicFactor(
                df,
                k_factors=k_factors,
                factor_order=factor_order,
                error_order=error_order,
                enforce_stationarity=False,
            )
            self.fitted_model = model.fit(disp=False, maxiter=maxiter)
        except Exception as e:
            print(f"Warning: Dynamic Factor model could not be fitted. Error: {e}")

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):
        """
        Generates forecasts for all variables using the fitted Dynamic Factor model.

        Note: This is an unconditional forecast. The 'columns' and 'values'
              arguments are ignored.
        """
        if not self.fitted_model:
            return [], []

        forecast = self.fitted_model.forecast(steps=horizon)

        predictable_cols = self.all_columns
        predicted_series_list = forecast.T.values.tolist()

        return predictable_cols, predicted_series_list


class SimpleVECM:
    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.fitted_model = None

        model = VECM(**kwargs)
        self.fitted_model = model.fit(y=df)

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):

        fh = np.arange(1, horizon + 1)
        forecast = self.fitted_model.predict(fh=fh)
        predictable_cols = self.all_columns
        predicted_series_list = forecast.T.values.tolist()

        return predictable_cols, predicted_series_list


# =============================================
# =============================================
class ClusterModel:
    def __init__(
        self,
        df: pd.DataFrame,
        clusters,
        predict_outside_cluster=True,
        **kwargs,
    ):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.sarima_columns = {
            "Огурцы тепличные, руб./кг",
            "Томаты тепличные, руб./кг",
        }
        self.linextr_model_cls = LinearExtrapolator
        self.sarima_model_cls = SARIMAModel
        self.clusters = clusters
        self.predict_outside_cluster = predict_outside_cluster
        self.params = kwargs

    def make_prediction(
        self, horizon: int, columns: List[str], values: List[float]
    ) -> Tuple[List[str], List[List[float]]]:
        conditioning_col = columns[0]
        predictions: Dict[str, List[float]] = {}

        for col in self.all_columns:
            if self.clusters[col] == self.clusters[conditioning_col]:
                # pass only conditioning column
                scale_coef = self.df[conditioning_col].iloc[-1] / self.df[col].iloc[-1]
                model = self.linextr_model_cls(self.df[[col]], **self.params)
                preds = model.make_prediction(horizon, [col], values / scale_coef)[1][0]
            elif col in self.sarima_columns:
                # pass only SARIMA column
                model = self.sarima_model_cls(self.df[[col]], **self.params)
                preds = model.make_prediction(horizon, [col], values)[1][0]
            else:
                # Naive
                if not self.predict_outside_cluster:
                    continue

                preds = [self.df[col].iloc[-1]] * horizon
            predictions[col] = preds

        return list(predictions.keys()), list(predictions.values())
