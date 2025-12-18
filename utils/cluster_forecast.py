from darts import TimeSeries
from darts.models import (
    NBEATSModel,
    TFTModel,
    NaiveSeasonal,
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    ExponentialSmoothing,
    Theta,
    RandomForestModel,
    # Chronos2Model # Requires python >= 3.10!
)
from pathlib import Path
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
import pandas as pd
from typing import Optional, Union
import warnings
import os
import pickle
import matplotlib.pyplot as plt
from pandas import DatetimeIndex
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from sklearn.linear_model import LinearRegression  # or any sklearn regressor
import numpy as np
import plotly.graph_objects as go

    
def plot_preds_on_curve_plt(preds: pd.DataFrame,
                        label: pd.DataFrame,
                        mape_val: float,
                        predictor_name: Union[str, list[str]],
                        cat_name: str,
                        save_dir: str):
    pred_names = "+".join([str(pred_name) for pred_name in predictor_name]) if isinstance(predictor_name, list) else predictor_name
        
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('lightgray')
    ax.plot(preds.index, preds, color="maroon", label=f"Sum index of {cat_name} preds")
    ax.plot(label.index, label, color="steelblue", label=f"Sum index of {cat_name} label")
    ax.legend()
    if predictor_name:
        joined = "|".join(predictor_name) if isinstance(predictor_name, list) else str(predictor_name)
        ax.set_title(f"Sum index {cat_name} preds vs labels with predictor {joined}; curr mape: {mape_val:.2f}")
    else:
        ax.set_title(f"Sum index {cat_name} preds vs labels without any predictors; curr mape: {mape_val:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sum")
    ax.set_facecolor('lightgray')
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cat_name_reg = cat_name.replace("/", "_")
        plt.savefig(f'{save_dir}/{cat_name_reg}_by_{pred_names}_sum_index_one_year.png')

    
def plot_preds_on_curve(preds: pd.DataFrame,
                               label: pd.DataFrame,
                               mape_val: float,
                               predictor_name: Union[str, list[str]],
                               cat_name: str,
                               save_dir: str):
    """
    Plot predictions vs labels using Plotly with hover functionality
    """
    # Prepare predictor names for title
    pred_names = "+".join([str(pred_name) for pred_name in predictor_name]) if isinstance(predictor_name, list) else predictor_name
    
    # Create figure
    fig = go.Figure()
    
    # Add prediction trace with hover information
    fig.add_trace(go.Scatter(
        x=preds.index,
        y=preds.iloc[:, 0] if preds.shape[1] == 1 else preds,
        mode='lines+markers',
        name=f"INDEX of {cat_name} preds",
        line=dict(color='maroon'),
        hovertemplate='<b>Prediction</b><br>' +
                     'Date: %{x}<br>' +
                     'Value: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add label trace with hover information
    fig.add_trace(go.Scatter(
        x=label.index,
        y=label.iloc[:, 0] if label.shape[1] == 1 else label,
        mode='lines+markers',
        name=f"Sum index of {cat_name} label",
        line=dict(color='steelblue'),
        hovertemplate='<b>Label</b><br>' +
                     'Date: %{x}<br>' +
                     'Value: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Configure layout
    joined = "|".join(predictor_name) if isinstance(predictor_name, list) else str(predictor_name)
    # title_text = f"Index {cat_name} preds vs labels with predictor {joined}; curr mape: {mape_val:.2f}" if predictor_name else f"Sum index {cat_name} preds vs labels without any predictors; curr mape: {mape_val:.2f}"
    title_text = f"Index {cat_name}; curr mape: {mape_val:.2f}" if predictor_name else f"Index {cat_name}; curr mape: {mape_val:.2f}"

    fig.update_layout(
        title=dict(text=title_text),
        xaxis_title="Date",
        yaxis_title="Sum",
        plot_bgcolor='lightgray',
        paper_bgcolor='lightgray',
        hovermode='x unified',  # Shows hover for all traces at same x-value[citation:1]
        showlegend=True
    )
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cat_name_reg = cat_name.replace("/", "_")
        # plot_preds_on_curve_plt(preds,
        #                        label,
        #                        mape_val,
        #                        predictor_name,
        #                        cat_name,
        #                        save_dir,
        #                        feature_to_cat_enc)
        # fig.write_html(f'{save_dir}/{cat_name_reg}_by_{pred_names}_sum_index_one_year.html')
        # fig.write_image(f'{save_dir}/{cat_name_reg}_by_{pred_names}_sum_index_one_year.png')
    # Display the figure
    fig.show()
    return fig


class ClusterForecaster:
    def __init__(self,
                 cluster_data: pd.DataFrame,
                 macro_data: pd.DataFrame,
                 category_map: dict[str, str],
                 freq: str = "M",
                 target_col_suffix: str = "_sum",
                 dir_to_save_plots: str = "darts_result",
                 dir_to_save_tables: str = "darts_result_tables",
                 future_macro_col: list[str] = [],
                 model_type: str = "NaiveSeasonal"):
        self.cluster_data = cluster_data
        self.macro_data = macro_data
        self.category_map = category_map
        self.freq = freq
        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()
        self.model = None
        self.model_type = model_type
        self.target_col_suffix = target_col_suffix
        self.dir_to_save_plots = dir_to_save_plots
        self.dir_to_save_tables = dir_to_save_tables
        self.future_macro_col = future_macro_col
        self._trained_ts = None
        self.future_cov_columns_used = None 

    def _prepare_series(self,
                        target_col: str,
                        train_end: pd.Timestamp = None,
                        covariates: Optional[pd.DataFrame] = None,
                        future_cov: Optional[pd.DataFrame] = None
                        ):
        future_cov_slice = None  # <-- ensure defined
        is_macro_data = target_col in self.macro_data.columns
        # slice up to train_end
        if train_end is not None:
            mask = self.cluster_data.index <= train_end
            if not is_macro_data:
                target_slice = self.cluster_data.loc[mask, f"{target_col}{self.target_col_suffix}"]
            else:
                target_slice = self.macro_data.loc[mask, f"{target_col}"]
            cov_slice = covariates.loc[mask, :] if covariates is not None else None
            future_cov_slice = future_cov
        else:
            if not is_macro_data:
                target_slice = self.cluster_data[f"{target_col}{self.target_col_suffix}"]
            else:
                target_slice = self.macro_data[f"{target_col}"]
            cov_slice = covariates
            future_cov_slice = future_cov

        target_ts = TimeSeries.from_series(target_slice, freq=self.freq)
        cov_ts = None
        future_cov_ts = None
        if cov_slice is not None:
            cov_ts = TimeSeries.from_dataframe(cov_slice, freq=self.freq)
        if future_cov_slice is not None:
            future_cov_ts = TimeSeries.from_dataframe(future_cov_slice, freq=self.freq)
        return target_ts, cov_ts, future_cov_ts

    def train(self,
              target_cluster: str,
              additional_clusters: Optional[list[str]] = None,
              use_macro: bool = True,
              past_covariate_lags: int = 12,
              train_end: pd.Timestamp = None,
              use_future_macro: bool = True,
              output_chunk_length_model: int = 12,
              is_backtest: bool = False,
              **model_kwargs):
        """
        Train model up to train_end (inclusive).
        self.model_type can be one of:
          - 'NaiveSeasonal' (requires K in model_kwargs; set K=12 for monthly seasonality)
          - 'NaiveDrift'
          - 'NaiveMean'
          - 'NaiveMovingAverage' (requires window parameter)
          - 'ExponentialSmoothing'
          - 'Theta'
          - 'NBEATS' (keeps previous implementation)
        """
        # build covariates DataFrame (the baseline models below WILL IGNORE covariates)
        cov_df = pd.DataFrame(index=self.cluster_data.index)
        future_cov_df = pd.DataFrame(index=self.cluster_data.index)
        if use_macro:
            cov_df = pd.concat([cov_df, self.macro_data], axis=1)
        if additional_clusters:
            for add in additional_clusters:
                col_name = f"{add}_lag1"
                if f"{add}{self.target_col_suffix}" in self.cluster_data:
                    cov_df[col_name] = self.cluster_data[f"{add}{self.target_col_suffix}"].shift(1)
                else:
                    print("Warning! The additional cluster column does not exists in cluster_data")
        # assert target_cluster, "Target column must be not emptry string or None!"
        # cov_df[f"{target_cluster}_lag1"] = self.cluster_data[f"{target_cluster}{self.target_col_suffix}"].shift(1)

        if use_macro and use_future_macro and self.future_macro_col:
             # Create future covariates that extend into the past
            future_cov_start = self.cluster_data.index[0]
            future_cov_end = self.cluster_data.index[-1]
            
            # For TFT/N-BEATS, we need future covariates that cover input window
            if self.model_type.lower() in ["tftmodel", "nbeats", "n-beats", "randomforest"]: #, "chronos"]:
                # Extend the start back by input_chunk_length
                future_cov_start = future_cov_start - pd.DateOffset(months=past_covariate_lags)
            
            future_cov_range = pd.date_range(start=future_cov_start, 
                                            end=future_cov_end, 
                                            freq=self.freq)
            future_cov_df = pd.DataFrame(index=future_cov_range)
            
            for future_col in self.future_macro_col:
                # Fill with available data
                if future_col in self.macro_data.columns:
                    future_cov_df[future_col] = self.macro_data.reindex(future_cov_range)[future_col]
                elif future_col in self.cluster_data.columns:
                    future_cov_df[future_col] = self.cluster_data.reindex(future_cov_range)[future_col]
            self.future_cov_columns_used = list(future_cov_df.columns)
        # fill missing values (bfill or ffill as needed)
        #print(cov_df)
        #raise ValueError("debugging")
        cov_df = cov_df.fillna(method="bfill").fillna(method="ffill")
        future_cov_df = future_cov_df.fillna(method="bfill").fillna(method="ffill")

        # convert to Darts time series
        target_ts, cov_ts, future_cov_ts = self._prepare_series(target_cluster,
                                                 train_end,
                                                 cov_df if not cov_df.empty else None,
                                                 future_cov_df if not future_cov_df.empty else None)
        self._trained_ts = target_ts
        # Choose and build model
        model_type_lower = self.model_type.lower()
        if model_type_lower in ("regression", "regressionmodel", "linearregression"):
            lags = model_kwargs.pop("lags", past_covariate_lags)
            lags_past_cov = model_kwargs.pop("lags_past_covariates", past_covariate_lags)
            # for future covariates: tuple (past_lags, future_lags)
            lags_future_cov = model_kwargs.pop("lags_future_covariates", (1, 6) if len(self.future_macro_col)!=0 else None)
            output_len = model_kwargs.pop("output_chunk_length", 1)

            model = LinearRegressionModel(
                lags=lags,
                lags_past_covariates=lags_past_cov,
                lags_future_covariates=lags_future_cov,
                output_chunk_length=output_len,
                **model_kwargs
            )

            uses_covariates = cov_ts is not None
            uses_future_covariates = future_cov_ts is not None
        elif model_type_lower == "nbeats" or model_type_lower == "n-beats":
            # keep your previous NBEATS behaviour
            model = NBEATSModel(input_chunk_length=past_covariate_lags,
                                output_chunk_length=output_chunk_length_model)
            # NBEATS supports covariates if configured (we pass past_covariates below)
            uses_covariates = True
            uses_future_covariates = False
        elif model_type_lower == "tftmodel":
            model = TFTModel(input_chunk_length=past_covariate_lags,
                            output_chunk_length=output_chunk_length_model,
                            add_relative_index=True,
                            add_encoders=None,  # Disable automatic encoders if you're handling covariates manually
                            **model_kwargs)
            model_kwargs.pop("lags_future_covariates", (1, 6))
            uses_covariates = True
            uses_future_covariates = True
        elif model_type_lower == "naiveseasonal":
            # user should provide K (seasonal period), default to 12 for monthly
            K = model_kwargs.pop("K", output_chunk_length_model)
            model = NaiveSeasonal(K=K)
            uses_covariates = False
            uses_future_covariates = False

        elif model_type_lower == "randomforest":
            lags = model_kwargs.pop("lags", past_covariate_lags)
            lags_past_cov = model_kwargs.pop("lags_past_covariates", past_covariate_lags)
            # for future covariates: tuple (past_lags, future_lags)
            lags_future_cov = model_kwargs.pop("lags_future_covariates", (1, 6) if len(self.future_macro_col)!=0 else None)
            output_len = model_kwargs.pop("output_chunk_length", output_chunk_length_model)
            if is_backtest:
                lags_future_cov = None
            model = RandomForestModel(
                lags=lags,
                lags_past_covariates=lags_past_cov,
                lags_future_covariates=lags_future_cov,
                output_chunk_length=output_len,
                **model_kwargs
            )
            

            uses_covariates = cov_ts is not None
            uses_future_covariates = future_cov_ts is not None
            
        elif model_type_lower == "chronos":
            model = Chronos2Model(input_chunk_length=past_covariate_lags,
                            output_chunk_length=output_chunk_length_model,
                            add_encoders=None,  # Disable automatic encoders if you're handling covariates manually
                            **model_kwargs)
            uses_covariates = True
            uses_future_covariates = True

        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented as baseline option")
        if is_backtest:
            uses_future_covariates = False
        # warn if covariates present but model doesn't use them
        if cov_ts is not None and not uses_covariates:
            warnings.warn(f"Model {self.model_type} does not support covariates; covariates will be ignored for fitting/prediction.")
        if future_cov_ts is not None and not uses_future_covariates:
            warnings.warn(f"Model {self.model_type} does not support future covariates; future covariates will be ignored for fitting/prediction.")
        # fit model
        if uses_covariates and uses_future_covariates and cov_ts is not None:
            model.fit(series=target_ts,
                      past_covariates=cov_ts,
                      future_covariates=future_cov_ts,
                      # verbose=True
                      )
        elif uses_covariates and not uses_future_covariates and cov_ts is not None:
            model.fit(series=target_ts,
                      past_covariates=cov_ts,
                     # verbose=True
                     )
        else:
            # baseline / many statistical models: just fit on series
            model.fit(series=target_ts, verbose=True)

        self.model = model
        # store the slices used for training
        self._train_target_ts = target_ts
        self._train_cov_ts = cov_ts
        self._train_end = train_end

    def forecast(self,
                 n: int = 12,
                 covariates_past: Optional[pd.DataFrame] = None,
                 covariates_future: Optional[pd.DataFrame] = None,
                 train_series: Optional[TimeSeries] = None,
                 is_backtest: bool = False
                 ):
        """
        Forecast the next n steps beyond training data.
        If covariates_future is provided, it will be ignored by baseline models that do not support covariates.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        # If the trained model supports past_covariates and covariates_future is provided as a DataFrame,
        # the user should convert to TimeSeries and pass it; otherwise, ignore.
        cov_future_ts = None
        cov_past_ts = None
        if covariates_past is not None:
            try:
                cov_past_ts = TimeSeries.from_dataframe(covariates_past, freq=self.freq)
            except Exception as e:
                print(f"Failed to convert covariates past into TimeSeries with error: {e}; Gonna skip")
                cov_past_ts = None
        if covariates_future is not None:
            try:
                cov_future_ts = TimeSeries.from_dataframe(covariates_future)
            except Exception as e:
                try:
                    print(f"Failed to convert covariates future into TimeSeries with error: {e}; Try conver into ts from series")
                    cov_future_ts = TimeSeries.from_series(covariates_future)
                except Exception as e:
                    print(f"Failed to convert covariates future into TimeSeries with error: {e}; Gonna skip")
                    cov_future_ts = None
        # If model has predict signature that accepts past_covariates / future_covariates, Darts will raise if passed incorrectly.
        # We'll attempt to call with covariates only when they were used in training (check attribute)
        try:
            # many baseline models: predict(n)
            if is_backtest:
                pred = self.model.predict(n=n,
                                      series=train_series if train_series is not None else self._trained_ts,
                                      past_covariates=cov_past_ts)
            else:
                pred = self.model.predict(n=n,
                                      series=train_series if train_series is not None else self._trained_ts,
                                      past_covariates=cov_past_ts,
                                      future_covariates=cov_future_ts)  # works for models that accept covariates
        except TypeError:
            # fallback: call without covariates
            pred = self.model.predict(n=n)

        values = pred.values(copy=True)
        values = np.maximum(values, 0.0)
        
        pred = TimeSeries.from_times_and_values(
            times=pred.time_index,
            values=values,
            freq=pred.freq,
            columns=pred.components
        )
        return pred

    def backtest(self,
                 target_cluster: str,
                 additional_clusters: Optional[list[str]] = None,
                 use_macro: bool = True,
                 past_covariate_lags: int = 12,
                 start_backtest: pd.Timestamp = None,
                 forecast_horizon: int = 12,
                 **model_kwargs):
        """
        Do a one-step backtest: train up to start_backtest - 1 month, forecast horizon steps,
        and compare with actuals.
        Returns (predicted_series, actual_series, dict of metrics).
        """


        if os.path.isdir(self.dir_to_save_tables):
            predictors_label = "none" if not additional_clusters else "_".join(additional_clusters)
            my_file = Path(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_preds.csv")
            if my_file.is_file():
                result_df = pd.read_csv(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_preds.csv")
                metrics_df = pd.read_csv(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_metrics.csv")
                return result_df[f"preds_{target_cluster}{self.target_col_suffix}"], result_df[f"true_{target_cluster}{self.target_col_suffix}"], metrics_df.to_dict()
        if start_backtest is None:
            raise ValueError("Please supply a start backtest date")

        # train up to just before backtest
        train_end = start_backtest - pd.offsets.MonthBegin(1)
        self.train(target_cluster=target_cluster,
                   additional_clusters=additional_clusters,
                   use_macro=use_macro,
                   past_covariate_lags=past_covariate_lags,
                   train_end=train_end,
                   **model_kwargs)

        # Prepare the covariates for the forecast period (may be ignored by baseline models)
        future_idx = pd.date_range(start=train_end + pd.DateOffset(months=1),
                                   periods=forecast_horizon,
                                   freq="MS")
        cov_past = pd.DataFrame(index=future_idx)
        if use_macro:
            cov_past = pd.concat([cov_past, self.macro_data], axis=1)
        if additional_clusters:
            for additional_cluster in additional_clusters:
                all_add = self.cluster_data[f"{additional_cluster}{self.target_col_suffix}"].shift(1)
                cov_past[f"{additional_cluster}_lag1"] = all_add.reindex(future_idx)

        cov_past = cov_past.fillna(method="bfill").fillna(method="ffill")
        pred_ts = self.forecast(n=forecast_horizon, covariates_past=cov_past)
        # actual in that period
        actual = self.cluster_data[f"{target_cluster}{self.target_col_suffix}"].reindex(future_idx)
        actual_ts = TimeSeries.from_series(actual, freq=self.freq)
        return self.plot_and_return_data_backtest(actual_ts, pred_ts, target_cluster, future_idx, additional_clusters)

    def plot_and_return_data_backtest(self,
                                      target_ts,
                                      pred_ts,
                                      target_cluster: str,
                                      future_idx: DatetimeIndex,
                                      additional_clusters: Optional[list[str]] = None):
        # metrics
        try:
            mape_val = mape(target_ts, pred_ts)
            mae_val = mae(target_ts, pred_ts)
            rmse_val = rmse(target_ts, pred_ts)
        except Exception as e:
            print(f"Could not calculate mape for prediction due to {e}")
            mape_val = 0.0
            mae_val = 0.0
            rmse_val = 0.0
        # (your plot function)
        plot_preds_on_curve(pred_ts.to_dataframe(),
                            target_ts.to_dataframe(),
                            mape_val,
                            additional_clusters,
                            target_cluster,
                            self.dir_to_save_plots)

        metrics = {"MAPE": mape_val, "MAE": mae_val, "RMSE": rmse_val}
        predictors_label = "none" if not additional_clusters else "_".join(additional_clusters)
        preds_df = pred_ts.to_dataframe().rename(columns={f"{target_cluster}{self.target_col_suffix}": 
                                                          f"preds_{target_cluster}{self.target_col_suffix}"})
        actual_df = target_ts.to_dataframe().rename(columns={f"{target_cluster}{self.target_col_suffix}":
                                                             f"true_{target_cluster}{self.target_col_suffix}"})
        result_df = pd.concat([actual_df, preds_df], axis=1)
        os.makedirs(self.dir_to_save_tables, exist_ok=True)
        target_cluster_name = target_cluster.replace("/", "_")
        result_df.to_csv(f"{self.dir_to_save_tables}/{target_cluster_name}_by_{predictors_label}_preds.csv", index=False)
        pd.DataFrame([metrics]).to_csv(f"{self.dir_to_save_tables}/{target_cluster_name}_by_{predictors_label}_metrics.csv", index=False)

        return pred_ts, target_ts, metrics

    def save(self, path: str):
        """
        Persist the ClusterForecaster + underlying Darts model.
        Two files are created:
          path + ".pkl"  -> pickled ClusterForecaster WITHOUT model
          path + "_model.pth" -> Darts model (Torch-based or baseline)
        """
        os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
        if self.model is None:
            raise ValueError("No model has been trained, nothing to save.")
        
        # save darts model separately
        model_path = path + "_model.pth"
        self.model.save(model_path)

        # temporarily remove the model so pickle does not break
        model_backup = self.model
        self.model = None

        with open(path + ".pkl", "wb") as f:
            pickle.dump(self, f)

        # restore the model in memory
        self.model = model_backup

        print(f"Saved forecaster to {path}.pkl and model to {model_path}")

    @classmethod
    def load(cls, path: str):
        """
        Load ClusterForecaster and restore the underlying model
        based on stored model_type.
        """
        model_path = path + "_model.pth"

        # Load forecaster object
        with open(path + ".pkl", "rb") as f:
            forecaster = pickle.load(f)

        model_type_lower = forecaster.model_type.lower()

        # Restore appropriate model from disk
        if model_type_lower in ("regression", "regressionmodel", "linearregression"):
            forecaster.model = LinearRegressionModel.load(model_path)

        elif model_type_lower in ("nbeats", "n-beats"):
            forecaster.model = NBEATSModel.load(model_path)

        elif model_type_lower == "tftmodel":
            forecaster.model = TFTModel.load(model_path)

        elif model_type_lower == "naiveseasonal":
            forecaster.model = NaiveSeasonal.load(model_path)

        elif model_type_lower == "naivedrift":
            forecaster.model = NaiveDrift.load(model_path)

        elif model_type_lower == "naivemean":
            forecaster.model = NaiveMean.load(model_path)

        elif model_type_lower == "naivemovingaverage":
            forecaster.model = NaiveMovingAverage.load(model_path)

        elif model_type_lower in ("exponentialsmoothing", "es"):
            forecaster.model = ExponentialSmoothing.load(model_path)

        elif model_type_lower == "theta":
            forecaster.model = Theta.load(model_path)

        elif model_type_lower == "randomforest":
            forecaster.model = RandomForestModel.load(model_path)
            
        elif model_type_lower == "chronos":
            forecaster.model = Chronos2Model.load(model_path)

        else:
            raise ValueError(f"Unknown model type for loading: {forecaster.model_type}")

        print(f"[OK] Loaded forecaster ← {path}.pkl")
        print(f"[OK] Loaded model      ← {model_path}")
        return forecaster