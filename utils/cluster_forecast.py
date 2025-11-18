from darts import TimeSeries
from darts.models import (
    NBEATSModel,
    TFTModel,
    NaiveSeasonal,
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    ExponentialSmoothing,
    Theta
)
from pathlib import Path
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
import pandas as pd
from typing import Optional
import warnings
from .constants import CAT_MAP_ENCODING
import os
import pickle
import matplotlib.pyplot as plt
from pandas import DatetimeIndex


def plot_preds_on_curve(preds: pd.DataFrame,
                        label: pd.DataFrame,
                        mape_val: float,
                        predictor_name: str|list[str],
                        cat_name: str,
                        save_dir: str,
                        feature_to_cat_enc: dict[str, str] = CAT_MAP_ENCODING):
    pred_names = "+".join([str(feature_to_cat_enc[pred_name]) for pred_name in predictor_name]) if isinstance(predictor_name, list) else predictor_name
        
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('lightgray')
    ax.plot(preds.index, preds, color="maroon", label=f"Sum index of {cat_name} preds")
    ax.plot(label.index, label, color="steelblue", label=f"Sum index of {cat_name} label")
    ax.legend()
    if predictor_name:
        ax.set_title(f"Sum index {cat_name} preds vs labels with predictor {"|".join(predictor_name)}; curr mape: {mape_val:.2f}")
    else:
        ax.set_title(f"Sum index {cat_name} preds vs labels without any predictors; curr mape: {mape_val:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sum")
    ax.set_facecolor('lightgray')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{cat_name}_by_{pred_names}_sum_index_one_year.png')

    
class ClusterForecaster:
    def __init__(self,
                 cluster_data: pd.DataFrame,
                 macro_data: pd.DataFrame,
                 category_map: dict[str, str],
                 freq: str = "M",
                 target_col_suffix: str = "_sum",
                 dir_to_save_plots: str = "darts_result",
                 dir_to_save_tables: str = "darts_result_tables",
                 future_macro_col: str = "",
                 features_cat_enc: dict[str, int] = CAT_MAP_ENCODING):
        self.cluster_data = cluster_data
        self.macro_data = macro_data
        self.category_map = category_map
        self.freq = freq
        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()
        self.model = None
        self.model_type = "NaiveSeasonal"
        self.target_col_suffix = target_col_suffix
        self.dir_to_save_plots = dir_to_save_plots
        self.dir_to_save_tables = dir_to_save_tables
        self.features_cat_enc = features_cat_enc
        self.future_macro_col = future_macro_col

    def _aggregate_clusters(self) -> pd.DataFrame:
        df = pd.DataFrame(index=self.cluster_data.index)
        for cluster in set(self.category_map.values()):
            cols = [c for c, cat in self.category_map.items()
                    if cat == cluster and c in self.cluster_data.columns]
            if cols:
                df[cluster] = self.cluster_data[cols].sum(axis=1)
        return df

    def _prepare_series(self,
                        target_col: str,
                        train_end: pd.Timestamp = None,
                        covariates: Optional[pd.DataFrame] = None,
                        future_cov: Optional[pd.DataFrame] = None
                        ):
        future_cov_slice = None  # <-- ensure defined
        # slice up to train_end
        if train_end is not None:
            mask = self.cluster_data.index <= train_end
            target_slice = self.cluster_data.loc[mask, f"{target_col}{self.target_col_suffix}"]
            cov_slice = covariates.loc[mask, :] if covariates is not None else None
            future_cov_slice = future_cov
        else:
            target_slice = self.cluster_data[f"{target_col}{self.target_col_suffix}"]
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
                cov_df[col_name] = self.cluster_data[f"{add}{self.target_col_suffix}"].shift(1)
        if use_macro and use_future_macro and self.future_macro_col:
            if self.future_macro_col not in self.macro_data.columns:
                raise ValueError(f"future_macro_col '{self.future_macro_col}' not found in macro_data columns")
            # only this column will be used as future covariate
            future_cov_df[self.future_macro_col] = self.macro_data[self.future_macro_col]
        # fill missing values (bfill or ffill as needed)
        cov_df = cov_df.fillna(method="bfill").fillna(method="ffill")
        future_cov_df = future_cov_df.fillna(method="bfill").fillna(method="ffill")

        # convert to Darts time series
        target_ts, cov_ts, future_cov_ts = self._prepare_series(target_cluster,
                                                 train_end,
                                                 cov_df if not cov_df.empty else None,
                                                 future_cov_df if not future_cov_df.empty else None)

        # Choose and build model
        model_type_lower = self.model_type.lower()
        if model_type_lower == "nbeats" or model_type_lower == "n-beats":
            # keep your previous NBEATS behaviour
            model = NBEATSModel(input_chunk_length=past_covariate_lags,
                                output_chunk_length=12,
                                **model_kwargs)
            # NBEATS supports covariates if configured (we pass past_covariates below)
            uses_covariates = True
            uses_future_covariates = False
        elif model_type_lower == "tftmodel":
            model = TFTModel(input_chunk_length=past_covariate_lags,
                            output_chunk_length=12,
                            **model_kwargs)
            uses_covariates = True
            uses_future_covariates = True
        elif model_type_lower == "naiveseasonal":
            # user should provide K (seasonal period), default to 12 for monthly
            K = model_kwargs.pop("K", 12)
            model = NaiveSeasonal(K=K)
            uses_covariates = False
            uses_future_covariates = False

        elif model_type_lower == "naivedrift":
            model = NaiveDrift()
            uses_covariates = False
            uses_future_covariates = False

        elif model_type_lower == "naivemean":
            model = NaiveMean()
            uses_covariates = False
            uses_future_covariates = False

        elif model_type_lower == "naivemovingaverage":
            window = model_kwargs.pop("window", None)
            if window is None:
                raise ValueError("NaiveMovingAverage requires 'window' argument (int).")
            model = NaiveMovingAverage(window=window)
            uses_covariates = False
            uses_future_covariates = False

        elif model_type_lower == "exponentialsmoothing" or model_type_lower == "es":
            model = ExponentialSmoothing()
            uses_covariates = False
            uses_future_covariates = False

        elif model_type_lower == "theta":
            model = Theta()
            uses_covariates = False
            uses_future_covariates = False

        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented as baseline option")

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
                      verbose=True)
        elif uses_covariates and not uses_future_covariates and cov_ts is not None:
            model.fit(series=target_ts,
                      past_covariates=cov_ts,
                      verbose=True)
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
                cov_future_ts = TimeSeries.from_dataframe(covariates_future, freq=self.freq)
            except Exception as e:
                print(f"Failed to convert covariates future into TimeSeries with error: {e}; Gonna skip")
                cov_future_ts = None
        # If model has predict signature that accepts past_covariates / future_covariates, Darts will raise if passed incorrectly.
        # We'll attempt to call with covariates only when they were used in training (check attribute)
        try:
            # many baseline models: predict(n)
            pred = self.model.predict(n=n,
                                      past_covariates=cov_past_ts,
                                      future_covariates=cov_future_ts)  # works for models that accept covariates
        except TypeError:
            # fallback: call without covariates
            pred = self.model.predict(n=n)

        return pred

    def forecast_to_date(
            self,
            last_timestamp: pd.Timestamp,
            target_date: Union[pd.Timestamp, str],
            covariates_past: Optional[pd.DataFrame] = None,
            covariates_future: Optional[pd.DataFrame] = None
    ):
        """
        Forecast up to a specific target date using the already trained model.
        Returns the forecasted value at exactly that date.
        """
    
        if self.model is None:
            raise RuntimeError("Model not trained yet")
    
        target_date = pd.to_datetime(target_date)
    
        if target_date <= last_timestamp:
            raise ValueError("target_date must be after last_timestamp")
    
        # -----------------------------
        # 1. Compute forecast horizon
        # -----------------------------
        # Assumes daily freq; if hourly/weekly/etc., you already know how to adapt:
        step_count = (target_date - last_timestamp).days
        if step_count <= 0:
            raise ValueError("Computed non-positive forecast horizon")
    
        # -----------------------------
        # 2. Convert covariates exactly like original forecast()
        # -----------------------------
        cov_future_ts = None
        cov_past_ts = None
    
        if covariates_past is not None:
            try:
                cov_past_ts = TimeSeries.from_dataframe(covariates_past, freq=self.freq)
            except Exception as e:
                print(f"Failed to convert covariates_past: {e}; skipping")
                cov_past_ts = None
    
        if covariates_future is not None:
            try:
                cov_future_ts = TimeSeries.from_dataframe(covariates_future, freq=self.freq)
            except Exception as e:
                print(f"Failed to convert covariates_future: {e}; skipping")
                cov_future_ts = None
    
        # -----------------------------
        # 3. Call model.predict() with identical logic to forecast()
        # -----------------------------
        try:
            pred_series = self.model.predict(
                n=step_count,
                past_covariates=cov_past_ts,
                future_covariates=cov_future_ts
            )
        except TypeError:
            # fallback for simple baseline models
            pred_series = self.model.predict(n=step_count)
    
        # -----------------------------
        # 4. Return only the value at target date
        # -----------------------------
        # pred_series is a TimeSeries → last point is the target_date
        return pred_series[-1]


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
        mape_val = mape(target_ts, pred_ts)
        mae_val = mae(target_ts, pred_ts)
        rmse_val = rmse(target_ts, pred_ts)
        # (your plot function)
        plot_preds_on_curve(pred_ts.to_dataframe(),
                            target_ts.to_dataframe(),
                            mape_val,
                            additional_clusters,
                            target_cluster,
                            self.dir_to_save_plots,
                            feature_to_cat_enc=self.features_cat_enc)

        metrics = {"MAPE": mape_val, "MAE": mae_val, "RMSE": rmse_val}
        predictors_label = "none" if not additional_clusters else "_".join(additional_clusters)
        preds_df = pred_ts.to_dataframe().rename(columns={f"{target_cluster}{self.target_col_suffix}": 
                                                          f"preds_{target_cluster}{self.target_col_suffix}"})
        actual_df = target_ts.to_dataframe().rename(columns={f"{target_cluster}{self.target_col_suffix}":
                                                             f"true_{target_cluster}{self.target_col_suffix}"})
        result_df = pd.concat([actual_df, preds_df], axis=1)
        os.makedirs(self.dir_to_save_tables, exist_ok=True)
        result_df.to_csv(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_preds.csv", index=False)
        pd.DataFrame([metrics]).to_csv(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_metrics.csv", index=False)

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
        if model_type_lower in ("nbeats", "n-beats"):
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

        else:
            raise ValueError(f"Unknown model type for loading: {forecaster.model_type}")

        print(f"[OK] Loaded forecaster ← {path}.pkl")
        print(f"[OK] Loaded model      ← {model_path}")
        return forecaster