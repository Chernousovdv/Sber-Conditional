from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
import pandas as pd
from typing import Optional


class ClusterForecaster:
    def __init__(self,
                 cluster_data: pd.DataFrame,
                 macro_data: pd.DataFrame,
                 category_map: dict[str, str],
                 freq: str = "M",
                 target_col_suffix: str = "_sum",
                 dir_to_save_plots: str = "darts_result",
                 dir_to_save_tables: str = "darts_result_tables"):
        self.cluster_data = cluster_data
        self.macro_data = macro_data
        self.category_map = category_map
        self.freq = freq
        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()
        self.model = None
        self.target_col_suffix = target_col_suffix
        self.dir_to_save_plots = dir_to_save_plots
        self.dir_to_save_tables = dir_to_save_tables

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
                        covariates: Optional[pd.DataFrame] = None):
        # slice up to train_end
        if train_end is not None:
            mask = self.cluster_data.index <= train_end
            target_slice = self.cluster_data.loc[mask, f"{target_col}{self.target_col_suffix}"]
            cov_slice = covariates.loc[mask, :]
        else:
            target_slice = self.cluster_data[f"{target_col}{self.target_col_suffix}"]
            cov_slice = covariates

        target_ts = TimeSeries.from_series(target_slice, freq='M')
        cov_ts = None
        if covariates is not None:
            cov_ts = TimeSeries.from_dataframe(covariates, freq='M')
        return target_ts, cov_ts

    def train(self,
              target_cluster: str,
              additional_clusters: Optional[list[str]] = None,
              use_macro: bool = True,
              past_covariate_lags: int = 12,
              train_end: pd.Timestamp = None,
              model_type: str = "NBEATS",
              **model_kwargs):
        """
        Train model up to train_end (inclusive).
        If train_end is None, use full series except last output chunk.
        """
        # build covariates DataFrame
        cov_df = pd.DataFrame(index=self.cluster_data.index)
        if use_macro:
            cov_df = pd.concat([cov_df, self.macro_data], axis=1)
        if additional_clusters:
            for add in additional_clusters:
                col_name = f"{add}_lag1"
                # shift by 1 so that covariate at time t corresponds to add at t-1
                cov_df[col_name] = self.cluster_data[f"{add}{self.target_col_suffix}"].shift(1)
       
        # fill missing values (bfill or ffill as needed)
        cov_df = cov_df.fillna(method="bfill").fillna(method="ffill")

        # convert to Darts time series
        target_ts, cov_ts = self._prepare_series(target_cluster, train_end, cov_df)

        # scaling
        # target_ts_scaled = self.scaler_target.fit_transform(target_ts)
        # cov_ts_scaled = None
        # if cov_ts is not None:
        #     cov_ts_scaled = self.scaler_cov.fit_transform(cov_ts[f"{target_cluster}_sum"])
        #     print(cov_ts_scaled)

        # choose model
        if model_type == "NBEATS":
            model = NBEATSModel(input_chunk_length=past_covariate_lags,
                                output_chunk_length=12,
                                **model_kwargs)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

        # fit model
        model.fit(series=target_ts, past_covariates=cov_ts, verbose=True)

        self.model = model
        # store the slices used for training
        self._train_target_ts = target_ts
        self._train_cov_ts = cov_ts
        self._train_end = train_end

    def forecast(self, n: int = 12, covariates_future: Optional[pd.DataFrame] = None):
        """
        Forecast the next n steps beyond training data.
        If covariates_future is provided, it should cover the future period.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        if covariates_future is not None:
            cov_future_ts = TimeSeries.from_dataframe(covariates_future, freq="M")
            # cov_future_ts = self.scaler_cov.transform(cov_future_ts)
        else:
            cov_future_ts = None

        pred = self.model.predict(n=n, past_covariates=cov_future_ts)
        # pred = self.scaler_target.inverse_transform(pred_scaled)
        return pred

    def backtest(self,
                 target_cluster: str,
                 additional_clusters: Optional[list[str]] = None,
                 use_macro: bool = True,
                 past_covariate_lags: int = 12,
                 start_backtest: pd.Timestamp = None,
                 forecast_horizon: int = 12,
                 model_type: str = "NBEATS",
                 **model_kwargs):
        """
        Do a one-step backtest: train on everything before `start_backtest`, forecast forecast_horizon steps,
        and compare with actuals.
        Returns (predicted_series, actual_series, dict of metrics).
        """
        if start_backtest is None:
            raise ValueError("Please supply a start backtest date")

        # train up to just before backtest
        train_end = start_backtest - pd.offsets.MonthBegin(1)
        self.train(target_cluster=target_cluster,
                   additional_clusters=additional_clusters,
                   use_macro=use_macro,
                   past_covariate_lags=past_covariate_lags,
                   train_end=train_end,
                   model_type=model_type,
                   **model_kwargs)

        # Prepare the covariates for the forecast period
        # covariates_future should be aligned to the next n periods after train_end
        future_idx = pd.date_range(start=train_end + pd.DateOffset(months=1),
                                   periods=forecast_horizon,
                                   freq="MS")
        cov_future = pd.DataFrame(index=future_idx)
        if use_macro:
            cov_future = pd.concat([cov_future, self.macro_data], axis=1)
        if additional_clusters:
            for additional_cluster in additional_clusters:
                # shift by one period so that at time t we use additional_cluster at t-1
                all_add = self.cluster_data[f"{additional_cluster}{self.target_col_suffix}"].shift(1)
                cov_future[f"{additional_cluster}_lag1"] = all_add.reindex(future_idx)

        cov_future = cov_future.fillna(method="bfill").fillna(method="ffill")

        # forecast
        pred_ts = self.forecast(n=forecast_horizon, covariates_future=cov_future)

        # actual in that period
        actual = self.cluster_data[f"{target_cluster}{self.target_col_suffix}"].reindex(future_idx)
        actual_ts = TimeSeries.from_series(actual, freq="M")

        # metrics
        mape_val = mape(actual_ts, pred_ts)
        mae_val = mae(actual_ts, pred_ts)
        rmse_val = rmse(actual_ts, pred_ts)
        plot_preds_on_curve(pred_ts.to_dataframe(),
                            actual_ts.to_dataframe(),
                            mape_val,
                            additional_clusters,
                            target_cluster,
                            self.dir_to_save_plots)
        metrics = {"MAPE": mape_val, "MAE": mae_val, "RMSE": rmse_val}
        predictors_label = "none" if not additional_clusters else "_".join(additional_clusters)
        preds_df = pred_ts.to_dataframe().rename(columns={f"{target_cluster}{self.target_col_suffix}": f"preds_{target_cluster}{self.target_col_suffix}"})
        actual_df = actual_ts.to_dataframe().rename(columns={f"{target_cluster}{self.target_col_suffix}": f"true_{target_cluster}{self.target_col_suffix}"})
        result_df = pd.concat([actual_df, preds_df], axis=1)
        result_df.to_csv(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_preds.csv", index=False)
        pd.DataFrame([metrics]).to_csv(f"{self.dir_to_save_tables}/{target_cluster}_by_{predictors_label}_metrics.csv", index=False)

        return pred_ts, actual_ts, metrics