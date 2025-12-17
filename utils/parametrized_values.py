from itertools import combinations
from typing import Type, Literal, Any, Optional
from darts import TimeSeries
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from utils.constants import (
    CATEGORY_MAP,
    BEST_PREDICTORS_FOR_INDEX
)
from utils.utils import (
    get_month_beginnings
)
from utils.TS_normalizations import transform_categories
from utils.data_prepocessing import _ensure_monthly_index_and_align_exog
from utils.cluster_forecast import ClusterForecaster
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # go up from utils/ to project root
DATA_DIR = os.path.join(BASE_DIR, "Data")

directory = f"{DATA_DIR}/"  # set directory path
d = {}
formed = {}

for i, entry in enumerate(os.scandir(directory)):  
    if entry.is_file() and entry.name.endswith('.xlsx'):  # check if it's a file
        d[entry.name] = pd.read_excel(f"{directory}{entry.name}")
    elif entry.is_file() and entry.name.endswith('.csv'):
        formed[entry.name] = pd.read_csv(f"{directory}{entry.name}")

global_macros = d["global_macro.xlsx"]
apk_nona = formed["apk_nona.csv"]

chemicals_nona = formed["chemicals_nona.csv"]
chemicals_nona["Date"] = pd.to_datetime(chemicals_nona["Date"])
chemicals_nona = chemicals_nona.set_index('Date')

global_macros = global_macros.drop(columns=["Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),  .1"])

apk_nona["Date"] = apk_nona["Date"]
apk_nona["Date"] = pd.to_datetime(apk_nona["Date"])

global_macros = global_macros.iloc[3:][::-1].reset_index(drop=True)
global_macros = global_macros.drop(columns=["Date"])
global_macros.index = apk_nona["Date"]

apk_nona = apk_nona.set_index('Date')


assert global_macros.shape[0] == apk_nona.shape[0]


apk_nona_en, global_macros_en = _ensure_monthly_index_and_align_exog(apk_nona, global_macros)
chemicals_nona_en, global_macros_en_chemicals = _ensure_monthly_index_and_align_exog(chemicals_nona[chemicals_nona.index<=global_macros.index.max()],
                                                                                     global_macros[global_macros.index>=chemicals_nona.index.min()])
global_macros_en.columns = list(map(lambda x: " ".join(x.split()), list(global_macros_en.columns)))  # убираем табы
apk_nona_en_rol = transform_categories(apk_nona_en)
chemicals_nona_en_rol = transform_categories(chemicals_nona_en)

macro_data = global_macros_en[global_macros_en.index >= apk_nona_en_rol.dropna().index[0]]
df_combined = pd.concat([apk_nona_en_rol.dropna(), chemicals_nona_en_rol.dropna()], axis=1)


def form_df_future(extended_index,
                   future_forecaster_ts_names: list[str],
                   future_forecaster_ts_values_raw: list[Any],
                   prediction_horizon: int,
                   df_combined: pd.DataFrame = df_combined,
                   macro_data: pd.DataFrame = macro_data):
    # Create a new DataFrame with the extended dates
    df_extended = pd.DataFrame(index=extended_index, columns=df_combined.columns)
    macro_extended = pd.DataFrame(index=extended_index, columns=macro_data.columns)

    # Combine the original and extended DataFrames
    df_updated = pd.concat([df_combined, df_extended])
    macro_updated = pd.concat([macro_data, macro_extended])
    future_forecaster_ts_values = []
    for idx, col_name in enumerate(future_forecaster_ts_names):
        if col_name in df_updated.columns:
            # Get the future values for this column
            future_value = future_forecaster_ts_values_raw[idx]
            future_values = np.linspace(df_combined[col_name][-1],
                            future_value,
                            prediction_horizon)
            future_forecaster_ts_values.append(future_values)
            # Update the values for the extended period
            # Make sure the length matches
            if len(future_values) == prediction_horizon:
                df_updated.loc[extended_index, col_name] = future_values
    
        elif col_name in macro_updated.columns:
            # Get the future values for this column
            future_value = future_forecaster_ts_values_raw[idx]
            future_values = np.linspace(macro_data[col_name][-1],
                            future_value,
                            prediction_horizon)
            future_forecaster_ts_values.append(future_values)
            if len(future_values) == prediction_horizon:
                macro_updated.loc[extended_index, col_name] = future_values
    return df_updated, macro_updated, future_forecaster_ts_values


def load_models(model_name: str,
                model_save_path: str,
                category_map: dict[str, str] = CATEGORY_MAP):
    f_dicts = {}
    for target_cat in set(category_map.values()):
        f = ClusterForecaster.load(model_save_path.format(model_name=model_name, category_name=target_cat.replace("/", "_")))
        f_dicts[target_cat] = f

    return f_dicts


def form_input_to_forecasting(df: pd.DataFrame,
                              macro_data: pd.DataFrame,
                              predictors_lst: list[str],
                              target_col_suffix: str,
                              future_forecaster_ts_name: list[str],
                              future_forecaster_ts_values: list[list[Any]],
                              past_idx,
                              future_idx,
                              use_future_macro,
                              is_backtest: bool,
                              model_type: str,
                              input_chunk_length: int = 12): 
    cov_past = pd.DataFrame(index=past_idx)
    cov_past = pd.concat([cov_past, macro_data[macro_data.index <= past_idx[-1]]], axis=1)  if is_backtest else pd.concat([cov_past, macro_data[macro_data.index < future_idx[0]]], axis=1) 
    if predictors_lst:
        for additional_cluster in predictors_lst:
            all_add = df[f"{additional_cluster}{target_col_suffix}"].shift(1)
            cov_past[f"{additional_cluster}_lag1"] = all_add.reindex(past_idx)

    cov_past = cov_past.fillna(method="bfill").fillna(method="ffill")
    cov_future = None
    if is_backtest:
        return cov_past, cov_future
    if use_future_macro and future_forecaster_ts_name:
        # For TFT/N-BEATS, future covariates must start earlier
        if model_type in ["tftmodel", "nbeats", "n-beats", "randomforest", "chronos"]:
            future_cov_start = future_idx[0] - pd.DateOffset(months=input_chunk_length)
        else:
            future_cov_start = future_idx[0]
        
        future_cov_end = future_idx[-1]
        future_cov_range = pd.date_range(start=future_cov_start,
                                        end=future_cov_end,
                                        freq='MS')
        
        cov_future = pd.DataFrame(index=future_cov_range)
        # Fill with known future values for future period
        # Fill with historical data for past period
        for i, col in enumerate(future_forecaster_ts_name):
            if col in macro_data.columns:
                # Get historical values for the lookback period
                hist_data = macro_data.loc[future_cov_start:future_idx[0] - pd.DateOffset(months=1), col]
                
                # Get future values (your linear interpolation values)
                future_vals = future_forecaster_ts_values[i]
                
                # Combine
                all_vals = list(hist_data.values) + list(future_vals)
                cov_future[col] = all_vals
            elif col in df.columns:
                # Get historical values for the lookback period
                hist_data = df.loc[future_cov_start:future_idx[0] - pd.DateOffset(months=1), col]
                
                # Get future values (your linear interpolation values)
                future_vals = future_forecaster_ts_values[i]
                
                # Combine
                all_vals = list(hist_data.values) + list(future_vals)
                cov_future[col] = all_vals
    
    return cov_past, cov_future


def train_model_and_eval_res(df_in: pd.DataFrame,
                             macro_data: pd.DataFrame,
                             past_idx,
                             future_idx,
                             predict_up_to,
                             category_map: dict ,
                             best_predictors_for_idx: dict,
                             target_col_suffix:str,
                             future_forecaster_ts_name: list[str],
                             future_forecaster_ts_values: list[list[Any]] ,
                             model_name: str ,
                             use_future_macro: bool,
                             prediction_horizon: int,
                             dir_to_save_plots: str,
                             dir_to_save_tables: str,
                             persist_model: bool,
                             refit: bool,
                             prediction_lag: int,
                             is_backtest: bool,
                             ts_freq: str = "MS",
                             pred_index: Optional[Any] = None
                            ):
    df = df_in.dropna()
    all_clusters = sorted(set(category_map.values()))
    forecasts_for_cluster: dict[str, Type[TimeSeries]] = {}
    forecasters_for_cluster: dict[str, Type[ClusterForecaster]] = {}
    if model_name not in ("regression"):
        past_data_for_train = df[df.index<=past_idx[-1]]
        macro_data_for_train = macro_data[macro_data.index<=past_idx[-1]]
    else:
        past_data_for_train = df[df.index<=past_idx[0]]
        macro_data_for_train = macro_data[macro_data.index<=past_idx[0]]
    trained_models: dict[str, Type[ClusterForecaster]] = {}
    for target_cat in all_clusters:
        macro_data_flag = target_cat in macro_data.columns

        target_cat_load_name = target_cat.replace("/", "_")
        model_save_path: str = f"{model_name}_dataset/{model_name}_for_{target_cat_load_name}__{prediction_horizon}"

        if macro_data_flag: 
            start_date = past_data_for_train.index.min()
            
            macro_data_for_train = macro_data_for_train.loc[start_date:]

        predictors_lst:list[str] = best_predictors_for_idx.get(target_cat)

        forecaster = ClusterForecaster(
            cluster_data=past_data_for_train,
            macro_data=macro_data_for_train,
            category_map=category_map,
            freq=ts_freq,
            target_col_suffix=target_col_suffix,
            dir_to_save_plots=dir_to_save_plots,
            dir_to_save_tables=dir_to_save_tables,
            future_macro_col=future_forecaster_ts_name,
            model_type=model_name
        )
        if os.path.isdir("".join(model_save_path.split("/")[:-1])) and not refit:
            try:
                forecaster = forecaster.load(model_save_path)
            except Exception as e:
                print(f"Did not found {model_save_path};Faild with error: {e}; Gonna train")
                forecaster.train(target_cat,
                                 predictors_lst,
                                 use_future_macro=use_future_macro,
                                 past_covariate_lags=prediction_lag,
                                 output_chunk_length_model=prediction_horizon,
                                 is_backtest=is_backtest
                                 )
        else:
            forecaster.train(target_cat,
                             predictors_lst,
                             use_future_macro=use_future_macro,
                             past_covariate_lags=prediction_lag,
                             output_chunk_length_model=prediction_horizon,
                             is_backtest=is_backtest
                            )
        cov_past, cov_future = form_input_to_forecasting(
            df,
            macro_data,
            predictors_lst,
            target_col_suffix,
            future_forecaster_ts_name,
            future_forecaster_ts_values,
            past_idx, future_idx,
            use_future_macro,
            is_backtest,
            model_type=model_name,
            input_chunk_length=prediction_lag
        )

        df_target: pd.DataFrame = macro_data if macro_data_flag else df
        col_name: str = f"{target_cat}" if macro_data_flag else f"{target_cat}{target_col_suffix}" 
        
        if use_future_macro and not is_backtest:
            target_s = df_target[df_target.index >= pd.to_datetime(future_idx[0])][col_name]
        elif is_backtest or not use_future_macro:
            target_s = df_target[df_target.index >= pd.to_datetime(past_idx[0])][col_name]
        if target_cat in future_forecaster_ts_name:
            for idx, ts_future_known in enumerate(future_forecaster_ts_name):

                if target_cat == ts_future_known:
                    preds = pd.DataFrame({f"{target_cat}_preds": future_forecaster_ts_values[idx]})
                    preds.index = pd.date_range(start=future_idx[-1] + pd.DateOffset(months=1),
                          end=predict_up_to,
                          freq='MS') if pred_index is None else pred_index
                    preds = TimeSeries.from_dataframe(preds)
                    break
        else:
            preds = forecaster.forecast(prediction_horizon,
                                cov_past,
                                cov_future if use_future_macro else None,
                                is_backtest=is_backtest)

        forecasters_for_cluster[target_cat] = forecaster
        forecasts_for_cluster[target_cat] = preds


        if use_future_macro and not is_backtest:
            preds.index = target_s.index
        if is_backtest:
            series_pd = preds.to_dataframe()
            series_pd.index = past_idx
            
            # Convert back to TimeSeries
            preds = TimeSeries.from_dataframe(series_pd, freq=ts_freq)
            # preds = preds.with_times(past_idx)

        preds_ts_no_neg = TimeSeries.from_dataframe(preds.to_dataframe().abs())

        pred_ts, actual_ts, metrics = forecaster.plot_and_return_data_backtest(TimeSeries.from_series(target_s.abs()),
                                                                               preds_ts_no_neg,
                                                                               target_cat,
                                                                               future_idx,
                                                                               predictors_lst)
        # persisting
        trained_models[target_cat] = forecaster
        if persist_model:
            forecaster.save(model_save_path)
    return trained_models
            

def disaggregate_predictions_to_TS(df_in: pd.DataFrame,
                                   pred_df: pd.DataFrame,
                                   cluster_name: str,
                                   train_end: pd.Timestamp,
                                   future_forecaster_changes: list[float],
                                   increase_by_changes: bool,
                                   is_backtest: bool,
                                   category_map: dict[str, str] = CATEGORY_MAP,
                                   prediction_horizon: int=12):
    df = df_in.copy()
    df.columns = [col_name.replace("/", " ") for col_name in df.columns]
    # Берем в качестве нормировки последние доступные значения
    
    if is_backtest:
        scaling_date = train_end - pd.DateOffset(months=prediction_horizon)
    else:
        scaling_date = train_end 
    print(scaling_date)
   
    scaled_vals = np.diff(pred_df[f"{cluster_name}_norm_sum"] / df[df.index == scaling_date][f"{cluster_name}_norm_sum"].iloc[0])
    scaled_vals = np.cumsum(np.array(list(scaled_vals)+[0])) + 1

    ts_to_scale = []
    for ts_name, cat_name in category_map.items():
        if cat_name == cluster_name:
            ts_to_scale.append(ts_name)
    for item in ts_to_scale:
        if item not in df.columns:
            continue
        df_cut = df[f"{item}"]
        # df_cut = np.mean(df_cut.loc[df_cut.index <= scaling_date].iloc[-6:-1])
        df_cut = np.mean(df_cut.loc[df_cut.index <= scaling_date].iloc[-1])
        
        if increase_by_changes:
            changes = (1 + np.array(future_forecaster_changes)*0.2)
            pred_df[f"{item}_preds"] = df_cut*(scaled_vals + changes)/2
        else:
            pred_df[f"{item}_preds"] = df_cut*scaled_vals

    return pred_df


def get_prediction_for_ts(df_in: pd.DataFrame,
                          model_category,
                          ts_to_predict_name: str,
                          model_name: str,
                          macro_data: pd.DataFrame,
                          past_idx,
                          future_idx,
                          category_map: dict,
                          best_predictors_for_idx: dict,
                          target_col_suffix:str,
                          future_forecaster_ts_name:str,
                          future_forecaster_ts_values: str,
                          use_future_macro: bool,
                          prediction_horizon: int,
                          is_backtest: bool,
                          prediction_lag: int,
                          predicted_index = None
):
    df = df_in.dropna()
    for idx, ts_name in enumerate(future_forecaster_ts_name):
        if ts_name == ts_to_predict_name:
            df_preds = pd.DataFrame({f"{ts_to_predict_name}_preds": future_forecaster_ts_values[idx]})
            if predicted_index is not None:
                df_preds.index = predicted_index
            return df_preds[f"{ts_to_predict_name}_preds"], df_preds

    ts_name_normalized = ts_to_predict_name.replace("/", " ")
    df_target = df if ts_name_normalized in df.columns else macro_data
    is_predicting_macro = ts_to_predict_name in macro_data.columns

    category_to_pred = category_map[ts_to_predict_name]
    predictors_lst = best_predictors_for_idx[category_to_pred]
    predictors_lst = predictors_lst if predictors_lst else []
    try:
        cov_past, cov_future = form_input_to_forecasting(
                                                        df,
                                                        macro_data,
                                                        predictors_lst,
                                                        target_col_suffix,
                                                        future_forecaster_ts_name,
                                                        future_forecaster_ts_values,
                                                        past_idx, future_idx,
                                                        use_future_macro,
                                                        is_backtest,
                                                        model_type=model_name,
                                                        input_chunk_length=prediction_lag
                                )
    except Exception as e:
        print(f"Failed with new error: {e}")
    

    predictions = model_category.forecast(prediction_horizon,
                                        cov_past,
                                        cov_future if use_future_macro else None,
                                        is_backtest=is_backtest).to_dataframe().reset_index(drop=True)
    if len(future_forecaster_ts_name) == 0:
        print(future_forecaster_ts_name)
        increase_by_changes = False
        changes = []
    else:
        increase_by_changes = category_map.get(ts_to_predict_name, "") == category_map.get(future_forecaster_ts_name[0], "")
        changes = [(x / future_forecaster_ts_values[0][0]) - 1 for x in future_forecaster_ts_values[0]]
        print(f'changes = {changes}')

    ts_to_cats = {k: category_map[k] for k in set(list(category_map.keys())) - set(['Подсолнечное масло (наливом) не бутилированное, не'])}

    if is_predicting_macro:
        predictions.index = predicted_index
        predictions[f"{ts_to_predict_name}_preds"] = predictions[ts_to_predict_name]
        return predictions[f"{ts_to_predict_name}_preds"], predictions

    else:
        df_per_ts = disaggregate_predictions_to_TS(
            df_target,
            predictions,
            category_to_pred,
            past_idx[-1],
            changes,
            increase_by_changes,
            is_backtest,
            category_map=category_map,
            prediction_horizon=prediction_horizon
        )

    if predicted_index is not None:
        df_per_ts.index = predicted_index
    # returns (TS prediction, metric result(if backtest), Cluster predictions)
    return df_per_ts[f"{ts_to_predict_name}_preds"], predictions
