from itertools import combinations
from typing import Type, Literal, Any
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
        # print(entry.name)
        d[entry.name] = pd.read_excel(f"{directory}{entry.name}")
    elif entry.is_file() and entry.name.endswith('.csv'):
        formed[entry.name] = pd.read_csv(f"{directory}{entry.name}")

global_macros = d["global_macro.xlsx"]
apk_nona = formed["apk_nona.csv"]

chemicals_nona = formed["chemicals_nona.csv"]
chemicals_nona["Date"] = pd.to_datetime(chemicals_nona["Date"])
chemicals_nona = chemicals_nona.set_index('Date')

global_macros = global_macros.drop(columns=["Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),  .1"])

apk_nona["Date"] = apk_nona["Unnamed: 0"]
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


PREDICTION_HORIZON = 12
PREDICTOR_LAG = 12

IS_BACKTEST: bool = False

SINGLE_DATAPOINT_TO_PRED = "2026-04-14"


TARGET_COL_SUFFIX = "_sum"
MODEL_NAME_FOR_FORECASTER: Literal["tftmodel", "regression", "nbeats", "naiveseasonal", "naivedrift", "naivemean", "naivemovingaverage"] = "tftmodel"
USE_FUTURE_COV = MODEL_NAME_FOR_FORECASTER in ("tftmodel", "regression")
TS_TO_PREDICT = "Соя"

FUTURE_DATE = apk_nona_en_rol.index[-1] - pd.DateOffset(months=PREDICTION_HORIZON)
BACKTEST_DATE = apk_nona_en_rol.index[-1] - pd.DateOffset(months=PREDICTION_HORIZON*2)

train_end = BACKTEST_DATE
past_idx = pd.date_range(start=train_end + pd.DateOffset(months=1),
                           periods=PREDICTION_HORIZON,
                           freq="MS")
future_idx = pd.date_range(start=past_idx[-1] + pd.DateOffset(months=1),
                           periods=PREDICTION_HORIZON,
                           freq="MS")

dates = get_month_beginnings(SINGLE_DATAPOINT_TO_PRED)
date_to_pred = dates[1] if len(dates) > 1 else dates[0]
predict_end = pd.to_datetime(date_to_pred)
last_date = df_combined.reset_index()["Date"].iloc[-1]

TOTAL_RANGE = (predict_end.year - last_date.year)*12 + (predict_end.month - last_date.month)

start_date = pd.to_datetime(SINGLE_DATAPOINT_TO_PRED) - relativedelta(months=TOTAL_RANGE)
end_date = pd.to_datetime(SINGLE_DATAPOINT_TO_PRED)

# Generate the list of monthly dates
DATE_LIST = pd.date_range(start=start_date + pd.DateOffset(months=1), end=end_date + pd.DateOffset(months=1), freq='MS').tolist()  # MS = Month Start

DATE_LIST[-1] = pd.to_datetime(SINGLE_DATAPOINT_TO_PRED)


## Parameters with known future values
FUTURE_FORECASTER_TS_NAME = ["Подсолнечник", "Ключевая ставка, годовых"]
FUTURE_FORECASTER_TS_VALUES_RAW = [22.1, 17.1]
FUTURE_FORECASTER_TS_VALUES = []
for idx in range(len(FUTURE_FORECASTER_TS_NAME)):
    pred_ts_name = FUTURE_FORECASTER_TS_NAME[idx]
    df_to_get = df_combined if pred_ts_name in df_combined.columns else macro_data
    FUTURE_FORECASTER_TS_VALUES.append(
        np.linspace(df_to_get[pred_ts_name][-1],
                    FUTURE_FORECASTER_TS_VALUES_RAW[idx],
                    TOTAL_RANGE)
    )

MODEL_SAVE_PATH = "{model_name}_dataset/{model_name}_for_{category_name}"

PERSIST_MODEL = True
REFIT_MODEL = False


def load_models(model_name: str = MODEL_NAME_FOR_FORECASTER,
                model_save_path: str = MODEL_SAVE_PATH,
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
                              future_forecaster_ts_names: list[str],
                              future_forecaster_ts_values: list[list[Any]],
                              past_idx,
                              future_idx,
                              use_future_macro
                              ):
    cov_past = pd.DataFrame(index=past_idx)
    cov_past = pd.concat([cov_past, macro_data[macro_data.index < future_idx[0]]], axis=1)
    if predictors_lst:
        for additional_cluster in predictors_lst:
            all_add = df[f"{additional_cluster}{target_col_suffix}"].shift(1)
            cov_past[f"{additional_cluster}_lag1"] = all_add.reindex(past_idx)

    cov_past = cov_past.fillna(method="bfill").fillna(method="ffill")
    if use_future_macro:
        cov_future_values = []
        for future_forecaster_ts_name, future_forecaster_single_ts_values in zip(future_forecaster_ts_names, future_forecaster_ts_values):
            if future_forecaster_ts_name in macro_data.columns:
                future_series_values = macro_data[future_forecaster_ts_name]
            elif future_forecaster_ts_name in df.columns:
                future_series_values = df[future_forecaster_ts_name]
            else:
                raise ValueError(f"future_forecaster_ts_name '{future_forecaster_ts_name}' not found in macro_data or df columns")
            if IS_BACKTEST:
                cov_future_values.append(pd.DataFrame(future_series_values[past_idx]))
                cov_future_values.append(pd.DataFrame(future_series_values[future_idx]))

            else:
                cov_future_values.append(pd.DataFrame(future_series_values[past_idx]))
                cov_future_values.append(pd.DataFrame(future_forecaster_single_ts_values,
                                                       index=future_idx,
                                                       columns=[future_forecaster_ts_name])
                                         )
                
        cov_future = pd.concat(cov_future_values)
        cov_future = cov_future.groupby(cov_future.index).first()
    else:
        cov_future = None

    return cov_past, cov_future


def train_model_and_eval_res(df_in: pd.DataFrame,
                             macro_data: pd.DataFrame,
                             past_idx = past_idx,
                             future_idx = future_idx,
                             category_map: dict = CATEGORY_MAP,
                             best_predictors_for_idx: dict = BEST_PREDICTORS_FOR_INDEX,
                             target_col_suffix:str = TARGET_COL_SUFFIX,
                             future_forecaster_ts_name: list[str] = FUTURE_FORECASTER_TS_NAME,
                             future_forecaster_ts_values: list[list[Any]] = FUTURE_FORECASTER_TS_VALUES,
                             model_name: str = MODEL_NAME_FOR_FORECASTER,
                             use_future_macro: bool = USE_FUTURE_COV,
                             prediction_horizon: int = PREDICTION_HORIZON,
                             dir_to_save_plots: str = f"darts_result_pairs_{MODEL_NAME_FOR_FORECASTER}",
                             dir_to_save_tables: str = f"darts_result_tables_pairs_{MODEL_NAME_FOR_FORECASTER}",
                             persist_model: bool = PERSIST_MODEL,
                             refit: bool = REFIT_MODEL,
                             prediction_lag: int = PREDICTOR_LAG
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
        model_save_path: str = f"{model_name}_dataset/{model_name}_for_{target_cat_load_name}"

        if macro_data_flag: 
            start_date = past_data_for_train.index.min()
            
            macro_data_for_train = macro_data_for_train.loc[start_date:]

        predictors_lst:list[str] = best_predictors_for_idx.get(target_cat)
    
        forecaster = ClusterForecaster(
            cluster_data=past_data_for_train,
            macro_data=macro_data_for_train,
            category_map=category_map,
            freq="M",
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
                forecaster.train(target_cat, predictors_lst, use_future_macro=use_future_macro, past_covariate_lags=PREDICTOR_LAG)
        else:
            forecaster.train(target_cat, predictors_lst, use_future_macro=use_future_macro, past_covariate_lags=PREDICTOR_LAG)

        cov_past, cov_future = form_input_to_forecasting(df,
                                                          macro_data,
                                                          predictors_lst,
                                                          target_col_suffix,
                                                          future_forecaster_ts_name,
                                                          future_forecaster_ts_values,
                                                          past_idx,
                                                          future_idx,
                                                          use_future_macro)
        print("@"*100)
        print(cov_future if use_future_macro else None)
        print("@"*100)
        preds = forecaster.forecast(prediction_horizon,
                            cov_past,
                            cov_future if use_future_macro else None)
        # if MODEL_NAME_FOR_FORECASTER in ("tftmodel"):
        #     preds *= 11
        forecasters_for_cluster[target_cat] = forecaster
        forecasts_for_cluster[target_cat] = preds

        col_name: str = f"{target_cat}" if macro_data_flag else f"{target_cat}{target_col_suffix}" 
        df_target: pd.DataFrame = macro_data if macro_data_flag else df
        if use_future_macro:
            target_s = df_target[df_target.index >= pd.to_datetime(future_idx[0])][col_name]
        else:
            target_s = df_target[df_target.index >= pd.to_datetime(past_idx[0])][col_name]
        if USE_FUTURE_COV:
            preds.index = target_s.index
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
                                   category_map: dict[str, str] = CATEGORY_MAP):
    df = df_in.copy()
    df.columns = [col_name.replace("/", " ") for col_name in df.columns]
    print(pred_df[f"{cluster_name}_norm_sum"])
    print(pred_df[f"{cluster_name}_norm_sum"].shape)
    # Берем в качестве нормировки последнее доступное значение из обучающей выборке
    scaled_vals = pred_df[f"{cluster_name}_norm_sum"] / df[df.index == train_end][f"{cluster_name}_norm_sum"].iloc[0] 
    print(scaled_vals)
    print(df.columns)
    ts_to_scale = []
    for ts_name, cat_name in category_map.items():
        if cat_name == cluster_name:
            ts_to_scale.append(ts_name)
    print(ts_to_scale)
    for item in ts_to_scale:
        df_cut = df[f"{item}"]
        df_cut = df_cut[df_cut.index == train_end].iloc[0]
        print(scaled_vals.shape)
        print("="*100)
        print(df_cut)
        print("="*100)
        pred_df[f"{item}_preds"] = df_cut*scaled_vals

    return pred_df
    

def get_prediction_for_ts(df_in: pd.DataFrame,
                          model_category,
                          ts_to_predict_name: str = TS_TO_PREDICT,
                          macro_data: pd.DataFrame = macro_data,
                          past_idx = past_idx,
                          future_idx = future_idx,
                          category_map: dict = CATEGORY_MAP,
                          best_predictors_for_idx: dict = BEST_PREDICTORS_FOR_INDEX,
                          target_col_suffix:str = TARGET_COL_SUFFIX,
                          future_forecaster_ts_name:str = FUTURE_FORECASTER_TS_NAME,
                          future_forecaster_ts_values: str = FUTURE_FORECASTER_TS_VALUES,
                          model_name: str = MODEL_NAME_FOR_FORECASTER,
                          use_future_macro: bool = USE_FUTURE_COV,
                          prediction_horizon: int = PREDICTION_HORIZON,
                          weight_method_apply: str = "last"
):
    df = df_in.dropna()
    is_predicting_macro = ts_to_predict_name in macro_data.columns

    category_to_pred = category_map[ts_to_predict_name]
    predictors_lst = best_predictors_for_idx[category_to_pred]
    predictors_lst = predictors_lst if predictors_lst else []

    cov_past, cov_future = form_input_to_forecasting(df,
                                                      macro_data,
                                                      predictors_lst,
                                                      target_col_suffix,
                                                      future_forecaster_ts_name,
                                                      future_forecaster_ts_values,
                                                      past_idx,
                                                      future_idx,
                                                      use_future_macro)
    
    predictions = model_category.forecast(TOTAL_RANGE,
                                        cov_past,
                                        cov_future if USE_FUTURE_COV else None).to_dataframe().reset_index(drop=True)
    ts_to_cats = {k: category_map[k] for k in set(list(category_map.keys())) - set(['Подсолнечное масло (наливом) не бутилированное, не'])}
    cols = [col for col, cat in ts_to_cats.items() if cat == category_to_pred and col in df.columns]

    # if MODEL_NAME_FOR_FORECASTER in ("tftmodel"):
    #     predictions *= 11
    if is_predicting_macro:
        future_idx[-1]
        predictions.index = pd.date_range(future_idx[-1], future_idx[-1] + pd.DateOffset(months=PREDICTION_HORIZON-1), freq='MS')
        return predictions, None, predictions

    else:
        df_per_ts = disaggregate_predictions_to_TS(
            df,
            predictions,
            category_to_pred,
            past_idx[-1],
            category_map=CATEGORY_MAP
        )
        print("@"*100)
        print(df_per_ts)
        print("@"*100)


    df_per_ts.index = DATE_LIST
    # returns (TS prediction, metric result(if backtest), Cluster predictions)
    return df_per_ts[f"{ts_to_predict_name}_preds"], predictions
