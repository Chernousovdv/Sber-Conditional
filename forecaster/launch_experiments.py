# scripts/run_forecasting.py
import os
from pathlib import Path
import pandas as pd
from forecasting.evaluator import ForecasterEvaluator
from forecasting.plotting import plot_horizons_same_lags
from utils.utils import plot_mode_top_features_grid
from .utils import ForecasterEvaluator
from typing import Callable


def save_results_to_csv(df_to_save:pd.DataFrame,
                        name: str,
                        save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    df_to_save.to_csv(os.path.join(save_dir, f"{name}_results.csv"), index=False, encoding="utf-8")
    print("Saved!")


def start_experiments(apk_nona_en: pd.DataFrame,
                      global_macros_en: pd.DataFrame,
                      regressors: dict[str, Callable],
                      forecaster_evaluator: ForecasterEvaluator,
                      lags_to_try: list[int] = [6, 12, 24],
                      horizons: list[int] = [12, 24],):
    for predict_col in apk_nona_en.drop(columns=["USDRUB", "Unnamed: 0"]).columns.tolist():
        print(f"{predict_col}")
        normed_predict_col_name = predict_col.replace("/", " ")
        
        print("="*100)
        for target_column in apk_nona_en.drop(columns=["USDRUB", "Unnamed: 0"]).columns.tolist():
            normed_target_col_name = target_column.replace("/", " ")
            
            for algo_name, regr in regressors.items():
                file_path = Path(f"results/{algo_name}_{normed_target_col_name}_{normed_predict_col_name}_tables/feature_imp_df_results.csv")
                if file_path.is_file():
                    print(f"Already exists skip {file_path}")
                    continue
                normed_target_col_name = target_column.replace("/", " ")
                if len(normed_predict_col_name) + len(normed_target_col_name) >= 120:
                    normed_predict_col_name = normed_predict_col_name[:50]
                    normed_target_col_name = normed_target_col_name[:50]
    
                print(f"Starting predicting {normed_target_col_name} by {normed_predict_col_name}")
                series_df = apk_nona_en[["USDRUB", target_column, predict_col]]
                series_df = series_df.rename(columns={target_column: normed_target_col_name,
                                                      predict_col: normed_predict_col_name})
    
                macro_df = global_macros_en.copy()
    
                summary_df, detailed_df, preds_df, feature_imp_df = forecaster_evaluator.evaluate_models_with_macros_kfolds_with_preds(
                    series_full = series_df,
                    exog_full   = macro_df,
                    target_level = normed_target_col_name,
                    predicted_by=normed_predict_col_name,
                    regressors = {algo_name: regr},
                    lags_to_try = lags_to_try,
                    horizons = horizons,
                    n_folds = 3,
                    fold_stride_months = None,
                    fallback_repeat_last_exog = False,
                    verbose = True
                )
                print(summary_df)
                print(detailed_df.head())
    
                try:
                    row = preds_df.iloc[0]
                    print(row['cutoff'], row['horizon_months'])
                    print(row['preds_series'].head())
                    print(row['actual_series'].head())
                    for lag in lags_to_try:
                        plot_horizons_same_lags(
                            preds_df = preds_df,
                            series_full = series_df,
                            model = algo_name,
                            lags = lag,
                            target_level = normed_target_col_name,
                            predicted_by = normed_predict_col_name,
                            horizons = horizons,   # which horizons to show
                            fold_selector = 'last',    # use each horizon's last fold
                            show_actual_full = True
                        )
                    save_results_to_csv(summary_df, name="summary_df",
                                        save_dir=f"results/{algo_name}_{normed_target_col_name}_{normed_predict_col_name}_tables")
                    save_results_to_csv(detailed_df, name="detailed_df",
                                        save_dir=f"results/{algo_name}_{normed_target_col_name}_{normed_predict_col_name}_tables")
                    save_results_to_csv(preds_df, name="preds_df",
                                        save_dir=f"results/{algo_name}_{normed_target_col_name}_{normed_predict_col_name}_tables")
                    save_results_to_csv(feature_imp_df, name="feature_imp_df",
                                        save_dir=f"results/{algo_name}_{normed_target_col_name}_{normed_predict_col_name}_tables")
                    try:
                        plot_mode_top_features_grid(normed_target_col_name,
                                normed_predict_col_name,
                                algo_name=algo_name,
                                figsize_per_ax=(12,10),
                                show=False,
                                save_dir=f"results/{algo_name}_{normed_target_col_name}_{normed_predict_col_name}_featureimportance_new")
                    except Exception as e:
                        print(e)
                        print("="*100)
                except Exception as e:
                    print(e)
                    print("*"*100)
                    continue