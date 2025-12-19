import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from io import BytesIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import combinations
from typing import Literal, Any, List, Dict, Optional

# Local imports
from utils.constants import (
    CATEGORY_MAP,
    BEST_PREDICTORS_FOR_INDEX
)
from utils.utils import (
    plot_known_feature_value,
    plot_graphics_for_each_ts,
    get_month_beginnings
)
from utils.parametrized_values import (
    form_df_future,
    load_models,
    form_input_to_forecasting,
    train_model_and_eval_res,
    get_prediction_for_ts,
    df_combined,
    macro_data
)
from utils.cluster_forecast import ClusterForecaster
import utils.cluster_forecast # Import module to monkeypatch

# Monkeypatch to preventutils/cluster_forecast.py from opening browser tabs with fig.show()
def quiet_plot(*args, **kwargs):
    return go.Figure()
utils.cluster_forecast.plot_preds_on_curve = quiet_plot

# Darts imports for Single Series mode
from darts import TimeSeries
from darts.models import (
    NaiveSeasonal, NaiveDrift, NaiveMean, NaiveMovingAverage,
    ExponentialSmoothing, Theta, RandomForestModel, NBEATSModel
)
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.metrics import mape as darts_mape, mae as darts_mae, rmse as darts_rmse

# Model parameters configuration for Single Series mode
MODEL_PARAMS_CONFIG = {
    "naiveseasonal": [
        {"key": "K", "label": "K (сезонный период)", "type": "number", "default": 12, "min": 1, "max": 365},
    ],
    "randomforest": [
        {"key": "lags", "label": "Лаги целевой переменной", "type": "number", "default": 12, "min": 1, "max": 48},
        {"key": "lags_past_covariates", "label": "Лаги ковариат", "type": "number", "default": 12, "min": 1, "max": 48},
        {"key": "output_chunk_length", "label": "Длина прогноза (chunk)", "type": "number", "default": 12, "min": 1, "max": 24},
    ],
    "linearregression": [
        {"key": "lags", "label": "Лаги целевой переменной", "type": "number", "default": 12, "min": 1, "max": 48},
        {"key": "lags_past_covariates", "label": "Лаги ковариат", "type": "number", "default": 12, "min": 1, "max": 48},
        {"key": "output_chunk_length", "label": "Длина прогноза (chunk)", "type": "number", "default": 1, "min": 1, "max": 24},
    ],
    "naivemovingaverage": [
        {"key": "window", "label": "Окно усреднения", "type": "number", "default": 3, "min": 1, "max": 48},
    ],
    "exponentialsmoothing": [
        {"key": "seasonal_periods", "label": "Сезонные периоды", "type": "number", "default": 12, "min": 1, "max": 365},
    ],
    "theta": [
        {"key": "theta", "label": "Theta", "type": "number", "default": 2, "min": 0, "max": 100},
        {"key": "seasonality_period", "label": "Сезонный период", "type": "number", "default": 12, "min": 1, "max": 365},
    ],
    "nbeats": [
        {"key": "input_chunk_length", "label": "Длина входа (input_chunk)", "type": "number", "default": 24, "min": 1, "max": 96},
        {"key": "output_chunk_length", "label": "Длина прогноза (output_chunk)", "type": "number", "default": 12, "min": 1, "max": 48},
        {"key": "n_epochs", "label": "Эпохи обучения", "type": "number", "default": 50, "min": 1, "max": 500},
    ],
}

# Models that support covariates
COVARIATE_MODELS = ["linearregression", "randomforest"]

# =====================
# Configuration & CSS
# =====================

st.set_page_config(
    page_title="GMTS Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS from example.py
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #08a652;
        margin-bottom: 1rem;
    }
    
    /* Main layout columns - large gap */
    .main > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:first-child {
        gap: 2rem;
    }
    
    /* Add some padding to column content */
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stVerticalBlock"]) {
        padding-right: 1rem;
    }
    
    /* Compact metrics display - fit 5 in one row */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.5rem 0.5rem;
        border-radius: 6px;
        border-left: 3px solid #08a652;
        min-width: 0;
    }
    
    [data-testid="stMetric"] > div {
        width: fit-content;
        gap: 0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        min-height: auto !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
    
    /* Reduce gap ONLY in blocks that directly contain metrics (not parent columns) */
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > div > [data-testid="stVerticalBlock"] > [data-testid="stMetric"]) {
        gap: 0.5rem !important;
    }
    
    /* Sidebar compact header - position logo next to collapse button */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Logo styling in sidebar header */
    .sidebar-logo {
        font-size: 1.3rem;
        font-weight: 700;
        color: #08a652;
        margin: 0;
        padding: 0;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Visualization Helpers
# =====================

def plot_forecast_chart(
    history_index: List[str],
    history_values: List[float],
    forecast_index: List[str],
    forecast_values: List[float],
    title: str = "Прогноз временного ряда",
    quantiles: Optional[Dict[str, List[float]]] = None,
    forecast_start: str = None,
) -> go.Figure:
    """Create forecast visualization."""
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=history_index,
        y=history_values,
        name="История",
        line=dict(color="#808080", width=2),  # Gray for historical
        mode="lines+markers"
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_values,
        name="Прогноз",
        line=dict(color="#08a652", width=2, dash="dash"),  # Green for forecast
        mode="lines+markers"
    ))

    # Confidence intervals
    if quantiles and "0.1" in quantiles and "0.9" in quantiles:
        # Align lengths if needed
        q_len = len(forecast_index)
        q10 = list(quantiles["0.1"])[:q_len]
        q90 = list(quantiles["0.9"])[:q_len]
        
        fig.add_trace(go.Scatter(
            x=list(forecast_index) + list(forecast_index)[::-1],
            y=q10 + q90[::-1],
            fill="toself",
            fillcolor="rgba(8, 166, 82, 0.2)",  # Light green
            line=dict(color="rgba(255,255,255,0)"),
            name="80% интервал"
        ))

    # Forecast start line
    if forecast_start:
        fig.add_vline(x=forecast_start, line_dash="dash", line_color="red")

    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Значение",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def normalize_to_month(date_str: str) -> Optional[str]:
    """Normalize date string to YYYY-MM format."""
    if not date_str:
        return None
    try:
        d = pd.to_datetime(date_str)
        return d.strftime("%Y-%m")
    except:
        return None


def calculate_monthly_mape(
    history_index: List[str],
    history_values: List[float],
    forecast_index: List[str],
    forecast_values: List[float]
) -> List[Dict]:
    """Calculate monthly MAPE metrics."""
    # Build actual lookup
    actual_lookup = {}
    for idx, val in zip(history_index, history_values):
        month_key = normalize_to_month(idx)
        if month_key:
            actual_lookup[month_key] = val

    metrics = []
    prev_forecast = None

    for i, (date_str, forecast_val) in enumerate(zip(forecast_index, forecast_values)):
        if forecast_val is None:
            continue

        month_key = normalize_to_month(date_str)
        if not month_key:
            continue

        actual_val = actual_lookup.get(month_key)

        error = forecast_val - actual_val if actual_val is not None else None
        mape = abs(error / actual_val) * 100 if actual_val and abs(actual_val) > 0.01 else None

        # Determine trend
        direction = "neutral"
        if prev_forecast is not None:
            if forecast_val > prev_forecast * 1.01:
                direction = "up"
            elif forecast_val < prev_forecast * 0.99:
                direction = "down"

        metrics.append({
            "month": month_key,
            "actual": actual_val,
            "forecast": forecast_val,
            "error": error,
            "mape": mape,
            "direction": direction
        })

        prev_forecast = forecast_val

    return metrics


def display_monthly_mape(
    history_index: List[str],
    history_values: List[float],
    forecast_index: List[str],
    forecast_values: List[float]
):
    """Display monthly MAPE analysis like React frontend."""
    monthly_metrics = calculate_monthly_mape(
        history_index, history_values, forecast_index, forecast_values
    )

    if not monthly_metrics:
        # If we can't align months (e.g. pure future forecast), just skip without noise
        return

    # Check if we have actuals to compare against
    has_actuals = any(m["actual"] is not None for m in monthly_metrics)
    if not has_actuals:
        return

    st.markdown("### 📊 Monthly MAPE Analysis")

    # MAPE Trend Chart
    months = [m["month"] for m in monthly_metrics]
    mape_values = [m["mape"] for m in monthly_metrics]

    fig = go.Figure()

    # MAPE line
    fig.add_trace(go.Scatter(
        x=months,
        y=mape_values,
        mode="lines+markers",
        name="MAPE %",
        line=dict(color="#3b82f6", width=2, shape="spline"),
        marker=dict(
            size=10,
            color=[
                "#10b981" if m is not None and m <= 5 else 
                "#f59e0b" if m is not None and m <= 10 else 
                "#ef4444" if m is not None else "#6b7280"
                for m in mape_values
            ]
        ),
        hovertemplate="%{x}<br>MAPE: %{y:.1f}%<extra></extra>"
    ))

    # Reference lines
    fig.add_hline(y=5, line_dash="dash", line_color="#10b981", 
                  annotation_text="Good (5%)", annotation_position="right")
    fig.add_hline(y=10, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Warning (10%)", annotation_position="right")

    fig.update_layout(
        title="MAPE Trend (%)",
        height=300,
        template="plotly_white",
        xaxis=dict(tickangle=-45),
        yaxis=dict(rangemode="tozero", title="MAPE %"),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)



    # Table
    st.markdown("#### Детальная таблица")

    table_data = []
    for m in monthly_metrics:
        mape_str = f"{m['mape']:.1f}%" if m['mape'] is not None else "-"
        error_str = f"{m['error']:+.2f}" if m['error'] is not None else "-"
        actual_str = f"{m['actual']:.2f}" if m['actual'] is not None else "-"

        # Trend emoji
        trend = "📈" if m["direction"] == "up" else "📉" if m["direction"] == "down" else "➖"

        table_data.append({
            "Месяц": m["month"],
            "Actual": actual_str,
            "Forecast": f"{m['forecast']:.2f}",
            "Error": error_str,
            "MAPE %": mape_str,
            "Тренд": trend
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

def display_metrics(metrics: Dict[str, float]) -> None:
    """Display metrics in a single row."""
    if not metrics:
        return

    valid_metrics = {k: v for k, v in metrics.items() if v is not None}
    if not valid_metrics:
        return

    metric_labels = {
        "mae": "MAE",
        "mape": "MAPE",
        "rmse": "RMSE",
        "wape": "WAPE",
        "smape": "sMAPE",
        "scaled_bias": "Bias"
    }

    # Display metrics in a single row using columns with gap="small"
    cols = st.columns(len(valid_metrics), gap="small")

    for i, (key, value) in enumerate(valid_metrics.items()):
        label = metric_labels.get(key, key.upper())
        try:
            # Handle numpy arrays or lists - take first element or mean
            if hasattr(value, '__iter__') and not isinstance(value, str):
                if hasattr(value, 'mean'):
                    val_float = float(value.mean())
                elif len(value) > 0:
                    val_float = float(value[0])
                else:
                    val_float = 0.0
            else:
                val_float = float(value)
            cols[i].metric(label, f"{val_float:.2f}")
        except:
            cols[i].metric(label, str(value))


# =====================
# Main App Layout
# =====================

# Sidebar Mode Toggle
with st.sidebar:
    st.markdown("## 📈 GMTS")
    
    # Mode toggle at very top
    is_gmts_mode = st.toggle(
        "GMTS Pipeline",
        value=True,
        help="Вкл — GMTS с предзагруженными данными. Выкл — Одиночный ряд с вашим файлом."
    )

# =====================
# Single Series Page (when toggle is OFF)
# =====================

if not is_gmts_mode:
    # Single Series mode - direct Darts forecasting
    
    # Initialize session state
    if "ss_forecast_result" not in st.session_state:
        st.session_state.ss_forecast_result = None
    if "ss_df" not in st.session_state:
        st.session_state.ss_df = None
    
    with st.sidebar:
        st.markdown("### 📁 Одиночный ряд")
        uploaded_file = st.file_uploader(
            "Загрузите CSV или Excel",
            type=["csv", "xlsx", "xls"],
        )
        
        df = None
        date_col = None
        target_col = None
        selected_covariates = []
        horizon = 12
        model_name = "NaiveSeasonal"
        model_params = {}
        is_backtest_ss = False
        forecast_start = None
        
        if uploaded_file:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Rename unnamed columns
            df.columns = ['Date' if str(c).startswith('Unnamed') or c == '' else c for c in df.columns]
            st.success(f"✅ {len(df)} строк, {len(df.columns)} колонок")
            
            # Detect columns
            all_cols = df.columns.tolist()
            date_cols = [c for c in all_cols if c == 'Date' or pd.api.types.is_datetime64_any_dtype(df[c]) or 
                         df[c].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any()]
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Column selection
            date_col = st.selectbox(
                "📅 Колонка даты",
                options=date_cols + [c for c in all_cols if c not in date_cols],
                index=0 if date_cols else 0
            )
            
            target_col = st.selectbox(
                "📊 Целевая колонка",
                options=numeric_cols,
                index=0 if numeric_cols else None
            )
            
            # Covariates
            available_covs = [c for c in numeric_cols if c != target_col]
            selected_covariates = st.multiselect(
                "Ковариаты (опц.)",
                options=available_covs
            )
            
            # Horizon
            horizon = st.number_input("Горизонт (мес.)", min_value=1, max_value=100, value=12)
            
            # Model selection
            model_options = [
                "NaiveSeasonal", "NaiveDrift", "NaiveMean", "NaiveMovingAverage",
                "ExponentialSmoothing", "Theta", "LinearRegression", "RandomForest", "NBEATS"
            ]
            model_name = st.selectbox("⚙️ Модель", model_options, index=0)
            model_lower = model_name.lower()
            
            # Model parameters
            param_config = MODEL_PARAMS_CONFIG.get(model_lower, [])
            if param_config:
                with st.expander("🔧 Параметры модели"):
                    for p in param_config:
                        if p["type"] == "number":
                            val = st.number_input(
                                p["label"],
                                min_value=int(p.get("min", 0)),
                                max_value=int(p.get("max", 1000)),
                                value=int(p["default"]),
                                key=f"ss_param_{p['key']}"
                            )
                            model_params[p["key"]] = int(val)
            
            # Backtest mode
            is_backtest_ss = st.checkbox("🔄 Бэктест", value=False)
            
            if is_backtest_ss:
                try:
                    df_work = df.copy()
                    df_work[date_col] = pd.to_datetime(df_work[date_col])
                    date_options = df_work[date_col].dt.strftime("%Y-%m-%d").tolist()
                    
                    if len(date_options) > 5:
                        default_idx = len(date_options) - horizon if len(date_options) > horizon else len(date_options) // 2
                        forecast_start = st.selectbox(
                            "Дата начала",
                            options=date_options,
                            index=default_idx
                        )
                except Exception as e:
                    st.warning(f"Ошибка: {e}")
            
            # Run button
            run_button = st.button("🚀 Прогноз", type="primary", use_container_width=True)
        else:
            st.info("👆 Загрузите файл")
            run_button = False
    
    # Main content for Single Series
    
    if uploaded_file and df is not None:
        # Prepare data
        try:
            df_viz = df.copy()
            df_viz[date_col] = pd.to_datetime(df_viz[date_col])
            df_viz = df_viz.sort_values(date_col)
        except Exception as e:
            st.warning(f"Не удалось подготовить данные: {e}")
            df_viz = None
        
        # Run forecast
        if run_button and df_viz is not None and target_col:
            with st.spinner("⏳ Выполняется прогноз..."):
                try:
                    # Prepare data
                    df_work = df_viz.set_index(date_col).sort_index()
                    df_work = df_work.dropna(subset=[target_col])
                    
                    # Convert to TimeSeries
                    target_series = df_work[target_col].astype(float)
                    target_ts = TimeSeries.from_series(target_series, freq='MS')
                    
                    # Prepare covariates if model supports them
                    cov_ts = None
                    if selected_covariates and model_lower in COVARIATE_MODELS:
                        cov_df = df_work[selected_covariates].astype(float)
                        cov_df = cov_df.fillna(method='ffill').fillna(method='bfill')
                        cov_ts = TimeSeries.from_dataframe(cov_df, freq='MS')
                    
                    # Create model
                    if model_lower == "naiveseasonal":
                        K = model_params.get("K", 12)
                        model = NaiveSeasonal(K=K)
                    elif model_lower == "naivedrift":
                        model = NaiveDrift()
                    elif model_lower == "naivemean":
                        model = NaiveMean()
                    elif model_lower == "naivemovingaverage":
                        window = model_params.get("window", 3)
                        model = NaiveMovingAverage(input_chunk_length=window)
                    elif model_lower == "exponentialsmoothing":
                        model = ExponentialSmoothing()
                    elif model_lower == "theta":
                        model = Theta()
                    elif model_lower == "linearregression":
                        lags = model_params.get("lags", 12)
                        output_len = model_params.get("output_chunk_length", 1)
                        model = LinearRegressionModel(lags=lags, output_chunk_length=output_len)
                    elif model_lower == "randomforest":
                        lags = model_params.get("lags", 12)
                        output_len = model_params.get("output_chunk_length", 12)
                        model = RandomForestModel(lags=lags, output_chunk_length=output_len)
                    elif model_lower == "nbeats":
                        input_len = model_params.get("input_chunk_length", 24)
                        output_len = model_params.get("output_chunk_length", 12)
                        n_epochs = model_params.get("n_epochs", 50)
                        model = NBEATSModel(
                            input_chunk_length=input_len,
                            output_chunk_length=output_len,
                            n_epochs=n_epochs,
                            random_state=42,
                            pl_trainer_kwargs={"accelerator": "cpu"}
                        )
                    else:
                        model = NaiveSeasonal(K=12)
                    
                    # Train and predict
                    if is_backtest_ss and forecast_start:
                        # Backtest mode
                        start_ts = pd.to_datetime(forecast_start)
                        train_ts = target_ts.slice(target_ts.start_time(), start_ts - pd.DateOffset(months=1))
                        actual_ts = target_ts.slice(start_ts, target_ts.end_time())
                        
                        model.fit(train_ts)
                        pred_ts = model.predict(n=min(horizon, len(actual_ts)))
                        
                        # Metrics
                        try:
                            # darts.mape already returns percentage (e.g., 5.7 means 5.7%)
                            mape_val = float(darts_mape(actual_ts[:len(pred_ts)], pred_ts))
                            mae_val = float(darts_mae(actual_ts[:len(pred_ts)], pred_ts))
                            rmse_val = float(darts_rmse(actual_ts[:len(pred_ts)], pred_ts))
                            metrics = {"mape": mape_val, "mae": mae_val, "rmse": rmse_val}
                        except:
                            metrics = {}
                    else:
                        # Future mode
                        model.fit(target_ts)
                        pred_ts = model.predict(n=horizon)
                        metrics = {}
                    
                    # Extract values
                    pred_index = pred_ts.time_index
                    pred_values = pred_ts.values().flatten()
                    
                    # Store result
                    st.session_state.ss_forecast_result = {
                        "pred_index": pred_index,
                        "pred_values": pred_values,
                        "metrics": metrics,
                        "history_index": target_series.index,
                        "history_values": target_series.values,
                        "model_name": model_name,
                        "target_col": target_col
                    }
                    st.session_state.ss_df = df_viz
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display results
        if st.session_state.ss_forecast_result is not None:
            result = st.session_state.ss_forecast_result
            df_saved = st.session_state.ss_df
            
            st.success("✅ Прогноз выполнен!")
            
            # Chart
            fig = go.Figure()
            
            # History (gray)
            fig.add_trace(go.Scatter(
                x=result["history_index"],
                y=result["history_values"],
                name="История",
                line=dict(color="#808080", width=2),
                mode="lines+markers"
            ))
            
            # Forecast (green)
            fig.add_trace(go.Scatter(
                x=result["pred_index"],
                y=result["pred_values"],
                name="Прогноз",
                line=dict(color="#08a652", width=2, dash="dash"),
                mode="lines+markers"
            ))
            
            fig.update_layout(
                title=f"📈 {result['target_col']} — {result['model_name']}",
                xaxis_title="Дата",
                yaxis_title="Значение",
                template="plotly_white",
                height=600,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            if result["metrics"]:
                st.markdown("### 📊 Метрики")
                display_metrics(result["metrics"])
                
                # Monthly MAPE Analysis button
                if st.button("📈 Детальный анализ MAPE по месяцам", use_container_width=True, key="ss_mape_btn"):
                    st.session_state.ss_show_mape_dialog = True
            
            # Monthly MAPE Dialog
            if st.session_state.get("ss_show_mape_dialog", False):
                @st.dialog("📊 Monthly MAPE Analysis", width="large")
                def show_mape_dialog():
                    # Calculate monthly MAPE
                    history_index = [str(d) for d in result["history_index"]]
                    history_values = list(result["history_values"])
                    forecast_index = [str(d) for d in result["pred_index"]]
                    forecast_values = list(result["pred_values"])
                    
                    monthly_metrics = calculate_monthly_mape(
                        history_index, history_values, forecast_index, forecast_values
                    )
                    
                    if not monthly_metrics:
                        st.warning("Нет данных для анализа MAPE по месяцам")
                        if st.button("Закрыть"):
                            st.session_state.ss_show_mape_dialog = False
                            st.rerun()
                        return
                    
                    # Check if we have actuals
                    has_actuals = any(m["actual"] is not None for m in monthly_metrics)
                    if not has_actuals:
                        st.info("В режиме прогноза на будущее нет фактических данных для сравнения")
                        if st.button("Закрыть"):
                            st.session_state.ss_show_mape_dialog = False
                            st.rerun()
                        return
                    
                    # MAPE Trend Chart
                    months = [m["month"] for m in monthly_metrics]
                    mape_values = [m["mape"] for m in monthly_metrics]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=months,
                        y=mape_values,
                        mode="lines+markers",
                        name="MAPE %",
                        line=dict(color="#3b82f6", width=2, shape="spline"),
                        marker=dict(
                            size=10,
                            color=[
                                "#10b981" if m is not None and m <= 5 else 
                                "#f59e0b" if m is not None and m <= 10 else 
                                "#ef4444" if m is not None else "#6b7280"
                                for m in mape_values
                            ]
                        ),
                        hovertemplate="%{x}<br>MAPE: %{y:.1f}%<extra></extra>"
                    ))
                    
                    # Reference lines
                    fig.add_hline(y=5, line_dash="dash", line_color="#10b981", 
                                  annotation_text="Хорошо (5%)", annotation_position="right")
                    fig.add_hline(y=10, line_dash="dash", line_color="#f59e0b",
                                  annotation_text="Внимание (10%)", annotation_position="right")
                    
                    fig.update_layout(
                        title="Тренд MAPE по месяцам (%)",
                        height=350,
                        template="plotly_white",
                        xaxis=dict(tickangle=-45),
                        yaxis=dict(rangemode="tozero", title="MAPE %"),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.markdown("#### Детальная таблица")
                    table_data = []
                    for m in monthly_metrics:
                        mape_str = f"{m['mape']:.1f}%" if m['mape'] is not None else "-"
                        error_str = f"{m['error']:+.2f}" if m['error'] is not None else "-"
                        actual_str = f"{m['actual']:.2f}" if m['actual'] is not None else "-"
                        trend = "📈" if m["direction"] == "up" else "📉" if m["direction"] == "down" else "➖"
                        
                        table_data.append({
                            "Месяц": m["month"],
                            "Факт": actual_str,
                            "Прогноз": f"{m['forecast']:.2f}",
                            "Ошибка": error_str,
                            "MAPE %": mape_str,
                            "Тренд": trend
                        })
                    
                    df_table = pd.DataFrame(table_data)
                    st.dataframe(df_table, use_container_width=True, hide_index=True)
                    
                    # Close button
                    if st.button("Закрыть", use_container_width=True):
                        st.session_state.ss_show_mape_dialog = False
                        st.rerun()
                
                show_mape_dialog()
                st.session_state.ss_show_mape_dialog = False
            
            # Export
            st.markdown("### 📥 Экспорт")
            export_df = pd.DataFrame({
    "Дата": list(result["history_index"]) + list(result["pred_index"]),
                result["target_col"]: list(result["history_values"]) + list(result["pred_values"]),
                "Тип": ["История"] * len(result["history_values"]) + ["Прогноз"] * len(result["pred_values"])
            })
            
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📄 Скачать CSV",
                csv_data,
                f"forecast_{result['model_name']}.csv",
                "text/csv",
                use_container_width=True,
                key="ss_download_csv"
            )
            
            # Clear button
            if st.button("🔄 Сбросить", use_container_width=True):
                st.session_state.ss_forecast_result = None
                st.rerun()
        
        elif df_viz is not None:
            # Show initial chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_viz[date_col],
                y=df_viz[target_col],
                mode="lines+markers",
                name=f"{target_col}",
                line=dict(color="#808080", width=2)
            ))
            fig.update_layout(
                title=f"📈 {target_col}",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Empty state placeholder for Single Series
        st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 60vh;
            text-align: center;
            color: #6c757d;
        ">
            <div style="font-size: 80px; margin-bottom: 20px;">📊</div>
            <h2 style="color: #495057; margin-bottom: 10px;">Добро пожаловать в режим Одиночного ряда</h2>
            <p style="font-size: 18px; max-width: 500px; margin-bottom: 30px;">
                Загрузите CSV или Excel файл в боковой панели слева, выберите колонки и модель, 
                затем нажмите <strong style="color: #08a652;">🚀 Прогноз</strong> для запуска.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # GMTS Mode (original code continues here)
    with st.sidebar:
        st.markdown("### Настройки")
        
        # 1. Model Selection
        model_options = ["randomforest", "nbeats", "tftmodel", "naiveseasonal", "chronos"]
        selected_model = st.selectbox("Тип модели", model_options, index=0)

        # 2. Date Selection
        current_date = datetime.now()
        last_data_date = df_combined.index[-1]
        default_date = datetime(2024, 1, 1) # Default to past for backtest demo
        target_date = st.date_input("Целевая дата прогноза", value=default_date)
        target_dt = pd.to_datetime(target_date)
        
        # 3. Auto-Backtest Logic
        # If target_date <= last_data_date → Backtest mode (auto)
        # If target_date > last_data_date → Future mode
        is_auto_backtest = target_dt <= last_data_date
        is_backtest = st.checkbox("Режим бэктеста", value=is_auto_backtest, disabled=is_auto_backtest, 
                                  help="Автоматически включается, если дата находится в пределах исторических данных")
        
        # 4. Prediction Horizon
        if is_backtest:
            prediction_horizon = st.number_input("Горизонт прогноза (месяцев)", min_value=1, value=12,
                                                 help="Отсечка = Целевая дата − Горизонт")
            # Calculate and show cutoff date
            cutoff_date = target_dt - relativedelta(months=prediction_horizon)
            st.caption(f"📅 Отсечка обучения: {cutoff_date.strftime('%Y-%m-%d')}")
            st.caption(f"→ Прогноз: с {cutoff_date.strftime('%Y-%m')} по {target_dt.strftime('%Y-%m')}")
        else:
            # Future mode: horizon = months from last_data_date to target_date
            # Calculate number of months between dates
            months_diff = (target_dt.year - last_data_date.year) * 12 + (target_dt.month - last_data_date.month)
            prediction_horizon = max(1, months_diff)  # At least 1 month
            st.info(f"🔮 Прогноз на будущее: {prediction_horizon} мес.")
            st.caption(f"С {last_data_date.strftime('%Y-%m')} по {target_dt.strftime('%Y-%m')}")

        # Define variable here but move input to Advanced or remove it
        target_col_suffix = "_norm_sum" # Hardcoded default as requested

        # 5. Advanced Settings
        with st.expander("🔧 Расширенные настройки"):
            predictor_lag = st.number_input("Лаг предикторов", value=12)
            force_retrain = st.checkbox("Принудительное переобучение", value=True)

        # 6. Target Selection (Multiselect) - ONLY ORIGINAL SERIES from CATEGORY_MAP
        # Get original series names from CATEGORY_MAP keys only, removing any suffixes
        # Also filter out aggregated cluster names (those that are also values in the map)
        cluster_names = set([str(v).strip() for v in CATEGORY_MAP.values()])  # e.g. "Овощи", "Сахар и мука" 
        
        def clean_series_name(name):
            """Remove all normalization/aggregation suffixes from series name robustly"""
            if not isinstance(name, str):
                return str(name)
            
            result = name.strip()
            # Recursive removal of suffixes
            suffixes = ["_norm_sum", "_norm", "_sum", str(target_col_suffix)]
            changed = True
            while changed:
                changed = False
                for suffix in suffixes:
                    if suffix and result.endswith(suffix):
                        result = result[:-len(suffix)].strip()
                        changed = True
            return result
        
        original_series_names = sorted(set([
            clean_series_name(name) 
            for name in df_combined.columns
            if (clean_series_name(name) not in cluster_names or clean_series_name(name) == "USDRUB" or "(USD)" in str(name))
            and not str(name).strip().endswith("_norm_sum") 
            and not str(name).strip().endswith("_norm")
        ] + [
            clean_series_name(name)
            for name in macro_data.columns
            if "(USD)" in str(name) or "Инфляция" in str(name)
        ]))
        
        selected_targets_clean = st.multiselect(
            "Выберите целевые ряды",
            options=original_series_names,
            default=[],
            help="Выберите ряды для прогнозирования. Если пусто — обрабатываются все индексы."
        )
        
        # Build mapping for later use (for raw column name lookup)
        RAW_TO_CLEAN = {}
        CLEAN_TO_RAW = {}
        for t in list(df_combined.columns) + list(macro_data.columns):
            clean_name = clean_series_name(t)
            RAW_TO_CLEAN[t] = clean_name
            if clean_name not in CLEAN_TO_RAW or t.endswith(target_col_suffix):
                 CLEAN_TO_RAW[clean_name] = t

        macro_scenarios_input = {}
        with st.expander("🎯 Сценарии макропоказателей"):
            st.caption("Задайте целевые значения макропоказателей (линейная интерполяция)")
            
            # Use columns from macro_data but explicitly add USDRUB if it exists in primary data
            macro_cols_to_show = list(macro_data.columns)
            if "USDRUB" in df_combined.columns and "USDRUB" not in macro_cols_to_show:
                macro_cols_to_show = ["USDRUB"] + macro_cols_to_show # Put it at the top
                
            for macro_name in macro_cols_to_show:
                val = st.number_input(
                    f"Цель для {macro_name}",
                    value=0.0,
                    key=f"macro_{macro_name}"
                )
                if val != 0.0:
                    macro_scenarios_input[macro_name] = val

        # 6. Future Values (Interpolation) - select series and set goal values
        target_goals_input = {}
        with st.expander("📊 Будущие значения рядов", expanded=True):
            st.caption("Выберите ряды и задайте их целевые значения на конец периода прогноза")
            
            # Multiselect to choose series for future values
            future_values_series = st.multiselect(
                "Выберите ряды для задания значений",
                options=original_series_names,
                default=[],
                key="future_values_series_select",
                help="Выберите один или несколько рядов для задания целевых значений"
            )
            
            if future_values_series:
                for t_clean in future_values_series:
                    val = st.number_input(
                        f"Целевое значение для {t_clean}",
                        value=0.0,
                        key=f"goal_{t_clean}"
                    )
                    if val != 0.0:
                        target_goals_input[t_clean] = val
            else:
                st.info("Выберите ряды выше, чтобы задать их целевые значения.")

        run_forecast_btn = st.button("🚀 Запустить прогноз", type="primary", use_container_width=True)

    # Initialize session state for forecast persistence
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = None
    
    if run_forecast_btn:
        with st.spinner("Running Forecast..."):
            try:
                # 1. Parse Inputs
                last_known_date = df_combined.index[-1]
                
                # In backtest mode: forecast from (target_date - horizon) + 1 to target_date
                # In future mode: forecast from (last_known_date + 1) to target_date
                if is_backtest:
                    # Cutoff = target_date - horizon
                    train_end = target_dt - relativedelta(months=prediction_horizon)
                    new_dates = pd.date_range(
                        start=train_end + relativedelta(months=1),
                        end=target_dt,
                        freq='MS'
                    )
                else:
                    # Future mode: forecast from last data point to target
                    train_end = last_known_date
                    new_dates = pd.date_range(
                        start=last_known_date + relativedelta(months=1),
                        end=target_dt,
                        freq='MS'
                    )
                
                actual_horizon = len(new_dates)
    
                # 2. Prepare Future Values
                future_forecaster_ts_name = []
                future_forecaster_ts_values_raw = []
                
                # From Macro Scenarios Inputs
                for macro_name, macro_val in macro_scenarios_input.items():
                    future_forecaster_ts_name.append(macro_name)
                    future_forecaster_ts_values_raw.append(float(macro_val))
                
                # From Target Goals Inputs
                for t_clean, t_val in target_goals_input.items():
                    # We need the RAW name (or the one used in dataframes) for the logic
                    # Try clean name first, then look it up
                    t_lookup = t_clean 
                    if t_clean in CLEAN_TO_RAW:
                         t_lookup = CLEAN_TO_RAW[t_clean]
                    else: 
                         t_lookup = f"{t_clean}{target_col_suffix}"
    
                    # The logic below expects names that match df columns (which have been cleaned of slashes mostly)
                    # But let's stick to the convention used in 'edited_df' loop: replace slash
                    t_final_name = t_lookup.replace("/", " ")
                    
                    # Avoid duplicates if user entered same thing in both places (Goal input takes precedence)
                    if t_final_name in future_forecaster_ts_name:
                        idx = future_forecaster_ts_name.index(t_final_name)
                        future_forecaster_ts_values_raw[idx] = float(t_val)
                    else:
                        future_forecaster_ts_name.append(t_final_name)
                        future_forecaster_ts_values_raw.append(float(t_val))
    
                # Create a cleaned version of CATEGORY_MAP for lookup
                CLEAN_CATEGORY_MAP = {k.replace("/", " ").strip(): v for k, v in CATEGORY_MAP.items()}
                
                # Validate Clusters
                selected_clusters_check = []
                for i in future_forecaster_ts_name:
                    if i in CLEAN_CATEGORY_MAP:
                        selected_clusters_check.append(CLEAN_CATEGORY_MAP[i])
                    elif i in macro_data.columns:
                        selected_clusters_check.append(i)
                
                if len(selected_clusters_check) != len(set(selected_clusters_check)):
                    st.warning("Warning: Some selected features belong to the same cluster. This might be overlapping.")
    
                # 3. Setup Logic
                
                # Dates Setup
                if is_backtest:
                    train_end = target_dt - relativedelta(months=actual_horizon)
                    
                    past_idx = pd.date_range(start=train_end + relativedelta(months=1),
                                           periods=actual_horizon,
                                           freq='MS')
                    future_idx = None
                else:
                    # Future Forecast Mode
                    train_end = last_known_date
                    # past_idx is the range we are predicting (from last data to target)
                    past_idx = new_dates
                    future_idx = new_dates  # Use the same dates for future forecast
    
                # 4. Prepare DataFrames
                if is_backtest:
                    df_updated = df_combined.copy()
                    df_updated.columns = [col.replace("/", " ") for col in df_updated.columns]
                    macro_updated = macro_data.copy()
                    
                    final_future_ts_values = []
                    
                    # Apply interpolation to MACRO indicators
                    for (macro_name, macro_val) in macro_scenarios_input.items():
                        df_target = macro_updated  # Macros are always in macro_data
                        
                        try:
                            start_val = df_target.loc[:train_end].iloc[-1][macro_name]
                        except:
                            start_val = 0
                            
                        scenario_values = np.linspace(start_val, macro_val, actual_horizon)
                        
                        existing_indices = [d for d in past_idx if d in df_target.index]
                        if len(existing_indices) == len(scenario_values):
                             df_target.loc[existing_indices, macro_name] = scenario_values
                        
                        final_future_ts_values.append(scenario_values)
                    
                    # Apply interpolation to Future Values (target series as covariates)
                    for (t_clean, t_val) in target_goals_input.items():
                        # Find column name in df_updated
                        t_lookup = t_clean
                        if t_clean in CLEAN_TO_RAW:
                            t_lookup = CLEAN_TO_RAW[t_clean]
                        else:
                            t_lookup = f"{t_clean}{target_col_suffix}"
                        t_final_name = t_lookup.replace("/", " ")
                        
                        # Check which dataframe contains this series
                        if t_final_name in df_updated.columns:
                            df_target = df_updated
                            col_name = t_final_name
                        elif t_lookup in df_updated.columns:
                            df_target = df_updated
                            col_name = t_lookup
                        else:
                            continue  # Skip if column not found
                        
                        try:
                            start_val = df_target.loc[:train_end].iloc[-1][col_name]
                        except:
                            start_val = 0
                            
                        scenario_values = np.linspace(start_val, float(t_val), actual_horizon)
                        
                        existing_indices = [d for d in past_idx if d in df_target.index]
                        if len(existing_indices) == len(scenario_values):
                            df_target.loc[existing_indices, col_name] = scenario_values
                        
                        final_future_ts_values.append(scenario_values)
                        
                else:
                     # Future Forecast Mode - NO interpolation for target goals
                     # Only extend dataframes with future dates, no value modification
                     df_updated, macro_updated, final_future_ts_values = form_df_future(
                         new_dates,
                         future_forecaster_ts_name,
                         future_forecaster_ts_values_raw,
                         actual_horizon
                     )
    
                # 5. Run Prediction Loop
                
                # Setup Results Storage
                index_results_data = [] # List of dicts: {name, chart_fig, metrics}
                target_results_data = [] # List of dicts: {name, chart_fig, metrics_mape_container}
    
                # Map selected CLEAN names back to RAW names for processing logic
                selected_targets_raw = []
                for clean in selected_targets_clean:
                    if clean in CLEAN_TO_RAW:
                        selected_targets_raw.append(CLEAN_TO_RAW[clean])
                    else:
                        # Fallback to appending suffix if missing (heuristic)
                        selected_targets_raw.append(f"{clean}{target_col_suffix}")
    
                # REQUIREMENT 2: Filter clusters based on selected targets
                if selected_targets_clean:
                    # Build set of clusters to process from selected targets
                    clusters_to_process = set()
                    for target_clean in selected_targets_clean:
                        # Fix: use CLEAN_CATEGORY_MAP with cleaned key (replace slash and strip)
                        lookup_key = target_clean.replace("/", " ").strip()
                        cluster = CLEAN_CATEGORY_MAP.get(lookup_key)
                        if cluster:
                            clusters_to_process.add(cluster)
                        else:
                            # Fallback to CATEGORY_MAP if CLEAN_CATEGORY_MAP lookup fails
                            cluster = CATEGORY_MAP.get(target_clean)
                            if cluster:
                                clusters_to_process.add(cluster)
                    
                    if clusters_to_process:
                        st.info(f"Processing clusters: {', '.join(clusters_to_process)}")
                    else:
                        st.warning("No clusters found for selected targets. Processing all.")
                        clusters_to_process = set(BEST_PREDICTORS_FOR_INDEX.keys())
                else:
                    # No targets selected = all clusters
                    clusters_to_process = set(BEST_PREDICTORS_FOR_INDEX.keys())
    
                # Iterate ONLY relevant clusters 
                for cluster_name, saved_predictors in BEST_PREDICTORS_FOR_INDEX.items():
                        # REQUIREMENT 2: Skip if not in clusters_to_process
                        if cluster_name not in clusters_to_process:
                            continue
                            
                        predictors = saved_predictors # Default
    
                        # Clear cache if force_retrain is enabled
                        if force_retrain:
                            try:
                                target_cluster_clean = cluster_name.replace("/", "_")
                                predictors_label = "none" if not predictors else "_".join(predictors)
                                cache_file_prefix = f"darts_result_tables/{target_cluster_clean}_by_{predictors_label}"
                                if os.path.exists(f"{cache_file_prefix}_preds.csv"):
                                    os.remove(f"{cache_file_prefix}_preds.csv")
                                if os.path.exists(f"{cache_file_prefix}_metrics.csv"):
                                    os.remove(f"{cache_file_prefix}_metrics.csv")
                            except Exception as e:
                                st.warning(f"Failed to clear cache for {cluster_name}: {e}")
                        
                        # Fix for IndexError: Ensure macro_data aligns with cluster_data
                        macro_aligned = macro_updated.reindex(df_updated.index).fillna(method='ffill').fillna(method='bfill')
    
                        # Fix for KeyError: Macro variables don't use suffix
                        current_target_suffix = "" if cluster_name in macro_aligned.columns else target_col_suffix
                        
                        # Fix for ValueError: Input y contains NaN (Cleaning data before passing to Forecaster)
                        df_updated_clean = df_updated.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                        
    
                        # Fix for KeyError in backtest: ClusterForecaster unconditionally looks for target in cluster_data
                        # So if we are targeting a macro variable, we must ensure it is present in cluster_data
                        # We merge ALL macro columns to also ensure predictors are found
                        for col in macro_aligned.columns:
                            df_updated_clean[col] = macro_aligned[col]
                        
                        # Fix for KeyError: Filter out missing predictors (e.g. 'Unexsiting category')
                        if predictors:
                            valid_predictors = []
                            for p in predictors:
                                if f"{p}{current_target_suffix}" in df_updated_clean.columns:
                                    valid_predictors.append(p)
                                else:
                                    pass # Silent ignore to reduce noise
                            predictors = valid_predictors
    
                        forecaster = ClusterForecaster(
                             cluster_data=df_updated_clean,
                             macro_data=macro_aligned,
                             category_map=CLEAN_CATEGORY_MAP,
                             target_col_suffix=current_target_suffix,
                             model_type=selected_model,
                             future_macro_col=future_forecaster_ts_name,
                             freq='MS'
                        )
                        
                            # Train/Forecast
                        if is_backtest:
                             # --- INDEX FORECAST ---
                             pred_ts, target_ts, metrics = forecaster.backtest(
                                 target_cluster=cluster_name,
                                 additional_clusters=predictors if predictors else None,
                                 past_covariate_lags=predictor_lag,
                                 start_backtest=past_idx[0],
                                 forecast_horizon=actual_horizon,
                                 use_future_macro=True,
                                 lags=predictor_lag,
                                 is_backtest=True
                             )
                             
                             # Process Index Result
                             def process_ts_result(ts_obj, default_index):
                                 if hasattr(ts_obj, "time_index"):
                                     return ts_obj.time_index, ts_obj.values()[:,0]
                                 elif hasattr(ts_obj, "values") and hasattr(ts_obj, "index"):
                                     if pd.api.types.is_numeric_dtype(ts_obj.index):
                                         return default_index, ts_obj.values
                                     return ts_obj.index, ts_obj.values
                                 else:
                                     return default_index, np.array(ts_obj)
    
                             pred_index, pred_values = process_ts_result(pred_ts, past_idx)
                             
                             # Prepare Data for Chart (INDEX)
                             # Show FULL available historical series for context
                             past_data = df_updated_clean[f"{cluster_name}{current_target_suffix}"]
                             history_data = past_data  # Full history for visualization
    
                             # Store Index Forecast
                             fig_index = plot_forecast_chart(
                                 history_index=history_data.index,
                                 history_values=history_data.values,
                                 forecast_index=pred_index,
                                 forecast_values=pred_values,
                                 title=f"Backtest Index: {cluster_name}",
                                 forecast_start=pred_index[0].strftime("%Y-%m-%d") if len(pred_index) > 0 else None
                             )
                             index_results_data.append({
                                 "name": cluster_name,
                                 "fig": fig_index,
                                 "metrics": metrics
                             })
    
                             # --- SPECIFIC TARGETS in this Cluster ---
                             # Find which selected targets belong to this cluster
                             current_cluster_targets_raw = []
                             for t_raw in selected_targets_raw:
                                 # Map raw target name to cluster to check if it belongs here
                                 
                                 # Heuristic to remove suffix for mapping check if needed, 
                                 # but CLEAN_CATEGORY_MAP usually has cleaned keys? 
                                 # Let's check CLEAN_CATEGORY_MAP keys.
                                 # If key in map describes the series name, it usually doesn't have suffix if mapped manually.
                                 
                                 # Try t_raw as is
                                 mapped_cluster = None
                                 if t_raw in CLEAN_CATEGORY_MAP:
                                     mapped_cluster = CLEAN_CATEGORY_MAP[t_raw]
                                 elif t_raw in macro_data.columns:
                                     if t_raw in CLEAN_CATEGORY_MAP:
                                          mapped_cluster = CLEAN_CATEGORY_MAP[t_raw]
                                     else:
                                          mapped_cluster = t_raw # Macro is its own cluster
                                 
                                 # Try stripping slash for lookup
                                 if mapped_cluster is None:
                                     t_clean = t_raw.replace(target_col_suffix, "").strip()
                                     t_clean_dashed = t_clean.replace("/", " ").strip()
                                     if t_clean_dashed in CLEAN_CATEGORY_MAP:
                                          mapped_cluster = CLEAN_CATEGORY_MAP[t_clean_dashed]
                                     elif t_clean in CLEAN_CATEGORY_MAP:
                                          mapped_cluster = CLEAN_CATEGORY_MAP[t_clean]
    
                                 if mapped_cluster == cluster_name:
                                     current_cluster_targets_raw.append(t_raw)
                             
                             # Ensure model is trained if there are targets to predict
                             if current_cluster_targets_raw and forecaster.model is None:
                                 forecaster.train(
                                     target_cluster=cluster_name,
                                     additional_clusters=predictors if predictors else None,
                                     use_macro=True,
                                     past_covariate_lags=predictor_lag,
                                     train_end=past_idx[0] - relativedelta(months=1),
                                     use_future_macro=True,
                                     lags=predictor_lag, # Pass args used in backtest logic
                                     is_backtest=True
                                 )
                             
                             for target_name in current_cluster_targets_raw:
                                 # Ensure mapping exists for get_prediction_for_ts
                                 if target_name not in CLEAN_CATEGORY_MAP:
                                     CLEAN_CATEGORY_MAP[target_name] = cluster_name
                                     
                                 target_series_pred, _ = get_prediction_for_ts(
                                      df_in=df_updated_clean, 
                                      model_category=forecaster,
                                      ts_to_predict_name=target_name,
                                      model_name=selected_model,
                                      macro_data=macro_aligned,
                                      past_idx=past_idx, 
                                      future_idx=pd.date_range(start=past_idx[-1] + relativedelta(months=1), periods=1, freq='MS'),
                                      category_map=CLEAN_CATEGORY_MAP,
                                      best_predictors_for_idx={cluster_name: predictors},
                                      target_col_suffix=current_target_suffix,
                                      future_forecaster_ts_name=future_forecaster_ts_name,
                                      future_forecaster_ts_values=final_future_ts_values,
                                      use_future_macro=True,
                                      prediction_horizon=actual_horizon,
                                      is_backtest=True,
                                      prediction_lag=predictor_lag,
                                      predicted_index=past_idx 
                                 )
                                 
                                 target_pred_values = target_series_pred.values
                                 if hasattr(target_pred_values, "flatten"):
                                      target_pred_values = target_pred_values.flatten()
                                 target_pred_index = target_series_pred.index
    
                                 # Historical for Target
                                 target_col_raw = target_name
                                 target_col_dashed = target_col_raw.replace("/", " ")
                                 if target_col_raw in df_updated_clean.columns:
                                      target_past_data = df_updated_clean[target_col_raw]
                                 elif target_col_dashed in df_updated_clean.columns:
                                      target_past_data = df_updated_clean[target_col_dashed]
                                 elif target_col_raw in macro_aligned.columns:
                                      target_past_data = macro_aligned[target_col_raw]
                                 else:
                                      target_past_data = pd.Series()
                                      
                                 # Metrics for Target
                                 target_metrics = {}
                                 if target_col_raw in df_updated_clean.columns:
                                       actuals_slice = df_updated_clean.loc[target_pred_index, target_col_raw]
                                 elif target_col_dashed in df_updated_clean.columns:
                                       actuals_slice = df_updated_clean.loc[target_pred_index, target_col_dashed]
                                 elif target_col_raw in macro_aligned.columns:
                                       actuals_slice = macro_aligned.loc[target_pred_index, target_col_raw]
                                 else:
                                       actuals_slice = None
                                  
                                 if actuals_slice is not None:
                                       from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
                                       try:
                                            y_true = actuals_slice.values
                                            y_pred = target_pred_values
                                            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                                            if mask.sum() > 0:
                                                 target_metrics = {
                                                      "mape": mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100,
                                                      "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
                                                      "rmse": root_mean_squared_error(y_true[mask], y_pred[mask])
                                                 }
                                       except:
                                            pass
                                
                                 # Store Result
                                 display_name = RAW_TO_CLEAN.get(target_name, target_name)
                                 # Show FULL available historical series for context
                                 target_history_data = target_past_data
                                 target_results_data.append({
                                     "name": display_name,
                                     "fig": plot_forecast_chart(
                                         history_index=target_history_data.index,
                                         history_values=target_history_data.values,
                                         forecast_index=target_pred_index,
                                         forecast_values=target_pred_values,
                                         title=f"Backtest Target: {display_name}",
                                         forecast_start=target_pred_index[0].strftime("%Y-%m-%d") if len(target_pred_index) > 0 else None
                                     ),
                                     "metrics": target_metrics,
                                     "mape_data": {
                                         "history_index": target_history_data.index,
                                         "history_values": target_history_data.values,
                                         "forecast_index": target_pred_index,
                                         "forecast_values": target_pred_values
                                     }
                                 })
    
                        else:
                            # Future Forecast
                            
                            # 1. Calculate Metrics on History (Backtest) to show model quality
                            with st.spinner(f"Calculating historical metrics for {cluster_name}..."):
                                _, _, metrics = forecaster.backtest(
                                     target_cluster=cluster_name,
                                     additional_clusters=predictors if predictors else None,
                                     past_covariate_lags=predictor_lag,
                                     start_backtest=train_end - relativedelta(months=actual_horizon*2), # Backtest on recent history
                                     forecast_horizon=actual_horizon,
                                     use_future_macro=True,
                                     lags=predictor_lag,
                                     is_backtest=True
                                )
                            
                            # Run Future Forecast for Index
                            # Run Future Forecast for Index
                            # First, TRAIN the model on the full available history
                            # We use is_backtest=True to force the model to rely on past_covariates (which we extend into future)
                            # rather than expecting separate future_covariates, simplifying data prep.
                            forecaster.train(
                                 target_cluster=cluster_name,
                                 additional_clusters=predictors if predictors else None,
                                 use_macro=True,
                                 past_covariate_lags=predictor_lag,
                                 train_end=train_end,
                                 use_future_macro=True,
                                 lags=predictor_lag,
                                 is_backtest=True
                            )
    
                            # Then, Prepare Covariates for Forecast
                            # Use past_idx (the immediate future horizon we are predicting)
                            # CRITICAL FIX: Darts requires past_covariates to include the lookback period (predictor_lag) BEFORE the forecast start.
                            lookback_start = past_idx[0] - relativedelta(months=predictor_lag)
                            full_cov_idx = pd.date_range(start=lookback_start, end=past_idx[-1], freq='MS')
                            
                            cov_past = pd.DataFrame(index=full_cov_idx)
                            if True: # use_macro is always True here
                                 cov_past = pd.concat([cov_past, macro_updated.reindex(full_cov_idx)], axis=1) # Use updated macro
                            
                            if predictors:
                                 for additional_cluster in predictors:
                                     # Ensure we use the correct dataframe for lags
                                     if f"{additional_cluster}{current_target_suffix}" in df_updated_clean:
                                         all_add = df_updated_clean[f"{additional_cluster}{current_target_suffix}"].shift(1)
                                         cov_past[f"{additional_cluster}_lag1"] = all_add.reindex(full_cov_idx)
                            
                            cov_past = cov_past.fillna(method="ffill").fillna(method="bfill")
                            
                            # Finally, Forecast
                            pred_ts = forecaster.forecast(
                                 n=actual_horizon,
                                 covariates_past=cov_past,
                                 is_backtest=True
                            )
                            
                            # Helper (redefined to ensure scope)
                            def process_ts_result(ts_obj, default_index):
                                 if hasattr(ts_obj, "time_index"):
                                     return ts_obj.time_index, ts_obj.values()[:,0]
                                 elif hasattr(ts_obj, "values") and hasattr(ts_obj, "index"):
                                     if pd.api.types.is_numeric_dtype(ts_obj.index):
                                         return default_index, ts_obj.values
                                     return ts_obj.index, ts_obj.values
                                 else:
                                     return default_index, np.array(ts_obj)
                            
                            pred_index, pred_values = process_ts_result(pred_ts, future_idx)
                            past_data = df_updated_clean[f"{cluster_name}{current_target_suffix}"]
                            # For future forecast, history is up to train_end (last_known_date)
                            history_data = past_data  # Full history for visualization
    
                            # Store Index Result
                            fig_index = plot_forecast_chart(
                                 history_index=history_data.index,
                                 history_values=history_data.values,
                                 forecast_index=pred_index,
                                 forecast_values=pred_values,
                                 title=f"Future Forecast Index: {cluster_name}",
                                 forecast_start=pred_index[0].strftime("%Y-%m-%d") if len(pred_index) > 0 else None
                            )
                            index_results_data.append({
                                 "name": cluster_name,
                                 "fig": fig_index,
                                 "metrics": metrics
                            })
    
    
                            # --- SPECIFIC TARGETS in this Cluster ---
                            current_cluster_targets_raw = []
                            for t_raw in selected_targets_raw:
                                 mapped_cluster = None
                                 if t_raw in CLEAN_CATEGORY_MAP:
                                     mapped_cluster = CLEAN_CATEGORY_MAP[t_raw]
                                 elif t_raw in macro_data.columns:
                                     if t_raw in CLEAN_CATEGORY_MAP:
                                          mapped_cluster = CLEAN_CATEGORY_MAP[t_raw]
                                     else:
                                          mapped_cluster = t_raw # Macro is its own cluster
                                 
                                 if mapped_cluster is None:
                                     t_clean = t_raw.replace(target_col_suffix, "").strip()
                                     t_clean_dashed = t_clean.replace("/", " ").strip()
                                     if t_clean_dashed in CLEAN_CATEGORY_MAP:
                                          mapped_cluster = CLEAN_CATEGORY_MAP[t_clean_dashed]
                                     elif t_clean in CLEAN_CATEGORY_MAP:
                                          mapped_cluster = CLEAN_CATEGORY_MAP[t_clean]
                                          
                                 if mapped_cluster == cluster_name:
                                     current_cluster_targets_raw.append(t_raw)
                            
                            for target_name in current_cluster_targets_raw:
                                 # Ensure mapping exists for get_prediction_for_ts
                                 if target_name not in CLEAN_CATEGORY_MAP:
                                     CLEAN_CATEGORY_MAP[target_name] = cluster_name
                                     
                                 target_series_pred, _ = get_prediction_for_ts(
                                      df_in=df_updated_clean,
                                      model_category=forecaster,
                                      ts_to_predict_name=target_name,
                                      model_name=selected_model,
                                      macro_data=macro_aligned,
                                      past_idx=past_idx, 
                                      future_idx=future_idx,
                                      category_map=CLEAN_CATEGORY_MAP,
                                      best_predictors_for_idx={cluster_name: predictors},
                                      target_col_suffix=current_target_suffix,
                                      future_forecaster_ts_name=future_forecaster_ts_name,
                                      future_forecaster_ts_values=final_future_ts_values,
                                      use_future_macro=False,
                                      prediction_horizon=actual_horizon,
                                      is_backtest=False,
                                      prediction_lag=predictor_lag,
                                      predicted_index=future_idx
                                 )
                                 
                                 target_pred_values = target_series_pred.values
                                 if hasattr(target_pred_values, "flatten"):
                                      target_pred_values = target_pred_values.flatten()
                                 target_pred_index = target_series_pred.index
                                 
                                 # Set historical data
                                 target_col_raw = target_name
                                 target_col_dashed = target_col_raw.replace("/", " ")
                                 if target_col_raw in df_updated_clean.columns:
                                      target_past_data = df_updated_clean[target_col_raw]
                                 elif target_col_dashed in df_updated_clean.columns:
                                      target_past_data = df_updated_clean[target_col_dashed]
                                 elif target_col_raw in macro_aligned.columns:
                                      target_past_data = macro_aligned[target_col_raw]
                                 else:
                                      target_past_data = pd.Series()
                                 
                                 # IMPORTANT: Limit history to train_end (last_known_date)
                                 target_history_data = target_past_data[target_past_data.index <= train_end]
                                 
                                 display_name = RAW_TO_CLEAN.get(target_name, target_name)
                                 # Store Result
                                 target_results_data.append({
                                     "name": display_name,
                                     "fig": plot_forecast_chart(
                                         history_index=target_history_data.index,
                                         history_values=target_history_data.values,
                                         forecast_index=target_pred_index,
                                         forecast_values=target_pred_values,
                                         title=f"Future Forecast Target: {display_name}",
                                         forecast_start=target_pred_index[0].strftime("%Y-%m-%d") if len(target_pred_index) > 0 else None
                                     ),
                                     "metrics": {}, # No metrics for future
                                     "mape_data": {
                                         "history_index": target_history_data.index,
                                         "history_values": target_history_data.values,
                                         "forecast_index": target_pred_index,
                                         "forecast_values": target_pred_values
                                     }
                                 })
    
                # Save results to session_state for persistence
                st.session_state.forecast_results = {
                    "target_results_data": target_results_data,
                    "index_results_data": index_results_data,
                    "macro_scenarios_input": macro_scenarios_input,
                    "target_goals_input": target_goals_input,
                    "df_updated": df_updated,
                    "macro_updated": macro_updated,
                    "selected_model": selected_model,
                    "target_dt": target_dt,
                    "actual_horizon": actual_horizon,
                    "is_backtest": is_backtest,
                    "predictor_lag": predictor_lag,
                    "CLEAN_TO_RAW": CLEAN_TO_RAW,
                    "target_col_suffix": target_col_suffix,
                }
    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    # Display results from session_state (persists after download)
    if st.session_state.forecast_results is not None:
        results = st.session_state.forecast_results
        target_results_data = results["target_results_data"]
        index_results_data = results["index_results_data"]
        macro_scenarios_input = results["macro_scenarios_input"]
        target_goals_input = results["target_goals_input"]
        df_updated = results["df_updated"]
        macro_updated = results["macro_updated"]
        selected_model = results["selected_model"]
        target_dt = results["target_dt"]
        actual_horizon = results["actual_horizon"]
        is_backtest = results["is_backtest"]
        predictor_lag = results["predictor_lag"]
        CLEAN_TO_RAW = results["CLEAN_TO_RAW"]
        target_col_suffix = results["target_col_suffix"]
        
        # === RENDER RESULTS ===
        # Build export DataFrame for all forecasts
        all_export_data = []
        all_metrics_data = []
                
        # Collect data from target results
        for data in target_results_data:
            name = data["name"]
            if "mape_data" in data:
                mape_data = data["mape_data"]
                # History
                for idx, val in zip(mape_data["history_index"], mape_data["history_values"]):
                    all_export_data.append({
                        "Date": idx,
                        "Series": name,
                        "Value": val,
                        "Type": "History"
                    })
                # Forecast
                for idx, val in zip(mape_data["forecast_index"], mape_data["forecast_values"]):
                    all_export_data.append({
                        "Date": idx,
                        "Series": name,
                        "Value": val,
                        "Type": "Forecast"
                    })
            # Metrics
            if data.get("metrics"):
                for metric_name, metric_val in data["metrics"].items():
                    if metric_val is not None:
                        all_metrics_data.append({
                            "Series": name,
                            "Metric": metric_name.upper(),
                            "Value": metric_val
                        })
        
        # Collect data from index results
        for idx_data in index_results_data:
            name = idx_data["name"] + " (Index)"
            if idx_data.get("metrics"):
                for metric_name, metric_val in idx_data["metrics"].items():
                    if metric_val is not None:
                        all_metrics_data.append({
                            "Series": name,
                            "Metric": metric_name.upper(),
                            "Value": metric_val
                        })
        
        # Create Excel file
        export_df = pd.DataFrame(all_export_data) if all_export_data else pd.DataFrame()
        metrics_df = pd.DataFrame(all_metrics_data) if all_metrics_data else pd.DataFrame()
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Forecast sheet (pivoted for readability) with covariates
            if not export_df.empty:
                # Pivot: Date as index, Series as columns
                pivot_df = export_df.pivot_table(
                    index='Date', 
                    columns=['Series', 'Type'], 
                    values='Value',
                    aggfunc='first'
                ).reset_index()
                # Flatten MultiIndex columns
                pivot_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else col[0] if isinstance(col, tuple) else col for col in pivot_df.columns]
                
                # Add interpolated macro data
                for macro_name in macro_scenarios_input.keys():
                    if macro_name in macro_updated.columns:
                        macro_series = macro_updated[macro_name]
                        pivot_df[f"{macro_name}_Macro"] = pivot_df['Date'].apply(
                            lambda d: macro_series.get(d, np.nan) if hasattr(macro_series, 'get') else (
                                macro_series.loc[d] if d in macro_series.index else np.nan
                            )
                        )
                
                # Add interpolated Future Values (target series as covariates)
                for t_clean in target_goals_input.keys():
                    t_lookup = t_clean
                    if t_clean in CLEAN_TO_RAW:
                        t_lookup = CLEAN_TO_RAW[t_clean]
                    else:
                        t_lookup = f"{t_clean}{target_col_suffix}"
                    t_final_name = t_lookup.replace("/", " ")
                    
                    # Find in df_updated
                    if t_final_name in df_updated.columns:
                        series_data = df_updated[t_final_name]
                        pivot_df[f"{t_clean}_FutureValue"] = pivot_df['Date'].apply(
                            lambda d: series_data.loc[d] if d in series_data.index else np.nan
                        )
                    elif t_lookup in df_updated.columns:
                        series_data = df_updated[t_lookup]
                        pivot_df[f"{t_clean}_FutureValue"] = pivot_df['Date'].apply(
                            lambda d: series_data.loc[d] if d in series_data.index else np.nan
                        )
                
                pivot_df.to_excel(writer, sheet_name='Прогноз', index=False)
            
            # Metrics sheet
            if not metrics_df.empty:
                metrics_df.to_excel(writer, sheet_name='Метрики', index=False)
            
            # Parameters sheet
            params_data = [
                {"Parameter": "Model", "Value": selected_model},
                {"Parameter": "Target Date", "Value": str(target_dt.date())},
                {"Parameter": "Prediction Horizon", "Value": actual_horizon},
                {"Parameter": "Backtest Mode", "Value": is_backtest},
                {"Parameter": "Predictor Lag", "Value": predictor_lag},
            ]
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        output.seek(0)
        
        # Header row with export button
        header_col, export_col = st.columns([3, 1])
        with header_col:
            st.markdown("### 📊 Forecast Results")
        with export_col:
            st.download_button(
                "📥 Export Excel",
                output.getvalue(),
                f"forecast_{selected_model}_{target_dt.strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_excel_forecast"
            )
        
        # 1. Target Tabs
        if target_results_data:
            tab_names = [d["name"] for d in target_results_data]
            tabs = st.tabs(tab_names)
            
            for i, tab in enumerate(tabs):
                data = target_results_data[i]
                with tab:
                    if data["metrics"]:
                        display_metrics(data["metrics"])
                    st.plotly_chart(data["fig"], use_container_width=True)
                    if "mape_data" in data and data["metrics"]: # Show detailed analysis if backtest available
                        display_monthly_mape(**data["mape_data"])
        else:
            st.info("No target results to display. Select target series and run forecast.")
    
        # 2. Indices Overview (Expander)
        with st.expander("📊 All Calculated Indices (Utility)", expanded=not bool(target_results_data)):
            if not index_results_data:
                st.info("No indices calculated.")
            else:
                for idx_res in index_results_data:
                    st.markdown(f"#### {idx_res['name']}")
                    st.caption(f"Cluster Hist. Metrics")
                    display_metrics(idx_res["metrics"])
                    st.plotly_chart(idx_res["fig"], use_container_width=True)
                    st.markdown("---")
        
        # Button to clear results
        if st.button("🔄 Clear Results & Run New Forecast"):
            st.session_state.forecast_results = None
            st.rerun()
    
    else:
        # Empty state placeholder
        st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 60vh;
            text-align: center;
            color: #6c757d;
        ">
            <div style="font-size: 80px; margin-bottom: 20px;">📈</div>
            <h2 style="color: #495057; margin-bottom: 10px;">Добро пожаловать в GMTS</h2>
            <p style="font-size: 18px; max-width: 500px; margin-bottom: 30px;">
                Настройте параметры прогнозирования в боковой панели слева и нажмите кнопку 
                <strong style="color: #08a652;">🚀 Прогноз</strong> для запуска.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
