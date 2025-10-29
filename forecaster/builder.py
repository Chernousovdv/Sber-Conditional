# forecasting/builder.py
from typing import Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirectMultiVariate


class ForecasterFactory:
    """Factory for creating ForecasterDirectMultiVariate instances."""

    @staticmethod
    def build_forecaster(
        regressor,
        target_level: str,
        lags: int | dict = 12,
        steps: int = 12,
        transformer_series = StandardScaler(),
        transformer_exog = StandardScaler(),
        n_jobs = 'auto',
        forecaster_id: Optional[str] = None
    ) -> ForecasterDirectMultiVariate:
        """
        Create a ForecasterDirectMultiVariate ready to fit.
        - regressor: any sklearn-compatible regressor or a Pipeline
        - target_level: name of the series (column) you want to predict, e.g. 'A'
        - lags: int or dict (different lags per series) as allowed by skforecast
        - steps: forecast horizon (12 for next 12 months)
        """
        forecaster = ForecasterDirectMultiVariate(
            regressor = regressor,
            level = target_level,
            lags = lags,
            steps = steps,
            transformer_series = transformer_series,
            transformer_exog = transformer_exog,
            n_jobs = n_jobs,
            forecaster_id = forecaster_id
        )
        return forecaster



# ---------------------------------------
# Backtest / evaluate a single forecaster
# ---------------------------------------







# ---------- Helper utilities ----------


