"""
Backtest: how well does GARCH(1,1) predict POST-EARNINGS volatility?

Definition of "correct" (this is what your 73% must mean):
For each historical earnings event we
  1. fit GARCH(1,1) on the trailing `train_window` trading days BEFORE the event,
  2. forecast volatility over the `post_window` trading days AFTER the event,
  3. compute the realized volatility actually observed in that post-event window,
  4. count the event as a HIT if realized vol falls inside the model's
     prediction band [forecast / tol, forecast * tol].
Accuracy = hits / total events. We report the real number, whatever it is.
"""
import numpy as np
import pandas as pd
from arch import arch_model
import logging

logger = logging.getLogger(__name__)


class EarningsVolatilityBacktest:
    def __init__(self, train_window: int = 250, post_window: int = 5, tol: float = 1.5):
        self.train_window = train_window   # trading days used to fit GARCH
        self.post_window = post_window     # trading days after earnings to evaluate
        self.tol = tol                     # band width multiplier

    def _annualized_vol(self, returns_pct: pd.Series) -> float:
        """Annualized vol (%) from a series of pct log-returns."""
        return float(returns_pct.std() * np.sqrt(252))

    def run(self, prices: pd.Series, earnings_dates: list) -> dict:
        """prices: close price Series indexed by date. earnings_dates: list of Timestamps."""
        prices = prices.sort_index()
        # Make the price index tz-naive so it compares cleanly with earnings dates
        if getattr(prices.index, "tz", None) is not None:
            prices.index = prices.index.tz_localize(None)
        prices.index = prices.index.normalize()
        log_ret_pct = (np.log(prices / prices.shift(1)).dropna()) * 100

        # Normalize earnings dates to tz-naive day-level timestamps
        eds = pd.to_datetime(list(earnings_dates))
        if getattr(eds, "tz", None) is not None:
            eds = eds.tz_localize(None)
        eds = eds.normalize()

        results = []
        for ed in sorted(eds):
            # locate the event in the return series
            before = log_ret_pct[log_ret_pct.index < ed]
            after = log_ret_pct[log_ret_pct.index >= ed]
            if len(before) < self.train_window or len(after) < self.post_window:
                continue

            train = before.iloc[-self.train_window:]
            post = after.iloc[:self.post_window]

            try:
                fit = arch_model(train, vol="Garch", p=1, q=1).fit(disp="off", show_warning=False)
                fc = fit.forecast(horizon=self.post_window, reindex=False)
                # mean daily forecast variance over the post window -> annualized vol
                daily_var = fc.variance.values[-1, :].mean()
                forecast_vol = float(np.sqrt(daily_var) * np.sqrt(252))
            except Exception as e:
                logger.warning(f"GARCH fit failed at {ed.date()}: {e}")
                continue

            realized_vol = self._annualized_vol(post)
            low, high = forecast_vol / self.tol, forecast_vol * self.tol
            hit = bool(low <= realized_vol <= high)

            results.append({
                "earnings_date": ed.date(),
                "forecast_vol": round(forecast_vol, 2),
                "realized_vol": round(realized_vol, 2),
                "hit": hit,
            })

        df = pd.DataFrame(results)
        accuracy = float(df["hit"].mean() * 100) if len(df) else float("nan")
        return {
            "events_tested": len(df),
            "accuracy_pct": round(accuracy, 1),
            "detail": df,
        }