import numpy as np
import timesfm
import torch


torch.set_float32_matmul_precision("high")


class TimesFMModel:

    def __init__(self, data, horizon, historical_lookup):

        self.data = np.asarray(data, dtype=np.float32).reshape(-1)
        self.horizon = horizon
        self.historical_lookup = historical_lookup
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.historical_lookup,
                max_horizon=self.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        self.context = self.data[-self.historical_lookup:]

    def predict(self, input=None):
        if input is None: input = self.context
        input = np.asarray(input, dtype=np.float32).reshape(-1)
        if len(input) > self.historical_lookup: input = input[-self.historical_lookup:]
        if len(input) < self.historical_lookup: raise ValueError(f"Input must contain at least ")
        self.context = input
        with torch.no_grad():
            point_forecast, quantile_forecast = self.model.forecast(horizon=self.horizon, inputs=[self.context])
        prediction = np.asarray(point_forecast[0], dtype=np.float32)
        return prediction

    def backtest(self):
        data = self.data
        n = len(data)
        minimum_length = self.historical_lookup + self.horizon
        if n < minimum_length: raise ValueError(f"Not enough data for backtesting.\n")
        predictions = []
        actuals = []
        contexts = []
        for i in range(self.historical_lookup, n - self.horizon + 1):
            context = data[i - self.historical_lookup:i]
            actual = data[i:i + self.horizon]
            prediction = self.predict(context)
            contexts.append(context)
            predictions.append(prediction)
            actuals.append(actual)
        contexts = np.asarray(contexts, dtype=np.float32)
        predictions = np.asarray(predictions, dtype=np.float32)
        actuals = np.asarray(actuals, dtype=np.float32)
        return contexts, predictions, actuals

    def metrics(self, current_pred, original):

        current_pred = np.asarray(
            current_pred,
            dtype=np.float32
        )

        original = np.asarray(
            original,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Validate shapes
        # --------------------------------------------------

        if current_pred.shape != original.shape:

            raise ValueError(
                f"Prediction shape {current_pred.shape} "
                f"does not match original shape "
                f"{original.shape}."
            )

        # --------------------------------------------------
        # Handle single prediction
        #
        # (6,) -> (1, 6)
        # --------------------------------------------------

        if current_pred.ndim == 1:

            current_pred = current_pred.reshape(1, -1)
            original = original.reshape(1, -1)

        # --------------------------------------------------
        # Error
        # --------------------------------------------------

        error = current_pred - original

        absolute_error = np.abs(error)

        squared_error = error ** 2

        # ==================================================
        # PER-HORIZON METRICS
        # ==================================================

        horizon_metrics = {}

        for h in range(self.horizon):

            pred_h = current_pred[:, h]
            actual_h = original[:, h]

            error_h = pred_h - actual_h

            # ----------------------------------------------
            # MAE
            # ----------------------------------------------

            mae = np.mean(
                np.abs(error_h)
            )

            # ----------------------------------------------
            # RMSE
            # ----------------------------------------------

            rmse = np.sqrt(
                np.mean(error_h ** 2)
            )

            # ----------------------------------------------
            # MAPE
            # ----------------------------------------------

            non_zero = actual_h != 0

            if np.any(non_zero):

                mape = np.mean(
                    np.abs(
                        error_h[non_zero]
                        / actual_h[non_zero]
                    )
                ) * 100

            else:

                mape = np.nan

            # ----------------------------------------------
            # R²
            # ----------------------------------------------

            ss_res = np.sum(
                error_h ** 2
            )

            ss_tot = np.sum(
                (
                    actual_h
                    - np.mean(actual_h)
                ) ** 2
            )

            if ss_tot == 0:

                r2 = np.nan

            else:

                r2 = 1 - (
                    ss_res / ss_tot
                )

            horizon_metrics[f"Day +{h + 1}"] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float(mape),
                "R2": float(r2),
            }

        # ==================================================
        # OVERALL METRICS
        # ==================================================

        flat_pred = current_pred.reshape(-1)
        flat_original = original.reshape(-1)

        flat_error = (
            flat_pred - flat_original
        )

        overall_mae = np.mean(
            np.abs(flat_error)
        )

        overall_rmse = np.sqrt(
            np.mean(flat_error ** 2)
        )

        non_zero = flat_original != 0

        if np.any(non_zero):

            overall_mape = np.mean(
                np.abs(
                    flat_error[non_zero]
                    / flat_original[non_zero]
                )
            ) * 100

        else:

            overall_mape = np.nan

        ss_res = np.sum(
            flat_error ** 2
        )

        ss_tot = np.sum(
            (
                flat_original
                - np.mean(flat_original)
            ) ** 2
        )

        if ss_tot == 0:

            overall_r2 = np.nan

        else:

            overall_r2 = 1 - (
                ss_res / ss_tot
            )

        overall_metrics = {
            "MAE": float(overall_mae),
            "RMSE": float(overall_rmse),
            "MAPE": float(overall_mape),
            "R2": float(overall_r2),
        }

        # ==================================================
        # RETURN
        # ==================================================

        return {
            "overall": overall_metrics,
            "horizon": horizon_metrics
        }