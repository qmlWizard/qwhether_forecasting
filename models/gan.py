# ============================================================
# BENCHMARK-READY QGAN + MULTISEQUENCEGAN (corrected)
# ============================================================
# Fixed forecast horizon: 30 (prediction_horizon only slices the first H values at inference/evaluation).
# Training modes: per_stock (independent model per stock), combined (shared model, no stock id), stock_id (shared model + trainable stock embedding).
# Data modes: univariate (target only), multivariate (target + covariates as input, TARGET ONLY as output).
# Univariate input: data = {"AAPL": close_array, ...}
# Multivariate input: target = {"AAPL": close_array, ...} and variables = {"AAPL": feature_matrix(T,V), ...}
# The model input is [target, variables] with shape (T, 1+V); the forecast output is always the target only: prediction (H,), samples (N_samples, H).
# ============================================================

import random
import time
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pennylane as qml
except Exception:
    qml = None

try:
    from scipy.stats import wasserstein_distance, ks_2samp
except Exception:
    wasserstein_distance = None
    ks_2samp = None

torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_HORIZON = 30
EPS = 1e-8


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# VALIDATION / DATA STANDARDIZATION
# ============================================================

def _validate_type(model_type):
    model_type = str(model_type).upper()
    if model_type not in {"CC", "QC", "CQ", "QQ"}:
        raise ValueError("type must be one of: CC, QC, CQ, QQ")
    return model_type


def _validate_modes(training_mode, data_mode):
    if training_mode not in {"per_stock", "combined", "stock_id"}:
        raise ValueError("training_mode must be 'per_stock', 'combined', or 'stock_id'")
    if data_mode not in {"univariate", "multivariate"}:
        raise ValueError("data_mode must be 'univariate' or 'multivariate'")


def _to_stock_dict(x, prefix="STOCK"):
    if isinstance(x, dict):
        source = list(x.items())
    elif isinstance(x, (list, tuple)):
        source = [(f"{prefix}_{i}", v) for i, v in enumerate(x)]
    else:
        source = [(f"{prefix}_0", x)]
    out = OrderedDict()
    for name, value in source:
        out[str(name)] = np.asarray(value, dtype=np.float32)
    return out


def _prepare_inputs(data, target, variables, data_mode):
    # Returns OrderedDict: stock -> {"target": (T,), "variables": (T,V), "features": (T,1+V)}.
    if data_mode == "univariate":
        if data is None:
            raise ValueError("data is required in univariate mode")
        raw = _to_stock_dict(data, "STOCK")
        stocks = OrderedDict()
        for stock, arr in raw.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.ndim != 1:
                raise ValueError(f"Univariate stock '{stock}' must have shape (T,) or (T,1), got {arr.shape}")
            if len(arr) == 0 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid target values for stock '{stock}'")
            stocks[stock] = {"target": arr, "variables": np.empty((len(arr), 0), dtype=np.float32), "features": arr[:, None]}
        return stocks
    if target is None or variables is None:
        raise ValueError("Multivariate mode requires BOTH target and variables.")
    target_dict = _to_stock_dict(target, "STOCK")
    variable_dict = _to_stock_dict(variables, "STOCK")
    if set(target_dict) != set(variable_dict):
        raise ValueError("target and variables must contain exactly the same stock names")
    stocks = OrderedDict()
    feature_count = None
    for stock in target_dict:
        y = np.asarray(target_dict[stock], dtype=np.float32)
        X = np.asarray(variable_dict[stock], dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"Target for '{stock}' must have shape (T,), got {y.shape}")
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"Variables for '{stock}' must have shape (T,V), got {X.shape}")
        if len(y) != len(X):
            raise ValueError(f"Target/variables length mismatch for '{stock}': {len(y)} vs {len(X)}")
        if not np.isfinite(y).all() or not np.isfinite(X).all():
            raise ValueError(f"NaN/Inf found in stock '{stock}'")
        if feature_count is None:
            feature_count = X.shape[1]
        elif X.shape[1] != feature_count:
            raise ValueError("All stocks must have the same number of covariates")
        features = np.concatenate([y[:, None], X], axis=1)
        stocks[stock] = {"target": y, "variables": X, "features": features.astype(np.float32)}
    return stocks


# ============================================================
# METRIC HELPERS
# ============================================================

def _safe_div(a, b):
    return a / np.where(np.abs(b) < EPS, EPS, b)


def _smape(pred, actual):
    den = (np.abs(pred) + np.abs(actual)) / 2.0
    return float(np.mean(_safe_div(np.abs(pred - actual), den)) * 100.0)


def _mase(pred, actual, contexts):
    # MASE against persistence: last observed target repeated H times. contexts must be (N,T).
    naive = np.repeat(contexts[:, -1:], actual.shape[1], axis=1)
    scale = np.mean(np.abs(actual - naive))
    return float(np.mean(np.abs(pred - actual)) / (scale + EPS))


def _acf(x, lag=1):
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) <= lag + 1:
        return np.nan
    x = x - x.mean()
    den = np.dot(x, x)
    if den < EPS:
        return np.nan
    return float(np.dot(x[:-lag], x[lag:]) / den)


def _dtw(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _mmd_rbf(x, y, max_n=500):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) > max_n:
        x = x[np.linspace(0, len(x) - 1, max_n).astype(int)]
    if len(y) > max_n:
        y = y[np.linspace(0, len(y) - 1, max_n).astype(int)]
    z = np.concatenate([x, y])
    d = np.abs(z[:, None] - z[None, :])
    nonzero = d[d > 0]
    sigma = np.median(nonzero) if len(nonzero) else 1.0
    K = np.exp(-(d ** 2) / (2.0 * sigma ** 2 + EPS))
    n = len(x)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2.0 * K[:n, n:].mean())


def _crps_ensemble(samples, actual):
    # samples (N,S,H), actual (N,H). Pairwise term uses the sorted-sample identity, so no (N,S,S,H) tensor is built.
    samples = np.asarray(samples, dtype=float)
    actual = np.asarray(actual, dtype=float)
    term1 = np.mean(np.abs(samples - actual[:, None, :]), axis=1)
    s = samples.shape[1]
    if s < 2:
        return float(term1.mean())
    sorted_s = np.sort(samples, axis=1)
    weights = (2.0 * np.arange(1, s + 1) - s - 1.0)[None, :, None]
    pair = 2.0 * np.sum(weights * sorted_s, axis=1) / (s * s)
    return float(np.mean(term1 - 0.5 * pair))


def _energy_score(samples, actual):
    samples = np.asarray(samples, dtype=float)
    actual = np.asarray(actual, dtype=float)
    t1 = np.mean(np.linalg.norm(samples - actual[:, None, :], axis=-1), axis=1)
    if samples.shape[1] < 2:
        return float(t1.mean())
    t2 = np.empty(samples.shape[0])
    for i in range(samples.shape[0]):
        t2[i] = np.linalg.norm(samples[i][:, None, :] - samples[i][None, :, :], axis=-1).mean()
    return float(np.mean(t1 - 0.5 * t2))


def _pinball(samples, actual, q):
    prediction = np.quantile(samples, q, axis=1)
    error = actual - prediction
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def _interval_metrics(samples, actual, alpha=0.10):
    lo = np.quantile(samples, alpha / 2.0, axis=1)
    hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
    covered = (actual >= lo) & (actual <= hi)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(hi - lo))
    pinaw = mpiw / (np.mean(np.abs(actual)) + EPS)
    score = (hi - lo) + (2.0 / alpha) * (lo - actual) * (actual < lo) + (2.0 / alpha) * (actual - hi) * (actual > hi)
    return {"PICP": picp, "MPIW": mpiw, "PINAW": float(pinaw), "Winkler Score": float(np.mean(score)), "Interval Score": float(np.mean(score))}


def _last(value):
    # Training histories store per-epoch lists; the benchmark dictionary reports the final epoch.
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[-1]) if len(value) else np.nan
    return np.nan if value is None else value


def _common_metrics(predictions, actuals, samples, contexts, history=None, quantum_info=None):
    # contexts must be the target-only history (N,T). Covariates are inputs only, so covariate-consistency metrics are N/A.
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(actuals, dtype=float)
    s = np.asarray(samples, dtype=float)
    c = np.asarray(contexts, dtype=float)
    if p.ndim != 2 or a.ndim != 2 or s.ndim != 3:
        raise ValueError("Expected predictions=(N,H), actuals=(N,H), samples=(N,S,H)")
    H = p.shape[1]
    err = p - a
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(_safe_div(err, a))) * 100.0)
    smape = _smape(p, a)
    wape = float(abs_err.sum() / (np.abs(a).sum() + EPS) * 100.0)
    mase = _mase(p, a, c)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + EPS))
    horizon_wise = {}
    coverage_vs_horizon = []
    for h in range(H):
        e = p[:, h] - a[:, h]
        horizon_wise[f"t+{h + 1}"] = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "sMAPE": _smape(p[:, h], a[:, h])}
        lo = np.quantile(s[:, :, h], 0.05, axis=1)
        hi = np.quantile(s[:, :, h], 0.95, axis=1)
        coverage_vs_horizon.append(float(np.mean((a[:, h] >= lo) & (a[:, h] <= hi))))
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    interval = _interval_metrics(s, a, alpha=0.10)
    nominal = []
    empirical = []
    for qlo, qhi in [(0.05, 0.95), (0.10, 0.90), (0.25, 0.75)]:
        lo = np.quantile(s, qlo, axis=1)
        hi = np.quantile(s, qhi, axis=1)
        empirical.append(float(np.mean((a >= lo) & (a <= hi))))
        nominal.append(qhi - qlo)
    calibration_error = float(np.mean(np.abs(np.asarray(empirical) - np.asarray(nominal))))
    real = a.reshape(-1)
    generated = s.reshape(-1)
    wasserstein = float(wasserstein_distance(real, generated)) if wasserstein_distance is not None else np.nan
    ks = float(ks_2samp(real, generated).statistic) if ks_2samp is not None else np.nan
    bins = np.histogram_bin_edges(np.concatenate([real, generated]), bins=50)
    hist_real, _ = np.histogram(real, bins=bins)
    hist_gen, _ = np.histogram(generated, bins=bins)
    prob_real = (hist_real + EPS) / np.sum(hist_real + EPS)
    prob_gen = (hist_gen + EPS) / np.sum(hist_gen + EPS)
    midpoint = 0.5 * (prob_real + prob_gen)
    js = float(0.5 * np.sum(prob_real * np.log(prob_real / midpoint)) + 0.5 * np.sum(prob_gen * np.log(prob_gen / midpoint)))
    scenario_errors = np.mean(np.abs(s - a[:, None, :]), axis=2)
    best_of_mm = float(np.mean(np.min(scenario_errors, axis=1)))
    diversity = float(np.mean(np.std(s, axis=1)))
    acf_errors = []
    spectral_distances = []
    dtw_distances = []
    for i in range(min(len(a), 100)):
        actual = a[i]
        generated_mean = s[i].mean(axis=0)
        acf_a = _acf(actual, 1)
        acf_g = _acf(generated_mean, 1)
        if np.isfinite(acf_a) and np.isfinite(acf_g):
            acf_errors.append(abs(acf_a - acf_g))
        fa = np.abs(np.fft.rfft(actual - actual.mean()))
        fg = np.abs(np.fft.rfft(generated_mean - generated_mean.mean()))
        fa = fa / (fa.mean() + EPS)
        fg = fg / (fg.mean() + EPS)
        spectral_distances.append(float(np.mean(np.abs(fa - fg))))
        dtw_distances.append(_dtw(actual, generated_mean))
    temporal = {"ACF error": float(np.mean(acf_errors)) if acf_errors else np.nan, "Cross-correlation error": np.nan, "Spectral distance": float(np.mean(spectral_distances)) if spectral_distances else np.nan, "DTW distance": float(np.mean(dtw_distances)) if dtw_distances else np.nan}
    multivariate_consistency = {"Correlation-matrix error": np.nan, "Constraint violation rate": np.nan}
    history = history or {}
    quantum_info = quantum_info or {}
    gan_specific = {"Discriminator accuracy": _last(history.get("discriminator_accuracy", np.nan)), "Generator loss": _last(history.get("generator_loss", np.nan)), "Discriminator loss": _last(history.get("discriminator_loss", np.nan)), "Gradient norm": _last(history.get("gradient_norm", np.nan)), "Mode-collapse indicator": float(1.0 / (1.0 + diversity))}
    efficiency = {"Parameter count": quantum_info.get("parameter_count", np.nan), "Training time": history.get("training_time", np.nan), "Inference time": history.get("inference_time", np.nan), "Memory usage": history.get("memory_usage", np.nan)}
    quantum_specific = {"Number of qubits": quantum_info.get("n_qubits", np.nan), "Circuit depth": quantum_info.get("circuit_depth", np.nan), "Number of trainable quantum parameters": quantum_info.get("quantum_parameters", np.nan), "Number of shots": quantum_info.get("shots", np.nan), "Circuit evaluations / epoch": quantum_info.get("circuit_evals_per_epoch", np.nan), "Noise robustness": quantum_info.get("noise_robustness", np.nan)}
    out = {}
    out["point_forecast"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape, "MASE": mase, "R2": r2}
    out["horizon_wise"] = horizon_wise
    out["probabilistic_forecast"] = {"CRPS": _crps_ensemble(s, a), "Pinball": {str(q): _pinball(s, a, q) for q in quantiles}, "NLL": np.nan, "Energy Score": _energy_score(s, a)}
    out["prediction_intervals"] = interval
    out["calibration"] = {"Calibration Error": calibration_error, "Reliability diagram": {"nominal": nominal, "empirical": empirical}, "Coverage-vs-Horizon": coverage_vs_horizon}
    out["generated_distribution"] = {"Wasserstein Distance": wasserstein, "KS statistic": ks, "Jensen-Shannon divergence": js, "Maximum Mean Discrepancy": _mmd_rbf(real, generated)}
    out["multimodal_scenario"] = {"Mode coverage": np.nan, "Scenario probability calibration": np.nan, "Diversity score": diversity, "Best-of-MM error": best_of_mm}
    out["temporal_realism"] = temporal
    out["multivariate_consistency"] = multivariate_consistency
    out["gan_specific"] = gan_specific
    out["efficiency"] = efficiency
    out["quantum_specific"] = quantum_specific
    return out


# ============================================================
# QUANTUM UTILITIES
# ============================================================

def quantum_circuit(n_qubits, n_layers, data_reuploading=True):
    if qml is None:
        raise ImportError("PennyLane is required for QC/CQ/QQ models.")
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        for layer in range(n_layers):
            if data_reuploading and layer > 0:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def _init_qweights(n_layers, n_qubits):
    return torch.randn(n_layers, n_qubits, 3) * 0.1


class BatchedQuantumRunner:
    def __init__(self, qnode):
        self.qnode = qnode
        self._batched = None

    def __call__(self, inputs, weights):
        if self._batched is not False:
            try:
                out = self.qnode(inputs, weights)
                if isinstance(out, (list, tuple)):
                    out = torch.stack(list(out), dim=-1)
                if out.ndim == 2 and out.shape[0] == inputs.shape[0]:
                    self._batched = True
                    return out
                self._batched = False
            except Exception:
                self._batched = False
        values = []
        for i in range(inputs.shape[0]):
            result = self.qnode(inputs[i], weights)
            if isinstance(result, (list, tuple)):
                result = torch.stack(list(result))
            values.append(result)
        return torch.stack(values)


# ============================================================
# STOCK EMBEDDING / TEMPORAL ENCODER
# ============================================================

class StockEmbedding(nn.Module):
    def __init__(self, num_stocks, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_stocks, embedding_dim)

    def forward(self, stock_ids):
        return self.embedding(stock_ids.long())


class TemporalEncoder(nn.Module):
    # Encodes a historical window (B,T,F) into (B,hidden).
    def __init__(self, features, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=features, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        _, h = self.rnn(x)
        return h[-1]


# ============================================================
# GENERATORS
# ============================================================

class ClassicalGenerator(nn.Module):
    def __init__(self, input_features, context_length, horizon, latent_size, hidden_size, condition_dim=0):
        super().__init__()
        self.encoder = TemporalEncoder(input_features, hidden_size)
        self.net = nn.Sequential(nn.Linear(hidden_size + latent_size + condition_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, context, noise, condition=None):
        h = self.encoder(context)
        parts = [h, noise]
        if condition is not None:
            parts.append(condition)
        return self.net(torch.cat(parts, dim=1))


class QuantumGenerator(nn.Module):
    def __init__(self, input_features, context_length, horizon, latent_size, n_qubits, quantum_layers, hidden_size, condition_dim=0):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for quantum models.")
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers
        self.encoder = TemporalEncoder(input_features, hidden_size)
        self.projection = nn.Sequential(nn.Linear(hidden_size + latent_size + condition_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_qubits))
        self.qweights = nn.Parameter(_init_qweights(quantum_layers, n_qubits))
        self.scale = nn.Parameter(torch.ones(n_qubits))
        self.qnode = BatchedQuantumRunner(quantum_circuit(n_qubits, quantum_layers))
        self.decoder = nn.Sequential(nn.Linear(n_qubits, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, context, noise, condition=None):
        h = self.encoder(context)
        parts = [h, noise]
        if condition is not None:
            parts.append(condition)
        z = torch.cat(parts, dim=1)
        angles = self.projection(z)
        angles = torch.tanh(angles * self.scale) * torch.pi
        qout = self.qnode(angles, self.qweights)
        # FIX: PennyLane returns float64; cast to the network dtype so the float32 decoder accepts it.
        qout = qout.to(device=angles.device, dtype=angles.dtype)
        return self.decoder(qout)


# ============================================================
# DISCRIMINATORS
# ============================================================

class ClassicalDiscriminator(nn.Module):
    def __init__(self, input_features, horizon, hidden_size, condition_dim=0):
        super().__init__()
        self.context_encoder = TemporalEncoder(input_features, hidden_size)
        self.future_encoder = nn.Sequential(nn.Linear(horizon, hidden_size), nn.ReLU())
        self.net = nn.Sequential(nn.Linear(hidden_size + hidden_size + condition_dim, hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, 1))

    def forward(self, context, future, condition=None):
        c = self.context_encoder(context)
        f = self.future_encoder(future)
        parts = [c, f]
        if condition is not None:
            parts.append(condition)
        return self.net(torch.cat(parts, dim=1))


class QuantumDiscriminator(nn.Module):
    def __init__(self, input_features, horizon, n_qubits, quantum_layers, hidden_size, condition_dim=0):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for quantum models.")
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers
        self.context_encoder = TemporalEncoder(input_features, hidden_size)
        self.future_encoder = nn.Sequential(nn.Linear(horizon, hidden_size), nn.ReLU())
        self.projection = nn.Sequential(nn.Linear(hidden_size + hidden_size + condition_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_qubits))
        self.qweights = nn.Parameter(_init_qweights(quantum_layers, n_qubits))
        self.scale = nn.Parameter(torch.ones(n_qubits))
        self.qnode = BatchedQuantumRunner(quantum_circuit(n_qubits, quantum_layers))
        self.classifier = nn.Sequential(nn.Linear(n_qubits, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, context, future, condition=None):
        c = self.context_encoder(context)
        f = self.future_encoder(future)
        parts = [c, f]
        if condition is not None:
            parts.append(condition)
        h = torch.cat(parts, dim=1)
        angles = self.projection(h)
        angles = torch.tanh(angles * self.scale) * torch.pi
        qout = self.qnode(angles, self.qweights)
        # FIX: same float64 -> float32 cast as in the quantum generator.
        qout = qout.to(device=angles.device, dtype=angles.dtype)
        return self.classifier(qout)


# ============================================================
# MULTISEQUENCE TREND FEATURES
# ============================================================

class TrendFeatureEncoder(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

    @staticmethod
    def slope(x):
        T = x.shape[1]
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        t = t - t.mean()
        return ((x - x.mean(dim=1, keepdim=True)) * t.unsqueeze(0)).sum(dim=1) / ((t ** 2).sum() + EPS)

    def forward(self, context):
        # FIX: use ONLY the target column (B,T). The old code fed the full (B,T,F) window, which produced [B,30] features next to [B,1] ones and crashed torch.stack.
        x = context[:, :, 0] if context.dim() == 3 else context
        slope = self.slope(x)
        if x.shape[1] > 1:
            last_delta = x[:, -1] - x[:, -2]
            volatility = (x[:, 1:] - x[:, :-1]).std(dim=1, unbiased=False)
        else:
            last_delta = torch.zeros_like(slope)
            volatility = torch.zeros_like(slope)
        level = x.mean(dim=1)
        features = torch.stack([slope, last_delta, volatility, level], dim=1)
        return self.net(features)


class NeuralTrajectorySelector(nn.Module):
    # Scores candidate futures from (context, candidate, realism). It never sees the observed future.
    def __init__(self, input_features, hidden_size=64):
        super().__init__()
        self.context_encoder = TemporalEncoder(input_features, hidden_size)
        self.future_encoder = nn.GRU(1, hidden_size, batch_first=True)
        self.net = nn.Sequential(nn.Linear(hidden_size + hidden_size + 1, hidden_size), nn.ReLU(), nn.Dropout(0.10), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))

    def forward(self, context, candidates, realism):
        B, K, H = candidates.shape
        context_h = self.context_encoder(context)
        future = candidates.reshape(B * K, H, 1)
        _, h = self.future_encoder(future)
        future_h = h[-1].reshape(B, K, -1)
        context_h = context_h[:, None, :].expand(-1, K, -1)
        z = torch.cat([context_h, future_h, realism.unsqueeze(-1)], dim=-1)
        return self.net(z).squeeze(-1)


# ============================================================
# BASE BENCHMARK MODEL
# ============================================================

class _BenchmarkGAN:
    MODEL_NAME = "GAN"
    # Extra conditioning width appended to the stock embedding (0 for the plain QGAN, trend width for MultiSequenceGAN).
    TREND_DIM = 0

    def __init__(self, data=None, target=None, variables=None, type="QC", historical_lookup=30, horizon=FIXED_HORIZON, latent_size=8, n_qubits=4, quantum_layers=1, hidden_size=64, epochs=50, batch_size=32, learning_rate=1e-3, train_ratio=0.8, training_mode="combined", data_mode="univariate", stock_embedding_dim=8, variety_k=5, recon_loss_weight=1.0, seed=42, device=None, lag=0, n_samples=100, shots=None):
        set_seed(seed)
        self.type = _validate_type(type)
        _validate_modes(training_mode, data_mode)
        if horizon != FIXED_HORIZON:
            raise ValueError(f"horizon is fixed at {FIXED_HORIZON}; received {horizon}")
        if historical_lookup < 1:
            raise ValueError("historical_lookup must be >= 1")
        if lag < 0:
            raise ValueError("lag must be >= 0")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        self.data_mode = data_mode
        self.training_mode = training_mode
        self.historical_lookup = int(historical_lookup)
        self.horizon = FIXED_HORIZON
        self.latent_size = int(latent_size)
        self.n_qubits = int(n_qubits)
        self.quantum_layers = int(quantum_layers)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.train_ratio = float(train_ratio)
        self.stock_embedding_dim = int(stock_embedding_dim)
        self.variety_k = int(variety_k)
        self.recon_loss_weight = float(recon_loss_weight)
        self.seed = int(seed)
        self.device = device or DEVICE
        self.lag = int(lag)
        self.n_samples = int(n_samples)
        self.shots = shots
        self.stocks = _prepare_inputs(data=data, target=target, variables=variables, data_mode=data_mode)
        self.stock_names = list(self.stocks.keys())
        self.stock_to_id = {s: i for i, s in enumerate(self.stock_names)}
        self.id_to_stock = {i: s for s, i in self.stock_to_id.items()}
        self.input_features = next(iter(self.stocks.values()))["features"].shape[1]
        self.num_stocks = len(self.stock_names)
        self.use_stock_embedding = training_mode == "stock_id"
        self.models = {}
        self.histories = {}
        self.history = {}
        self.trained = False
        if training_mode != "per_stock":
            self._initialize_single_model(self.stocks)

    # ========================================================
    # WINDOW CREATION
    # ========================================================

    def _create_windows(self, features):
        required = self.historical_lookup + self.lag + self.horizon
        if len(features) < required:
            raise ValueError(f"Need at least {required} observations; received {len(features)}")
        X = []
        y = []
        for i in range(len(features) - required + 1):
            context_end = i + self.historical_lookup
            future_start = context_end + self.lag
            future_end = future_start + self.horizon
            X.append(features[i:context_end])
            y.append(features[future_start:future_end, 0])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def _prepare_windows(self, stock_data):
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        train_sid = []
        test_sid = []
        raw_test_contexts = []
        raw_test_targets = []
        self.stats = {}
        required = self.historical_lookup + self.lag + self.horizon
        for stock, obj in stock_data.items():
            sid = self.stock_to_id[stock]
            features = obj["features"]
            split = int(len(features) * self.train_ratio)
            if split < required or len(features) - split < required:
                raise ValueError(f"Stock '{stock}' does not have enough observations for both train and test windows.")
            trX, try_ = self._create_windows(features[:split])
            teX, tey = self._create_windows(features[split:])
            input_mean = trX.reshape(-1, self.input_features).mean(axis=0)
            input_std = trX.reshape(-1, self.input_features).std(axis=0) + EPS
            target_mean = try_.mean()
            target_std = try_.std() + EPS
            self.stats[sid] = {"input_mean": input_mean.astype(np.float32), "input_std": input_std.astype(np.float32), "target_mean": float(target_mean), "target_std": float(target_std)}
            train_X.append((trX - input_mean) / input_std)
            train_y.append((try_ - target_mean) / target_std)
            test_X.append((teX - input_mean) / input_std)
            test_y.append((tey - target_mean) / target_std)
            train_sid.append(np.full(len(trX), sid, dtype=np.int64))
            test_sid.append(np.full(len(teX), sid, dtype=np.int64))
            raw_test_contexts.append(teX)
            raw_test_targets.append(tey)
        return {"X_train": np.concatenate(train_X), "y_train": np.concatenate(train_y), "sid_train": np.concatenate(train_sid), "X_test": np.concatenate(test_X), "y_test": np.concatenate(test_y), "sid_test": np.concatenate(test_sid), "raw_X_test": np.concatenate(raw_test_contexts), "raw_y_test": np.concatenate(raw_test_targets)}

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    def _initialize_single_model(self, stock_data):
        self.local_stocks = stock_data
        windows = self._prepare_windows(stock_data)
        self.X_train = torch.tensor(windows["X_train"], dtype=torch.float32)
        self.y_train = torch.tensor(windows["y_train"], dtype=torch.float32)
        self.sid_train = torch.tensor(windows["sid_train"], dtype=torch.long)
        self.X_test = torch.tensor(windows["X_test"], dtype=torch.float32)
        self.y_test = torch.tensor(windows["y_test"], dtype=torch.float32)
        self.sid_test = windows["sid_test"]
        self.raw_X_test = windows["raw_X_test"]
        self.raw_y_test = windows["raw_y_test"]
        # FIX: the condition width now includes TREND_DIM so MultiSequenceGAN's [stock_embedding, trend] conditioning matches the network input size.
        condition_dim = (self.stock_embedding_dim if self.use_stock_embedding else 0) + self.TREND_DIM
        if self.use_stock_embedding:
            self.stock_embedding = StockEmbedding(self.num_stocks, self.stock_embedding_dim).to(self.device)
        else:
            self.stock_embedding = None
        if self.type in {"QC", "QQ"}:
            self.generator = QuantumGenerator(input_features=self.input_features, context_length=self.historical_lookup, horizon=self.horizon, latent_size=self.latent_size, n_qubits=self.n_qubits, quantum_layers=self.quantum_layers, hidden_size=self.hidden_size, condition_dim=condition_dim)
        else:
            self.generator = ClassicalGenerator(input_features=self.input_features, context_length=self.historical_lookup, horizon=self.horizon, latent_size=self.latent_size, hidden_size=self.hidden_size, condition_dim=condition_dim)
        if self.type in {"CQ", "QQ"}:
            self.discriminator = QuantumDiscriminator(input_features=self.input_features, horizon=self.horizon, n_qubits=self.n_qubits, quantum_layers=self.quantum_layers, hidden_size=self.hidden_size, condition_dim=condition_dim)
        else:
            self.discriminator = ClassicalDiscriminator(input_features=self.input_features, horizon=self.horizon, hidden_size=self.hidden_size, condition_dim=condition_dim)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        g_params = list(self.generator.parameters())
        d_params = list(self.discriminator.parameters())
        if self.stock_embedding is not None:
            g_params += list(self.stock_embedding.parameters())
            d_params += list(self.stock_embedding.parameters())
        self.g_optimizer = torch.optim.Adam(g_params, lr=self.learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(d_params, lr=self.learning_rate * 2.0, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def _all_modules(self):
        modules = [self.generator, self.discriminator]
        if self.stock_embedding is not None:
            modules.append(self.stock_embedding)
        return modules

    # ========================================================
    # CONDITIONING
    # ========================================================

    def _condition(self, sid):
        if self.stock_embedding is None:
            return None
        return self.stock_embedding(sid.to(self.device))

    def _full_condition(self, context, sid):
        # Condition used at inference. MultiSequenceGAN overrides this to append the trend features.
        return self._condition(sid)

    def _point_estimate(self, context, sid, samples, H):
        # Point forecast from the full (N,30) sample pool. MultiSequenceGAN overrides this with the selector.
        return samples[:, :H].mean(axis=0)

    # ========================================================
    # TRAIN ONE SHARED MODEL
    # ========================================================

    def _train_single(self):
        dataset = TensorDataset(self.X_train, self.y_train, self.sid_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        history = {"generator_loss": [], "discriminator_loss": [], "reconstruction_loss": [], "gradient_norm": [], "discriminator_accuracy": []}
        t0 = time.perf_counter()
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.epochs):
            g_total = 0.0
            d_total = 0.0
            r_total = 0.0
            grad_total = 0.0
            acc_total = 0.0
            batches = 0
            for context, real, sid in loader:
                context = context.to(self.device)
                real = real.to(self.device)
                sid = sid.to(self.device)
                B = context.shape[0]
                # ---------------- discriminator ----------------
                self.d_optimizer.zero_grad()
                condition = self._condition(sid)
                real_logits = self.discriminator(context, real, condition)
                real_loss = self.criterion(real_logits, torch.ones_like(real_logits) * 0.9)
                noise = torch.randn(B, self.latent_size, device=self.device)
                with torch.no_grad():
                    fake = self.generator(context, noise, condition)
                fake_logits = self.discriminator(context, fake, condition)
                fake_loss = self.criterion(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (real_loss + fake_loss) / 2.0
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.d_optimizer.step()
                # ---------------- generator ----------------
                self.g_optimizer.zero_grad()
                K = self.variety_k
                context_k = context[:, None].expand(-1, K, -1, -1).reshape(B * K, context.shape[1], context.shape[2])
                sid_k = sid[:, None].expand(-1, K).reshape(-1)
                condition_k = self._condition(sid_k)
                noise_k = torch.randn(B * K, self.latent_size, device=self.device)
                candidates = self.generator(context_k, noise_k, condition_k).reshape(B, K, self.horizon)
                candidate_flat = candidates.reshape(B * K, self.horizon)
                fake_logits_g = self.discriminator(context_k, candidate_flat, condition_k)
                adversarial_loss = self.criterion(fake_logits_g, torch.ones_like(fake_logits_g))
                distances = torch.mean(torch.abs(candidates - real[:, None, :]), dim=2)
                reconstruction_loss = distances.min(dim=1).values.mean()
                g_loss = adversarial_loss + self.recon_loss_weight * reconstruction_loss
                g_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0))
                self.g_optimizer.step()
                d_acc = ((real_logits.detach() > 0).float().mean() + (fake_logits.detach() < 0).float().mean()) / 2.0
                g_total += float(g_loss.item())
                d_total += float(d_loss.item())
                r_total += float(reconstruction_loss.item())
                grad_total += grad_norm
                acc_total += float(d_acc.item())
                batches += 1
            history["generator_loss"].append(g_total / max(batches, 1))
            history["discriminator_loss"].append(d_total / max(batches, 1))
            history["reconstruction_loss"].append(r_total / max(batches, 1))
            history["gradient_norm"].append(grad_total / max(batches, 1))
            history["discriminator_accuracy"].append(acc_total / max(batches, 1))
        history["training_time"] = time.perf_counter() - t0
        return history

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):
        if self.training_mode == "per_stock":
            self.models = {}
            self.histories = {}
            t0 = time.perf_counter()
            for stock in self.stock_names:
                univariate = self.data_mode == "univariate"
                data_arg = {stock: self.stocks[stock]["target"]} if univariate else None
                target_arg = None if univariate else {stock: self.stocks[stock]["target"]}
                variables_arg = None if univariate else {stock: self.stocks[stock]["variables"]}
                child = self.__class__(data=data_arg, target=target_arg, variables=variables_arg, type=self.type, historical_lookup=self.historical_lookup, horizon=FIXED_HORIZON, latent_size=self.latent_size, n_qubits=self.n_qubits, quantum_layers=self.quantum_layers, hidden_size=self.hidden_size, epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate, train_ratio=self.train_ratio, training_mode="combined", data_mode=self.data_mode, stock_embedding_dim=self.stock_embedding_dim, variety_k=self.variety_k, recon_loss_weight=self.recon_loss_weight, seed=self.seed, device=self.device, lag=self.lag, n_samples=self.n_samples, shots=self.shots)
                child.train()
                self.models[stock] = child
                self.histories[stock] = child.history
            self.history = {"per_stock": self.histories, "training_time": time.perf_counter() - t0}
            self.trained = True
            return self.history
        self.history = self._train_single()
        self.trained = True
        return self.history

    # ========================================================
    # SAMPLE GENERATION
    # ========================================================

    def _inverse_target(self, values, sid):
        stats = self.stats[int(sid)]
        return np.asarray(values) * stats["target_std"] + stats["target_mean"]

    def _generate_samples(self, context, sid, n_samples):
        context = np.asarray(context, dtype=np.float32)
        stats = self.stats[int(sid)]
        context_norm = (context - stats["input_mean"]) / stats["input_std"]
        x = torch.tensor(context_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = x.repeat(n_samples, 1, 1)
        sid_tensor = torch.full((n_samples,), int(sid), dtype=torch.long, device=self.device)
        noise = torch.randn(n_samples, self.latent_size, device=self.device)
        self.generator.eval()
        with torch.no_grad():
            condition = self._full_condition(x, sid_tensor)
            normalized_samples = self.generator(x, noise, condition).cpu().numpy()
        return self._inverse_target(normalized_samples, sid)

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, context, stock, prediction_horizon=None, n_samples=None):
        # Always generates 30 target values; prediction_horizon=H returns only [:H]. Multivariate context is (historical_lookup, 1+V) with the target in column 0.
        H = self.horizon if prediction_horizon is None else int(prediction_horizon)
        if not 1 <= H <= self.horizon:
            raise ValueError(f"prediction_horizon must be in [1,{self.horizon}]")
        if self.training_mode == "per_stock":
            if stock not in self.models:
                raise ValueError(f"Unknown stock '{stock}'")
            return self.models[stock].predict(context, stock=stock, prediction_horizon=H, n_samples=n_samples)
        if isinstance(stock, str):
            if stock not in self.stock_to_id:
                raise ValueError(f"Unknown stock '{stock}'")
            sid = self.stock_to_id[stock]
        else:
            sid = int(stock)
        context = np.asarray(context, dtype=np.float32)
        if self.data_mode == "univariate" and context.ndim == 1:
            context = context[:, None]
        if context.shape != (self.historical_lookup, self.input_features):
            raise ValueError(f"context must have shape ({self.historical_lookup}, {self.input_features}); got {context.shape}")
        N = self.n_samples if n_samples is None else int(n_samples)
        t0 = time.perf_counter()
        samples = self._generate_samples(context, sid, N)
        prediction = self._point_estimate(context, sid, samples, H)
        self.history["inference_time"] = time.perf_counter() - t0
        return prediction.astype(np.float32), samples[:, :H].astype(np.float32)

    # ========================================================
    # QUANTUM RESOURCE INFO
    # ========================================================

    def _quantum_info(self):
        if self.type not in {"QC", "CQ", "QQ"}:
            return {"n_qubits": np.nan, "circuit_depth": np.nan, "quantum_parameters": np.nan, "shots": np.nan, "circuit_evals_per_epoch": np.nan, "noise_robustness": np.nan}
        quantum_parameters = 0
        for module in self._all_modules():
            for name, p in module.named_parameters():
                if "qweights" in name:
                    quantum_parameters += p.numel()
        shots = self.shots if self.shots is not None else np.nan
        return {"n_qubits": self.n_qubits, "circuit_depth": self.quantum_layers * 2, "quantum_parameters": quantum_parameters, "shots": shots, "circuit_evals_per_epoch": math.ceil(len(self.X_train) / self.batch_size), "noise_robustness": np.nan}

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest(self, n_samples=None, prediction_horizon=None):
        # Common return contract for both models: model, type, training_mode, data_mode, fixed_horizon, prediction_horizon, stock_names, stock_ids, contexts, predictions, actuals, samples, metrics, history.
        H = self.horizon if prediction_horizon is None else int(prediction_horizon)
        if not 1 <= H <= self.horizon:
            raise ValueError(f"prediction_horizon must be in [1,{self.horizon}]")
        if self.training_mode == "per_stock":
            result_by_stock = {}
            for stock, model in self.models.items():
                result_by_stock[stock] = model.backtest(n_samples=n_samples, prediction_horizon=H)
            return self._aggregate_per_stock(result_by_stock, H)
        N = self.n_samples if n_samples is None else int(n_samples)
        predictions = []
        actuals = []
        samples = []
        contexts = []
        stock_ids = []
        stock_names = []
        inference_times = []
        for i in range(len(self.raw_X_test)):
            sid = int(self.sid_test[i])
            t0 = time.perf_counter()
            full = self._generate_samples(self.raw_X_test[i], sid, N)
            pred = self._point_estimate(self.raw_X_test[i], sid, full, H)
            inference_times.append(time.perf_counter() - t0)
            predictions.append(pred)
            actuals.append(self.raw_y_test[i, :H])
            samples.append(full[:, :H])
            contexts.append(self.raw_X_test[i])
            stock_ids.append(sid)
            stock_names.append(self.id_to_stock[sid])
        predictions = np.asarray(predictions, dtype=np.float32)
        actuals = np.asarray(actuals, dtype=np.float32)
        samples = np.asarray(samples, dtype=np.float32)
        contexts = np.asarray(contexts, dtype=np.float32)
        self.history["inference_time"] = float(np.mean(inference_times))
        parameter_count = sum(p.numel() for m in self._all_modules() for p in m.parameters())
        qinfo = self._quantum_info()
        qinfo["parameter_count"] = parameter_count
        metrics = _common_metrics(predictions, actuals, samples, contexts[:, :, 0], history=self.history, quantum_info=qinfo)
        return {"model": self.MODEL_NAME, "type": self.type, "training_mode": self.training_mode, "data_mode": self.data_mode, "fixed_horizon": self.horizon, "prediction_horizon": H, "stock_names": np.asarray(stock_names), "stock_ids": np.asarray(stock_ids, dtype=np.int64), "contexts": contexts, "predictions": predictions, "actuals": actuals, "samples": samples, "metrics": metrics, "history": self.history}

    # ========================================================
    # PER-STOCK AGGREGATION
    # ========================================================

    def _aggregate_per_stock(self, results, H):
        predictions = []
        actuals = []
        samples = []
        contexts = []
        stock_ids = []
        stock_names = []
        metrics_by_stock = {}
        for stock, result in results.items():
            predictions.append(result["predictions"])
            actuals.append(result["actuals"])
            samples.append(result["samples"])
            contexts.append(result["contexts"])
            sid = self.stock_to_id[stock]
            stock_ids.extend([sid] * len(result["predictions"]))
            stock_names.extend([stock] * len(result["predictions"]))
            metrics_by_stock[stock] = result["metrics"]
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        samples = np.concatenate(samples)
        contexts = np.concatenate(contexts)
        target_context = contexts[:, :, 0]
        child_times = [m.history.get("inference_time") for m in self.models.values() if m.history.get("inference_time") is not None]
        if child_times:
            self.history["inference_time"] = float(np.mean(child_times))
        parameter_count = sum(p.numel() for m in self.models.values() for mod in m._all_modules() for p in mod.parameters())
        qinfo = next(iter(self.models.values()))._quantum_info()
        qinfo["parameter_count"] = parameter_count
        metrics = _common_metrics(predictions, actuals, samples, target_context, history=self.history, quantum_info=qinfo)
        return {"model": self.MODEL_NAME, "type": self.type, "training_mode": "per_stock", "data_mode": self.data_mode, "fixed_horizon": self.horizon, "prediction_horizon": H, "stock_names": np.asarray(stock_names), "stock_ids": np.asarray(stock_ids, dtype=np.int64), "contexts": contexts, "predictions": predictions, "actuals": actuals, "samples": samples, "metrics": metrics, "metrics_by_stock": metrics_by_stock, "history": self.history}


# ============================================================
# QGAN MODEL
# ============================================================
# Plain benchmark QGAN: shared temporal encoder, generator/discriminator, fixed 30-step output, stock modes, multivariate conditioning, target-only forecast. No trend module or trajectory selector.

class QGanModel(_BenchmarkGAN):
    MODEL_NAME = "QGanModel"


# ============================================================
# MULTISEQUENCEGAN
# ============================================================
# Adds to QGanModel: (1) causal local trend features, (2) trend conditioning of generator and discriminator, (3) a neural multi-candidate trajectory selector used for the point forecast.

class MultiSequenceGAN(_BenchmarkGAN):
    MODEL_NAME = "MultiSequenceGAN"
    TREND_DIM = 32

    def _initialize_single_model(self, stock_data):
        super()._initialize_single_model(stock_data)
        self.trend_encoder = TrendFeatureEncoder(hidden_size=self.TREND_DIM).to(self.device)
        self.selector = NeuralTrajectorySelector(input_features=self.input_features, hidden_size=self.hidden_size).to(self.device)
        g_params = list(self.generator.parameters()) + list(self.trend_encoder.parameters()) + list(self.selector.parameters())
        if self.stock_embedding is not None:
            g_params += list(self.stock_embedding.parameters())
        self.g_optimizer = torch.optim.Adam(g_params, lr=self.learning_rate, betas=(0.5, 0.999))
        self.selector_fitted = False

    def _all_modules(self):
        return super()._all_modules() + [self.trend_encoder, self.selector]

    def _full_condition(self, context, sid):
        trend = self.trend_encoder(context)
        condition = self._condition(sid)
        return trend if condition is None else torch.cat([condition, trend], dim=1)

    def _train_single(self):
        dataset = TensorDataset(self.X_train, self.y_train, self.sid_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        history = {"generator_loss": [], "discriminator_loss": [], "reconstruction_loss": [], "selector_loss": [], "gradient_norm": [], "discriminator_accuracy": []}
        t0 = time.perf_counter()
        self.generator.train()
        self.discriminator.train()
        self.selector.train()
        self.trend_encoder.train()
        for epoch in range(self.epochs):
            g_total = 0.0
            d_total = 0.0
            r_total = 0.0
            s_total = 0.0
            grad_total = 0.0
            acc_total = 0.0
            batches = 0
            for context, real, sid in loader:
                context = context.to(self.device)
                real = real.to(self.device)
                sid = sid.to(self.device)
                B = context.shape[0]
                trend = self.trend_encoder(context)
                # ---------------- discriminator (trend detached so the D backward pass never frees the generator-step graph) ----------------
                self.d_optimizer.zero_grad()
                condition = self._condition(sid)
                trend_d = trend.detach()
                full_condition = trend_d if condition is None else torch.cat([condition, trend_d], dim=1)
                real_logits = self.discriminator(context, real, full_condition)
                real_loss = self.criterion(real_logits, torch.ones_like(real_logits) * 0.9)
                noise = torch.randn(B, self.latent_size, device=self.device)
                with torch.no_grad():
                    fake = self.generator(context, noise, full_condition.detach())
                fake_logits = self.discriminator(context, fake, full_condition)
                fake_loss = self.criterion(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (real_loss + fake_loss) / 2.0
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.d_optimizer.step()
                # ---------------- generator + selector ----------------
                self.g_optimizer.zero_grad()
                K = self.variety_k
                context_k = context[:, None].expand(-1, K, -1, -1).reshape(B * K, context.shape[1], context.shape[2])
                sid_k = sid[:, None].expand(-1, K).reshape(-1)
                trend_k = trend[:, None].expand(-1, K, -1).reshape(B * K, -1)
                condition_k = self._condition(sid_k)
                full_condition_k = trend_k if condition_k is None else torch.cat([condition_k, trend_k], dim=1)
                noise_k = torch.randn(B * K, self.latent_size, device=self.device)
                candidates = self.generator(context_k, noise_k, full_condition_k).reshape(B, K, self.horizon)
                candidate_flat = candidates.reshape(B * K, self.horizon)
                fake_logits_g = self.discriminator(context_k, candidate_flat, full_condition_k)
                adversarial_loss = self.criterion(fake_logits_g, torch.ones_like(fake_logits_g))
                distances = torch.mean(torch.abs(candidates - real[:, None, :]), dim=2)
                reconstruction_loss = distances.min(dim=1).values.mean()
                realism = fake_logits_g.reshape(B, K).detach()
                selector_scores = self.selector(context, candidates, realism)
                selector_weights = torch.softmax(selector_scores, dim=1)
                selector_target = torch.softmax(-distances.detach(), dim=1)
                selector_loss = -(selector_target * torch.log(selector_weights + EPS)).sum(dim=1).mean()
                g_loss = adversarial_loss + self.recon_loss_weight * reconstruction_loss + 0.25 * selector_loss
                g_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0))
                self.g_optimizer.step()
                d_acc = ((real_logits.detach() > 0).float().mean() + (fake_logits.detach() < 0).float().mean()) / 2.0
                g_total += float(g_loss.item())
                d_total += float(d_loss.item())
                r_total += float(reconstruction_loss.item())
                s_total += float(selector_loss.item())
                grad_total += grad_norm
                acc_total += float(d_acc.item())
                batches += 1
            history["generator_loss"].append(g_total / max(batches, 1))
            history["discriminator_loss"].append(d_total / max(batches, 1))
            history["reconstruction_loss"].append(r_total / max(batches, 1))
            history["selector_loss"].append(s_total / max(batches, 1))
            history["gradient_norm"].append(grad_total / max(batches, 1))
            history["discriminator_accuracy"].append(acc_total / max(batches, 1))
        history["training_time"] = time.perf_counter() - t0
        self.selector_fitted = True
        return history

    def _point_estimate(self, context, sid, samples, H):
        # Selector-weighted point forecast over the generated candidate pool; the full pool is still returned for probabilistic metrics.
        stats = self.stats[int(sid)]
        context_norm = (np.asarray(context, dtype=np.float32) - stats["input_mean"]) / stats["input_std"]
        x = torch.tensor(context_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        samples_norm = torch.tensor((samples - stats["target_mean"]) / stats["target_std"], dtype=torch.float32, device=self.device)
        n = samples_norm.shape[0]
        sid_tensor = torch.full((1,), int(sid), dtype=torch.long, device=self.device)
        self.discriminator.eval()
        self.selector.eval()
        with torch.no_grad():
            condition = self._full_condition(x, sid_tensor)
            realism = self.discriminator(x.repeat(n, 1, 1), samples_norm, condition.repeat(n, 1)).reshape(1, n)
            scores = self.selector(x, samples_norm.unsqueeze(0), realism)
            weights = torch.softmax(scores, dim=1)
            point_norm = (samples_norm.unsqueeze(0) * weights.unsqueeze(-1)).sum(dim=1).squeeze(0).cpu().numpy()
        point = point_norm * stats["target_std"] + stats["target_mean"]
        return point[:H]


# ============================================================
# BENCHMARK METRICS ALIAS
# ============================================================

def benchmark_metrics(predictions, actuals, samples, contexts, history=None, quantum_info=None):
    return _common_metrics(predictions, actuals, samples, contexts, history=history, quantum_info=quantum_info)


# ============================================================
# EXAMPLE USAGE
# ============================================================
# Univariate:   model = QGanModel(data={"AAPL": close}, type="QC", training_mode="combined", data_mode="univariate"); model.train(); result = model.backtest(n_samples=100, prediction_horizon=10)
# Multivariate: model = MultiSequenceGAN(target={"AAPL": close}, variables={"AAPL": features}, type="QC", training_mode="stock_id", data_mode="multivariate"); model.train()
# Predict:      prediction, samples = model.predict(context=np.column_stack([close[-30:], features[-30:]]), stock="AAPL", prediction_horizon=5, n_samples=100)
# ============================================================