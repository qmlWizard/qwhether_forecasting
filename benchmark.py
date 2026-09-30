from __future__ import annotations

import inspect
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ray.tune import Callback as TuneCallback
except ImportError:
    TuneCallback = object


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS = ["coalindia", "tcs", "hindalco", "jswsteel"]
DATA_DIR = "data"

# Forecasting (FIXED_HORIZON is never tuned; PREDICTION_HORIZON only limits evaluation to the first H steps).
HISTORICAL_LOOKUP = 30
FIXED_HORIZON = 30
PREDICTION_HORIZON = 30
LAG = 0

# Set to a dict {stock: "path/to/features.npy"} (each file shaped (T, V)) to also benchmark "multivariate".
VARIABLE_FILES = None

TRAINING_MODES = ["per_stock", "combined", "stock_id"]
MODEL_NAMES = ["QGanModel", "MultiSequenceGAN"]

# Default model configuration (used for every value that is not tuned).
MODEL_TYPE = "QC"
LATENT_SIZE = 8
N_QUBITS = 6
QUANTUM_LAYERS = 4
HIDDEN_SIZE = 32
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
STOCK_EMBEDDING_DIM = 8
VARIETY_K = 5
RECON_LOSS_WEIGHT = 1.0
SEED = 42
DEVICE = None
N_SAMPLES = 100
SHOTS = None

# Ray.
RAY_MAX_CONCURRENT_TRIALS = 25
# Number of random hyperparameter samples drawn for EACH (model, training_mode, data_mode) combination.
# Total trials = len(MODEL_NAMES) * len(TRAINING_MODES) * len(data_modes) * RAY_SAMPLES_PER_COMBO.
RAY_SAMPLES_PER_COMBO = 3
RAY_CPUS_PER_TRIAL = 1
RAY_GPUS_PER_TRIAL = 1
RAY_USE_GPU = True
RAY_METRIC = "MAE"
RAY_MODE = "min"
# "random": grid over model/training/data mode + random hyperparameters (balanced benchmark, recommended).
# "optuna": Optuna picks everything (concentrates on the best combinations, so the benchmark is unbalanced).
RAY_SEARCH_ALGORITHM = "random"
RAY_STORAGE_PATH = "./ray_results"
RAY_CSV_PATH = "ray_benchmark_results.csv"

# Objective reported when a trial crashes, so the crash is still recorded in the CSV.
FAILURE_PENALTY = 1e12


# ============================================================
# DATA LOADING
# ============================================================

def load_target_data(stocks=STOCKS, data_dir=DATA_DIR):
    target = {}
    for stock in stocks:
        path = Path(data_dir) / f"{stock}_l10y.npy"
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {path}")
        target[stock] = np.asarray(np.load(path)).reshape(-1)
    return target


def load_variable_data(target, data_mode, variable_files=None):
    variables = {}
    if data_mode == "univariate":
        for stock, y in target.items():
            variables[stock] = y.reshape(-1, 1)
        return variables
    if variable_files is None:
        raise ValueError("data_mode='multivariate' requires variable_files.")
    for stock, y in target.items():
        if stock not in variable_files:
            raise KeyError(f"Missing variable file for {stock}.")
        path = Path(variable_files[stock])
        if not path.exists():
            raise FileNotFoundError(f"Variable file not found: {path}")
        x = np.asarray(np.load(path))
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"{stock}: variables must be (T,V), got {x.shape}")
        if x.shape[0] != len(y):
            raise ValueError(f"{stock}: target length={len(y)}, variables length={x.shape[0]}")
        variables[stock] = x
    return variables


def prepare_data(data_dir=DATA_DIR, variable_files=None):
    target = load_target_data(stocks=STOCKS, data_dir=data_dir)
    for stock in STOCKS:
        if len(target[stock]) <= HISTORICAL_LOOKUP + FIXED_HORIZON:
            raise ValueError(f"{stock}: insufficient observations for history={HISTORICAL_LOOKUP}, horizon={FIXED_HORIZON}.")
    data_modes = ["univariate"]
    if variable_files is not None:
        data_modes.append("multivariate")
    variables_by_mode = {mode: load_variable_data(target, mode, variable_files) for mode in data_modes}
    return target, variables_by_mode, data_modes


def print_data_statistics(target, variables_by_mode):
    rows = []
    widest = variables_by_mode.get("multivariate", variables_by_mode["univariate"])
    for stock, y in target.items():
        rows.append({"Stock": stock, "N": len(y), "Mean": np.mean(y), "Std": np.std(y), "Variance": np.var(y), "Min": np.min(y), "Max": np.max(y), "Input Variables": widest[stock].shape[1]})
    print("\n" + "=" * 110)
    print("INPUT DATA STATISTICS")
    print("=" * 110)
    print(pd.DataFrame(rows).to_string(index=False))
    all_values = np.concatenate(list(target.values()))
    print("\n" + "=" * 110)
    print("COMBINED DATA INFORMATION")
    print("=" * 110)
    print(pd.DataFrame([{"Stocks": len(target), "Total Observations": len(all_values), "Mean": np.mean(all_values), "Std": np.std(all_values), "Variance": np.var(all_values), "Min": np.min(all_values), "Max": np.max(all_values)}]).to_string(index=False))


# ============================================================
# MODEL IMPORT / INTROSPECTION
# ============================================================

def import_models():
    from models.gan import QGanModel, MultiSequenceGAN
    return QGanModel, MultiSequenceGAN


def get_init_defaults(model_class):
    signature = inspect.signature(model_class.__init__)
    defaults = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        defaults[name] = None if parameter.default is inspect.Parameter.empty else parameter.default
    return defaults


def get_init_parameter_names(model_class):
    return list(get_init_defaults(model_class))


def scalarize(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 and np.isfinite(value.item()) else json.dumps(value.tolist(), default=str)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return str(value)


def flatten_dict(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: scalarize(value)} if prefix else {}
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            out.update(flatten_dict(item, name))
        else:
            out[name] = scalarize(item)
    return out


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def make_base_config(model_class, target, variables, training_mode, data_mode):
    defaults = get_init_defaults(model_class)
    init_names = set(defaults)
    config = dict(defaults)
    overrides = {}
    overrides["type"] = MODEL_TYPE
    overrides["historical_lookup"] = HISTORICAL_LOOKUP
    overrides["horizon"] = FIXED_HORIZON
    overrides["latent_size"] = LATENT_SIZE
    overrides["n_qubits"] = N_QUBITS
    overrides["quantum_layers"] = QUANTUM_LAYERS
    overrides["hidden_size"] = HIDDEN_SIZE
    overrides["epochs"] = EPOCHS
    overrides["batch_size"] = BATCH_SIZE
    overrides["learning_rate"] = LEARNING_RATE
    overrides["train_ratio"] = TRAIN_RATIO
    overrides["stock_embedding_dim"] = STOCK_EMBEDDING_DIM
    overrides["variety_k"] = VARIETY_K
    overrides["recon_loss_weight"] = RECON_LOSS_WEIGHT
    overrides["seed"] = SEED
    overrides["device"] = DEVICE
    overrides["lag"] = LAG
    overrides["n_samples"] = N_SAMPLES
    overrides["shots"] = SHOTS
    overrides["training_mode"] = training_mode
    overrides["data_mode"] = data_mode
    overrides["prediction_horizon"] = PREDICTION_HORIZON
    for key, value in overrides.items():
        if key in init_names:
            config[key] = value
    if "target" in init_names:
        config["target"] = target
    if "variables" in init_names:
        config["variables"] = variables
    if "data" in init_names:
        config["data"] = target
    return {key: value for key, value in config.items() if key in init_names}


def enforce_benchmark_constraints(kwargs, init_names, training_mode, data_mode):
    # Tuning can never change these.
    if "horizon" in init_names:
        kwargs["horizon"] = FIXED_HORIZON
    if "prediction_horizon" in init_names:
        kwargs["prediction_horizon"] = PREDICTION_HORIZON
    if "training_mode" in init_names:
        kwargs["training_mode"] = training_mode
    if "data_mode" in init_names:
        kwargs["data_mode"] = data_mode
    return kwargs


def csv_metadata(model_class, kwargs, training_mode, data_mode, trial_id):
    # Large arrays are never written to the CSV; only their source is recorded.
    row = {}
    row["trial_id"] = trial_id
    row["model"] = model_class.__name__
    row["training_mode"] = training_mode
    row["data_mode"] = data_mode
    row["fixed_horizon"] = FIXED_HORIZON
    row["prediction_horizon"] = PREDICTION_HORIZON
    row["stocks"] = json.dumps(STOCKS)
    row["n_stocks"] = len(STOCKS)
    row["target_source"] = f"{DATA_DIR}/<stock>_l10y.npy"
    row["variable_source"] = "target itself" if data_mode == "univariate" else "variable_files"
    for name in get_init_parameter_names(model_class):
        if name in {"data", "target"}:
            row[f"arg_{name}"] = "target_dict"
        elif name == "variables":
            row[f"arg_{name}"] = "variables_dict"
        else:
            row[f"arg_{name}"] = scalarize(kwargs.get(name))
    return row


# ============================================================
# RAY SEARCH SPACE
# ============================================================

def make_search_space(tune, data_modes, use_grid):
    chooser = tune.grid_search if use_grid else tune.choice
    space = {}
    space["model_class"] = chooser(list(MODEL_NAMES))
    space["training_mode"] = chooser(list(TRAINING_MODES))
    space["data_mode"] = chooser(list(data_modes))
    space["latent_size"] = tune.choice([LATENT_SIZE, 4, 16])
    space["n_qubits"] = tune.choice([N_QUBITS, 2, 12])
    space["quantum_layers"] = tune.choice([QUANTUM_LAYERS, 2, 10])
    space["hidden_size"] = tune.choice([HIDDEN_SIZE, 32, 128])
    space["batch_size"] = tune.choice([BATCH_SIZE, 16, 64])
    space["learning_rate"] = tune.loguniform(1e-4, 1e-2)
    space["stock_embedding_dim"] = tune.choice([STOCK_EMBEDDING_DIM, 4, 16])
    space["variety_k"] = tune.choice([VARIETY_K, 3, 10])
    space["recon_loss_weight"] = tune.loguniform(1e-2, 1)
    return space


# ============================================================
# MODEL-RETURNED RESULTS -> CSV ROWS
# ============================================================

def history_rows(history, metadata):
    rows = []
    if not isinstance(history, dict):
        return rows

    def visit(obj, prefix):
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj)
            if arr.ndim == 1 and arr.dtype != object:
                for i, value in enumerate(arr, start=1):
                    rows.append({**metadata, "record_type": "epoch", "epoch": i, "metric_scope": "training", "metric_name": prefix, "metric_value": scalarize(value)})
                return
        rows.append({**metadata, "record_type": "training_metric", "epoch": None, "metric_scope": "training", "metric_name": prefix, "metric_value": scalarize(obj)})

    visit(history, "")
    return rows


def backtest_metric_rows(backtest_result, metadata):
    rows = []
    for metric_name, metric_value in flatten_dict(backtest_result.get("metrics", {})).items():
        rows.append({**metadata, "record_type": "backtest_metric", "epoch": None, "metric_scope": "backtest", "metric_name": metric_name, "metric_value": metric_value})
    metrics_by_stock = backtest_result.get("metrics_by_stock")
    if isinstance(metrics_by_stock, dict):
        for stock, stock_metrics in metrics_by_stock.items():
            for metric_name, metric_value in flatten_dict(stock_metrics).items():
                rows.append({**metadata, "record_type": "backtest_metric_per_stock", "epoch": None, "metric_scope": "backtest_per_stock", "stock": stock, "metric_name": metric_name, "metric_value": metric_value})
    return rows


def find_objective_metric(metrics, metric_name):
    lowered = {key.lower(): value for key, value in flatten_dict(metrics).items()}
    for candidate in (metric_name.lower(), f"point_forecast.{metric_name}".lower()):
        if lowered.get(candidate) is not None:
            return float(lowered[candidate])
    raise KeyError(f"Metric '{metric_name}' was not returned (or was NaN/inf). Available metrics: {list(lowered)[:30]}")


# ============================================================
# RAY COMPATIBILITY HELPERS
# ============================================================

def get_trial_id():
    try:
        from ray import train
        return train.get_context().get_trial_id()
    except Exception:
        from ray.air import session
        return session.get_trial_id()


def report_metrics(metrics):
    try:
        from ray import train
        train.report(metrics)
    except Exception:
        from ray.air import session
        session.report(metrics)


# ============================================================
# SINGLE RAY TRAINABLE (ONE TRIAL = ONE COMPLETE EXPERIMENT)
# ============================================================

def ray_trainable(config, target=None, variables_by_mode=None):
    config = dict(config)
    model_name = config["model_class"]
    training_mode = config["training_mode"]
    data_mode = config["data_mode"]
    variables = variables_by_mode[data_mode]
    QGanModel, MultiSequenceGAN = import_models()
    model_class = {"QGanModel": QGanModel, "MultiSequenceGAN": MultiSequenceGAN}[model_name]
    init_names = set(get_init_parameter_names(model_class))
    kwargs = make_base_config(model_class, target, variables, training_mode, data_mode)
    kwargs.update({key: value for key, value in config.items() if key in init_names})
    kwargs = enforce_benchmark_constraints(kwargs, init_names, training_mode, data_mode)
    trial_id = get_trial_id()
    metadata = csv_metadata(model_class, kwargs, training_mode, data_mode, trial_id)
    penalty = FAILURE_PENALTY if RAY_MODE == "min" else -FAILURE_PENALTY
    objective = penalty
    status = "ok"
    training_wall_time = None
    backtest_wall_time = None
    rows = []
    try:
        train_start = time.perf_counter()
        model = model_class(**kwargs)
        history = model.train()
        training_wall_time = time.perf_counter() - train_start
        rows.extend(history_rows(history, metadata))
        backtest_start = time.perf_counter()
        backtest_result = model.backtest(n_samples=N_SAMPLES, prediction_horizon=PREDICTION_HORIZON)
        backtest_wall_time = time.perf_counter() - backtest_start
        if not isinstance(backtest_result, dict):
            raise TypeError("model.backtest() must return a dictionary.")
        rows.extend(backtest_metric_rows(backtest_result, metadata))
        returned_history = backtest_result.get("history")
        if returned_history is not None:
            rows.extend(history_rows(returned_history, metadata))
        objective = find_objective_metric(backtest_result.get("metrics", {}), RAY_METRIC)
    except Exception as exc:
        status = "error"
        rows.append({**metadata, "record_type": "error", "epoch": None, "metric_scope": "error", "metric_name": type(exc).__name__, "metric_value": f"{exc} | {traceback.format_exc()[-1500:]}"})
    rows.append({**metadata, "record_type": "efficiency", "epoch": None, "metric_scope": "timing", "metric_name": "Training Time", "metric_value": training_wall_time})
    rows.append({**metadata, "record_type": "efficiency", "epoch": None, "metric_scope": "timing", "metric_name": "Inference Time", "metric_value": backtest_wall_time})
    rows.append({**metadata, "record_type": "objective", "epoch": None, "metric_scope": "ray", "metric_name": RAY_METRIC, "metric_value": objective})
    report_metrics({RAY_METRIC: objective, "status": status, "training_wall_time": training_wall_time, "backtest_wall_time": backtest_wall_time, "benchmark_rows_json": json.dumps(rows, default=str)})


# ============================================================
# INCREMENTAL CSV WRITING (DRIVER SIDE, SAFE ACROSS NODES)
# ============================================================

def write_trial_part(trial_id, last_result, parts_dir):
    payload = (last_result or {}).get("benchmark_rows_json")
    if not payload:
        return
    rows = json.loads(payload)
    if not rows:
        return
    Path(parts_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(parts_dir) / f"{trial_id}.csv", index=False)


class TrialPartsWriter(TuneCallback):
    # Writes each trial's rows to its own CSV the moment the trial finishes, so nothing is lost if the run is interrupted.
    def __init__(self, parts_dir):
        self.parts_dir = str(parts_dir)

    def on_trial_complete(self, iteration, trials, trial, **info):
        write_trial_part(trial.trial_id, trial.last_result, self.parts_dir)


def merge_parts(parts_dir, csv_path):
    files = sorted(Path(parts_dir).glob("*.csv"))
    if not files:
        warnings.warn(f"No per-trial CSV parts found in {parts_dir}.")
        return pd.DataFrame()
    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    first = [c for c in ["trial_id", "model", "training_mode", "data_mode", "record_type", "metric_scope", "stock", "epoch", "metric_name", "metric_value"] if c in merged.columns]
    merged = merged[first + [c for c in merged.columns if c not in first]]
    merged.to_csv(csv_path, index=False)
    return merged