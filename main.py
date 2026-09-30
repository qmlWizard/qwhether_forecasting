import argparse
import inspect
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import yaml
from ray import train as ray_train
from ray import tune

try:
    from ray.train import RunConfig
except ImportError:  # older Ray
    from ray.air import RunConfig

from models.timefm import TimesFMModel  # noqa: F401
from models.lstm import LSTMModel, QLSTMModel  # noqa: F401
from models.gan import QGanModel, MultiSequenceGAN


# ============================================================
# STATIC CONFIG
# ============================================================

model_classes = {
    "QGanModel": QGanModel,
    "MultiSequenceGAN": MultiSequenceGAN,
}

# Constructor-arg name  ->  config key it should be read from (when names differ)
ARG_ALIASES = {
    "horizon": "fixed_horizon",
    "n_samples": "samples",
    "type": "train_type",       # YAML uses train_type (QC / CC); constructor arg is `type`
}


def get_device():
    """Resolved inside the worker so it matches the GPU Ray assigned to the trial."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# ============================================================
# DATA LOADING
# ============================================================

def load_target_data(stocks, data_dir):
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


def prepare_data(stocks, data_dir, variable_files, historical_lookup, fixed_horizon, data_modes):
    """Loads only the data modes that will actually be used by the search space.
    `historical_lookup` should be the LARGEST value that will be tried."""
    target = load_target_data(stocks, data_dir=data_dir)
    for stock in stocks:
        if len(target[stock]) <= historical_lookup + fixed_horizon:
            raise ValueError(
                f"{stock}: insufficient observations for history={historical_lookup}, horizon={fixed_horizon}."
            )
    variables_by_mode = {mode: load_variable_data(target, mode, variable_files) for mode in data_modes}
    return target, variables_by_mode


def print_data_statistics(target, variables_by_mode):
    rows = []
    widest = variables_by_mode.get("multivariate") or next(iter(variables_by_mode.values()))
    for stock, y in target.items():
        rows.append({
            "Stock": stock, "N": len(y), "Mean": np.mean(y), "Std": np.std(y),
            "Variance": np.var(y), "Min": np.min(y), "Max": np.max(y),
            "Input Variables": widest[stock].shape[1],
        })
    print("\n" + "=" * 110)
    print("INPUT DATA STATISTICS")
    print("=" * 110)
    print(pd.DataFrame(rows).to_string(index=False))
    all_values = np.concatenate(list(target.values()))
    print("\n" + "=" * 110)
    print("COMBINED DATA INFORMATION")
    print("=" * 110)
    print(pd.DataFrame([{
        "Stocks": len(target), "Total Observations": len(all_values),
        "Mean": np.mean(all_values), "Std": np.std(all_values),
        "Variance": np.var(all_values), "Min": np.min(all_values), "Max": np.max(all_values),
    }]).to_string(index=False))


# ============================================================
# SEARCH SPACE
# ============================================================

def build_param(spec, random_mode=False):
    """
    Convert one YAML entry into a Ray Tune sampler.

      - plain scalar              -> constant
      - plain list                -> grid_search(list)
      - {type: grid,   values: [...]}
      - {type: choice, values: [...]}
      - {type: uniform|loguniform, low: a, high: b}
      - {type: randint, low: a, high: b}          (high exclusive)
      - {type: quniform|qloguniform, low, high, q}
    """
    grid = tune.choice if random_mode else tune.grid_search  # random mode: sample instead of enumerate
    if isinstance(spec, list):
        return grid(spec)
    if not isinstance(spec, dict) or "type" not in spec:
        return spec

    t = spec["type"].lower()
    if t == "grid":
        return grid(spec["values"])
    if t == "choice":
        return tune.choice(spec["values"])
    if t == "uniform":
        return tune.uniform(spec["low"], spec["high"])
    if t == "loguniform":
        return tune.loguniform(spec["low"], spec["high"], spec.get("base", 10))
    if t == "randint":
        return tune.randint(spec["low"], spec["high"])
    if t == "quniform":
        return tune.quniform(spec["low"], spec["high"], spec["q"])
    if t == "qloguniform":
        return tune.qloguniform(spec["low"], spec["high"], spec["q"], spec.get("base", 10))
    raise ValueError(f"Unknown search-space type '{spec['type']}'")


def spec_values(spec, default=None):
    """Candidate values of a search-space entry (list / grid / choice / scalar).
    Returns None for continuous samplers."""
    if spec is None:
        return default
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        if str(spec.get("type", "")).lower() in ("grid", "choice"):
            return spec["values"]
        return None
    return [spec]


def max_candidate(spec):
    vals = spec_values(spec)
    if vals is not None:
        return max(vals)
    if isinstance(spec, dict) and "high" in spec:
        return spec["high"]
    raise ValueError(f"Cannot determine the maximum of search-space entry: {spec}")


def count_trials(space, num_samples):
    """Total trials = product of all grid sizes x num_samples."""
    n = 1
    for v in space.values():
        if isinstance(v, dict) and "grid_search" in v:
            n *= len(v["grid_search"])
    return n * num_samples


def build_search_space(cfg, data_modes, stocks, random_mode=False):
    """
    Fixed params (cfg['training'] + cfg['dataset'] scalars) are added as constants.
    Anything listed under cfg['search_space'] overrides / adds a sampler.
    data_mode and model default to sensible grids if not specified.
    """
    space = {}

    # constants
    for section in ("training", "model_params"):
        for k, v in (cfg.get(section) or {}).items():
            space[k] = v
    ds = cfg["dataset"]
    if "historical_lookup" in ds:  # otherwise it comes from search_space
        space["historical_lookup"] = ds["historical_lookup"]
    space["fixed_horizon"] = ds["fixed_horizon"]
    space["stocks"] = list(stocks)

    # defaults driven by the loaded data
    grid = tune.choice if random_mode else tune.grid_search
    space["data_mode"] = grid(data_modes)
    space["model"] = grid(list(model_classes.keys()))

    # user-defined samplers (override everything above)
    for k, spec in (cfg.get("search_space") or {}).items():
        space[k] = build_param(spec, random_mode)

    return space


# ============================================================
# MODEL CONSTRUCTION VIA SIGNATURE
# ============================================================

def build_model_kwargs(cls, config, extras=None):
    """
    Inspect cls.__init__ and build kwargs from `config` (+ `extras`) that match
    its parameter names. Uses ARG_ALIASES when a name differs from the config key.
    Raises if a required argument can't be filled; logs config keys that were unused.
    """
    sig = inspect.signature(cls.__init__)
    pool = {**config, **(extras or {})}

    kwargs, missing = {}, []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in pool:
            kwargs[name] = pool[name]
        elif name in ARG_ALIASES and ARG_ALIASES[name] in pool:
            kwargs[name] = pool[ARG_ALIASES[name]]
        elif p.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise TypeError(f"{cls.__name__}: required arguments not found in config: {missing}")

    used = set(kwargs) | {ARG_ALIASES[k] for k in kwargs if k in ARG_ALIASES}
    unused = sorted(set(config) - used)
    if unused:
        print(f"[{cls.__name__}] config keys not consumed by constructor: {unused}")
    return kwargs


# ============================================================
# METRIC HANDLING
# ============================================================

def flatten_metrics(d, prefix=""):
    """Flatten nested dicts, keep only scalar numbers (Ray needs flat, serialisable metrics)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_metrics(v, prefix=f"{key}_"))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out[key] = float(v)
    return out


# ============================================================
# TRIAL FUNCTION
# ============================================================

def train_fn(config, target=None, variables_by_mode=None):
    # target / variables_by_mode come in via tune.with_parameters (object store),
    # NOT as globals -- globals don't exist on Ray workers.
    Model = model_classes[config["model"]]
    variables = variables_by_mode[config["data_mode"]]

    extras = {
        "data": config["stocks"],
        "target": target,
        "variables": variables,
        "device": get_device(),
    }
    model = Model(**build_model_kwargs(Model, config, extras))

    t0 = time.perf_counter()
    model.train()
    training_wall_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    backtest_result = model.backtest(
        n_samples=config["samples"],
        prediction_horizon=config["prediction_horizon"],
    )
    backtest_wall_time = time.perf_counter() - t0

    report = flatten_metrics(backtest_result.get("metrics", {}), prefix="metric_")
    report["training_wall_time"] = training_wall_time
    report["backtest_wall_time"] = backtest_wall_time
    ray_train.report(report)


# ============================================================
# RESULTS -> CSV
# ============================================================

def save_combined_csv(results, path, exp_name=None):
    df = results.get_dataframe()  # one row per trial: metrics + 'config/<param>' columns
    df = df.rename(columns=lambda c: c.replace("config/", "cfg_", 1) if c.startswith("config/") else c)

    if exp_name:
        df.insert(0, "exp_name", exp_name)

    cfg_cols = sorted(c for c in df.columns if c.startswith("cfg_"))
    metric_cols = sorted(c for c in df.columns if c.startswith("metric_") or c.endswith("_wall_time"))
    other = [c for c in df.columns if c not in cfg_cols + metric_cols]
    first = [c for c in ("exp_name", "trial_id") if c in df.columns]
    df = df[first + cfg_cols + metric_cols + [c for c in other if c not in first]]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nCombined config + metrics CSV written to: {path.resolve()}  ({len(df)} trials)")

    if results.errors:
        print(f"WARNING: {len(results.errors)} trial(s) failed and are not in the CSV.")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receives the yaml config file")
    parser.add_argument("--config", default="configs/checkerboard.yaml")
    parser.add_argument("--force", action="store_true",
                        help="launch even if the trial count exceeds ray_config.max_trials")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ray.init(address="auto", log_to_driver=False, ignore_reinit_error=True)

    ds = cfg["dataset"]
    ss = cfg.get("search_space") or {}

    # historical_lookup may live in the dataset section or in the search space
    hl_spec = ss.get("historical_lookup", ds.get("historical_lookup"))
    if hl_spec is None:
        raise ValueError("historical_lookup must be set under dataset or search_space.")

    # load only the data modes the search space will actually use
    data_modes = spec_values(ss.get("data_mode")) or (
        ["univariate"] + (["multivariate"] if ds.get("variable_files") else [])
    )
    if "multivariate" in data_modes and not ds.get("variable_files"):
        raise ValueError("data_mode includes 'multivariate' but dataset.variable_files is not set.")

    target, variables_by_mode = prepare_data(
        stocks=ds["stocks"],
        data_dir=ds["data_dir"],
        variable_files=ds.get("variable_files"),
        historical_lookup=max_candidate(hl_spec),  # validate against the largest window
        fixed_horizon=ds["fixed_horizon"],
        data_modes=data_modes,
    )
    print_data_statistics(target, variables_by_mode)

    # ray_config.num_trials set -> random mode: every grid entry becomes a random choice and
    # exactly num_trials configs are sampled. Otherwise: full grid x ray_num_trial_samples.
    num_trials = cfg["ray_config"].get("num_trials")
    search_space = build_search_space(cfg, data_modes, ds["stocks"], random_mode=bool(num_trials))
    num_samples = int(num_trials) if num_trials else cfg["ray_config"].get("ray_num_trial_samples", 1)

    rc = cfg["ray_config"]
    resources = ray.cluster_resources()
    available_cpu = resources.get("CPU", 0)
    available_gpu = resources.get("GPU", 0)
    cpu_capacity = int(available_cpu // rc["num_cpus"]) if rc["num_cpus"] else 0
    gpu_capacity = int(available_gpu // rc["num_gpus"]) if rc["num_gpus"] else 0

    print("\n" + "=" * 110)
    print("RAY CLUSTER")
    print("=" * 110)
    print(f"CPU resources       : {available_cpu}  (fits {cpu_capacity} concurrent trials)")
    print(f"GPU resources       : {available_gpu}  (fits {gpu_capacity} concurrent trials)")

    n_trials = count_trials(search_space, num_samples)
    max_trials = rc.get("max_trials", 10000)
    print(f"Total trials        : {n_trials:,}")

    if rc["num_cpus"] > available_cpu or (rc.get("num_gpus") or 0) > available_gpu:
        ray.shutdown()
        raise SystemExit(
            f"Per-trial request (cpu={rc['num_cpus']}, gpu={rc.get('num_gpus')}) exceeds cluster "
            f"resources (cpu={available_cpu}, gpu={available_gpu}); trials would never be scheduled."
        )
    if n_trials > max_trials and not args.force:
        ray.shutdown()
        raise SystemExit(
            f"{n_trials:,} trials exceeds max_trials={max_trials:,}. Shrink the grid, "
            f"raise ray_config.max_trials, or pass --force."
        )

    # ---- experiment naming: <exp_name>_<date>, used for the Ray run, results dir and CSV ----
    exp_name = re.sub(r"[^\w\-.]", "_", str(cfg.get("exp_name", "exp")))
    run_name = f"{exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    base_dir = Path(rc.get("results_dir") or Path(rc.get("results_csv", "results/x.csv")).parent)
    out_dir = base_dir / run_name
    csv_path = out_dir / f"{run_name}.csv"
    print(f"Run name            : {run_name}")
    print(f"Results CSV         : {csv_path}")

    trainable = tune.with_parameters(train_fn, target=target, variables_by_mode=variables_by_mode)
    trainable = tune.with_resources(trainable, {"cpu": rc["num_cpus"], "gpu": rc["num_gpus"]})

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            trial_dirname_creator=lambda t: f"{t.trial_id}",
        ),
        param_space=search_space,
        run_config=RunConfig(name=run_name),
    )

    results = tuner.fit()
    save_combined_csv(results, csv_path, exp_name=exp_name)
    ray.shutdown()