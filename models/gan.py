import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

torch.set_default_dtype(torch.float32)
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def quantum_circuit(n_qubits=4, n_layers=1, data_reuploading=True, entangling=True):
    """
    Hardware-efficient variational circuit with optional data re-uploading
    and optional entanglement ("strongly entangling" ansatz).

    `inputs` is re-embedded at the start of every layer (instead of only
    once, before layer 0) when `data_reuploading=True` (default).
    Interleaving trainable rotations with repeated data encoding is the
    technique from Perez-Salinas et al., "Data re-uploading for a universal
    quantum classifier" (Quantum 4, 226, 2020). With n_layers=1 this has
    no effect, since there is nothing to re-upload into; it starts helping
    once quantum_layers > 1.

    `entangling=True` turns each layer into a genuine "strongly
    entangling layer": every qubit gets a full single-qubit rotation
    (`qml.Rot`, i.e. RZ-RY-RZ, 3 free angles) and the layer ends with a
    ring of CNOTs across all qubits. Without entanglement the circuit is
    a stack of independent single-qubit rotations -- mathematically it
    factorizes into n_qubits separate 1-qubit models and can never
    represent correlations between qubits, no matter how many layers or
    qubits are added.

    Because `qml.Rot` takes 3 angles per qubit (vs. 2 for RY+RZ), the
    trainable weight tensor for this circuit has shape
    (n_layers, n_qubits, 3). This is consistent across every call site
    (`q_weights` in all four quantum modules below).
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        for layer in range(n_layers):
            if data_reuploading and layer > 0:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
            if entangling and n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def _init_strongly_entangling_weights(quantum_layers, n_qubits, near_identity=True):
    """
    Shared weight-initialization helper for the (n_layers, n_qubits, 3)
    `qml.Rot` weight tensor.

    `near_identity=True` (default) initializes weights close to zero
    (small random noise) instead of uniformly across the full
    [-pi/2, pi/2] range. With entangling CNOTs in the circuit, a fully
    random initialization is exactly the regime where "barren plateaus"
    (McClean et al., Nat. Commun. 2018) are most severe. Starting near
    the identity (Grant et al., 2019) keeps the circuit close to a
    trivial, well-conditioned starting point and lets entanglement build
    up gradually as training proceeds.
    """
    if near_identity:
        return torch.randn(quantum_layers, n_qubits, 3) * 0.1
    return -torch.pi / 2 + torch.rand(quantum_layers, n_qubits, 3) * torch.pi


class _BatchedQuantumRunner:
    """
    Wraps a PennyLane qnode so a whole batch of angles can be evaluated in
    a single call when the installed PennyLane/device combination supports
    parameter broadcasting, instead of a Python-level per-sample loop.

    `weights` (the trainable `q_weights`) is shared across the batch and
    is passed through unbatched -- PennyLane's supported "many inputs, one
    set of trainable weights" broadcasting pattern. The first call tries
    the batched form; if the installed PennyLane version/device doesn't
    support it (or produces an unexpected shape), it falls back
    permanently to the explicit per-sample loop -- same numerical result
    either way, just slower.
    """

    def __init__(self, qnode):
        self.qnode = qnode
        self._batching_supported = None

    def __call__(self, angles, weights):
        if self._batching_supported is not False:
            try:
                out = torch.stack(self.qnode(angles, weights), dim=-1)
                if out.shape[0] == angles.shape[0]:
                    self._batching_supported = True
                    return out.to(device=angles.device, dtype=angles.dtype)
            except Exception:
                self._batching_supported = False

        outputs = [torch.stack(self.qnode(angles[i], weights)) for i in range(angles.shape[0])]
        return torch.stack(outputs, dim=0).to(device=angles.device, dtype=angles.dtype)


def set_seed(seed):
    """
    Fix every source of randomness used in this file (Python's `random`,
    numpy, and torch's CPU/CUDA generators), plus cuDNN's deterministic
    flags, so that two runs constructed with the same `seed` -- including
    across the CC/QC/CQ/QQ architecture choices in `MultiSequenceGAN` and
    `QGanModel`, which otherwise get independently-random weight
    initializations and noise draws -- are reproducible and therefore
    actually comparable to each other. Both models' `__init__` call this
    once, before building the generator/discriminator, using their `seed`
    constructor argument.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalProjection(nn.Module):
    """
    Temporal front-end for the quantum modules' `input_projection`,
    replacing a flat MLP (`Linear -> ReLU -> Linear`).

    An MLP treats the historical window (and, for the discriminator, the
    candidate future) as an unordered feature vector: a `Linear` layer's
    weight matrix has one independent weight per (timestep, output) pair
    and no notion that position t comes right before position t+1. A
    single-layer GRU processes the window step by step and carries a
    running hidden state forward, the standard inductive bias behind most
    classical RNN-based time-series models (trend, momentum, mean-
    reversion are all naturally expressed as "what happened most recently
    matters, and how it relates to what came before it matters too").
    This gives the raw numbers going into the quantum circuit a chance to
    already encode temporal structure, instead of asking Linear layers to
    rediscover it from scratch with no inductive bias for order at all.

    `seq` is the raw time series (context alone for the generator,
    context+future concatenated for the discriminator -- see each
    module's forward()), shape (batch, seq_len). `cond` is an optional
    non-temporal conditioning vector (the generator's noise vector)
    concatenated onto the GRU's final hidden state before the n_qubits
    projection; pass `cond=None` / `cond_size=0` when there is none (the
    discriminators, which have already folded everything temporal into
    `seq`).
    """

    def __init__(self, n_qubits, cond_size=0, hidden_size=32, rnn_hidden_size=None, rnn_type="gru"):
        super().__init__()
        rnn_hidden_size = rnn_hidden_size or max(8, hidden_size // 2)
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=1, hidden_size=rnn_hidden_size, batch_first=True)
        self.cond_size = cond_size
        self.head = nn.Sequential(nn.Linear(rnn_hidden_size + cond_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_qubits))

    def forward(self, seq, cond=None):
        seq = seq.unsqueeze(-1)  # (batch, seq_len) -> (batch, seq_len, 1)
        _, h_n = self.rnn(seq)
        h_n = h_n[0] if isinstance(h_n, tuple) else h_n  # LSTM returns (h_n, c_n)
        h = h_n[-1]  # last layer's final hidden state: (batch, rnn_hidden_size)
        h = torch.cat([h, cond], dim=1) if (self.cond_size > 0 and cond is not None) else h
        return self.head(h)


class QuantumGenerator(nn.Module):
    def __init__(self, context_size=6, horizon=6, latent_size=4, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnode = _BatchedQuantumRunner(quantum_circuit(n_qubits, quantum_layers))
        self.input_projection = TemporalProjection(n_qubits=n_qubits, cond_size=latent_size, hidden_size=hidden_size)
        self.q_weights = nn.Parameter(_init_strongly_entangling_weights(quantum_layers, n_qubits))
        self.input_scale = nn.Parameter(torch.ones(n_qubits))
        self.decoder = nn.Sequential(nn.Linear(n_qubits, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, context, noise):
        angles = self.input_projection(context, noise)
        angles = torch.tanh(angles * self.input_scale) * torch.pi
        quantum_outputs = self.qnode(angles, self.q_weights)
        return self.decoder(quantum_outputs)


class ClassicalGenerator(nn.Module):
    def __init__(self, context_size=6, horizon=6, latent_size=4, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(context_size + latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, context, noise):
        x = torch.cat([context, noise], dim=1)
        return self.network(x)


class ClassicalDiscriminator(nn.Module):
    def __init__(self, context_size=6, horizon=6, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(context_size + horizon, hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, 1))

    def forward(self, context, future):
        x = torch.cat([context, future], dim=1)
        return self.network(x)


class QuantumDiscriminator(nn.Module):
    def __init__(self, context_size=6, horizon=6, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnode = _BatchedQuantumRunner(quantum_circuit(n_qubits, quantum_layers))
        self.input_projection = TemporalProjection(n_qubits=n_qubits, cond_size=0, hidden_size=hidden_size)
        self.q_weights = nn.Parameter(_init_strongly_entangling_weights(quantum_layers, n_qubits))
        self.input_scale = nn.Parameter(torch.ones(n_qubits))
        self.classifier = nn.Sequential(nn.Linear(n_qubits, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, context, future):
        seq = torch.cat([context, future], dim=1)
        angles = self.input_projection(seq)
        angles = torch.tanh(angles * self.input_scale) * torch.pi
        quantum_outputs = self.qnode(angles, self.q_weights)
        return self.classifier(quantum_outputs)


# ============================================================
# QGAN MODEL
# ============================================================

class QGanModel:
    """
    FIXES applied vs. the original version:

    1. Normalization no longer leaks test data into train statistics.
       Previously `data_mean`/`data_std` were computed from
       `self.data[:split_raw]` where `split_raw = train_ratio * len(raw
       series)`, while the actual train/test boundary used everywhere
       else was `split = train_ratio * len(sequences)` (sequences are
       shorter than the raw series by `historical_lookup + horizon - 1`).
       Because `split_raw > split`, part of what later becomes the test
       set was included when fitting the normalization statistics. Now,
       sequences are built first from RAW (unnormalized) data, split by
       sequence count, and normalization statistics are computed only
       from the training sequences -- mirroring the leak-free approach
       already used in `MultiSequenceGAN`.

    2. The generator's reconstruction term used to be a plain MSE against
       the single observed real future, for every random noise draw, at
       a weight (5.0) that dominates the adversarial term. That pushes
       the generator to collapse to a near-deterministic mapping from
       context to future and ignore its noise input -- directly
       undermining the "stochastic forecast" framing used in `predict()`
       and `backtest()`. This is replaced with a best-of-`variety_k`
       reconstruction loss (same mechanism `MultiSequenceGAN` already
       uses): the generator only needs ONE of `variety_k` sampled
       candidates to be close to the real future, so noise is still free
       to produce a genuine spread of trajectories.

    3. `set_seed(seed)` is now called in `__init__` (was previously only
       done in `MultiSequenceGAN`), so QGanModel runs are reproducible.

    4. `metrics()` guards against a zero (or near-zero) variance target
       window, which previously produced `inf`/`nan` R2 silently.

    5. `lag` controls the spacing of historical input observations while
       keeping the forecast horizon consecutive.
    """

    def __init__(self, data, type="QC", historical_lookup=6, horizon=6, latent_size=4, n_qubits=4, quantum_layers=1, hidden_size=32, epochs=50, batch_size=16, learning_rate=1e-3, train_ratio=0.8, recon_loss_weight=5.0, variety_k=5, label_smoothing=0.9, seed=42, device=None, lag=1):
        set_seed(seed)
        self.seed = seed
        self.data = np.asarray(data, dtype=np.float32).reshape(-1)
        self.type = type.upper()
        self.historical_lookup = historical_lookup
        self.horizon = horizon
        self.lag = int(lag)
        if self.lag < 1:
            raise ValueError("lag must be an integer >= 1")
        if self.historical_lookup < 1:
            raise ValueError("historical_lookup must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.latent_size = latent_size
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.recon_loss_weight = recon_loss_weight
        self.variety_k = variety_k
        self.label_smoothing = label_smoothing
        self.device = device or DEVICE
        if self.type not in ["QC", "QQ", "CQ", "CC"]:
            raise ValueError("type must be QC, QQ, CQ or CC")

        # Build sequences from RAW (unnormalized) data first, then split
        # by sequence count, then fit normalization stats only on the
        # training sequences -- see fix #1 in the class docstring.
        self.contexts_raw, self.targets_raw = self._create_sequences()
        split = int(len(self.contexts_raw) * train_ratio)
        self.sequence_split = split

        train_values = np.concatenate([self.contexts_raw[:split].numpy().reshape(-1), self.targets_raw[:split].numpy().reshape(-1)])
        self.data_mean = float(train_values.mean())
        self.data_std = float(train_values.std() + 1e-8)

        self.contexts = (self.contexts_raw - self.data_mean) / self.data_std
        self.targets = (self.targets_raw - self.data_mean) / self.data_std

        self.X_train = self.contexts[:split]
        self.y_train = self.targets[:split]
        self.X_test = self.contexts[split:]
        self.y_test = self.targets[split:]

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------
        if self.type in ["QC", "QQ"]:
            self.generator = QuantumGenerator(context_size=historical_lookup, horizon=horizon, latent_size=latent_size, n_qubits=n_qubits, quantum_layers=quantum_layers, hidden_size=hidden_size)
        else:
            self.generator = ClassicalGenerator(context_size=historical_lookup, horizon=horizon, latent_size=latent_size, hidden_size=hidden_size)

        # ----------------------------------------------------
        # Discriminator
        # ----------------------------------------------------
        if self.type in ["CQ", "QQ"]:
            self.discriminator = QuantumDiscriminator(context_size=historical_lookup, horizon=horizon, n_qubits=n_qubits, quantum_layers=quantum_layers, hidden_size=hidden_size)
        else:
            self.discriminator = ClassicalDiscriminator(context_size=historical_lookup, horizon=horizon, hidden_size=hidden_size)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # ----------------------------------------------------
        # Optimizers (TTUR: discriminator trains at 2x the generator lr)
        # ----------------------------------------------------
        self.d_lr_multiplier = 2.0
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate * self.d_lr_multiplier, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def _create_sequences(self):
        # Sequences are built from the RAW series; normalization happens
        # afterward in __init__ using training-sequence-only statistics.
        #
        # `lag` is a FORECAST-GAP parameter, not an input sampling stride.
        # The historical input itself is always consecutive. `lag` specifies
        # how many observations immediately before the forecast origin are
        # skipped.
        #
        # Example: historical_lookup=6, horizon=6, lag=2
        #   data:    ... 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ...
        #   input:       7, 8, 9, 10, 11, 12
        #   skipped:                                  13, 14
        #   target:                                             15, 16, 17, 18, 19, 20
        #
        # Equivalently, if the input is displayed from most recent to oldest,
        # it is [12, 11, 10, 9, 8, 7]. The model itself receives the
        # chronological ordering [7, 8, 9, 10, 11, 12].
        X, Y = [], []
        n_windows = len(self.data) - self.historical_lookup - self.lag - self.horizon + 1

        for i in range(n_windows):
            context_start = i
            context_end = context_start + self.historical_lookup
            future_start = context_end + self.lag
            future_end = future_start + self.horizon

            X.append(self.data[context_start:context_end])
            Y.append(self.data[future_start:future_end])

        return torch.tensor(np.asarray(X), dtype=torch.float32), torch.tensor(np.asarray(Y), dtype=torch.float32)

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        history = {"generator_loss": [], "discriminator_loss": [], "reconstruction_loss": []}

        for epoch in range(self.epochs):
            g_total, d_total, recon_total = 0.0, 0.0, 0.0

            for context, real_future in loader:
                context = context.to(self.device)
                real_future = real_future.to(self.device)
                batch_size = context.shape[0]

                # ============================================
                # DISCRIMINATOR
                # ============================================
                self.d_optimizer.zero_grad()

                real_logits = self.discriminator(context, real_future)
                real_labels = torch.ones_like(real_logits) * self.label_smoothing
                real_loss = self.criterion(real_logits, real_labels)

                noise = torch.randn(batch_size, self.latent_size, device=self.device)
                fake_future = self.generator(context, noise)
                fake_logits = self.discriminator(context, fake_future.detach())
                fake_labels = torch.zeros_like(fake_logits)
                fake_loss = self.criterion(fake_logits, fake_labels)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                # Gradient clipping guards against occasional sharp gradient
                # spikes from the quantum layers (e.g. near AngleEmbedding
                # wrap points, or from the entangling CNOTs).
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()

                # ============================================
                # GENERATOR
                # ============================================
                self.g_optimizer.zero_grad()

                # Best-of-k reconstruction loss instead of single-sample
                # MSE, so the generator isn't forced to collapse to a
                # near-deterministic mapping and ignore its noise input
                # (fix #2 in the class docstring).
                k = self.variety_k
                context_k = context.unsqueeze(1).expand(-1, k, -1).reshape(batch_size * k, -1)
                noise_k = torch.randn(batch_size * k, self.latent_size, device=self.device)
                candidates = self.generator(context_k, noise_k).view(batch_size, k, self.horizon)
                candidates_flat = candidates.view(batch_size * k, self.horizon)

                fake_logits = self.discriminator(context_k, candidates_flat)
                generator_labels = torch.ones_like(fake_logits)
                adversarial_loss = self.criterion(fake_logits, generator_labels)

                distances_to_real = torch.norm(candidates - real_future.unsqueeze(1), dim=2)
                reconstruction_loss = distances_to_real.min(dim=1).values.mean()

                g_loss = adversarial_loss + self.recon_loss_weight * reconstruction_loss
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.g_optimizer.step()

                g_total += g_loss.item()
                d_total += d_loss.item()
                recon_total += reconstruction_loss.item()

            g_avg, d_avg, recon_avg = g_total / len(loader), d_total / len(loader), recon_total / len(loader)
            history["generator_loss"].append(g_avg)
            history["discriminator_loss"].append(d_avg)
            history["reconstruction_loss"].append(recon_avg)

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"[{self.type}] Epoch {epoch+1}/{self.epochs} G={g_avg:.4f} D={d_avg:.4f} Recon={recon_avg:.4f}")

        return history

    # ========================================================
    # PREDICT ONE 6-DAY SEQUENCE
    # ========================================================

    def predict(self, context, n_samples=1):
        """
        `n_samples`: the generator takes a random noise vector, so a
        single call returns one stochastic sample of the conditional
        output distribution -- not its expected value. For a point-
        forecast metric like MAPE/RMSE the best point estimate is the
        conditional mean E_z[G(context, z)], approximated here by
        averaging `n_samples` independent noise draws. `n_samples=1`
        (default) reproduces the original single-sample behavior;
        `n_samples=20` to `50` gives a materially less noisy point
        forecast at the cost of that many extra forward passes.
        """
        self.generator.eval()

        context = torch.tensor(context, dtype=torch.float32) if isinstance(context, np.ndarray) else context
        context = context.unsqueeze(0) if context.ndim == 1 else context
        context = context.to(self.device)
        batch_size = context.shape[0]

        context_repeated = context.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1)
        noise = torch.randn(batch_size * n_samples, self.latent_size, device=self.device)
        prediction = self.generator(context_repeated, noise)
        prediction = prediction.view(batch_size, n_samples, self.horizon).mean(dim=1)
        prediction = prediction.cpu().detach().numpy()

        # Undo the standardization applied in __init__ so the caller
        # always gets predictions back in the original data units.
        return prediction * self.data_std + self.data_mean

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest(self, n_samples=20):
        """
        `n_samples` (default 20, was implicitly 1). Same rationale as
        `predict()`'s `n_samples` -- a single noise draw per test point
        adds noise-induced variance straight into MAE/RMSE/MAPE that has
        nothing to do with model quality. Pass `n_samples=1` to reproduce
        the original single-sample behavior.
        """
        self.generator.eval()
        predictions, contexts, actuals = [], [], []

        for i in range(len(self.X_test)):
            context = self.X_test[i].unsqueeze(0).to(self.device)
            target = self.y_test[i].numpy()
            context_repeated = context.repeat(n_samples, 1)
            noise = torch.randn(n_samples, self.latent_size, device=self.device)
            prediction = self.generator(context_repeated, noise).mean(dim=0)
            predictions.append(prediction.cpu().detach().numpy())
            contexts.append(self.X_test[i].numpy())
            actuals.append(target)

        contexts, predictions, actuals = np.asarray(contexts), np.asarray(predictions), np.asarray(actuals)

        # X_test/y_test live in standardized space; undo that here so
        # metrics() computes MAE/RMSE/MAPE/R2 in the original data units.
        contexts = contexts * self.data_std + self.data_mean
        predictions = predictions * self.data_std + self.data_mean
        actuals = actuals * self.data_std + self.data_mean

        return contexts, predictions, actuals

    # ========================================================
    # METRICS
    # ========================================================

    @staticmethod
    def metrics(predictions, actuals):
        predictions, actuals = np.asarray(predictions), np.asarray(actuals)
        error = predictions - actuals

        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        denominator = np.where(np.abs(actuals) < 1e-8, 1e-8, np.abs(actuals))
        mape = np.mean(np.abs(error) / denominator) * 100
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / max(ss_tot, 1e-12))  # guarded against a degenerate (near-constant) actuals window

        result = {"overall": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}, "per_horizon": {}}

        for h in range(predictions.shape[1]):
            p, a = predictions[:, h], actuals[:, h]
            e = p - a
            h_mae = np.mean(np.abs(e))
            h_rmse = np.sqrt(np.mean(e ** 2))
            h_denominator = np.where(np.abs(a) < 1e-8, 1e-8, np.abs(a))
            h_mape = np.mean(np.abs(e) / h_denominator) * 100
            h_ss_res = np.sum(e ** 2)
            h_ss_tot = np.sum((a - np.mean(a)) ** 2)
            h_r2 = 1 - (h_ss_res / max(h_ss_tot, 1e-12))  # same guard, per horizon
            result["per_horizon"][f"Day +{h+1}"] = {"MAE": h_mae, "RMSE": h_rmse, "MAPE": h_mape, "R2": h_r2}

        return result

    # ========================================================
    # DIAGNOSTICS: IS 66% MAPE ACTUALLY BAD FOR THIS DATA?
    # ========================================================

    @staticmethod
    def naive_persistence_baseline(contexts, horizon):
        """
        The simplest possible forecaster -- repeat the last observed
        value for every step of the horizon. Any trained model that
        cannot beat this on MAE/RMSE/MAPE is not learning anything useful
        yet.
        """
        contexts = np.asarray(contexts)
        return np.repeat(contexts[:, -1:], horizon, axis=1)

    @staticmethod
    def smape(predictions, actuals):
        """
        Symmetric MAPE. Plain MAPE divides by |actual| alone, so it
        explodes whenever the true value is small in magnitude even if
        the absolute error is tiny. SMAPE divides by the average
        magnitude of prediction and actual instead, bounding it to
        [0, 200%].
        """
        predictions, actuals = np.asarray(predictions), np.asarray(actuals)
        denominator = np.where((np.abs(predictions) + np.abs(actuals)) / 2.0 < 1e-8, 1e-8, (np.abs(predictions) + np.abs(actuals)) / 2.0)
        return np.mean(np.abs(predictions - actuals) / denominator) * 100


# ============================================================
# MULTI-SEQUENCE GENERATOR / DISCRIMINATOR
# ============================================================

class ClassicalMultiSequenceGenerator(nn.Module):
    """
    `trend_feature_size` (default 1): the residual GAN previously saw
    ONLY the detrended (residual) context -- the local trend itself
    (equivalently, its slope) was computed, subtracted out, and then
    discarded before the network ever saw it. That leaves the generator
    with no way to know whether it is being asked to correct a flat
    window or a steeply-moving one, even though the two call for very
    different residual behavior (see the class docstring on
    `MultiSequenceGAN` for the trend-aware rationale). `trend_feature`
    (the normalized local slope, shape (batch, trend_feature_size)) is
    now concatenated onto the input alongside the noise vector.
    """

    def __init__(self, context_size=6, horizon=6, latent_size=8, trend_feature_size=1, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(context_size + latent_size + trend_feature_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, context, noise, trend_feature):
        x = torch.cat([context, noise, trend_feature], dim=1)
        return self.network(x)


class QuantumMultiSequenceGenerator(nn.Module):
    """See `ClassicalMultiSequenceGenerator` docstring for `trend_feature_size` rationale."""

    def __init__(self, context_size=6, horizon=6, latent_size=8, trend_feature_size=1, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnode = _BatchedQuantumRunner(quantum_circuit(n_qubits=n_qubits, n_layers=quantum_layers))
        self.input_projection = TemporalProjection(n_qubits=n_qubits, cond_size=latent_size + trend_feature_size, hidden_size=hidden_size)
        self.q_weights = nn.Parameter(_init_strongly_entangling_weights(quantum_layers, n_qubits))
        self.input_scale = nn.Parameter(torch.ones(n_qubits))
        self.decoder = nn.Sequential(nn.Linear(n_qubits, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, context, noise, trend_feature):
        cond = torch.cat([noise, trend_feature], dim=1)
        angles = self.input_projection(context, cond)
        angles = torch.tanh(angles * self.input_scale) * torch.pi
        quantum_outputs = self.qnode(angles, self.q_weights)
        return self.decoder(quantum_outputs)


class ClassicalMultiSequenceDiscriminator(nn.Module):
    """
    `trend_feature_size` (default 1): gives the discriminator the same
    normalized local-slope signal as the generator (see
    `ClassicalMultiSequenceGenerator`), so it can judge realism
    conditioned on how much trend movement was actually happening --
    e.g. a near-zero residual future is very plausible after a flat
    window but not after a steeply trending one.
    """

    def __init__(self, context_size=6, horizon=6, trend_feature_size=1, hidden_size=64):
        super().__init__()
        self.horizon = horizon
        self.shared = nn.Sequential(nn.Linear(context_size + horizon + trend_feature_size, hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2))
        self.horizon_head = nn.Linear(hidden_size, horizon)
        self.sequence_head = nn.Linear(hidden_size, 1)

    def forward(self, context, future, trend_feature):
        x = torch.cat([context, future, trend_feature], dim=1)
        features = self.shared(x)
        return torch.cat([self.horizon_head(features), self.sequence_head(features)], dim=1)


class QuantumMultiSequenceDiscriminator(nn.Module):
    """See `ClassicalMultiSequenceDiscriminator` docstring for `trend_feature_size` rationale."""

    def __init__(self, context_size=6, horizon=6, trend_feature_size=1, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.horizon = horizon
        self.n_qubits = n_qubits
        self.qnode = _BatchedQuantumRunner(quantum_circuit(n_qubits=n_qubits, n_layers=quantum_layers))
        self.input_projection = TemporalProjection(n_qubits=n_qubits, cond_size=trend_feature_size, hidden_size=hidden_size)
        self.q_weights = nn.Parameter(_init_strongly_entangling_weights(quantum_layers, n_qubits))
        self.input_scale = nn.Parameter(torch.ones(n_qubits))
        self.shared = nn.Sequential(nn.Linear(n_qubits, hidden_size), nn.ReLU())
        self.horizon_head = nn.Linear(hidden_size, horizon)
        self.sequence_head = nn.Linear(hidden_size, 1)

    def forward(self, context, future, trend_feature):
        seq = torch.cat([context, future], dim=1)
        angles = self.input_projection(seq, trend_feature)
        angles = torch.tanh(angles * self.input_scale) * torch.pi
        quantum_outputs = self.qnode(angles, self.q_weights)
        features = self.shared(quantum_outputs)
        return torch.cat([self.horizon_head(features), self.sequence_head(features)], dim=1)



# ============================================================
# NEURAL TRAJECTORY SELECTOR
# ============================================================

class NeuralSelectionModel(nn.Module):
    """
    Temporal- and trend-dynamics-aware candidate trajectory selector.

    The selector does NOT receive the observed future.

    For every generated candidate it receives:
        1. historical context,
        2. candidate future trajectory,
        3. horizon-wise discriminator realism,
        4. sequence-level discriminator realism.

    It explicitly models temporal dynamics with GRUs and explicit trend
    features including:
        - context least-squares slope,
        - candidate least-squares slope,
        - slope change / acceleration,
        - context-to-future transition,
        - cumulative context-to-future movement,
        - mean future first difference,
        - future first-difference volatility,
        - deviation from the extrapolated historical trend.

    The selector is trained jointly with the GAN.
    """

    def __init__(
        self,
        context_size=6,
        horizon=6,
        hidden_size=64,
        fusion_size=128,
        dropout=0.10
    ):
        super().__init__()

        self.context_size = context_size
        self.horizon = horizon
        self.hidden_size = hidden_size

        # Each timestep contains:
        # value, first difference, second difference, |first difference|
        temporal_input_size = 4

        self.context_encoder = nn.GRU(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.future_encoder = nn.GRU(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Explicit dynamics:
        # context slope
        # future slope
        # slope change
        # transition
        # cumulative movement
        # mean future delta
        # future delta std
        # deviation from extrapolated context trend
        trend_feature_count = 8

        self.trend_encoder = nn.Sequential(
            nn.Linear(trend_feature_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # horizon realism + sequence realism
        self.realism_encoder = nn.Sequential(
            nn.Linear(horizon + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        fusion_input = (
            hidden_size +   # context temporal embedding
            hidden_size +   # future temporal embedding
            32 +             # explicit trend/dynamics
            32               # discriminator realism
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_size, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(),

            nn.Linear(fusion_size // 2, 1)
        )

    @staticmethod
    def _temporal_features(sequence):
        """
        sequence: (batch, T)

        Returns:
            (batch, T, 4)
        """
        x = sequence.unsqueeze(-1)

        delta = torch.zeros_like(x)
        if sequence.shape[1] > 1:
            delta[:, 1:] = x[:, 1:] - x[:, :-1]

        acceleration = torch.zeros_like(x)
        if sequence.shape[1] > 2:
            acceleration[:, 2:] = delta[:, 2:] - delta[:, 1:-1]

        return torch.cat(
            [
                x,
                delta,
                acceleration,
                torch.abs(delta)
            ],
            dim=-1
        )

    @staticmethod
    def _least_squares_slope(sequence):
        """
        Least-squares slope along the time axis.
        """
        length = sequence.shape[1]
        t = torch.arange(
            length,
            device=sequence.device,
            dtype=sequence.dtype
        )

        t = t - t.mean()
        denominator = torch.sum(t * t).clamp_min(1e-8)

        centered = sequence - sequence.mean(dim=1, keepdim=True)

        return torch.sum(
            centered * t.unsqueeze(0),
            dim=1
        ) / denominator

    def _trend_features(self, context, sequence):
        """
        Explicit temporal/trend dynamics.

        context:
            (batch, context_size)

        sequence:
            (batch, horizon)
        """

        context_slope = self._least_squares_slope(context)
        future_slope = self._least_squares_slope(sequence)

        # Change in slope = trajectory acceleration at the trend level.
        slope_change = future_slope - context_slope

        # Immediate transition from history to forecast.
        transition = sequence[:, 0] - context[:, -1]

        # Total movement from the last observed value.
        cumulative_movement = sequence[:, -1] - context[:, -1]

        future_delta = sequence[:, 1:] - sequence[:, :-1]

        if future_delta.shape[1] > 0:
            mean_future_delta = future_delta.mean(dim=1)
            future_delta_std = future_delta.std(
                dim=1,
                unbiased=False
            )
        else:
            mean_future_delta = torch.zeros_like(future_slope)
            future_delta_std = torch.zeros_like(future_slope)

        # Extrapolate the historical least-squares line into the future.
        context_length = context.shape[1]
        horizon = sequence.shape[1]

        t_context = torch.arange(
            context_length,
            device=context.device,
            dtype=context.dtype
        )

        context_mean = context.mean(dim=1)
        t_mean = t_context.mean()

        intercept = (
            context_mean -
            context_slope * t_mean
        )

        t_future = torch.arange(
            context_length,
            context_length + horizon,
            device=context.device,
            dtype=context.dtype
        )

        extrapolated_trend = (
            intercept.unsqueeze(1) +
            context_slope.unsqueeze(1) * t_future.unsqueeze(0)
        )

        trend_deviation = torch.sqrt(
            torch.mean(
                (sequence - extrapolated_trend) ** 2,
                dim=1
            ).clamp_min(1e-12)
        )

        return torch.stack(
            [
                context_slope,
                future_slope,
                slope_change,
                transition,
                cumulative_movement,
                mean_future_delta,
                future_delta_std,
                trend_deviation
            ],
            dim=-1
        )

    def forward(
        self,
        context,
        sequence,
        horizon_realism_score,
        sequence_realism_score
    ):
        # Temporal representation of history.
        context_temporal = self._temporal_features(context)

        _, context_hidden = self.context_encoder(
            context_temporal
        )

        context_embedding = context_hidden[-1]

        # Temporal representation of candidate future.
        sequence_temporal = self._temporal_features(sequence)

        _, future_hidden = self.future_encoder(
            sequence_temporal
        )

        future_embedding = future_hidden[-1]

        # Explicit dynamics.
        trend_features = self._trend_features(
            context,
            sequence
        )

        trend_embedding = self.trend_encoder(
            trend_features
        )

        # Discriminator realism.
        realism = torch.cat(
            [
                horizon_realism_score,
                sequence_realism_score
            ],
            dim=-1
        )

        realism_embedding = self.realism_encoder(
            realism
        )

        # Final candidate representation.
        fused = torch.cat(
            [
                context_embedding,
                future_embedding,
                trend_embedding,
                realism_embedding
            ],
            dim=-1
        )

        return self.fusion(fused).squeeze(-1)


class MultiSequenceGAN:
    """Multi-sequence conditional GAN with lagged historical inputs.

    `lag` is the number of observations skipped between the final
    historical input and the forecast origin. The historical context itself
    remains consecutive, and the target horizon remains consecutive.

    Example with historical_lookup=6 and lag=2:
        context = [7, 8, 9, 10, 11, 12]
        skipped = [13, 14]
        target  = [15, 16, 17, 18, 19, 20]
    """

    def __init__(self, data, type="QC", historical_lookup=6, horizon=6, latent_size=8, n_qubits=4, quantum_layers=1, hidden_size=64, epochs=100, batch_size=32, learning_rate=1e-4, train_ratio=0.8, variety_k=5, variety_loss_weight=1.0, diversity_loss_weight=0.5, label_smoothing=0.9, mape_zero_threshold=1.0, trend_feature_size=1, seed=42, device=None, lag=1,
                 selector_hidden_size=64,
                 selector_fusion_size=128,
                 selector_learning_rate=1e-3,
                 selector_loss_weight=0.25,
                 selector_dropout=0.10):
        set_seed(seed)
        self.seed = seed
        self.data = np.asarray(data, dtype=np.float32).reshape(-1)
        self.type = type.upper()
        if self.type not in ["CC", "QC", "CQ", "QQ"]:
            raise ValueError("type must be one of: CC, QC, CQ, QQ")

        self.historical_lookup = historical_lookup
        self.horizon = horizon
        self.lag = int(lag)
        if self.lag < 1:
            raise ValueError("lag must be an integer >= 1")
        if self.historical_lookup < 1:
            raise ValueError("historical_lookup must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.latent_size = latent_size
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.variety_k = variety_k
        self.variety_loss_weight = variety_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.label_smoothing = label_smoothing
        self.mape_zero_threshold = mape_zero_threshold
        self.trend_feature_size = trend_feature_size
        self.device = device or DEVICE

        # ============================================================
        # JOINT NEURAL TRAJECTORY SELECTOR
        # ============================================================
        self.selector_hidden_size = selector_hidden_size
        self.selector_fusion_size = selector_fusion_size
        self.selector_learning_rate = selector_learning_rate
        self.selector_loss_weight = selector_loss_weight
        self.selector_dropout = selector_dropout

        self.neural_selector = NeuralSelectionModel(
            context_size=historical_lookup,
            horizon=horizon,
            hidden_size=selector_hidden_size,
            fusion_size=selector_fusion_size,
            dropout=selector_dropout
        ).to(self.device)

        self.selector_optimizer = torch.optim.Adam(
            self.neural_selector.parameters(),
            lr=selector_learning_rate
        )

        self.neural_selector_fitted = False

        # ============================================================
        # BUILD CAUSAL TREND / RESIDUAL DATASET
        # ============================================================
        self.raw_contexts, self.raw_targets, self.contexts_raw, self.targets_raw, self.trend_contexts, self.trend_futures, self.slopes = self._create_sequences()

        # Sequence-level chronological split. No random splitting is used.
        split = int(len(self.contexts_raw) * train_ratio)
        self.sequence_split = split

        # ------------------------------------------------------------
        # Residual normalization
        #
        # IMPORTANT: statistics are fitted ONLY on training residuals.
        # Both historical and future residuals from training windows are
        # used to estimate the residual scale.
        # ------------------------------------------------------------
        train_context_residuals = self.contexts_raw[:split]
        train_future_residuals = self.targets_raw[:split]
        train_residual_values = np.concatenate([train_context_residuals.reshape(-1), train_future_residuals.reshape(-1)])
        self.residual_mean = float(train_residual_values.mean())
        self.residual_std = float(train_residual_values.std() + 1e-8)

        # ------------------------------------------------------------
        # Trend (slope) normalization -- fitted on TRAINING slopes only,
        # same leakage-free pattern as the residual stats above. This
        # normalized slope is the `trend_feature` fed into the generator
        # and discriminator (see the ENHANCEMENT note in the class
        # docstring).
        # ------------------------------------------------------------
        train_slopes = self.slopes[:split]
        self.slope_mean = float(train_slopes.mean())
        self.slope_std = float(train_slopes.std() + 1e-8)

        # Normalize residual sequences.
        contexts_norm = (self.contexts_raw - self.residual_mean) / self.residual_std
        targets_norm = (self.targets_raw - self.residual_mean) / self.residual_std
        self.contexts = torch.tensor(contexts_norm, dtype=torch.float32)
        self.targets = torch.tensor(targets_norm, dtype=torch.float32)

        # Train/test split.
        self.X_train = self.contexts
        self.y_train = self.targets
        self.X_test = self.contexts[split:]
        self.y_test = self.targets[split:]

        # Raw temperature contexts/targets.
        self.raw_contexts_train = self.raw_contexts
        self.raw_targets_train = self.raw_targets
        self.raw_contexts_test = self.raw_contexts[split:]
        self.raw_targets_test = self.raw_targets[split:]

        # Residual contexts/targets.
        self.contexts_raw_train = self.contexts_raw
        self.targets_raw_train = self.targets_raw
        self.contexts_raw_test = self.contexts_raw[split:]
        self.targets_raw_test = self.targets_raw[split:]

        self.trend_contexts_train = self.trend_contexts
        self.trend_futures_train = self.trend_futures
        self.trend_contexts_test = self.trend_contexts[split:]
        self.trend_futures_test = self.trend_futures[split:]

        self.slopes_train = self.slopes
        self.slopes_test = self.slopes[split:]

        # Normalized slope tensor for training, aligned index-for-index
        # with X_train/y_train so it stays in sync when DataLoader
        # shuffles (see `_normalize_slope` / the trend-aware training
        # loop in `train()`).
        self.slopes_train_norm = torch.tensor(
            self._normalize_slope(self.slopes_train),
            dtype=torch.float32
        ).unsqueeze(1)

        # Raw-domain tensors are used ONLY by the selector to learn
        # physically meaningful temporal/trend dynamics. The GAN itself
        # continues to operate in normalized residual space.
        self.raw_contexts_train_tensor = torch.tensor(
            self.raw_contexts_train,
            dtype=torch.float32
        )
        self.raw_targets_train_tensor = torch.tensor(
            self.raw_targets_train,
            dtype=torch.float32
        )
        self.trend_futures_train_tensor = torch.tensor(
            self.trend_futures_train,
            dtype=torch.float32
        )

        # ============================================================
        # GENERATOR
        # ============================================================
        if self.type in ["QC", "QQ"]:
            self.generator = QuantumMultiSequenceGenerator(context_size=historical_lookup, horizon=horizon, latent_size=latent_size, trend_feature_size=trend_feature_size, n_qubits=n_qubits, quantum_layers=quantum_layers, hidden_size=hidden_size)
        else:
            self.generator = ClassicalMultiSequenceGenerator(context_size=historical_lookup, horizon=horizon, latent_size=latent_size, trend_feature_size=trend_feature_size, hidden_size=hidden_size)

        # ============================================================
        # DISCRIMINATOR
        # ============================================================
        if self.type in ["CQ", "QQ"]:
            self.discriminator = QuantumMultiSequenceDiscriminator(context_size=historical_lookup, horizon=horizon, trend_feature_size=trend_feature_size, n_qubits=n_qubits, quantum_layers=quantum_layers, hidden_size=hidden_size)
        else:
            self.discriminator = ClassicalMultiSequenceDiscriminator(context_size=historical_lookup, horizon=horizon, trend_feature_size=trend_feature_size, hidden_size=hidden_size)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # ============================================================
        # OPTIMIZERS (TTUR: discriminator lr = 2 * generator lr)
        # ============================================================
        self.d_lr_multiplier = 2.0
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate * self.d_lr_multiplier, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    # ================================================================
    # CAUSAL LOCAL TREND
    # ================================================================

    def _fit_local_trend(self, context):
        """
        Fit a causal local linear trend using ONLY the consecutive historical
        context.

        `lag` is the number of observations skipped between the final
        historical input and the first forecast target. Therefore the
        historical coordinates are consecutive [0, ..., history-1], while
        the future coordinates begin at history + lag.

        Example (historical_lookup=6, lag=2):
            context times = [0, 1, 2, 3, 4, 5]
            skipped times = [6, 7]
            future times  = [8, 9, 10, 11, 12, 13]

        No future target observations are used.
        """
        context = np.asarray(context, dtype=np.float32).reshape(-1)
        if len(context) != self.historical_lookup:
            raise ValueError(f"Expected context length {self.historical_lookup}, got {len(context)}")

        x = np.arange(self.historical_lookup, dtype=np.float32)
        slope, intercept = np.polyfit(x, context, 1)
        trend_context = slope * x + intercept

        future_start = self.historical_lookup + self.lag
        future_x = np.arange(
            future_start,
            future_start + self.horizon,
            dtype=np.float32
        )
        trend_future = slope * future_x + intercept

        return trend_context.astype(np.float32), trend_future.astype(np.float32), float(slope)

    # ================================================================
    # DATASET CREATION
    # ================================================================

    def _create_sequences(self):
        """
        Construct causal trend/residual forecasting windows.

        The historical input is ALWAYS consecutive. `lag` specifies the
        number of observations skipped between the final historical input and
        the forecast origin.

        Example with historical_lookup=6, horizon=6, lag=2:
            raw_context = [7, 8, 9, 10, 11, 12]
            skipped     = [13, 14]
            raw_future  = [15, 16, 17, 18, 19, 20]

        If the historical input is displayed backwards, the same context is
        [12, 11, 10, 9, 8, 7], but the model receives chronological order.

        A local trend is fitted ONLY to raw_context. Residuals:
            context_residual = raw_context - trend_context
            future_residual  = raw_future - extrapolated_trend

        Returns both raw temperatures and residuals. Residual normalization
        happens afterward using training-only residual statistics.
        """
        raw_contexts, raw_futures = [], []
        context_residuals, future_residuals = [], []
        trend_contexts, trend_futures, slopes = [], [], []

        n_windows = len(self.data) - self.historical_lookup - self.lag - self.horizon + 1

        for i in range(n_windows):
            context_start = i
            context_end = context_start + self.historical_lookup
            future_start = context_end + self.lag
            future_end = future_start + self.horizon

            raw_context = self.data[context_start:context_end]
            raw_future = self.data[future_start:future_end]
            trend_context, trend_future, slope = self._fit_local_trend(raw_context)

            raw_contexts.append(raw_context)
            raw_futures.append(raw_future)
            context_residuals.append(raw_context - trend_context)
            future_residuals.append(raw_future - trend_future)
            trend_contexts.append(trend_context)
            trend_futures.append(trend_future)
            slopes.append(slope)

        return (
            np.asarray(raw_contexts, dtype=np.float32),
            np.asarray(raw_futures, dtype=np.float32),
            np.asarray(context_residuals, dtype=np.float32),
            np.asarray(future_residuals, dtype=np.float32),
            np.asarray(trend_contexts, dtype=np.float32),
            np.asarray(trend_futures, dtype=np.float32),
            np.asarray(slopes, dtype=np.float32),
        )

    # ================================================================
    # RESIDUAL NORMALIZATION HELPERS
    # ================================================================

    def _normalize_residual(self, x):
        return (np.asarray(x, dtype=np.float32) - self.residual_mean) / self.residual_std

    def _inverse_residual(self, x):
        return np.asarray(x, dtype=np.float32) * self.residual_std + self.residual_mean

    def _normalize_slope(self, slope):
        """
        Normalize a local trend slope (or array of slopes) using
        TRAINING-slope statistics (`self.slope_mean`/`self.slope_std`,
        fit in `__init__`). This is the `trend_feature` fed into the
        generator/discriminator -- see the ENHANCEMENT note in the
        `MultiSequenceGAN` class docstring.
        """
        return (np.asarray(slope, dtype=np.float32) - self.slope_mean) / self.slope_std

    # ================================================================
    # VARIETY + DIVERSITY LOSS
    # ================================================================

    def _variety_and_diversity_loss(self, context, real_future, trend_feature):
        """
        Sample `variety_k` residual futures per context.

        variety_loss: best-of-k L2 distance to the real residual future.
        diversity_loss: negative pairwise distance between candidate
        residual futures. Both operate entirely in normalized residual
        space. `trend_feature` (normalized local slope, shape (batch,
        trend_feature_size)) is passed to every generator call so
        candidates are conditioned on how strong the local trend was --
        see the ENHANCEMENT note in the class docstring.
        """
        k = self.variety_k
        batch_size = context.shape[0]

        context_expanded = context.unsqueeze(1).expand(-1, k, -1).reshape(batch_size * k, -1)
        trend_feature_expanded = trend_feature.unsqueeze(1).expand(-1, k, -1).reshape(batch_size * k, -1)
        noise_k = torch.randn(batch_size * k, self.latent_size, device=self.device)
        candidates = self.generator(context_expanded, noise_k, trend_feature_expanded).view(batch_size, k, self.horizon)

        distances_to_real = torch.norm(candidates - real_future.unsqueeze(1), dim=2)
        variety_loss = distances_to_real.min(dim=1).values.mean()

        if k > 1:
            diff = candidates.unsqueeze(2) - candidates.unsqueeze(1)
            pairwise_dist = torch.norm(diff, dim=-1)
            mean_pairwise_dist = pairwise_dist.sum(dim=(1, 2)) / (k * (k - 1))
            diversity_loss = -mean_pairwise_dist.mean()
        else:
            diversity_loss = torch.zeros((), device=self.device)

        return variety_loss, diversity_loss, candidates, context_expanded, trend_feature_expanded

    # ================================================================
    # JOINT NEURAL SELECTOR LOSS
    # ================================================================

    def _neural_selector_loss(
        self,
        context,
        real_future,
        trend_feature,
        raw_context,
        raw_future,
        raw_future_trend,
        residual_candidates=None
    ):
        """
        Joint selector loss.

        GAN variables are residual/normalized. The selector, however,
        operates in ORIGINAL temperature space so that its temporal and
        trend features have direct physical meaning.

        raw_context:
            (batch, historical_lookup)

        raw_future:
            (batch, horizon)
            Actual future in original temperature units. It is used only
            to construct the supervised selector target.

        raw_future_trend:
            (batch, horizon)
            Causal extrapolated trend in original temperature units.

        The observed target is used only to create the training ranking
        target. It is never passed to the selector.
        """

        k = self.variety_k
        batch_size = context.shape[0]

        context_expanded = (
            context.unsqueeze(1)
            .expand(-1, k, -1)
            .reshape(batch_size * k, -1)
        )

        trend_expanded = (
            trend_feature.unsqueeze(1)
            .expand(-1, k, -1)
            .reshape(batch_size * k, -1)
        )

        if residual_candidates is None:
            noise = torch.randn(
                batch_size * k,
                self.latent_size,
                device=self.device,
                dtype=context.dtype
            )

            residual_candidates = self.generator(
                context_expanded,
                noise,
                trend_expanded
            ).view(
                batch_size,
                k,
                self.horizon
            )

        # Convert residual candidates back to original temperature units.
        candidate_temperature = (
            residual_candidates *
            self.residual_std +
            self.residual_mean
        )

        candidate_temperature = (
            candidate_temperature +
            raw_future_trend.unsqueeze(1)
        )

        # ------------------------------------------------------------
        # Discriminator realism.
        # ------------------------------------------------------------

        disc_output = self.discriminator(
            context_expanded,
            residual_candidates.reshape(
                batch_size * k,
                self.horizon
            ),
            trend_expanded
        )

        disc_output = disc_output.view(
            batch_size,
            k,
            self.horizon + 1
        )

        horizon_scores = torch.sigmoid(
            disc_output[:, :, :self.horizon]
        ).detach()

        sequence_scores = torch.sigmoid(
            disc_output[:, :, self.horizon]
        ).detach()

        # ------------------------------------------------------------
        # Selector input in ORIGINAL temperature space.
        # ------------------------------------------------------------

        selector_context = (
            raw_context.unsqueeze(1)
            .expand(-1, k, -1)
            .reshape(batch_size * k, -1)
        )

        selector_sequence = candidate_temperature.reshape(
            batch_size * k,
            self.horizon
        )

        selector_horizon_scores = horizon_scores.reshape(
            batch_size * k,
            self.horizon
        )

        selector_sequence_scores = sequence_scores.reshape(
            batch_size * k,
            1
        )

        selector_logits = self.neural_selector(
            selector_context,
            selector_sequence,
            selector_horizon_scores,
            selector_sequence_scores
        ).view(batch_size, k)

        # ------------------------------------------------------------
        # Training target.
        #
        # IMPORTANT:
        # raw_future is never passed into neural_selector().
        # It is used only here to determine the best candidate.
        # ------------------------------------------------------------

        with torch.no_grad():
            candidate_mae = torch.mean(
                torch.abs(
                    candidate_temperature -
                    raw_future.unsqueeze(1)
                ),
                dim=-1
            )

            target_index = torch.argmin(
                candidate_mae,
                dim=1
            )

        selector_loss = nn.functional.cross_entropy(
            selector_logits,
            target_index
        )

        return selector_loss


    # ================================================================
    # TRAIN
    # ================================================================

    def train(self):
        """
        Joint GAN + neural selector training.

        Every training iteration performs:

            1. discriminator update
            2. generator adversarial/variety/diversity update
            3. neural selector update

        The selector loss is also included in the generator-side objective
        with `selector_loss_weight`, so the generator and selector are
        trained together rather than in two separate phases.
        """

        dataset = TensorDataset(
            self.X_train,
            self.y_train,
            self.slopes_train_norm,
            self.raw_contexts_train_tensor,
            self.raw_targets_train_tensor,
            self.trend_futures_train_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        history = {
            "generator_loss": [],
            "discriminator_loss": [],
            "variety_loss": [],
            "diversity_loss": [],
            "selector_loss": []
        }

        print(
            f"\nTraining Model C MultiSequenceGAN type={self.type}"
        )
        print(
            f"Generator: "
            f"{'Quantum' if self.type in ['QC', 'QQ'] else 'Classical'}"
        )
        print(
            f"Discriminator: "
            f"{'Quantum' if self.type in ['CQ', 'QQ'] else 'Classical'}"
        )
        print("Trend: causal local linear (trend-aware residual GAN)")
        print("Selector: jointly trained neural trajectory selector")

        for epoch in range(self.epochs):

            g_total = 0.0
            d_total = 0.0
            variety_total = 0.0
            diversity_total = 0.0
            selector_total = 0.0

            for (
                context,
                real_future,
                trend_feature,
                raw_context,
                raw_future,
                trend_future
            ) in loader:

                context = context.to(self.device)
                real_future = real_future.to(self.device)
                trend_feature = trend_feature.to(self.device)

                raw_context = raw_context.to(self.device)
                raw_future = raw_future.to(self.device)
                trend_future = trend_future.to(self.device)

                batch_size = context.shape[0]

                # ====================================================
                # DISCRIMINATOR UPDATE
                # ====================================================

                self.d_optimizer.zero_grad()

                real_output = self.discriminator(
                    context,
                    real_future,
                    trend_feature
                )

                real_labels = (
                    torch.ones_like(real_output)
                    * self.label_smoothing
                )

                real_loss = self.criterion(
                    real_output,
                    real_labels
                )

                noise = torch.randn(
                    batch_size,
                    self.latent_size,
                    device=self.device
                )

                fake_future = self.generator(
                    context,
                    noise,
                    trend_feature
                )

                fake_output = self.discriminator(
                    context,
                    fake_future.detach(),
                    trend_feature
                )

                fake_labels = torch.zeros_like(fake_output)

                fake_loss = self.criterion(
                    fake_output,
                    fake_labels
                )

                d_loss = (
                    real_loss + fake_loss
                ) / 2.0

                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=1.0
                )

                self.d_optimizer.step()

                # ====================================================
                # GENERATOR + SELECTOR UPDATE
                # ====================================================

                self.g_optimizer.zero_grad()
                self.selector_optimizer.zero_grad()

                (
                    variety_loss,
                    diversity_loss,
                    candidates,
                    context_expanded,
                    trend_feature_expanded
                ) = self._variety_and_diversity_loss(
                    context,
                    real_future,
                    trend_feature
                )

                k = self.variety_k

                candidates_flat = candidates.view(
                    batch_size * k,
                    self.horizon
                )

                fake_output = self.discriminator(
                    context_expanded,
                    candidates_flat,
                    trend_feature_expanded
                )

                generator_labels = torch.ones_like(
                    fake_output
                )

                adversarial_loss = self.criterion(
                    fake_output,
                    generator_labels
                )

                # ----------------------------------------------------
                # Joint neural selector loss.
                # ----------------------------------------------------

                selector_loss = self._neural_selector_loss(
                    context=context,
                    real_future=real_future,
                    trend_feature=trend_feature,
                    raw_context=raw_context,
                    raw_future=raw_future,
                    raw_future_trend=trend_future,
                    residual_candidates=candidates
                )

                g_loss = (
                    adversarial_loss
                    + self.variety_loss_weight * variety_loss
                    + self.diversity_loss_weight * diversity_loss
                    + self.selector_loss_weight * selector_loss
                )

                g_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    max_norm=1.0
                )

                torch.nn.utils.clip_grad_norm_(
                    self.neural_selector.parameters(),
                    max_norm=1.0
                )

                self.g_optimizer.step()
                self.selector_optimizer.step()

                g_total += g_loss.item()
                d_total += d_loss.item()
                variety_total += variety_loss.item()
                diversity_total += diversity_loss.item()
                selector_total += selector_loss.item()

            g_avg = g_total / max(len(loader), 1)
            d_avg = d_total / max(len(loader), 1)
            variety_avg = variety_total / max(len(loader), 1)
            diversity_avg = diversity_total / max(len(loader), 1)
            selector_avg = selector_total / max(len(loader), 1)

            history["generator_loss"].append(g_avg)
            history["discriminator_loss"].append(d_avg)
            history["variety_loss"].append(variety_avg)
            history["diversity_loss"].append(diversity_avg)
            history["selector_loss"].append(selector_avg)

            if (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or epoch == self.epochs - 1
            ):
                print(
                    f"[{self.type}] "
                    f"Epoch {epoch + 1}/{self.epochs} "
                    f"G={g_avg:.4f} "
                    f"D={d_avg:.4f} "
                    f"Variety={variety_avg:.4f} "
                    f"Diversity={diversity_avg:.4f} "
                    f"Selector={selector_avg:.4f}"
                )

        self.neural_selector_fitted = True

        return history

    # ================================================================
    # GENERATE N TEMPERATURE SEQUENCES
    # ================================================================

    def generate_sequences(self, context, n_sequences=100):
        """
        Generate N temperature trajectories.

        Pipeline: raw context -> causal local trend -> historical
        residual -> normalize -> GAN (conditioned on the normalized
        local slope, `trend_feature` -- see the ENHANCEMENT note in the
        class docstring) -> future residual -> inverse normalize ->
        + local future trend -> temperature ensemble.

        Returns:
            futures: (horizon, n_sequences), original temperature units
            horizon_realism_score: (horizon, n_sequences)
            sequence_realism_score: (n_sequences,)
            trend_future: (horizon,)
            residual_futures: (horizon, n_sequences), original residual units
        """
        self.generator.eval()
        self.discriminator.eval()

        context = np.asarray(context, dtype=np.float32).reshape(-1)
        if len(context) != self.historical_lookup:
            raise ValueError(f"Expected context length {self.historical_lookup}, got {len(context)}")

        # 1. Fit causal local trend.
        trend_context, trend_future, slope = self._fit_local_trend(context)

        # 2. Historical residual.
        context_residual = context - trend_context

        # 3. Normalize residual and slope.
        context_residual_norm = (context_residual - self.residual_mean) / self.residual_std
        context_tensor = torch.tensor(context_residual_norm, dtype=torch.float32, device=self.device)
        context_batch = context_tensor.unsqueeze(0).repeat(n_sequences, 1)

        slope_norm = self._normalize_slope(slope)
        trend_feature = torch.tensor([[slope_norm]], dtype=torch.float32, device=self.device).repeat(n_sequences, 1)

        # 4. Latent variables.
        noise = torch.randn(n_sequences, self.latent_size, device=self.device)

        # 5. Generate future residuals.
        with torch.no_grad():
            residual_future_norm = self.generator(context_batch, noise, trend_feature)
            discriminator_output = self.discriminator(context_batch, residual_future_norm, trend_feature)

        # 6. Realism scores.
        horizon_logits = discriminator_output[:, :self.horizon]
        sequence_logits = discriminator_output[:, self.horizon]
        horizon_realism_score = torch.sigmoid(horizon_logits).cpu().numpy().T
        sequence_realism_score = torch.sigmoid(sequence_logits).cpu().numpy()

        # 7. Inverse residual normalization.
        residual_futures = residual_future_norm.cpu().numpy() * self.residual_std + self.residual_mean  # shape: (n_sequences, horizon)

        # 8. Reconstruct temperature.
        futures = residual_futures + trend_future[None, :]

        # 9. Return in standard ensemble format.
        return futures.T.astype(np.float32), horizon_realism_score.astype(np.float32), sequence_realism_score.astype(np.float32), trend_future.astype(np.float32), residual_futures.T.astype(np.float32)

    # ================================================================
    # SELECT BEST TRAJECTORY
    # ================================================================

    @staticmethod
    def select_best_trajectory(horizon_realism_score, sequence_realism_score, alpha=0.5, horizon_agg="min"):
        if horizon_agg == "min":
            horizon_agg_score = horizon_realism_score.min(axis=0)
        elif horizon_agg == "mean":
            horizon_agg_score = horizon_realism_score.mean(axis=0)
        elif horizon_agg == "geo_mean":
            horizon_agg_score = np.exp(np.mean(np.log(np.clip(horizon_realism_score, 1e-8, 1.0)), axis=0))
        else:
            raise ValueError("horizon_agg must be one of: 'min', 'mean', 'geo_mean'")

        composite_score = alpha * sequence_realism_score + (1.0 - alpha) * horizon_agg_score
        best_index = int(np.argmax(composite_score))
        return best_index, composite_score

    def neural_selection(self, context, sequences, horizon_realism_score, sequence_realism_score):
        if not self.neural_selector_fitted: raise RuntimeError("Neural selector has not been trained. Run model.train() first.")
        context = np.asarray(context, dtype=np.float32).reshape(-1)
        sequences = np.asarray(sequences, dtype=np.float32)
        horizon_realism_score = np.asarray(horizon_realism_score, dtype=np.float32)
        sequence_realism_score = np.asarray(sequence_realism_score, dtype=np.float32).reshape(-1)
        if sequences.ndim != 2:
            raise ValueError(
                f"Expected sequences with shape "
                f"({self.horizon}, N), got {sequences.shape}"
            )

        if sequences.shape[0] != self.horizon:
            raise ValueError(
                f"Expected sequences first dimension to be "
                f"{self.horizon}, got {sequences.shape[0]}"
            )

        if horizon_realism_score.shape != sequences.shape:
            raise ValueError(
                "horizon_realism_score must have the same shape as sequences"
            )

        if sequence_realism_score.shape[0] != sequences.shape[1]:
            raise ValueError(
                "sequence_realism_score length must equal number of candidates"
            )

        n_sequences = sequences.shape[1]

        contexts = np.repeat(context.reshape(1, -1), n_sequences, axis=0)
        candidate_sequences = sequences.T
        candidate_horizon_scores = horizon_realism_score.T
        candidate_sequence_scores = (sequence_realism_score.reshape(-1, 1))

        contexts = torch.tensor(contexts, dtype=torch.float32, device=self.device)
        candidate_sequences = torch.tensor(candidate_sequences, dtype=torch.float32, device=self.device)
        candidate_horizon_scores = torch.tensor(candidate_horizon_scores, dtype=torch.float32, device=self.device)
        candidate_sequence_scores = torch.tensor(candidate_sequence_scores, dtype=torch.float32, device=self.device)

        self.neural_selector.eval()

        with torch.no_grad():
            logits = self.neural_selector(contexts, candidate_sequences, candidate_horizon_scores, candidate_sequence_scores)
            neural_score = torch.sigmoid(logits).cpu().numpy()
        best_index = int(np.argmax(neural_score))
        return best_index, neural_score

    def predict(self, context, n_sequences=100, selection="composite", alpha=0.75, horizon_agg="mean"):
        sequences, horizon_realism_score, sequence_realism_score, trend_future, residual_futures = self.generate_sequences(context, n_sequences)
        if selection == "composite":
            best_index, selection_score = self.select_best_trajectory(horizon_realism_score, sequence_realism_score, alpha=alpha, horizon_agg=horizon_agg)
        elif selection == "sequence_only":
            best_index = int(np.argmax(sequence_realism_score))
            selection_score = sequence_realism_score
        elif selection == "neural_selection":
            best_index, selection_score = self.neural_selection(context, sequences, horizon_realism_score, sequence_realism_score)
        else:
            raise ValueError("selection must be one of: 'composite', 'sequence_only', 'neural_selection'")

        return {
            "all_sequences": sequences,
            "residual_sequences": residual_futures,
            "trend": trend_future,
            "horizon_realism_score": horizon_realism_score,
            "sequence_realism_score": sequence_realism_score,
            "selection_score": selection_score,
            "composite_score": selection_score,
            "best_index": best_index,
            "best_sequence": sequences[:, best_index],
            "best_residual": residual_futures[:, best_index],
            "best_horizon_realism_score": horizon_realism_score[:, best_index],
            "best_sequence_realism_score": sequence_realism_score[best_index],
        }

    # ================================================================
    # BACKTEST
    # ================================================================

    def backtest(self, n_sequences=100, return_all_sequences=True, selection="composite", alpha=0.5, horizon_agg="min"):
        """
        Chronological backtest. Everything returned here is in ORIGINAL
        temperature units.

        `all_sequences`: (n_test, horizon, n_sequences)
        `predictions`: selected best trajectory (see `selection`,
            `alpha`, `horizon_agg` -- passed through to `predict()`)
        `trend`: causal local trend extrapolation
        `residual_sequences`: generated residual ensemble
        """
        all_contexts, all_predictions = [], []
        all_horizon_realism_scores, all_sequence_realism_scores = [], []
        all_actuals, all_trends, all_residuals = [], [], []
        all_ensembles = [] if return_all_sequences else None

        for i in range(len(self.raw_contexts_test)):
            context_raw = self.raw_contexts_test[i]
            actual_raw = self.raw_targets_test[i]
            result = self.predict(context_raw, n_sequences, selection=selection, alpha=alpha, horizon_agg=horizon_agg)

            all_contexts.append(context_raw)
            all_predictions.append(result["best_sequence"])
            all_horizon_realism_scores.append(result["best_horizon_realism_score"])
            all_sequence_realism_scores.append(result["best_sequence_realism_score"])
            all_actuals.append(actual_raw)
            all_trends.append(result["trend"])
            all_residuals.append(result["residual_sequences"])
            if return_all_sequences:
                all_ensembles.append(result["all_sequences"])

        output = {
            "contexts": np.asarray(all_contexts, dtype=np.float32),
            "predictions": np.asarray(all_predictions, dtype=np.float32),
            "horizon_realism_score": np.asarray(all_horizon_realism_scores, dtype=np.float32),
            "sequence_realism_score": np.asarray(all_sequence_realism_scores, dtype=np.float32),
            "actuals": np.asarray(all_actuals, dtype=np.float32),
            "trend": np.asarray(all_trends, dtype=np.float32),
            "residual_sequences": np.asarray(all_residuals, dtype=np.float32),
        }

        if return_all_sequences:
            output["all_sequences"] = np.asarray(all_ensembles, dtype=np.float32)

        return output

    # ================================================================
    # POINT METRICS
    # ================================================================

    def metrics(self, predictions=None, actuals=None, mape_zero_threshold=None):
        if predictions is None or actuals is None:
            raise ValueError("metrics() requires predictions and actuals.")

        threshold = self.mape_zero_threshold if mape_zero_threshold is None else mape_zero_threshold
        predictions, actuals = np.asarray(predictions, dtype=np.float32), np.asarray(actuals, dtype=np.float32)
        error = predictions - actuals

        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))

        valid_mask = np.abs(actuals) >= threshold
        mape = float(np.mean(np.abs(error[valid_mask]) / np.abs(actuals[valid_mask])) * 100) if valid_mask.sum() > 0 else float("nan")
        mape_excluded_fraction = float(1.0 - valid_mask.mean())

        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

        result = {"overall": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MAPE_excluded_fraction": mape_excluded_fraction, "R2": r2}, "per_horizon": {}}

        for h in range(predictions.shape[1]):
            p, a = predictions[:, h], actuals[:, h]
            e = p - a
            h_mae = float(np.mean(np.abs(e)))
            h_rmse = float(np.sqrt(np.mean(e ** 2)))
            h_valid = np.abs(a) >= threshold
            h_mape = float(np.mean(np.abs(e[h_valid]) / np.abs(a[h_valid])) * 100) if h_valid.sum() > 0 else float("nan")
            h_excluded_fraction = float(1.0 - h_valid.mean())
            h_ss_res = np.sum(e ** 2)
            h_ss_tot = np.sum((a - np.mean(a)) ** 2)
            h_r2 = float(1.0 - h_ss_res / max(h_ss_tot, 1e-12))
            result["per_horizon"][f"Day +{h+1}"] = {"MAE": h_mae, "RMSE": h_rmse, "MAPE": h_mape, "MAPE_excluded_fraction": h_excluded_fraction, "R2": h_r2}

        return result

    # ================================================================
    # SMAPE
    # ================================================================

    @staticmethod
    def smape(predictions, actuals):
        predictions, actuals = np.asarray(predictions), np.asarray(actuals)
        denominator = (np.abs(predictions) + np.abs(actuals)) / 2.0
        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
        return float(np.mean(np.abs(predictions - actuals) / denominator) * 100)

    # ================================================================
    # NAIVE PERSISTENCE BASELINE
    # ================================================================

    @staticmethod
    def naive_persistence_baseline(contexts, horizon):
        contexts = np.asarray(contexts)
        return np.repeat(contexts[:, -1:], horizon, axis=1)

    # ================================================================
    # PROBABILISTIC METRICS: CRPS
    # ================================================================

    @staticmethod
    def probabilistic_metrics(all_sequences, actuals):
        """
        Sample-estimator CRPS.

        all_sequences: (n_test, horizon, n_sequences)
        actuals: (n_test, horizon)
        All values must be in ORIGINAL temperature units.

        FIX: the pairwise |ensemble_i - ensemble_j| term (`term2`) is
        averaged over n*(n-1) off-diagonal pairs, not n^2 -- the original
        code included the n zero-valued diagonal terms (i == j) in the
        average, which biases the CRPS estimate low (the bias shrinks as
        n_sequences grows, but is nonzero for any finite ensemble).
        """
        all_sequences = np.asarray(all_sequences, dtype=np.float64)
        actuals = np.asarray(actuals, dtype=np.float64)
        n_test, horizon, n_sequences = all_sequences.shape
        if actuals.shape != (n_test, horizon):
            raise ValueError(f"actuals must have shape ({n_test}, {horizon})")

        crps_per_horizon = np.zeros(horizon, dtype=np.float64)

        for h in range(horizon):
            crps_values = np.zeros(n_test, dtype=np.float64)
            for t in range(n_test):
                ensemble = all_sequences[t, h, :]
                y = actuals[t, h]
                term1 = np.mean(np.abs(ensemble - y))
                pairwise_diff = np.abs(ensemble[:, None] - ensemble[None, :])
                term2 = pairwise_diff.sum() / (2.0 * n_sequences * (n_sequences - 1)) if n_sequences > 1 else 0.0
                crps_values[t] = term1 - term2
            crps_per_horizon[h] = np.mean(crps_values)

        return {"overall_CRPS": float(np.mean(crps_per_horizon)), "per_horizon_CRPS": {f"Day +{h+1}": float(crps_per_horizon[h]) for h in range(horizon)}}

    # ================================================================
    # PREDICTION INTERVAL METRICS
    # ================================================================

    @staticmethod
    def prediction_interval_metrics(all_sequences, actuals, confidence=0.9):
        all_sequences, actuals = np.asarray(all_sequences), np.asarray(actuals)
        if all_sequences.ndim != 3:
            raise ValueError("all_sequences must have shape (n_test, horizon, n_sequences)")

        n_test, horizon, n_sequences = all_sequences.shape
        if actuals.shape != (n_test, horizon):
            raise ValueError("actuals shape does not match all_sequences")

        alpha = 1.0 - confidence
        lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
        lower = np.quantile(all_sequences, lower_q, axis=2)
        upper = np.quantile(all_sequences, upper_q, axis=2)
        inside = (actuals >= lower) & (actuals <= upper)
        width = upper - lower
        picp_per_horizon = inside.mean(axis=0)
        piw_per_horizon = width.mean(axis=0)

        return {
            "confidence_level": confidence,
            "overall_PICP": float(inside.mean()),
            "overall_PIW": float(width.mean()),
            "per_horizon_PICP": {f"Day +{h+1}": float(picp_per_horizon[h]) for h in range(horizon)},
            "per_horizon_PIW": {f"Day +{h+1}": float(piw_per_horizon[h]) for h in range(horizon)},
        }

    # ================================================================
    # CALIBRATION CURVE
    # ================================================================

    @staticmethod
    def calibration_curve(all_sequences, actuals, confidence_levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
        curve = []
        for level in confidence_levels:
            result = MultiSequenceGAN.prediction_interval_metrics(all_sequences, actuals, confidence=level)
            curve.append({"nominal_confidence": level, "empirical_PICP": result["overall_PICP"], "gap": (result["overall_PICP"] - level), "PIW": result["overall_PIW"]})
        return curve

    # ================================================================
    # MODEL C DIAGNOSTICS
    # ================================================================

    def decomposition_diagnostics(self, n_examples=5):
        """
        Print a few causal trend/residual decompositions from the test set.
        Useful for verifying that Model C is actually learning residuals.
        """
        n_examples = min(n_examples, len(self.contexts_raw_test))

        for i in range(n_examples):
            context = self.raw_contexts_test[i]
            actual = self.raw_targets_test[i]
            trend_context = self.trend_contexts_test[i]
            trend_future = self.trend_futures_test[i]
            context_residual = context - trend_context
            future_residual = actual - trend_future

            print(f"\nExample {i}")
            print("Context:", np.round(context, 4))
            print("Historical trend:", np.round(trend_context, 4))
            print("Context residual:", np.round(context_residual, 4))
            print("Future trend:", np.round(trend_future, 4))
            print("Actual future:", np.round(actual, 4))
            print("Actual future residual:", np.round(future_residual, 4))
            print("Local slope:", float(self.slopes_test[i]))

        return {
            "contexts": self.contexts_raw_test[:n_examples],
            "historical_trends": self.trend_contexts_test[:n_examples],
            "future_trends": self.trend_futures_test[:n_examples],
            "slopes": self.slopes_test[:n_examples],
        }