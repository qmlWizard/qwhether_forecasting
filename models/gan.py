import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

torch.set_default_dtype(torch.float32)
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

def quantum_circuit(n_qubits=4, n_layers=1):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    return circuit

class QuantumGenerator(nn.Module):
    def __init__( self, context_size=6, horizon=6, latent_size=4, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnode = quantum_circuit(n_qubits, quantum_layers)
        self.input_projection = nn.Sequential(
            nn.Linear(context_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits))
        self.q_weights = nn.Parameter(0.01 * torch.randn(quantum_layers, n_qubits, 2))
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, context, noise):
        x = torch.cat([context, noise], dim=1)
        angles = self.input_projection(x)
        quantum_outputs = []
        for i in range(angles.shape[0]):
            q_out = self.qnode(angles[i], self.q_weights)
            q_out = torch.stack(q_out)
            q_out = q_out.to(device=angles.device, dtype=angles.dtype)
            quantum_outputs.append(q_out)
        quantum_outputs = torch.stack(quantum_outputs, dim=0)
        quantum_outputs = quantum_outputs.to(device=angles.device, dtype=angles.dtype)
        return self.decoder(quantum_outputs)

class ClassicalGenerator(nn.Module):
    def __init__(self, context_size=6, horizon=6, latent_size=4, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, context, noise):
        x = torch.cat([context, noise], dim=1)
        return self.network(x)

class ClassicalDiscriminator(nn.Module):
    def __init__(self, context_size=6, horizon=6, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_size + horizon, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, context, future):
        x = torch.cat([context, future], dim=1)
        return self.network(x)

class QuantumDiscriminator(nn.Module):
    def __init__(self, context_size=6, horizon=6, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnode = quantum_circuit(n_qubits, quantum_layers)
        self.input_projection = nn.Sequential(
            nn.Linear(context_size + horizon, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits)
        )

        self.q_weights = nn.Parameter(0.01 * torch.randn(quantum_layers, n_qubits, 2))
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, context, future):
        x = torch.cat([context, future], dim=1)
        angles = self.input_projection(x)
        quantum_outputs = []
        for i in range(angles.shape[0]):
            q_out = self.qnode(angles[i], self.q_weights)
            q_out = torch.stack(q_out)
            q_out = q_out.to(device=angles.device, dtype=angles.dtype)
            quantum_outputs.append(q_out)
        quantum_outputs = torch.stack(quantum_outputs, dim=0)
        quantum_outputs = quantum_outputs.to(device=angles.device, dtype=angles.dtype)
        return self.classifier(quantum_outputs)

# ============================================================
# QGAN MODEL
# ============================================================

class QGanModel:

    def __init__(
        self,
        data,
        type="QC",
        historical_lookup=6,
        horizon=6,
        latent_size=4,
        n_qubits=4,
        quantum_layers=1,
        hidden_size=32,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        train_ratio=0.8,
        device=None
    ):

        self.data = np.asarray(data, dtype=np.float32).reshape(-1)
        self.type = type.upper()
        self.historical_lookup = historical_lookup
        self.horizon = horizon
        self.latent_size = latent_size
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.device = device or DEVICE
        if self.type not in ["QC", "QQ", "CQ", "CC"]: raise ValueError("type must be QC, QQ, CQ or CC")

        self.contexts, self.targets = (self._create_sequences())
        split = int(len(self.contexts) * train_ratio)

        self.X_train = self.contexts[:split]
        self.y_train = self.targets[:split]

        self.X_test = self.contexts[split:]
        self.y_test = self.targets[split:]

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        if self.type in ["QC", "QQ"]:
            self.generator = QuantumGenerator(
                context_size=historical_lookup,
                horizon=horizon,
                latent_size=latent_size,
                n_qubits=n_qubits,
                quantum_layers=quantum_layers,
                hidden_size=hidden_size
            )
        else:
            self.generator = ClassicalGenerator(
                context_size=historical_lookup,
                horizon=horizon,
                latent_size=latent_size,
                hidden_size=hidden_size
            )

        # ----------------------------------------------------
        # Discriminator
        # ----------------------------------------------------

        if self.type in ["CQ", "QQ"]:
            self.discriminator = (
                QuantumDiscriminator(
                    context_size=historical_lookup,
                    horizon=horizon,
                    n_qubits=n_qubits,
                    quantum_layers=quantum_layers,
                    hidden_size=hidden_size
                )
            )
        else:
            self.discriminator = (
                ClassicalDiscriminator(
                    context_size=historical_lookup,
                    horizon=horizon,
                    hidden_size=hidden_size
                )
            )
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # ----------------------------------------------------
        # Optimizers
        # ----------------------------------------------------

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _create_sequences(self):

        X = []
        Y = []

        for i in range(len(self.data) - self.historical_lookup - self.horizon + 1):
            X.append(self.data[i:i + self.historical_lookup])
            Y.append(self.data[i + self.historical_lookup: i + self.historical_lookup + self.horizon])

        return (
                torch.tensor(np.asarray(X), dtype=torch.float32), 
                torch.tensor(np.asarray(Y), dtype=torch.float32)
            )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

        dataset = TensorDataset(
            self.X_train,
            self.y_train
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        history = {
            "generator_loss": [],
            "discriminator_loss": []
        }

        for epoch in range(
            self.epochs
        ):

            g_total = 0.0
            d_total = 0.0

            for context, real_future in loader:

                context = context.to(
                    self.device
                )

                real_future = real_future.to(
                    self.device
                )

                batch_size = (
                    context.shape[0]
                )

                # ============================================
                # DISCRIMINATOR
                # ============================================

                self.d_optimizer.zero_grad()

                real_logits = (
                    self.discriminator(
                        context,
                        real_future
                    )
                )

                real_labels = torch.ones_like(
                    real_logits
                )

                real_loss = (
                    self.criterion(
                        real_logits,
                        real_labels
                    )
                )

                noise = torch.randn(
                    batch_size,
                    self.latent_size,
                    device=self.device
                )

                fake_future = (
                    self.generator(
                        context,
                        noise
                    )
                )

                fake_logits = (
                    self.discriminator(
                        context,
                        fake_future.detach()
                    )
                )

                fake_labels = torch.zeros_like(
                    fake_logits
                )

                fake_loss = (
                    self.criterion(
                        fake_logits,
                        fake_labels
                    )
                )

                d_loss = (
                    real_loss + fake_loss
                ) / 2

                d_loss.backward()

                self.d_optimizer.step()

                # ============================================
                # GENERATOR
                # ============================================

                self.g_optimizer.zero_grad()

                noise = torch.randn(
                    batch_size,
                    self.latent_size,
                    device=self.device
                )

                fake_future = (
                    self.generator(
                        context,
                        noise
                    )
                )

                fake_logits = (
                    self.discriminator(
                        context,
                        fake_future
                    )
                )

                generator_labels = (
                    torch.ones_like(
                        fake_logits
                    )
                )

                g_loss = (
                    self.criterion(
                        fake_logits,
                        generator_labels
                    )
                )

                g_loss.backward()

                self.g_optimizer.step()

                g_total += g_loss.item()

                d_total += d_loss.item()

            g_avg = (
                g_total / len(loader)
            )

            d_avg = (
                d_total / len(loader)
            )

            history[
                "generator_loss"
            ].append(g_avg)

            history[
                "discriminator_loss"
            ].append(d_avg)

            if (
                epoch == 0
                or (epoch + 1) % 10 == 0
            ):

                print(
                    f"[{self.type}] "
                    f"Epoch {epoch+1}/{self.epochs} "
                    f"G={g_avg:.4f} "
                    f"D={d_avg:.4f}"
                )

        return history

    # ========================================================
    # PREDICT ONE 6-DAY SEQUENCE
    # ========================================================

    def predict(
        self,
        context
    ):

        self.generator.eval()

        if isinstance(
            context,
            np.ndarray
        ):

            context = torch.tensor(
                context,
                dtype=torch.float32
            )

        if context.ndim == 1:

            context = context.unsqueeze(0)

        context = context.to(
            self.device
        )

        noise = torch.randn(
            context.shape[0],
            self.latent_size,
            device=self.device
        )

        prediction = self.generator(
            context,
            noise
        )

        return prediction.cpu().detach().numpy()

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest(self):

        self.generator.eval()

        predictions = []

        contexts = []

        actuals = []

        for i in range(
            len(self.X_test)
        ):

            context = (
                self.X_test[i]
                .unsqueeze(0)
                .to(self.device)
            )

            target = (
                self.y_test[i]
                .numpy()
            )

            noise = torch.randn(
                1,
                self.latent_size,
                device=self.device
            )

            prediction = (
                self.generator(
                    context,
                    noise
                )
            )

            predictions.append(
                prediction.squeeze(0)
                .cpu()
                .detach()
                .numpy()
            )

            contexts.append(
                self.X_test[i].numpy()
            )

            actuals.append(
                target
            )

        return (
            np.asarray(contexts),
            np.asarray(predictions),
            np.asarray(actuals)
        )

    # ========================================================
    # METRICS
    # ========================================================

    @staticmethod
    def metrics(
        predictions,
        actuals
    ):

        predictions = np.asarray(
            predictions
        )

        actuals = np.asarray(
            actuals
        )

        error = (
            predictions - actuals
        )

        mae = np.mean(
            np.abs(error)
        )

        rmse = np.sqrt(
            np.mean(
                error ** 2
            )
        )

        denominator = np.where(
            np.abs(actuals) < 1e-8,
            1e-8,
            np.abs(actuals)
        )

        mape = np.mean(
            np.abs(error) / denominator
        ) * 100

        ss_res = np.sum(
            error ** 2
        )

        ss_tot = np.sum(
            (
                actuals
                - np.mean(actuals)
            ) ** 2
        )

        r2 = 1 - (
            ss_res / ss_tot
        )

        result = {

            "overall": {

                "MAE": mae,

                "RMSE": rmse,

                "MAPE": mape,

                "R2": r2
            },

            "per_horizon": {}
        }

        for h in range(
            predictions.shape[1]
        ):

            p = predictions[:, h]

            a = actuals[:, h]

            e = p - a

            h_mae = np.mean(
                np.abs(e)
            )

            h_rmse = np.sqrt(
                np.mean(
                    e ** 2
                )
            )

            h_denominator = np.where(
                np.abs(a) < 1e-8,
                1e-8,
                np.abs(a)
            )

            h_mape = np.mean(
                np.abs(e)
                / h_denominator
            ) * 100

            h_ss_res = np.sum(
                e ** 2
            )

            h_ss_tot = np.sum(
                (
                    a - np.mean(a)
                ) ** 2
            )

            h_r2 = 1 - (
                h_ss_res
                / h_ss_tot
            )

            result[
                "per_horizon"
            ][f"Day +{h+1}"] = {

                "MAE": h_mae,

                "RMSE": h_rmse,

                "MAPE": h_mape,

                "R2": h_r2
            }

        return result

class MultiSequenceGenerator(nn.Module):

    """
    One noise vector = one possible future trajectory.

    Input:
        context : (batch, 6)
        noise   : (batch, latent_size)

    Output:
        future  : (batch, 6)
    """

    def __init__(
        self,
        context_size=6,
        horizon=6,
        latent_size=8,
        hidden_size=64
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                context_size + latent_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                horizon
            )
        )

    def forward(
        self,
        context,
        noise
    ):

        x = torch.cat(
            [context, noise],
            dim=1
        )

        return self.network(x)


# ============================================================
# MULTI-SEQUENCE DISCRIMINATOR
# ============================================================

class ClassicalMultiSequenceGenerator(nn.Module):

    def __init__(self, context_size=6, horizon=6, latent_size=8, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, context, noise):
        x = torch.cat([context, noise], dim=1)
        return self.network(x)


class QuantumMultiSequenceGenerator(nn.Module):

    def __init__(
        self,
        context_size=6,
        horizon=6,
        latent_size=8,
        n_qubits=4,
        quantum_layers=1,
        hidden_size=32
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnode = quantum_circuit(n_qubits=n_qubits, n_layers=quantum_layers)
        # Context + latent noise
        self.input_projection = nn.Sequential(
            nn.Linear(context_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits))
        self.q_weights = nn.Parameter(0.01 * torch.randn(quantum_layers, n_qubits, 2))
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon))

    def forward(self, context, noise):
        x = torch.cat([context, noise], dim=1)
        angles = self.input_projection(x)
        quantum_outputs = []
        for i in range(angles.shape[0]):
            q_out = self.qnode(angles[i], self.q_weights)
            q_out = torch.stack(q_out).to(dtype=torch.float32)
            quantum_outputs.append(q_out)
        quantum_outputs = torch.stack(quantum_outputs, dim=0)
        return self.decoder(quantum_outputs)

class ClassicalMultiSequenceDiscriminator(nn.Module):

    def __init__(self, context_size=6, horizon=6, hidden_size=64):
        super().__init__()
        self.horizon = horizon
        self.shared = nn.Sequential(
            nn.Linear(context_size + horizon, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.horizon_head = nn.Linear(hidden_size, horizon)
        self.sequence_head = nn.Linear(hidden_size, 1)

    def forward(self, context, future):
        x = torch.cat([context, future], dim=1)
        features = self.shared(x)
        horizon_logits = (self.horizon_head(features))
        sequence_logit = (self.sequence_head(features))
        return torch.cat([horizon_logits, sequence_logit], dim=1)


class QuantumMultiSequenceDiscriminator(nn.Module):
    def __init__(self, context_size=6, horizon=6, n_qubits=4, quantum_layers=1, hidden_size=32):
        super().__init__()
        self.horizon = horizon
        self.n_qubits = n_qubits
        self.qnode = quantum_circuit(n_qubits=n_qubits, n_layers=quantum_layers)
        # Context + candidate future
        self.input_projection = nn.Sequential(
            nn.Linear(context_size + horizon, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits)
        )
        self.q_weights = nn.Parameter(0.01 * torch.randn(quantum_layers, n_qubits, 2))
        self.shared = nn.Sequential(
            nn.Linear(n_qubits, hidden_size),
            nn.ReLU()
        )
        self.horizon_head = nn.Linear(hidden_size, horizon)
        self.sequence_head = nn.Linear(hidden_size, 1)

    def forward(self, context, future):
        x = torch.cat([context, future], dim=1)
        angles = self.input_projection(x)
        quantum_outputs = []
        for i in range(angles.shape[0]):
            q_out = self.qnode(angles[i], self.q_weights)
            q_out = torch.stack(q_out).to(dtype=torch.float32)
            quantum_outputs.append(q_out)
        quantum_outputs = torch.stack(quantum_outputs, dim=0)
        features = self.shared(quantum_outputs)
        horizon_logits = (self.horizon_head(features))
        sequence_logit = (self.sequence_head(features))
        return torch.cat([horizon_logits, sequence_logit], dim=1)

class MultiSequenceGAN:

    """
    Conditional Multi-Sequence GAN.

    Supported architectures:

        CC = Classical Generator
             Classical Discriminator

        QC = Quantum Generator
             Classical Discriminator

        CQ = Classical Generator
             Quantum Discriminator

        QQ = Quantum Generator
             Quantum Discriminator


    Input:

        6 historical temperatures

    Output:

        N different 6-step futures


    predictions:

        shape = (6, N)


    horizon_confidence:

        shape = (6, N)


    sequence_confidence:

        shape = (N,)


    best_sequence:

        shape = (6,)
    """

    def __init__(
        self,
        data,

        type="QC",

        historical_lookup=6,
        horizon=6,

        latent_size=8,

        n_qubits=4,
        quantum_layers=1,

        hidden_size=64,

        epochs=100,
        batch_size=32,

        learning_rate=1e-4,

        train_ratio=0.8,

        device=None
    ):

        self.data = np.asarray(
            data,
            dtype=np.float32
        ).reshape(-1)

        self.type = type.upper()

        if self.type not in [
            "CC",
            "QC",
            "CQ",
            "QQ"
        ]:

            raise ValueError(
                "type must be one of: "
                "CC, QC, CQ, QQ"
            )

        self.historical_lookup = (
            historical_lookup
        )

        self.horizon = horizon

        self.latent_size = latent_size

        self.n_qubits = n_qubits

        self.quantum_layers = (
            quantum_layers
        )

        self.hidden_size = hidden_size

        self.epochs = epochs

        self.batch_size = batch_size

        self.learning_rate = (
            learning_rate
        )

        self.train_ratio = train_ratio

        self.device = device or DEVICE

        # ====================================================
        # DATASET
        # ====================================================

        (
            self.contexts,
            self.targets
        ) = self._create_sequences()

        split = int(
            len(self.contexts)
            * train_ratio
        )

        self.X_train = (
            self.contexts[:split]
        )

        self.y_train = (
            self.targets[:split]
        )

        self.X_test = (
            self.contexts[split:]
        )

        self.y_test = (
            self.targets[split:]
        )

        # ====================================================
        # GENERATOR
        # ====================================================

        if self.type in ["QC", "QQ"]:

            self.generator = (
                QuantumMultiSequenceGenerator(

                    context_size=
                        historical_lookup,

                    horizon=horizon,

                    latent_size=latent_size,

                    n_qubits=n_qubits,

                    quantum_layers=
                        quantum_layers,

                    hidden_size=
                        hidden_size
                )
            )

        else:

            self.generator = (
                ClassicalMultiSequenceGenerator(

                    context_size=
                        historical_lookup,

                    horizon=horizon,

                    latent_size=latent_size,

                    hidden_size=
                        hidden_size
                )
            )

        # ====================================================
        # DISCRIMINATOR
        # ====================================================

        if self.type in ["CQ", "QQ"]:

            self.discriminator = (
                QuantumMultiSequenceDiscriminator(

                    context_size=
                        historical_lookup,

                    horizon=horizon,

                    n_qubits=n_qubits,

                    quantum_layers=
                        quantum_layers,

                    hidden_size=
                        hidden_size
                )
            )

        else:

            self.discriminator = (
                ClassicalMultiSequenceDiscriminator(

                    context_size=
                        historical_lookup,

                    horizon=horizon,

                    hidden_size=
                        hidden_size
                )
            )

        # ====================================================
        # MOVE TO DEVICE
        # ====================================================

        self.generator = (
            self.generator.to(
                self.device
            )
        )

        self.discriminator = (
            self.discriminator.to(
                self.device
            )
        )

        # ====================================================
        # OPTIMIZERS
        # ====================================================

        self.g_optimizer = torch.optim.Adam(

            self.generator.parameters(),

            lr=learning_rate,

            betas=(0.5, 0.999)
        )

        self.d_optimizer = torch.optim.Adam(

            self.discriminator.parameters(),

            lr=learning_rate,

            betas=(0.5, 0.999)
        )

        self.criterion = (
            nn.BCEWithLogitsLoss()
        )

    # ========================================================
    # DATASET
    # ========================================================

    def _create_sequences(self):

        X = []
        Y = []

        for i in range(
            len(self.data)
            - self.historical_lookup
            - self.horizon
            + 1
        ):

            context = self.data[
                i:
                i + self.historical_lookup
            ]

            future = self.data[
                i + self.historical_lookup:
                i + self.historical_lookup
                + self.horizon
            ]

            X.append(context)
            Y.append(future)

        return (

            torch.tensor(
                np.asarray(X),
                dtype=torch.float32
            ),

            torch.tensor(
                np.asarray(Y),
                dtype=torch.float32
            )
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

        dataset = TensorDataset(
            self.X_train,
            self.y_train
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        history = {
            "generator_loss": [],
            "discriminator_loss": []
        }

        print(
            f"\nTraining MultiSequenceGAN "
            f"type={self.type}"
        )

        print(
            f"Generator: "
            f"{'Quantum' if self.type in ['QC', 'QQ'] else 'Classical'}"
        )

        print(
            f"Discriminator: "
            f"{'Quantum' if self.type in ['CQ', 'QQ'] else 'Classical'}"
        )

        for epoch in range(
            self.epochs
        ):

            g_total = 0.0
            d_total = 0.0

            for context, real_future in loader:

                context = context.to(
                    self.device
                )

                real_future = real_future.to(
                    self.device
                )

                batch_size = (
                    context.shape[0]
                )

                # ==================================================
                # DISCRIMINATOR
                # ==================================================

                self.d_optimizer.zero_grad()

                # Real trajectories
                real_output = (
                    self.discriminator(
                        context,
                        real_future
                    )
                )

                real_labels = (
                    torch.ones_like(
                        real_output
                    )
                )

                real_loss = (
                    self.criterion(
                        real_output,
                        real_labels
                    )
                )

                # Fake trajectories
                noise = torch.randn(
                    batch_size,
                    self.latent_size,
                    device=self.device
                )

                fake_future = (
                    self.generator(
                        context,
                        noise
                    )
                )

                fake_output = (
                    self.discriminator(
                        context,
                        fake_future.detach()
                    )
                )

                fake_labels = (
                    torch.zeros_like(
                        fake_output
                    )
                )

                fake_loss = (
                    self.criterion(
                        fake_output,
                        fake_labels
                    )
                )

                d_loss = (
                    real_loss + fake_loss
                ) / 2

                d_loss.backward()

                self.d_optimizer.step()

                # ==================================================
                # GENERATOR
                # ==================================================

                self.g_optimizer.zero_grad()

                noise = torch.randn(
                    batch_size,
                    self.latent_size,
                    device=self.device
                )

                fake_future = (
                    self.generator(
                        context,
                        noise
                    )
                )

                fake_output = (
                    self.discriminator(
                        context,
                        fake_future
                    )
                )

                generator_labels = (
                    torch.ones_like(
                        fake_output
                    )
                )

                g_loss = (
                    self.criterion(
                        fake_output,
                        generator_labels
                    )
                )

                g_loss.backward()

                self.g_optimizer.step()

                g_total += g_loss.item()

                d_total += d_loss.item()

            g_avg = (
                g_total / len(loader)
            )

            d_avg = (
                d_total / len(loader)
            )

            history[
                "generator_loss"
            ].append(g_avg)

            history[
                "discriminator_loss"
            ].append(d_avg)

            if (
                epoch == 0
                or (epoch + 1) % 10 == 0
            ):

                print(
                    f"[{self.type}] "
                    f"Epoch "
                    f"{epoch+1}/{self.epochs} "
                    f"G={g_avg:.4f} "
                    f"D={d_avg:.4f}"
                )

        return history

    # ========================================================
    # GENERATE N SEQUENCES
    # ========================================================

    def generate_sequences(
        self,
        context,
        n_sequences=100
    ):

        self.generator.eval()
        self.discriminator.eval()

        if isinstance(
            context,
            np.ndarray
        ):

            context = torch.tensor(
                context,
                dtype=torch.float32
            )

        context = context.float()

        if context.ndim == 1:

            context = context.unsqueeze(0)

        context = context.to(
            self.device
        )

        # Repeat same historical context
        context_batch = context.repeat(
            n_sequences,
            1
        )

        # Different latent vector
        # => different future trajectory
        noise = torch.randn(
            n_sequences,
            self.latent_size,
            device=self.device
        )

        # ----------------------------------------------------
        # Generate N futures
        # ----------------------------------------------------

        futures = self.generator(
            context_batch,
            noise
        )

        # futures:
        #
        # (N, 6)

        # ----------------------------------------------------
        # Discriminator
        # ----------------------------------------------------

        discriminator_output = (
            self.discriminator(
                context_batch,
                futures
            )
        )

        # ----------------------------------------------------
        # Six horizon confidence values
        # ----------------------------------------------------

        horizon_logits = (
            discriminator_output[
                :, :self.horizon
            ]
        )

        horizon_confidence = (
            torch.sigmoid(
                horizon_logits
            )
        )

        # ----------------------------------------------------
        # Complete sequence confidence
        # ----------------------------------------------------

        sequence_logits = (
            discriminator_output[
                :, self.horizon
            ]
        )

        sequence_confidence = (
            torch.sigmoid(
                sequence_logits
            )
        )

        # ----------------------------------------------------
        # Convert:
        #
        # (N,6) -> (6,N)
        # ----------------------------------------------------

        futures = (
            futures
            .cpu()
            .detach()
            .numpy()
            .T
        )

        horizon_confidence = (
            horizon_confidence
            .cpu()
            .detach()
            .numpy()
            .T
        )

        sequence_confidence = (
            sequence_confidence
            .cpu()
            .detach()
            .numpy()
        )

        return (
            futures,
            horizon_confidence,
            sequence_confidence
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        context,
        n_sequences=100
    ):

        (
            sequences,
            horizon_confidence,
            sequence_confidence
        ) = self.generate_sequences(
            context,
            n_sequences
        )

        # Highest confidence sequence
        best_index = np.argmax(
            sequence_confidence
        )

        best_sequence = (
            sequences[:, best_index]
        )

        best_horizon_confidence = (
            horizon_confidence[
                :, best_index
            ]
        )

        best_sequence_confidence = (
            sequence_confidence[
                best_index
            ]
        )

        return {

            "all_sequences":
                sequences,

            "horizon_confidence":
                horizon_confidence,

            "sequence_confidence":
                sequence_confidence,

            "best_index":
                best_index,

            "best_sequence":
                best_sequence,

            "best_horizon_confidence":
                best_horizon_confidence,

            "best_sequence_confidence":
                best_sequence_confidence
        }

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest(
        self,
        n_sequences=100
    ):

        all_contexts = []

        all_predictions = []

        all_horizon_confidences = []

        all_sequence_confidences = []

        all_actuals = []

        for i in range(
            len(self.X_test)
        ):

            context = (
                self.X_test[i]
                .numpy()
            )

            actual = (
                self.y_test[i]
                .numpy()
            )

            result = self.predict(
                context,
                n_sequences
            )

            all_contexts.append(
                context
            )

            all_predictions.append(
                result[
                    "best_sequence"
                ]
            )

            all_horizon_confidences.append(
                result[
                    "best_horizon_confidence"
                ]
            )

            all_sequence_confidences.append(
                result[
                    "best_sequence_confidence"
                ]
            )

            all_actuals.append(
                actual
            )

        return {

            "contexts":
                np.asarray(
                    all_contexts
                ),

            "predictions":
                np.asarray(
                    all_predictions
                ),

            "horizon_confidence":
                np.asarray(
                    all_horizon_confidences
                ),

            "sequence_confidence":
                np.asarray(
                    all_sequence_confidences
                ),

            "actuals":
                np.asarray(
                    all_actuals
                )
        }

    # ========================================================
    # METRICS
    # ========================================================

    @staticmethod
    def metrics(
        predictions,
        actuals
    ):

        predictions = np.asarray(
            predictions
        )

        actuals = np.asarray(
            actuals
        )

        error = (
            predictions - actuals
        )

        mae = np.mean(
            np.abs(error)
        )

        rmse = np.sqrt(
            np.mean(
                error ** 2
            )
        )

        denominator = np.where(
            np.abs(actuals) < 1e-8,
            1e-8,
            np.abs(actuals)
        )

        mape = np.mean(
            np.abs(error)
            / denominator
        ) * 100

        ss_res = np.sum(
            error ** 2
        )

        ss_tot = np.sum(
            (
                actuals
                - np.mean(actuals)
            ) ** 2
        )

        r2 = 1 - (
            ss_res / ss_tot
        )

        result = {

            "overall": {

                "MAE": mae,

                "RMSE": rmse,

                "MAPE": mape,

                "R2": r2
            },

            "per_horizon": {}
        }

        for h in range(
            predictions.shape[1]
        ):

            p = predictions[:, h]

            a = actuals[:, h]

            e = p - a

            h_mae = np.mean(
                np.abs(e)
            )

            h_rmse = np.sqrt(
                np.mean(
                    e ** 2
                )
            )

            h_denominator = np.where(
                np.abs(a) < 1e-8,
                1e-8,
                np.abs(a)
            )

            h_mape = np.mean(
                np.abs(e)
                / h_denominator
            ) * 100

            h_ss_res = np.sum(
                e ** 2
            )

            h_ss_tot = np.sum(
                (
                    a - np.mean(a)
                ) ** 2
            )

            h_r2 = 1 - (
                h_ss_res
                / h_ss_tot
            )

            result[
                "per_horizon"
            ][f"Day +{h+1}"] = {

                "MAE": h_mae,

                "RMSE": h_rmse,

                "MAPE": h_mape,

                "R2": h_r2
            }

        return result