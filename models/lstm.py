import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml


# =========================================================
# LSTM NETWORK
# =========================================================

class LSTMNetwork(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        horizon=6,
        dropout=0.1
    ):

        super().__init__()

        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.fc = nn.Linear(
            hidden_size,
            horizon
        )

    def forward(self, x):

        # x:
        # (batch, historical_lookup, 1)

        output, (hidden, cell) = self.lstm(x)

        # Last timestep
        last_output = output[:, -1, :]

        # Predict all future points
        prediction = self.fc(last_output)

        # (batch, horizon)
        return prediction


# =========================================================
# LSTM MODEL
# =========================================================

class LSTMModel:

    def __init__(
        self,
        data,
        horizon=6,
        historical_lookup=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        train_ratio=0.8,
        device=None
    ):

        # -------------------------------------------------
        # Data
        # -------------------------------------------------

        self.data = np.asarray(
            data,
            dtype=np.float32
        ).reshape(-1)

        self.horizon = horizon
        self.historical_lookup = historical_lookup

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_ratio = train_ratio

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(device)

        print(
            f"LSTM device: {self.device}"
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout
        ).to(self.device)

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        self.criterion = nn.MSELoss()

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # -------------------------------------------------
        # Create datasets
        # -------------------------------------------------

        (
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test
        ) = self._create_dataset()

    # =====================================================
    # CREATE SLIDING WINDOWS
    # =====================================================

    def _create_dataset(self):

        data = self.data

        n = len(data)

        minimum_length = (
            self.historical_lookup
            + self.horizon
        )

        if n < minimum_length:

            raise ValueError(
                f"Not enough data.\n"
                f"Required: {minimum_length}\n"
                f"Available: {n}"
            )

        X = []
        Y = []

        # -------------------------------------------------
        # Create 6 -> 6 windows
        # -------------------------------------------------

        for i in range(
            self.historical_lookup,
            n - self.horizon + 1
        ):

            X.append(
                data[
                    i - self.historical_lookup:i
                ]
            )

            Y.append(
                data[
                    i:i + self.horizon
                ]
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        Y = np.asarray(
            Y,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Chronological split
        #
        # IMPORTANT:
        # Don't randomly shuffle time-series data
        # before splitting.
        # -------------------------------------------------

        split = int(
            len(X) * self.train_ratio
        )

        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]

        print(
            f"Total samples : {len(X)}"
        )

        print(
            f"Training      : {len(X_train)}"
        )

        print(
            f"Testing       : {len(X_test)}"
        )

        return (
            X_train,
            Y_train,
            X_test,
            Y_test
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        X_train = torch.tensor(
            self.X_train,
            dtype=torch.float32
        )

        Y_train = torch.tensor(
            self.Y_train,
            dtype=torch.float32
        )

        # -------------------------------------------------
        # LSTM expects:
        #
        # (batch, sequence_length, features)
        #
        # So:
        #
        # (N, 6) -> (N, 6, 1)
        # -------------------------------------------------

        X_train = X_train.unsqueeze(-1)

        dataset = TensorDataset(
            X_train,
            Y_train
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model.train()

        history = []

        for epoch in range(
            self.epochs
        ):

            epoch_loss = 0.0

            for X_batch, Y_batch in loader:

                X_batch = X_batch.to(
                    self.device
                )

                Y_batch = Y_batch.to(
                    self.device
                )

                # -----------------------------------------
                # Forward
                # -----------------------------------------

                prediction = self.model(
                    X_batch
                )

                # -----------------------------------------
                # Loss
                # -----------------------------------------

                loss = self.criterion(
                    prediction,
                    Y_batch
                )

                # -----------------------------------------
                # Backprop
                # -----------------------------------------

                self.optimizer.zero_grad()

                loss.backward()

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                epoch_loss += (
                    loss.item()
                    * len(X_batch)
                )

            epoch_loss /= len(
                dataset
            )

            history.append(
                epoch_loss
            )

            if (
                (epoch + 1) % 10 == 0
                or epoch == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch + 1:03d}/"
                    f"{self.epochs} "
                    f"- Loss: "
                    f"{epoch_loss:.6f}"
                )

        return history

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, input):

        input = np.asarray(
            input,
            dtype=np.float32
        ).reshape(-1)

        if len(input) < self.historical_lookup:

            raise ValueError(
                f"Input must contain at least "
                f"{self.historical_lookup} values."
            )

        # Take most recent observations
        input = input[
            -self.historical_lookup:
        ]

        # (6,) -> (1, 6, 1)
        X = torch.tensor(
            input,
            dtype=torch.float32
        ).reshape(
            1,
            self.historical_lookup,
            1
        )

        X = X.to(
            self.device
        )

        self.model.eval()

        prediction = self.model(X)

        prediction = (
            prediction
            .cpu()
            .detach()
            .numpy()[0]
        )

        return prediction

    # =====================================================
    # BACKTEST
    # =====================================================

    def backtest(self):

        predictions = []

        actuals = []

        contexts = []

        self.model.eval()

        for i in range(
            len(self.X_test)
        ):

            context = self.X_test[i]

            actual = self.Y_test[i]

            prediction = self.predict(
                context
            )

            contexts.append(
                context
            )

            predictions.append(
                prediction
            )

            actuals.append(
                actual
            )

        contexts = np.asarray(
            contexts,
            dtype=np.float32
        )

        predictions = np.asarray(
            predictions,
            dtype=np.float32
        )

        actuals = np.asarray(
            actuals,
            dtype=np.float32
        )

        return (
            contexts,
            predictions,
            actuals
        )

    # =====================================================
    # METRICS
    # =====================================================

    def metrics(
        self,
        current_pred,
        original
    ):

        current_pred = np.asarray(
            current_pred,
            dtype=np.float32
        )

        original = np.asarray(
            original,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Handle single prediction
        # -------------------------------------------------

        if current_pred.ndim == 1:

            current_pred = (
                current_pred.reshape(1, -1)
            )

            original = (
                original.reshape(1, -1)
            )

        if (
            current_pred.shape
            != original.shape
        ):

            raise ValueError(
                f"Prediction shape "
                f"{current_pred.shape} "
                f"does not match "
                f"original shape "
                f"{original.shape}."
            )

        # =================================================
        # HORIZON METRICS
        # =================================================

        horizon_metrics = {}

        for h in range(
            self.horizon
        ):

            pred = current_pred[:, h]

            actual = original[:, h]

            error = pred - actual

            # ---------------------------------------------
            # MAE
            # ---------------------------------------------

            mae = np.mean(
                np.abs(error)
            )

            # ---------------------------------------------
            # RMSE
            # ---------------------------------------------

            rmse = np.sqrt(
                np.mean(error ** 2)
            )

            # ---------------------------------------------
            # MAPE
            # ---------------------------------------------

            non_zero = (
                actual != 0
            )

            if np.any(non_zero):

                mape = np.mean(
                    np.abs(
                        error[non_zero]
                        / actual[non_zero]
                    )
                ) * 100

            else:

                mape = np.nan

            # ---------------------------------------------
            # R2
            # ---------------------------------------------

            ss_res = np.sum(
                error ** 2
            )

            ss_tot = np.sum(
                (
                    actual
                    - np.mean(actual)
                ) ** 2
            )

            if ss_tot == 0:

                r2 = np.nan

            else:

                r2 = (
                    1
                    - ss_res / ss_tot
                )

            horizon_metrics[
                f"Day +{h + 1}"
            ] = {

                "MAE": float(mae),

                "RMSE": float(rmse),

                "MAPE": float(mape),

                "R2": float(r2)
            }

        # =================================================
        # OVERALL
        # =================================================

        pred_flat = (
            current_pred.reshape(-1)
        )

        actual_flat = (
            original.reshape(-1)
        )

        error = (
            pred_flat
            - actual_flat
        )

        overall_mae = np.mean(
            np.abs(error)
        )

        overall_rmse = np.sqrt(
            np.mean(error ** 2)
        )

        non_zero = (
            actual_flat != 0
        )

        if np.any(non_zero):

            overall_mape = np.mean(
                np.abs(
                    error[non_zero]
                    / actual_flat[non_zero]
                )
            ) * 100

        else:

            overall_mape = np.nan

        ss_res = np.sum(
            error ** 2
        )

        ss_tot = np.sum(
            (
                actual_flat
                - np.mean(actual_flat)
            ) ** 2
        )

        if ss_tot == 0:

            overall_r2 = np.nan

        else:

            overall_r2 = (
                1
                - ss_res / ss_tot
            )

        overall = {

            "MAE": float(
                overall_mae
            ),

            "RMSE": float(
                overall_rmse
            ),

            "MAPE": float(
                overall_mape
            ),

            "R2": float(
                overall_r2
            )
        }

        return {

            "overall": overall,

            "horizon": horizon_metrics
        }

# ============================================================
# QLSTM CELL
# ============================================================

class QuantumLSTMCell(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=4,
        n_qubits=4,
        quantum_layers=1
    ):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers

        # ----------------------------------------------------
        # Classical -> Quantum projection
        #
        # [x_t, h_(t-1)] -> n_qubits
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            input_size + hidden_size,
            n_qubits
        )

        # ----------------------------------------------------
        # Quantum device
        # ----------------------------------------------------

        self.dev = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        # ----------------------------------------------------
        # Quantum circuit
        #
        # IMPORTANT:
        # This QNode accepts a BATCH of inputs.
        # ----------------------------------------------------

        @qml.qnode(
            self.dev,
            interface="torch",
            diff_method="backprop"
        )
        def quantum_circuit(inputs, weights):

            # inputs shape:
            #
            # (batch, n_qubits)

            qml.AngleEmbedding(
                inputs,
                wires=range(n_qubits),
                rotation="Y"
            )

            # ------------------------------------------------
            # Variational layers
            # ------------------------------------------------

            for layer in range(
                quantum_layers
            ):

                for qubit in range(
                    n_qubits
                ):

                    qml.RY(
                        weights[layer, qubit, 0],
                        wires=qubit
                    )

                    qml.RZ(
                        weights[layer, qubit, 1],
                        wires=qubit
                    )

                # Ring entanglement

                for qubit in range(
                    n_qubits - 1
                ):

                    qml.CNOT(
                        wires=[
                            qubit,
                            qubit + 1
                        ]
                    )

                qml.CNOT(
                    wires=[
                        n_qubits - 1,
                        0
                    ]
                )

            # ------------------------------------------------
            # Measurements
            # ------------------------------------------------

            return [
                qml.expval(
                    qml.PauliZ(qubit)
                )
                for qubit in range(
                    n_qubits
                )
            ]

        self.quantum_circuit = quantum_circuit

        # ----------------------------------------------------
        # Trainable quantum parameters
        # ----------------------------------------------------

        self.q_weights = nn.Parameter(
            0.01 * torch.randn(
                quantum_layers,
                n_qubits,
                2,
                dtype=torch.float32
            )
        )

        # ----------------------------------------------------
        # Quantum -> LSTM gates
        #
        # n_qubits -> 4 * hidden_size
        #
        # 4 gates:
        #
        # forget
        # input
        # candidate
        # output
        # ----------------------------------------------------

        self.gate_projection = nn.Linear(
            n_qubits,
            4 * hidden_size
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        hidden=None
    ):

        # ----------------------------------------------------
        # x shape:
        #
        # (batch, input_size)
        # ----------------------------------------------------

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # Initialize states
        # ----------------------------------------------------

        if hidden is None:

            h = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )

            c = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )

        else:

            h, c = hidden

        # ----------------------------------------------------
        # Concatenate input and hidden state
        #
        # (batch, input_size + hidden_size)
        # ----------------------------------------------------

        combined = torch.cat(
            [
                x,
                h
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Classical -> quantum
        #
        # (batch, n_qubits)
        # ----------------------------------------------------

        quantum_input = self.input_projection(
            combined
        )

        quantum_input = torch.tanh(
            quantum_input
        )

        # Map approximately [-1,1] -> [-pi,pi]

        quantum_input = (
            quantum_input * np.pi
        )

        # ----------------------------------------------------
        # Quantum circuit
        #
        # Entire batch goes through the circuit.
        # ----------------------------------------------------

        quantum_output = self.quantum_circuit(
            quantum_input,
            self.q_weights
        )

        # PennyLane can return a list of expectation values.
        #
        # Stack along feature dimension.
        #
        # Desired:
        #
        # (batch, n_qubits)
        # ----------------------------------------------------

        if isinstance(
            quantum_output,
            (list, tuple)
        ):

            quantum_output = torch.stack(
                quantum_output,
                dim=-1
            )

        # ----------------------------------------------------
        # Handle possible dimension ordering
        # ----------------------------------------------------

        if quantum_output.ndim == 1:

            quantum_output = (
                quantum_output
                .unsqueeze(0)
            )

        # ----------------------------------------------------
        # Ensure dtype/device compatibility
        # ----------------------------------------------------

        quantum_output = quantum_output.to(
            device=x.device,
            dtype=x.dtype
        )

        # ----------------------------------------------------
        # Quantum -> classical
        # ----------------------------------------------------

        gates = self.gate_projection(
            quantum_output
        )

        # ----------------------------------------------------
        # Split into four LSTM gates
        # ----------------------------------------------------

        i_gate, f_gate, g_gate, o_gate = torch.chunk(
            gates,
            4,
            dim=1
        )

        # ----------------------------------------------------
        # LSTM nonlinearities
        # ----------------------------------------------------

        i_gate = torch.sigmoid(
            i_gate
        )

        f_gate = torch.sigmoid(
            f_gate
        )

        g_gate = torch.tanh(
            g_gate
        )

        o_gate = torch.sigmoid(
            o_gate
        )

        # ----------------------------------------------------
        # Cell state
        # ----------------------------------------------------

        c = (
            f_gate * c
            +
            i_gate * g_gate
        )

        # ----------------------------------------------------
        # Hidden state
        # ----------------------------------------------------

        h = (
            o_gate
            * torch.tanh(c)
        )

        return h, c


# ============================================================
# QLSTM NETWORK
# ============================================================

class QuantumLSTMNetwork(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=4,
        n_qubits=4,
        quantum_layers=1,
        horizon=6
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.horizon = horizon

        self.qlstm = QuantumLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            quantum_layers=quantum_layers
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            horizon
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # x:
        #
        # (batch, sequence_length, 1)

        h = None
        c = None

        # ----------------------------------------------------
        # Process sequence
        # ----------------------------------------------------

        for t in range(
            x.shape[1]
        ):

            x_t = x[:, t, :]

            if h is None:

                h, c = self.qlstm(
                    x_t,
                    hidden=None
                )

            else:

                h, c = self.qlstm(
                    x_t,
                    hidden=(h, c)
                )

        # ----------------------------------------------------
        # Final hidden state -> horizon
        # ----------------------------------------------------

        prediction = self.fc(
            h
        )

        return prediction


# ============================================================
# QLSTM MODEL
# ============================================================

class QLSTMModel:

    def __init__(
        self,
        data,
        horizon=6,
        historical_lookup=6,
        hidden_size=4,
        n_qubits=4,
        quantum_layers=1,
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        train_ratio=0.8,
        device=None
    ):

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        self.data = np.asarray(
            data,
            dtype=np.float32
        ).reshape(-1)

        self.horizon = horizon
        self.historical_lookup = historical_lookup

        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        print(
            f"QLSTM device: {self.device}"
        )

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        self.model = QuantumLSTMNetwork(
            input_size=1,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            quantum_layers=quantum_layers,
            horizon=horizon
        ).to(self.device)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.criterion = nn.MSELoss()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        (
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test
        ) = self._create_dataset()

    # ========================================================
    # DATASET
    # ========================================================

    def _create_dataset(self):

        X = []
        Y = []

        for i in range(
            self.historical_lookup,
            len(self.data) - self.horizon + 1
        ):

            X.append(
                self.data[
                    i - self.historical_lookup:i
                ]
            )

            Y.append(
                self.data[
                    i:i + self.horizon
                ]
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        Y = np.asarray(
            Y,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Chronological split
        # ----------------------------------------------------

        split = int(
            len(X) * self.train_ratio
        )

        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]

        print(
            f"Total samples : {len(X)}"
        )

        print(
            f"Training      : {len(X_train)}"
        )

        print(
            f"Testing       : {len(X_test)}"
        )

        print(
            f"X_train       : {X_train.shape}"
        )

        print(
            f"Y_train       : {Y_train.shape}"
        )

        return (
            X_train,
            Y_train,
            X_test,
            Y_test
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

        X_train = torch.from_numpy(
            self.X_train
        ).unsqueeze(-1)

        Y_train = torch.from_numpy(
            self.Y_train
        )

        dataset = TensorDataset(
            X_train,
            Y_train
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        history = []

        self.model.train()

        for epoch in range(
            self.epochs
        ):

            epoch_loss = 0.0

            for X_batch, Y_batch in loader:

                X_batch = X_batch.to(
                    self.device,
                    dtype=torch.float32
                )

                Y_batch = Y_batch.to(
                    self.device,
                    dtype=torch.float32
                )

                # --------------------------------------------
                # Forward
                # --------------------------------------------

                prediction = self.model(
                    X_batch
                )

                # --------------------------------------------
                # Loss
                # --------------------------------------------

                loss = self.criterion(
                    prediction,
                    Y_batch
                )

                # --------------------------------------------
                # Backprop
                # --------------------------------------------

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                epoch_loss += (
                    loss.item()
                    * X_batch.size(0)
                )

            epoch_loss /= len(
                dataset
            )

            history.append(
                epoch_loss
            )

            print(
                f"Epoch "
                f"{epoch + 1:03d}/"
                f"{self.epochs} "
                f"- Loss: "
                f"{epoch_loss:.6f}"
            )

        return history

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        input
    ):

        input = np.asarray(
            input,
            dtype=np.float32
        ).reshape(-1)

        if len(input) < self.historical_lookup:

            raise ValueError(
                f"Input must contain at least "
                f"{self.historical_lookup} observations."
            )

        input = input[
            -self.historical_lookup:
        ]

        X = torch.from_numpy(
            input
        ).reshape(
            1,
            self.historical_lookup,
            1
        )

        X = X.to(
            self.device,
            dtype=torch.float32
        )

        self.model.eval()

        prediction = self.model(X)

        return (
            prediction
            .cpu()
            .detach()
            .numpy()[0]
        )

    # ========================================================
    # BATCH PREDICT
    # ========================================================

    def predict_batch(
        self,
        inputs
    ):

        inputs = np.asarray(
            inputs,
            dtype=np.float32
        )

        if inputs.ndim != 2:

            raise ValueError(
                "inputs must have shape "
                "(samples, historical_lookup)"
            )

        inputs = inputs[
            :, -self.historical_lookup:
        ]

        X = torch.from_numpy(
            inputs
        ).unsqueeze(-1)

        X = X.to(
            self.device,
            dtype=torch.float32
        )

        self.model.eval()

        prediction = self.model(X)

        return (
            prediction
            .cpu()
            .detach()
            .numpy()
        )

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest(self):

        self.model.eval()

        # ----------------------------------------------------
        # IMPORTANT:
        # Process the entire test set in batches.
        #
        # Do NOT call predict() one sample at a time.
        # ----------------------------------------------------

        predictions = []

        batch_size = self.batch_size

        for start in range(
            0,
            len(self.X_test),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(self.X_test)
            )

            X_batch = self.X_test[
                start:end
            ]

            prediction = self.predict_batch(
                X_batch
            )

            predictions.append(
                prediction
            )

        predictions = np.concatenate(
            predictions,
            axis=0
        )

        return (
            self.X_test.copy(),
            predictions,
            self.Y_test.copy()
        )

    # ========================================================
    # METRICS
    # ========================================================

    def metrics(
        self,
        current_pred,
        original
    ):

        current_pred = np.asarray(
            current_pred,
            dtype=np.float32
        )

        original = np.asarray(
            original,
            dtype=np.float32
        )

        if current_pred.ndim == 1:

            current_pred = (
                current_pred.reshape(1, -1)
            )

            original = (
                original.reshape(1, -1)
            )

        if current_pred.shape != original.shape:

            raise ValueError(
                f"Prediction shape "
                f"{current_pred.shape} "
                f"does not match "
                f"original shape "
                f"{original.shape}"
            )

        horizon_metrics = {}

        # ----------------------------------------------------
        # Per-horizon metrics
        # ----------------------------------------------------

        for h in range(
            self.horizon
        ):

            pred = current_pred[:, h]

            actual = original[:, h]

            error = pred - actual

            mae = np.mean(
                np.abs(error)
            )

            rmse = np.sqrt(
                np.mean(error ** 2)
            )

            non_zero = actual != 0

            if np.any(non_zero):

                mape = np.mean(
                    np.abs(
                        error[non_zero]
                        / actual[non_zero]
                    )
                ) * 100

            else:

                mape = np.nan

            ss_res = np.sum(
                error ** 2
            )

            ss_tot = np.sum(
                (
                    actual
                    - np.mean(actual)
                ) ** 2
            )

            if ss_tot == 0:

                r2 = np.nan

            else:

                r2 = (
                    1
                    - ss_res / ss_tot
                )

            horizon_metrics[
                f"Day +{h + 1}"
            ] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float(mape),
                "R2": float(r2)
            }

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        pred_flat = current_pred.reshape(-1)

        actual_flat = original.reshape(-1)

        error = (
            pred_flat
            - actual_flat
        )

        mae = np.mean(
            np.abs(error)
        )

        rmse = np.sqrt(
            np.mean(error ** 2)
        )

        non_zero = actual_flat != 0

        if np.any(non_zero):

            mape = np.mean(
                np.abs(
                    error[non_zero]
                    / actual_flat[non_zero]
                )
            ) * 100

        else:

            mape = np.nan

        ss_res = np.sum(
            error ** 2
        )

        ss_tot = np.sum(
            (
                actual_flat
                - np.mean(actual_flat)
            ) ** 2
        )

        if ss_tot == 0:

            r2 = np.nan

        else:

            r2 = (
                1
                - ss_res / ss_tot
            )

        return {
            "overall": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float(mape),
                "R2": float(r2)
            },
            "horizon": horizon_metrics
        }