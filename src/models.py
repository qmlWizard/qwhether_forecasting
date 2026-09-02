import pennylane as qml
import torch
import torch.nn as nn

from .circuits.variational_circuits import apply_variational_circuit, PARAMS_PER_WIRE
from .circuits.windspeed_circuits import build_wind_speed_circuit

class QNN(nn.Module):
    def __init__(
        self,
        n_qubits,
        n_layers,
        ansatz="he",
        reuploading=True,
        input_scaling=True,
        n_features=None,
        n_outputs=None,
        diff_method="backprop",
        device_name="default.qubit",
    ):
        super().__init__()

        if ansatz not in PARAMS_PER_WIRE:
            raise ValueError(f"Unknown ansatz '{ansatz}'. Choose from {list(PARAMS_PER_WIRE)}.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.reuploading = reuploading
        self.input_scaling = input_scaling
        self.n_features = n_features or n_qubits
        self.n_outputs = n_outputs or n_qubits

        if self.n_outputs > n_qubits:
            raise ValueError("n_outputs cannot exceed n_qubits in a fully-quantum readout.")

        params_per_wire = PARAMS_PER_WIRE[ansatz]
        wires = list(range(n_qubits))
        readout_wires = wires[: self.n_outputs]
        dev = qml.device(device_name, wires=n_qubits)

        n_encodings = n_layers if reuploading else 1

        if input_scaling:
            def circuit(inputs, layer_weights, scaling):
                weights = {"layer_weights": layer_weights, "scaling": scaling}
                apply_variational_circuit(
                    inputs, n_layers, reuploading, input_scaling, weights,
                    ansatz=ansatz, wires=wires,
                )
                return qml.expval(qml.PauliZ(0))
        else:
            def circuit(inputs, layer_weights):
                weights = {"layer_weights": layer_weights}
                apply_variational_circuit(
                    inputs, n_layers, reuploading, input_scaling, weights,
                    ansatz=ansatz, wires=wires,
                )
                return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)
        self.layer_weights = nn.Parameter(
            torch.empty(n_layers, n_qubits, params_per_wire).uniform_(0, 2 * torch.pi)
        )
        if input_scaling:
            self.scaling = nn.Parameter(torch.ones(n_encodings, self.n_features))
        else:
            self.scaling = None

    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.layer_weights.dtype, device=self.layer_weights.device)
        if self.input_scaling:
            results = self.qnode(x, self.layer_weights, self.scaling)
        else:
            results = self.qnode(x, self.layer_weights)
        return results


class WindSpeedModel(nn.Module):
    def __init__(
            self,
            n_qubits,
            n_layers,
            ansatz="circular",
            reuploading=True,
            input_scaling=True,
            n_features=None,
            n_outputs=None,
            diff_method="backprop",
            device_name="default.qubit",
        ):
            super().__init__()
    
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.ansatz = ansatz
            self.reuploading = reuploading
            self.input_scaling = input_scaling
            self.n_features = n_features or n_qubits
            self.n_outputs = n_outputs or n_qubits
    
            if self.n_outputs > n_qubits:
                raise ValueError("n_outputs cannot exceed n_qubits in a fully-quantum readout.")

            params_per_wire = 3
            wires = list(range(n_qubits))
            readout_wires = wires[: self.n_outputs]
            dev = qml.device(device_name, wires=n_qubits)
    
            n_encodings = n_layers if reuploading else 1
    
            if input_scaling:
                def circuit(inputs, layer_weights, scaling):
                    weights = {"layer_weights": layer_weights, "scaling": scaling}
                    build_wind_speed_circuit(inputs, n_layers, reuploading, input_scaling, weights, ansatz=ansatz, wires=wires)
                    return qml.expval(qml.PauliZ(0))
            else:
                def circuit(inputs, layer_weights):
                    weights = {"layer_weights": layer_weights}
                    build_wind_speed_circuit(inputs, n_layers, reuploading, input_scaling, weights, ansatz=ansatz, wires=wires)
                    return qml.expval(qml.PauliZ(0))
    
            self.qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)
            self.layer_weights = nn.Parameter(torch.empty(n_layers, n_qubits, params_per_wire).uniform_(0, 2 * torch.pi))
            if input_scaling:
                self.scaling = nn.Parameter(torch.ones(n_encodings, self.n_features))
            else:
                self.scaling = None

            self.n_quantum_outputs = n_qubits
            self.dense = nn.Sequential( nn.Linear(self.n_quantum_outputs, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 6))
    
    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.layer_weights.dtype, device=self.layer_weights.device)
        if self.input_scaling:
            self.n_quantum_outputs = self.qnode(x, self.layer_weights, self.scaling)
        else:
            self.n_quantum_outputs = self.qnode(x, self.layer_weights)
        #classical post-processing 
        results = self.dense(self.n_quantum_outputs)
        return results