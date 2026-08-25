import pennylane as qml
import torch
import torch.nn as nn

from ..circuits.variational_circuits import apply_variational_circuit, PARAMS_PER_WIRE


class QNN(nn.Module):
    """
    Fully-quantum neural network. This module *is* the model — there are
    no classical nn.Linear layers before or after the circuit, and no
    qml.qnn.TorchLayer wrapper either: the circuit's trainable angles
    live directly as nn.Parameter tensors on this module, and the QNode
    is called directly in forward(). Input features are angle-encoded
    onto the qubits and the circuit's per-qubit expectation values are
    the model's output (e.g. fed straight into a loss function, or
    through torch.sigmoid/softmax outside this module for classification).

    Because there's no classical pre-net, `n_features` (the input
    dimensionality) must already match what the encoding expects:
    each of the n_qubits wires is encoded with one feature via
    `x[i % len(x)]`, so n_features should equal n_qubits (or a divisor/
    multiple of it if you intentionally want feature re-use/wrapping).

    Handles:
      - device / qnode construction
      - weight_shapes bookkeeping for each ansatz (he / quack / ra)
      - optional trainable input scaling (still a quantum-circuit
        parameter — it rescales rotation angles, not a classical layer)
      - optional data re-uploading
      - optional readout over a subset of wires (n_outputs), so the
        output dimensionality can differ from n_qubits without adding
        any classical weights
    """

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

        # Trainable parameters live directly on the module as nn.Parameter
        # (no qml.qnn.TorchLayer involved). Rotation angles start uniform
        # in [0, 2*pi); scaling starts at 1.0 so it's a no-op until trained.
        self.layer_weights = nn.Parameter(
            torch.empty(n_layers, n_qubits, params_per_wire).uniform_(0, 2 * torch.pi)
        )
        if input_scaling:
            self.scaling = nn.Parameter(torch.ones(n_encodings, self.n_features))
        else:
            self.scaling = None

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_outputs)
        # qml.qnn.TorchLayer used to silently cast `inputs` to a torch
        # tensor before calling the qnode. Without it, PennyLane's
        # interface="torch" does NOT force-convert positional args on its
        # own -- if x arrives as a numpy array, a list, or a
        # pennylane.numpy array, it stays that type and breaks when it
        # hits `x_vec * weights["scaling"]` (a torch.Tensor) inside the
        # circuit. So we coerce explicitly here.
        x = torch.as_tensor(x, dtype=self.layer_weights.dtype, device=self.layer_weights.device)

        # PennyLane broadcasts automatically when `inputs` carries a leading
        # batch dimension and the weight arguments don't; the qnode returns
        # one tensor of shape (batch,) per measurement, which we stack.
        if self.input_scaling:
            results = self.qnode(x, self.layer_weights, self.scaling)
        else:
            results = self.qnode(x, self.layer_weights)
        return results