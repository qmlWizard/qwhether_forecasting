import pennylane as qml
from pennylane import numpy as np


def he_layer(weights, wires):
    """
    Hardware-Efficient Ansatz (HEA) layer.
    RY + RZ rotation on every wire, followed by a ring of CNOTs.

    weights: shape (n_wires, 2) -> [theta_y, theta_z] per wire
    """
    n = len(wires)
    for i, w in enumerate(wires):
        qml.RY(weights[i, 0], wires=w)
        qml.RZ(weights[i, 1], wires=w)

    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def quack_paper_layer(weights, wires):
    """
    Ansatz used in the QUACK (Quantum Kernels) paper.
    RZ-RY-RZ single-qubit rotation per wire, followed by a *linear*
    (non-circular) chain of CNOTs.

    weights: shape (n_wires, 3) -> [theta_z1, theta_y, theta_z2] per wire
    """
    n = len(wires)
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)

    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def ra_layer(weights, wires):
    """
    Real Amplitudes ansatz (Qiskit-style).
    Only RY rotations (keeps the statevector real), linear CNOT chain.

    weights: shape (n_wires,) -> theta_y per wire
    """
    n = len(wires)
    for i, w in enumerate(wires):
        qml.RY(weights[i], wires=w)

    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


LAYER_FNS = {
    "he": he_layer,
    "quack": quack_paper_layer,
    "ra": ra_layer,
}

# number of trainable params per wire, per ansatz -- needed by model.py
# to compute weight_shapes for qml.qnn.TorchLayer
PARAMS_PER_WIRE = {
    "he": 2,
    "quack": 3,
    "ra": 1,
}


def apply_variational_circuit(x, layers, reuploading, input_scaling, weights,
                               ansatz="he", wires=None):
    """
    Apply the gate sequence only (encoding + ansatz layers) — no
    measurements. Use this directly when the caller wants full control
    over what gets measured and in what order (e.g. reading out a
    subset of wires, or non-PauliZ observables), since PennyLane queues
    every `qml.expval(...)` the moment it's constructed, whether or not
    it's returned. `build_circuit` below is a thin convenience wrapper
    around this for the common "measure every wire" case.

    Parameters
    ----------
    x : array-like, shape (n_features,)
        Classical input to angle-encode.
    layers : int
        Number of variational layers.
    reuploading : bool
        Re-encode x before every layer if True; encode once before the
        first layer if False.
    input_scaling : bool
        If True, multiply x by a trainable vector before each encoding.
        Requires weights["scaling"] of shape (n_encodings, n_features),
        n_encodings = layers if reuploading else 1.
    weights : dict
        weights["layer_weights"]: shape (layers, n_wires, params_per_wire)
        weights["scaling"]: only needed if input_scaling=True
    ansatz : str
        "he", "quack", or "ra".
    wires : Sequence[int] or None
        Defaults to range(len(x)).

    Returns
    -------
    The wires used (list), for convenience when the caller didn't pass any.
    """
    if ansatz not in LAYER_FNS:
        raise ValueError(f"Unknown ansatz '{ansatz}'. Choose from {list(LAYER_FNS)}.")
    layer_fn = LAYER_FNS[ansatz]

    if wires is None:
        wires = list(range(len(x)))

    layer_weights = weights["layer_weights"]

    def encode(x_vec, encoding_idx):
        if input_scaling:
            x_vec = x_vec * weights["scaling"][encoding_idx]
        for i, w in enumerate(wires):
            qml.RX(x_vec[i % len(x_vec)], wires=w)

    if reuploading:
        for l in range(layers):
            encode(x, encoding_idx=l)
            layer_fn(layer_weights[l], wires)
    else:
        encode(x, encoding_idx=0)
        for l in range(layers):
            layer_fn(layer_weights[l], wires)

    return wires


def build_circuit(x, layers, reuploading, input_scaling, weights,
                   ansatz="he", wires=None):
    """
    Convenience wrapper around `apply_variational_circuit` that measures
    every wire with PauliZ. Do NOT call this if you also plan to define
    your own measurements downstream (e.g. a subset-wire readout) —
    the expvals it creates get queued onto the tape regardless of
    whether you use its return value, and will collide with yours.
    Use `apply_variational_circuit` directly in that case.

    Returns
    -------
    list of qml.expval(PauliZ) measurements, one per wire.
    """
    used_wires = apply_variational_circuit(
        x, layers, reuploading, input_scaling, weights, ansatz=ansatz, wires=wires,
    )
    return qml.expval(qml.PauliZ(0))