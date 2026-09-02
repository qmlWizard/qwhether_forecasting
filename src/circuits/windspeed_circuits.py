import pennylane as qml
from pennylane import numpy as np

def encoding(inputs, wires):
    if len(inputs) != len(wires): raise ValueError("Number of inputs must match number of wires.")
    if type(wires) is not list: wires = range(wires)
    for i, w in enumerate(wires): qml.H(wires=w)
    qml.templates.AngleEmbedding(inputs, rotation='Y', wires=wires)

def variational_circuit(weights, wires):
    if type(wires) is not list: wires = range(wires)
    # Variational Layer: RY rotations
    for i, w in enumerate(wires): qml.Rot(weights[i], wires=w)


def circular_entangling_layer(wires):
    if type(wires) is not list: wires = range(wires)
    # Entangling Layer: Circular CNOTs
    for i, w in enumerate(wires[:-1]): qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[wires[-1], wires[0]])


def parallel_entangling_layer(wires):
    if type(wires) is not list: wires = range(wires)
    # Entangling Layer: Parallel CNOTs
    for i in range(0, len(wires) - 1, 2): qml.CNOT(wires=[wires[i], wires[i + 1]])
    for i in range(0, len(wires) - 1, 2): qml.CNOT(wires=[wires[i], wires[i + 1]])

def build_wind_speed_circuit(inputs, layers, reuploading, input_scaling, weights, entangling, wires):

    if input_scaling:
        scaling = weights["scaling"]
        scaled_inputs = inputs * scaling
    else:
        scaled_inputs = inputs

    encoding(scaled_inputs, wires)

    if entangling == "circular": entangling_layer = circular_entangling_layer
    elif entangling == "parallel": entangling_layer = parallel_entangling_layer
    else: raise ValueError(f"Unknown entangling type '{entangling}'. Choose from ['circular', 'parallel'].")

    for layer in range(layers):
        entangling_layer(wires)
        variational_circuit(weights["layer_weights"][layer], wires)
        if reuploading and layer < len(weights["layer_weights"]) - 1: encoding(scaled_inputs, wires)

    return qml.expval(qml.PauliZ(w) for w in range(len(wires)))