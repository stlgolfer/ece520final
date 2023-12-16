import tensorflow as tf
import tensorflow_quantum as tfq

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


if __name__ == "__main__":
    a, b = sympy.symbols('a b')

    # Create two qubits
    q0, q1 = cirq.GridQubit.rect(1, 2)

    # Create a circuit on these qubits using the parameters you created above.
    circuit = cirq.Circuit(
        cirq.rx(a).on(q0),
        cirq.ry(b).on(q1), cirq.CNOT(control=q0, target=q1))

    SVGCircuit(circuit)

    # Calculate a state vector with a=0.5 and b=-0.5.
    resolver = cirq.ParamResolver({a: 0.5, b: -0.5})
    output_state_vector = cirq.Simulator().simulate(circuit, resolver).final_state_vector
    output_state_vector

    z0 = cirq.Z(q0)

    qubit_map = {q0: 0, q1: 1}

    z0.expectation_from_state_vector(output_state_vector, qubit_map).real