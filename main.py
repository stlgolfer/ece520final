import tensorflow as tf
import tensorflow_quantum as tfq

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
from keras.utils.vis_utils import plot_model

def run_experiment(epochs, layers):
    # Parameters that the classical NN will feed values into.
    control_params = sympy.symbols('theta_1 theta_2 theta_3')

    # Create the parameterized circuit.
    qubit = cirq.GridQubit(0, 0)
    model_circuit = cirq.Circuit(
        cirq.rz(control_params[0])(qubit),
        cirq.ry(control_params[1])(qubit),
        cirq.rx(control_params[2])(qubit))
    print(model_circuit)
    SVGCircuit(model_circuit)

    # The classical neural network layers.
    controller = tf.keras.Sequential([
        tf.keras.layers.Dense(layers, activation='elu'),
        tf.keras.layers.Dense(3)
    ])

    controller(tf.constant([[0.0], [1.0]])).numpy()

    # This input is the simulated miscalibration that the model will learn to correct.
    circuits_input = tf.keras.Input(shape=(),
                                    # The circuit-tensor has dtype `tf.string`
                                    dtype=tf.string,
                                    name='circuits_input')

    # Commands will be either `0` or `1`, specifying the state to set the qubit to.
    commands_input = tf.keras.Input(shape=(1,),
                                    dtype=tf.dtypes.float32,
                                    name='commands_input')
    dense_2 = controller(commands_input)

    # TFQ layer for classically controlled circuits.
    expectation_layer = tfq.layers.ControlledPQC(model_circuit,
                                                 # Observe Z
                                                 operators=cirq.Z(qubit))
    expectation = expectation_layer([circuits_input, dense_2])
    # The full Keras model is built from our layers.
    model = tf.keras.Model(inputs=[circuits_input, commands_input],
                           outputs=expectation)
    print(model)
    plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)
    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)
    # The command input values to the classical NN.
    commands = np.array([[0], [1]], dtype=np.float32)

    # The desired Z expectation value at output of quantum circuit.
    expected_outputs = np.array([[1], [-1]], dtype=np.float32)
    random_rotations = np.random.uniform(0, 2 * np.pi, 3)
    noisy_preparation = cirq.Circuit(
        cirq.rx(random_rotations[0])(qubit),
        cirq.ry(random_rotations[1])(qubit),
        cirq.rz(random_rotations[2])(qubit)
    )
    print(noisy_preparation)
    datapoint_circuits = tfq.convert_to_tensor([
                                                   noisy_preparation
                                               ] * 2)  # Make two copied of this circuit
    model([datapoint_circuits, commands]).numpy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(x=[datapoint_circuits, commands],
                        y=expected_outputs,
                        epochs=epochs,
                        verbose=0)
    # plt.plot(history.history['loss'])
    # plt.title("Learning to Control a Qubit")
    # plt.xlabel("Iterations")
    # plt.ylabel("Error in Control")
    # plt.show()

    cirq_simulator = cirq.Simulator()
    def check_error(full_circuit, command_values, desired_values):
        """Based on the value in `command_value` see how well you could prepare
        the full circuit to have `desired_value` when taking expectation w.r.t. Z."""
        params_to_prepare_output = controller(command_values).numpy()
        # full_circuit = noisy_preparation + model_circuit

        # Test how well you can prepare a state to get expectation the expectation
        # value in `desired_values`
        expectations = []
        for index in [0, 1]:
            state = cirq_simulator.simulate(
                full_circuit,
                {s: v for (s, v) in zip(control_params, params_to_prepare_output[index])}
            ).final_state_vector
            expt = cirq.Z(qubit).expectation_from_state_vector(state, {qubit: 0}).real
            expectations.append(expt)
            print(f'For a desired output (expectation) of {desired_values[index]} with'
                  f' noisy preparation, the controller\nnetwork found the following '
                  f'values for theta: {params_to_prepare_output[index]}\nWhich gives an'
                  f' actual expectation of: {expt}\n')
        return expectations


    return check_error(noisy_preparation + model_circuit, commands, expected_outputs)

def plot_epoch_sensitivity(layers):
    epoch_domain = [x for x in range(2, 25)]
    outcomes = np.array([run_experiment(epoch_domain[0],layers)])
    for e in epoch_domain[1:]:
        outcomes = np.vstack((outcomes, run_experiment(e, layers)))
    outcomes = np.abs(outcomes)
    e_fig, e_ax = plt.subplots(1, 1)
    e_ax.plot(epoch_domain, outcomes.T[0], label='Up')
    e_ax.plot(epoch_domain, outcomes.T[1], label='Down')
    e_ax.set_xlabel('Number of Epochs')
    e_ax.set_ylabel('Fidelity')
    e_fig.suptitle('Hybrid System Epoch Sensitivity')
    e_fig.legend()
    e_fig.show()
    return outcomes

def plot_layers_sensitivity():
    layers_domain = [x for x in range(10,20)]

    outcomes = np.average(plot_epoch_sensitivity(layers_domain[0]),axis=1).T

    for l in layers_domain[1:]:
        outcomes = np.vstack((outcomes, np.average(plot_epoch_sensitivity(l),axis=1).T))

        # layer_outcomes = np.average(np.array([run_experiment(epoch_domain[0], l)]),axis=1)
        # for e in epoch_domain[1:]:
        #     layer_outcomes = np.average(np.vstack((outcomes, run_experiment(e,l))),1)
        # outcomes = np.hstack((outcomes, layer_outcomes))
        # outcomes = np.vstack((outcomes, run_experiment(40, e)))
    outcomes = np.abs(outcomes)
    e_fig, e_ax = plt.subplots(1, 1)
    # e_ax.plot(layers_domain, outcomes.T[0], label='Up')
    # e_ax.plot(layers_domain, outcomes.T[1], label='Down')
    e_ax.imshow(outcomes, cmap='Greys')
    e_ax.set_ylabel('Number of Layers')
    e_ax.set_xlabel('Epochs')
    e_ax.set_xticks([x for x in range(0, 23)], [x+2 for x in range(0, 23)])
    e_ax.set_yticks([x for x in range(0, 10)], [x + 10 for x in range(0, 10)])
    e_fig.suptitle('Hybrid System Epoch/Layer Sensitivity')
    e_fig.legend()
    e_fig.show()

if __name__ == "__main__":
    # plot_epoch_sensitivity()
    plot_layers_sensitivity()
    # now let's consider the epoch and layer sensitivities together in a 2d plot
    # expts =  # gets expectation values (absolute value is fidelity)