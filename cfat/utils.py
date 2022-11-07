import os

def muscle_activation_model_secondorder(activation, activation_first_derivative, control, dt, t_activation=0.04, t_excitation=0.03):
    """
    Returns activation/torque signal and its first derivative at next time step using second-order (muscle) activation dynamics.
    For details see https://journals.physiology.org/doi/pdf/10.1152/jn.00652.2003.
    """
    assert len(activation) == len(control)  # == sim.sim.model.nu

    return activation + dt * activation_first_derivative,\
           -(dt/(t_excitation * t_activation)) * activation + \
           (1 - ((dt*(t_excitation + t_activation))/(t_excitation * t_activation))) * activation_first_derivative + \
           dt * (control/(t_excitation * t_activation))

def muscle_activation_model_secondorder_inverse(activation_sequence, activation_first_derivative_sequence, dt, t_activation=0.04, t_excitation=0.03):
    """
    Returns actual control vector sequence given sequence of activation/torque signals and its first derivative (in discrete domain; forward Euler solution) using second-order (muscle) activation dynamics.
    For details see https://journals.physiology.org/doi/pdf/10.1152/jn.00652.2003.
    """
    assert len(activation_sequence) == len(activation_first_derivative_sequence)
    # assert dt <= np.sqrt(t_activation*t_excitation), f"IMPORTANT WARNING: Second-order (muscle) activation " \
    #                                                  f"dynamics might not work appropriate due to too large " \
    #                                                  f"time step 'dt' (is {dt}, should not be larger than {np.sqrt(t_activation*t_excitation)})."

    return ((t_excitation * t_activation) / dt) * (activation_first_derivative_sequence[1:] - (1 - ((dt*(t_excitation + t_activation))/(t_excitation * t_activation))) * activation_first_derivative_sequence[:-1] + (dt/(t_excitation * t_activation)) * (activation_sequence[:-1]))

def get_unique_filename(filename):
    filename_counter = 0
    filename_new = filename
    while os.path.isfile(filename_new):
        filename_counter += 1
        filename_new = ".".join(filename.split('.')[:-1]) + "({}).".format(filename_counter) + filename.split('.')[-1]
    return filename_new

def store_trajectories_table(filename, trajectories_table, unique_filename=True):
    if unique_filename:
        filename = get_unique_filename(filename)
    return trajectories_table.to_csv(filename, sep=',', mode='w', index=True, header=True,
                                     float_format='%.8f')