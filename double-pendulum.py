import numpy as np


def derivatives(t, state, m1, m2, l1, l2, g=9.81):
    """Return the state derivative; angles are in radians."""
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2

    mass_matrix = np.array([
        [(m1 + m2) * l1**2, m2 * l1 * l2 * np.cos(delta)],
        [m2 * l1 * l2 * np.cos(delta), m2 * l2**2],
    ])

    forces = np.array([
        -m2 * l1 * l2 * np.sin(delta) * omega2**2
        - (m1 + m2) * g * l1 * np.sin(theta1),

        m2 * l1 * l2 * np.sin(delta) * omega1**2
        - m2 * g * l2 * np.sin(theta2),
    ])

    alpha1, alpha2 = np.linalg.solve(mass_matrix, forces)

    return np.array([omega1, omega2, alpha1, alpha2])
