from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from double_pendulum import derivatives


# Physical parameters.
m1 = m2 = 1.0
l1 = l2 = 1.0
g = 9.81
parameters = (m1, m2, l1, l2, g)

# Two initial states differing only in the first angle.
state_a = np.array([
    np.deg2rad(120),
    np.deg2rad(-10),
    0.0,
    0.0,
])

epsilon = 1e-6  # Radians.
state_b = state_a.copy()
state_b[0] += epsilon

duration = 30.0
fps = 15
times = np.linspace(0, duration, int(duration * fps) + 1)


def simulate(initial_state, rtol=1e-10, atol=1e-12):
    solution = solve_ivp(
        derivatives,
        (0, duration),
        initial_state,
        t_eval=times,
        args=parameters,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y


trajectory_a = simulate(state_a)
trajectory_b = simulate(state_b)


def coordinates(trajectory):
    theta1, theta2 = trajectory[:2]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return x1, y1, x2, y2


coords_a = coordinates(trajectory_a)
coords_b = coordinates(trajectory_b)

# Dimensionless distance in the full four-dimensional state space.
# Angular differences are wrapped into [-pi, pi].
# Scale angular velocities by the natural time sqrt(l1/g).
natural_time = np.sqrt(l1 / g)


def state_distance(first, second):
    angle_difference = first[:2] - second[:2]
    angle_difference = np.arctan2(
        np.sin(angle_difference),
        np.cos(angle_difference),
    )

    velocity_difference = natural_time * (first[2:] - second[2:])

    return np.sqrt(
        np.sum(angle_difference**2, axis=0)
        + np.sum(velocity_difference**2, axis=0)
    )


separation = state_distance(trajectory_a, trajectory_b)

# Repeat A with tighter tolerances as a numerical sensitivity check.
trajectory_a_tighter = simulate(
    state_a, rtol=1e-12, atol=1e-14
)
numerical_difference = state_distance(
    trajectory_a, trajectory_a_tighter
)

# Avoid taking the logarithm of zero.
separation = np.maximum(separation, 1e-16)
numerical_difference = np.maximum(numerical_difference, 1e-16)

# Animation settings.
plt.rcParams["animation.embed_limit"] = 100

fig, (ax_motion, ax_phase, ax_distance) = plt.subplots(
    1, 3, figsize=(14, 4.5), dpi=70
)

blue = "tab:blue"
orange = "tab:orange"

# Panel 1: physical motion.
reach = l1 + l2
ax_motion.set(
    xlim=(-1.1 * reach, 1.1 * reach),
    ylim=(-1.1 * reach, 1.1 * reach),
    xlabel="x (m)",
    ylabel="y (m)",
    title="Nearby initial conditions",
)
ax_motion.set_aspect("equal")
ax_motion.grid(alpha=0.25)
ax_motion.plot(0, 0, "ko", markersize=4)

rods_a, = ax_motion.plot(
    [], [], "o-", color=blue, lw=2, label="A"
)
rods_b, = ax_motion.plot(
    [], [], "o--", color=orange, lw=2, label="B", alpha=0.8
)
ax_motion.legend()

clock = ax_motion.text(
    0.04, 0.96, "",
    transform=ax_motion.transAxes,
    va="top",
)

# Panel 2: phase-space projection.
ax_phase.set(
    xlabel=r"$\theta_1$ (rad, unwrapped)",
    ylabel=r"$\omega_1$ (rad/s)",
    title="Phase-space projection",
)
ax_phase.grid(alpha=0.25)

phase_data = np.column_stack((
    np.concatenate((trajectory_a[0], trajectory_b[0])),
    np.concatenate((trajectory_a[2], trajectory_b[2])),
))
ax_phase.update_datalim(phase_data)
ax_phase.autoscale_view()
ax_phase.margins(0.08)

path_a, = ax_phase.plot([], [], color=blue, lw=1, label="A")
path_b, = ax_phase.plot(
    [], [], color=orange, lw=1, alpha=0.8, label="B"
)
point_a, = ax_phase.plot([], [], "o", color=blue)
point_b, = ax_phase.plot([], [], "o", color=orange)
ax_phase.legend()

# Panel 3: separation and numerical sensitivity.
ax_distance.set(
    xlim=(0, duration),
    xlabel="Time (s)",
    ylabel="Dimensionless state distance",
    title="Separation of trajectories",
)
ax_distance.set_yscale("log")
ax_distance.set_ylim(
    1e-16,
    10 * max(separation.max(), numerical_difference.max()),
)
ax_distance.grid(alpha=0.25, which="both")

distance_line, = ax_distance.plot(
    [], [], color="crimson", label="A versus B"
)
numerical_line, = ax_distance.plot(
    [], [], "--", color="gray",
    label="A versus A with tighter tolerances",
)
ax_distance.legend(fontsize=7)

fig.tight_layout()


def update(i):
    for rods, coords in ((rods_a, coords_a), (rods_b, coords_b)):
        x1, y1, x2, y2 = coords
        rods.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

    path_a.set_data(trajectory_a[0, :i + 1], trajectory_a[2, :i + 1])
    path_b.set_data(trajectory_b[0, :i + 1], trajectory_b[2, :i + 1])

    point_a.set_data([trajectory_a[0, i]], [trajectory_a[2, i]])
    point_b.set_data([trajectory_b[0, i]], [trajectory_b[2, i]])

    distance_line.set_data(times[:i + 1], separation[:i + 1])
    numerical_line.set_data(
        times[:i + 1], numerical_difference[:i + 1]
    )

    clock.set_text(f"Time: {times[i]:.2f} s")

    return (
        rods_a, rods_b, path_a, path_b,
        point_a, point_b, distance_line, numerical_line, clock,
    )


animation = FuncAnimation(
    fig,
    update,
    frames=len(times),
    interval=1000 / fps,
    blit=True,
    repeat=False,
)

output = Path("comparison.html")
output.write_text(
    animation.to_jshtml(fps=fps, default_mode="once"),
    encoding="utf-8",
)

plt.close(fig)
print(f"Animation saved to {output.resolve()}")
