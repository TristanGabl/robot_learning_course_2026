import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODE_LENGTH = 100


def get_trajectory(t=100, base_pos=np.zeros(3)):
    dist = np.random.uniform(0.25, 0.3)
    dy   = np.random.uniform(-0.1, 0.1)
    phi  = np.random.uniform(-np.pi, np.pi)
    psi  = np.random.uniform(-np.pi, np.pi)
    r    = np.random.uniform(0.05, 0.15)
    dz   = r + np.random.uniform(0.15, 0.20)

    cx = base_pos[0] + dist * np.cos(psi) - dy * np.sin(psi)
    cy = base_pos[1] + dist * np.sin(psi) + dy * np.cos(psi)
    cz = base_pos[2] + dz
    omega = 2 * np.pi / MAX_EPISODE_LENGTH

    steps    = np.arange(t)
    local_y  = r * np.cos(omega * steps + phi)
    local_z  = r * np.sin(omega * steps + phi)
    xs = cx - local_y * np.sin(psi)
    ys = cy + local_y * np.cos(psi)
    zs = cz + local_z
    return np.stack([xs, ys, zs], axis=1)


N = 8
fig = plt.figure(figsize=(14, 6))

# --- 3D view ---
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
for _ in range(N):
    traj = get_trajectory()
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.7)
ax3d.scatter([0], [0], [0], c="red", s=80, zorder=5, label="base")
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
ax3d.set_title("3D trajectories")
ax3d.legend()

# --- Top-down XY view ---
ax2d = fig.add_subplot(1, 2, 2)
for _ in range(N):
    traj = get_trajectory()
    ax2d.plot(traj[:, 0], traj[:, 1], alpha=0.7)
    ax2d.plot(traj[0, 0], traj[0, 1], "o", markersize=4)   # start marker
ax2d.scatter([0], [0], c="red", s=80, zorder=5, label="base")
ax2d.set_aspect("equal")
ax2d.set_xlabel("X"); ax2d.set_ylabel("Y")
ax2d.set_title("Top-down (XY)")
ax2d.legend()

plt.tight_layout()
plt.savefig("trajectory_test.png", dpi=150)
plt.show()
