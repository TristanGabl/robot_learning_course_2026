from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import time

from exercises.ex3 import *

class SO100TrackEnv(gym.Env):
    xml_path: Path
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: Path, render_mode=None, real_time=True):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Timestep & Episode
        self.sim_timestep = self.model.opt.timestep # 0.002s (500 Hz)
        self.ctrl_decimation = 50 # makes control frequency 10 Hz
        self.ctrl_timestep = self.sim_timestep * self.ctrl_decimation # 0.1
        self.max_episode_length_s = 10
        self.max_episode_length = int(self.max_episode_length_s / self.ctrl_timestep) # 100 steps per episode
        self.current_step = 0

        # Define Observation and Action Spaces
        self._prev_target_pos = self.data.mocap_pos[0].copy() # bonus
        obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None

        # Deafult robot home position
        self.default_qpos = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])

        # Evaluation metrics
        self.ee_tracking_error = 0.0

        # bonus
        self.real_time = real_time
        self.current_trajectory = self.get_trajectory(t=self.max_episode_length)


        
    # bonus
    def get_trajectory(self, t=100, turns=5):
        # random variables
        dist = np.random.uniform(0.22, 0.27)       
        dy   = np.random.uniform(-0.1, 0.1)       
        phi = np.random.uniform(-np.pi, np.pi)    
        psi = np.random.uniform(-np.pi * 0.4, np.pi * 0.4) # cannot rotate 360, keeping the targets in front of the robot
        r = np.random.uniform(0.1, 0.2)         
        dz = r + np.random.uniform(0.1, 0.15)

        base_pos = self.data.body("Base").xpos.copy()
        cx = base_pos[0] + dist * np.cos(psi) - dy * np.sin(psi)
        cy = base_pos[1] + dist * np.sin(psi) + dy * np.cos(psi)
        cz = base_pos[2] + dz
        omega = 2 * np.pi / self.max_episode_length  # one full loop per episode

        steps = np.arange(t)
        # circle lies in the plane perpendicular to the XY radial direction
        local_y = r * np.cos(omega * steps * turns + phi)  # tangential (in XY plane)
        local_z = r * np.sin(omega * steps * turns + phi)  # vertical
        xs = cx - local_y * np.sin(psi)
        ys = cy + local_y * np.cos(psi)
        zs = cz + local_z
        return np.stack([xs, ys, zs], axis=1)  # shape (t, 3)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)

        # Reset Robot to home position
        self.data.qpos[:] = reset_robot(self.default_qpos)
        mujoco.mj_forward(self.model, self.data)

        # Reset target position around robot base
        base_pos = self.data.body("Base").xpos.copy()
        self.data.mocap_pos[0] = reset_target_position(base_pos)

        self.current_step = 0

        # bonus
        self.current_trajectory = self.get_trajectory(t=self.max_episode_length)

        return self._get_obs(), {}

    def _process_action(self, action):
        return process_action(action, self.model.jnt_range)

    def compute_reward(self):
        return compute_reward(self.ee_tracking_error)

    def step(self, action):
        t_start = time.time()

        # bonus
        # advance mocap target along the pre-generated trajectory
        self.data.mocap_pos[0] = self.current_trajectory[self.current_step]

        self.data.ctrl[:] = self._process_action(action)
        for _ in range(self.ctrl_decimation): 
            mujoco.mj_step(self.model, self.data)
        self.ee_tracking_error = np.linalg.norm(self.data.site("ee_site").xpos - self.data.mocap_pos[0])
        reward = self.compute_reward()

        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            truncated = True
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()

        # Extra info as metrics for evaluation
        info = {"ee_tracking_error": self.ee_tracking_error.item()}

        if self.render_mode == "human" and self.real_time:
            elapsed = time.time() - t_start
            sleep_time = self.ctrl_timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos.flat[:].copy()
        ee_pos_w = self.data.site("ee_site").xpos.copy()
        ee_rot_w = self.data.site("ee_site").xmat.reshape(3, 3)
        base_pos_w = self.data.body("Base").xpos.copy()
        base_rot_w = self.data.body("Base").xmat.reshape(3, 3)
        target_pos_w = self.data.mocap_pos[0].copy()

        # bonus
        target_vel_w = (self.data.mocap_pos[0] - self._prev_target_pos) / self.ctrl_timestep
        self._prev_target_pos = self.data.mocap_pos[0].copy()
        
        return get_obs(qpos, ee_pos_w, ee_rot_w, base_pos_w, base_rot_w, target_pos_w, target_vel_w)

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Update the viewer's copy of the data
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None