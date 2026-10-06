"""Gymnasium/PyBullet environment for a ducted micro-turbine drone.

The flight model uses PyBullet for rigid-body integration while applying the
thrust, vane, reaction-torque, wind, and gravity forces analytically.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - exercised by installation errors
    raise ImportError("Install PyBullet with `pip install pybullet`.") from exc


class DuctedTurbineEnv(gym.Env):
    """A 200 Hz, force-based drone interception environment."""

    metadata = {"render_modes": ["human", "direct"], "render_fps": 200}

    DT = 0.005
    PHYSICS_HZ = 200
    VISION_PERIOD = 1.0 / 30.0
    MAX_EPISODE_SECONDS = 10.0
    MAX_THRUST = 5.0
    DUCT_FACTOR = 1.18
    EXIT_RADIUS = 0.0635
    VANE_AREA = 0.0012
    VANE_CL_ALPHA = 2.5
    MAX_VANE_ANGLE = math.radians(30.0)
    MASS = 0.083
    INERTIA = (8.0e-5, 8.0e-5, 5.0e-5)

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        if render_mode not in (None, "human", "direct"):
            raise ValueError("render_mode must be None, 'human', or 'direct'")
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.full(12, -np.inf, dtype=np.float32),
            high=np.full(12, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.client_id = -1
        self.drone_id = -1
        self.target_id = -1
        self.target_position = np.zeros(3, dtype=np.float64)
        self.mass = self.MASS
        self.motor_level = 0.0
        self.previous_action = np.zeros(3, dtype=np.float64)
        self.servo_queue: deque[np.ndarray] = deque(maxlen=3)
        self.vision_error = np.zeros(2, dtype=np.float64)
        self.pending_vision_error = np.zeros(2, dtype=np.float64)
        self.vision_timer = 0.0
        self.vision_delay_frames = 0
        self.wind = np.zeros(3, dtype=np.float64)
        self.gyro_measurement = np.zeros(3, dtype=np.float64)
        self.wind_timer = 0.0
        self.step_count = 0

    def _connect(self) -> None:
        if self.client_id >= 0:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.DT, numSolverIterations=20, physicsClientId=self.client_id
        )

    def _create_bodies(self) -> None:
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.0635, height=0.0762, physicsClientId=self.client_id
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.0635, length=0.0762,
            rgbaColor=(0.16, 0.22, 0.28, 1.0), physicsClientId=self.client_id
        )
        self.drone_id = p.createMultiBody(
            baseMass=self.mass, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual, basePosition=(0.0, 0.0, 0.7),
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            self.drone_id, -1, mass=self.mass, localInertiaDiagonal=self.INERTIA,
            linearDamping=0.02, angularDamping=0.01, physicsClientId=self.client_id,
        )
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.045, rgbaColor=(1.0, 0.2, 0.05, 1.0),
            physicsClientId=self.client_id
        )
        self.target_id = p.createMultiBody(
            baseMass=0.0, baseVisualShapeIndex=target_visual,
            basePosition=self.target_position, physicsClientId=self.client_id
        )
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=35, cameraPitch=-25,
                cameraTargetPosition=(0.0, 0.0, 0.65), physicsClientId=self.client_id
            )

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._connect()
        if self.drone_id >= 0:
            p.removeBody(self.drone_id, physicsClientId=self.client_id)
        if self.target_id >= 0:
            p.removeBody(self.target_id, physicsClientId=self.client_id)
        self.mass = self.MASS * self.np_random.uniform(0.92, 1.08)
        self.target_position = np.array(
            [self.np_random.uniform(-0.7, 0.7), self.np_random.uniform(-0.7, 0.7),
             self.np_random.uniform(0.8, 1.5)], dtype=np.float64
        )
        self._create_bodies()
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)
        self.motor_level = 0.0
        self.previous_action.fill(0.0)
        self.servo_queue.clear()
        self.servo_queue.extend([np.zeros(2), np.zeros(2), np.zeros(2)])
        self.vision_error.fill(0.0)
        self.pending_vision_error.fill(0.0)
        self.vision_timer = 0.0
        self.vision_delay_frames = int(self.np_random.integers(0, 2))
        self.wind = np.zeros(3)
        self.gyro_measurement.fill(0.0)
        self.wind_timer = self.np_random.uniform(0.05, 0.5)
        self.step_count = 0
        return self._observation(), {"mass": self.mass, "vision_delay_frames": self.vision_delay_frames}

    def _observation(self) -> np.ndarray:
        position, orientation = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        velocity, angular_velocity = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        euler = p.getEulerFromQuaternion(orientation)
        offset = self.target_position - np.asarray(position)
        return np.asarray(
            [*offset, *velocity, *euler, *self.vision_error, self.motor_level], dtype=np.float32
        )

    def _update_vision(self, position: np.ndarray, orientation: tuple[float, ...]) -> None:
        self.vision_timer += self.DT
        if self.vision_timer < self.VISION_PERIOD:
            return
        self.vision_timer -= self.VISION_PERIOD
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
        relative = self.target_position - position
        body_relative = rotation.T @ relative
        focal_length = 1.0
        if body_relative[2] > 0.05:
            current = np.clip(
                [focal_length * body_relative[0] / body_relative[2],
                 focal_length * body_relative[1] / body_relative[2]], -1.0, 1.0
            )
        else:
            current = np.clip(relative[:2], -1.0, 1.0)
        if self.vision_delay_frames:
            self.vision_error = self.pending_vision_error.copy()
        else:
            self.vision_error = current
        self.pending_vision_error = current

    def _apply_forces(self, action: np.ndarray, vane_action: np.ndarray) -> None:
        position, orientation = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
        motor_command = float(np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0))
        self.motor_level += (motor_command - self.motor_level) * (self.DT / 0.02)
        thrust = self.DUCT_FACTOR * self.MAX_THRUST * self.motor_level
        body_thrust = np.array([0.0, 0.0, thrust])
        p.applyExternalForce(self.drone_id, -1, rotation @ body_thrust, position, p.WORLD_FRAME, physicsClientId=self.client_id)

        area_exit = math.pi * self.EXIT_RADIUS**2
        slip_velocity = math.sqrt(max(0.0, 2.0 * thrust / (1.225 * area_exit)))
        dynamic_pressure = 0.5 * 1.225 * slip_velocity**2
        pitch_force = dynamic_pressure * self.VANE_AREA * self.VANE_CL_ALPHA * vane_action[0]
        yaw_force = dynamic_pressure * self.VANE_AREA * self.VANE_CL_ALPHA * vane_action[1]
        local_force = np.array([pitch_force, yaw_force, 0.0])
        p.applyExternalForce(self.drone_id, -1, rotation @ local_force, position, p.WORLD_FRAME, physicsClientId=self.client_id)
        p.applyExternalTorque(self.drone_id, p.WORLD_FRAME, rotation @ np.array([0.0, 0.0, -0.004 * thrust]), physicsClientId=self.client_id)
        p.applyExternalForce(self.drone_id, -1, self.wind, position, p.WORLD_FRAME, physicsClientId=self.client_id)

    def _draw_debug(self) -> None:
        if self.render_mode != "human":
            return
        position, orientation = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        start = np.asarray(position)
        p.addUserDebugLine(start, start + rotation @ np.array([0, 0, self.motor_level * 0.35]), [0, 1, 0], 2, 0.05, physicsClientId=self.client_id)
        p.addUserDebugLine(start, start + self.wind * 0.5, [0, 0.5, 1], 2, 0.05, physicsClientId=self.client_id)
        p.addUserDebugLine(start, self.target_position, [1, 0.2, 0], 1, 0.02, physicsClientId=self.client_id)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(3).clip(-1.0, 1.0)
        self.servo_queue.append(action[1:3].copy())
        vane_action = self.servo_queue.popleft()
        self._apply_forces(action, vane_action)
        self.wind_timer -= self.DT
        if self.wind_timer <= 0.0:
            direction = self.np_random.normal(size=3)
            direction[2] *= 0.2
            self.wind = direction / max(np.linalg.norm(direction), 1e-9) * self.np_random.uniform(0.2, 0.5)
            self.wind_timer = self.np_random.uniform(0.25, 1.0)
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_count += 1
        position, orientation = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        velocity, angular_velocity = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        position = np.asarray(position)
        velocity = np.asarray(velocity)
        angular_velocity = np.asarray(angular_velocity)
        self.gyro_measurement = angular_velocity + self.np_random.normal(0.0, 0.02, size=3)
        euler = np.asarray(p.getEulerFromQuaternion(orientation))
        self._update_vision(position, orientation)
        offset = self.target_position - position
        distance = float(np.linalg.norm(offset))
        line_of_sight = offset / max(distance, 1e-9)
        reward = -distance + 0.8 * float(np.dot(velocity, line_of_sight))
        reward -= 0.2 * float(np.sum(angular_velocity[:2] ** 2)) + 0.5 * float(angular_velocity[2] ** 2)
        reward -= 0.01 * float(np.sum((action - self.previous_action) ** 2))
        self.previous_action = action
        intercepted = distance < 0.1
        crashed = abs(euler[0]) > math.radians(75) or abs(euler[1]) > math.radians(75) or position[2] <= 0.0
        terminated = intercepted or crashed
        truncated = self.step_count >= int(self.MAX_EPISODE_SECONDS / self.DT)
        if intercepted:
            reward += 100.0
        if crashed:
            reward -= 50.0
        self._draw_debug()
        info = {
            "distance": distance,
            "wind": self.wind.copy(),
            "gyro_measurement": self.gyro_measurement.copy(),
            "intercepted": intercepted,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1
            self.drone_id = -1
            self.target_id = -1


if __name__ == "__main__":
    from stable_baselines3 import PPO

    environment = DuctedTurbineEnv(render_mode="human")
    model = PPO("MlpPolicy", environment, n_steps=512, batch_size=64, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("ducted_turbine_ppo")
    environment.close()