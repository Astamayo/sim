"""Gymnasium + PyBullet environment for a ducted micro-turbine interceptor drone.

The vehicle is an 83 g ducted fan with two orthogonal control vanes sitting in
the exit plume, plus an upward-looking camera used for terminal guidance.
PyBullet integrates the rigid body at 200 Hz while every aerodynamic effect
(ducted thrust, slipstream vane forces, motor reaction torque, body drag, wind
gusts) is computed analytically and injected as an external force or torque.

Key modelling choices
---------------------
* Vane forces are applied at the duct **exit plane**, one half-height below the
  center of mass, so deflecting the plume produces both a side force and the
  control moment that actually steers the airframe (like thrust-vector control
  on a rocket).
* Vane deflection is mapped to a real angle (+/- 30 deg) and the lift
  coefficient is ``C_L = C_L_alpha * sin(delta)``, evaluated against the
  dynamic pressure of the duct exit jet.
* The commanded vane angle reaches the vane after a 15 ms servo transport
  delay (3 physics steps at 200 Hz).
* The vision error is refreshed at 30 Hz from a pinhole camera model and may be
  held back by one vision frame (domain randomization).

Run ``python ducted_turbine_env.py --help`` for the training / playback CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - exercised by installation errors
    raise ImportError("Install PyBullet with `pip install pybullet`.") from exc

AIR_DENSITY = 1.225  # kg/m^3 at sea level
GRAVITY = 9.81  # m/s^2


@dataclass
class DuctedTurbineConfig:
    """Every physical, task and randomization parameter in one place.

    Defaults match the airframe specification; tweak a copy of this dataclass
    instead of editing the environment when experimenting.
    """

    # --- airframe -----------------------------------------------------------
    mass: float = 0.083  # kg
    inertia: tuple[float, float, float] = (8.0e-5, 8.0e-5, 5.0e-5)  # kg*m^2
    ring_diameter: float = 0.127  # 5.0 in base ring
    inlet_diameter: float = 0.0508  # 2.0 in top inlet
    height: float = 0.0762  # 3.0 in total height
    exit_radius: float = 0.0635  # m, duct exit used for the slipstream model
    drag_coefficient: float = 0.8  # bluff-body drag on the duct shell

    # --- propulsion ---------------------------------------------------------
    max_thrust: float = 5.0  # N at 100 % motor power
    duct_factor: float = 1.18  # lip-suction thrust augmentation
    motor_time_constant: float = 0.02  # s, first-order spool-up lag
    reaction_torque_coefficient: float = 0.004  # N*m of rotor torque per N thrust
    # Fraction of the rotor torque absorbed by the duct stator vanes. Without
    # stators the raw 0.004 N*m/N against I_zz = 5e-5 would spin the airframe up
    # at ~470 rad/s^2, which no 2-vane control scheme can trim out.
    stator_recovery: float = 0.95
    yaw_damping: float = 3.0e-4  # N*m per rad/s from swirl drag on the stator
    body_rate_damping: float = 1.5e-4  # N*m per rad/s about the lateral axes

    # --- control vanes ------------------------------------------------------
    vane_area: float = 0.0012  # m^2 per vane
    vane_cl_alpha: float = 2.5  # per radian
    max_vane_angle: float = math.radians(30.0)
    vane_efficiency: float = 0.9  # fraction of the plume actually turned
    servo_delay: float = 0.015  # s of transport delay
    servo_rate_limit: float = math.radians(600.0)  # rad/s slew rate of the micro servos

    # --- loop rates ---------------------------------------------------------
    physics_hz: float = 200.0
    vision_hz: float = 30.0
    control_decimation: int = 1  # physics steps per agent action
    episode_seconds: float = 10.0

    # --- onboard camera -----------------------------------------------------
    camera_fov: float = math.radians(90.0)  # full field of view, body +Z axis
    camera_resolution: tuple[int, int] = (96, 96)

    # --- observation --------------------------------------------------------
    # The airframe is attitude-unstable (thrust vectoring below the center of
    # mass), so a memoryless policy needs gyro feedback to damp it. Set this to
    # False for the strict 12-element layout without body rates.
    include_gyro: bool = True

    # --- task ---------------------------------------------------------------
    intercept_radius: float = 0.1  # m
    max_tilt: float = math.radians(75.0)
    max_spin_rate: float = 40.0  # rad/s, tumbling is unrecoverable
    arena_radius: float = 3.0  # m, horizontal fly-away limit
    arena_ceiling: float = 4.0  # m, vertical fly-away limit
    start_position: tuple[float, float, float] = (0.0, 0.0, 0.7)

    # --- domain randomization ----------------------------------------------
    randomize: bool = True
    mass_jitter: float = 0.08  # +/- 8 % per episode
    thrust_jitter: float = 0.05  # +/- 5 % motor/ESC spread
    vane_jitter: float = 0.10  # +/- 10 % vane effectiveness
    gyro_noise_std: float = 0.02  # rad/s
    attitude_noise_std: float = 0.01  # rad, IMU attitude estimate noise
    wind_force_range: tuple[float, float] = (0.2, 0.5)  # N
    wind_interval_range: tuple[float, float] = (0.25, 1.0)  # s
    start_position_jitter: float = 0.05  # m
    start_tilt_jitter: float = math.radians(5.0)
    target_xy_range: float = 0.7  # m
    target_z_range: tuple[float, float] = (0.8, 1.5)  # m

    # --- reward weights -----------------------------------------------------
    w_distance: float = 1.0
    w_approach: float = 0.8
    w_spin_xy: float = 0.2
    w_spin_z: float = 0.5
    # Body rates are normalized by this reference before the quadratic spin
    # penalty. With I = 8e-5 kg*m^2 a normal maneuver already reaches ~10 rad/s,
    # where a raw 0.2 * w^2 term is ~100x the distance term and the optimal
    # policy becomes "never rotate", i.e. fall out of the sky. Set to 1.0 for
    # the literal specification weights.
    spin_reference_rate: float = 10.0  # rad/s
    w_smooth: float = 0.01
    intercept_bonus: float = 100.0
    crash_penalty: float = 50.0
    out_of_bounds_penalty: float = 50.0
    reward_clip: float = 20.0  # bound on the per-step shaping terms

    # --- rendering ----------------------------------------------------------
    realtime_human: bool = False  # pace the GUI to wall-clock time
    debug_draw_hz: float = 60.0

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def vision_period(self) -> float:
        return 1.0 / self.vision_hz

    @property
    def servo_delay_steps(self) -> int:
        return max(1, int(round(self.servo_delay * self.physics_hz)))

    @property
    def max_episode_steps(self) -> int:
        steps = self.episode_seconds * self.physics_hz / self.control_decimation
        return max(1, int(round(steps)))

    @property
    def exit_area(self) -> float:
        return math.pi * self.exit_radius**2

    @property
    def vane_arm(self) -> float:
        """Distance from the center of mass down to the exit plane."""
        return 0.5 * self.height

    @property
    def reference_area(self) -> float:
        """Frontal area used by the body drag model."""
        return self.ring_diameter * self.height


class DuctedTurbineEnv(gym.Env):
    """200 Hz force-based interception environment for a ducted micro turbine.

    Observation (15,), or (12,) with ``config.include_gyro=False``
        ``0:3``    position offset to the target [m]
        ``3:6``    linear velocity in world frame [m/s]
        ``6:9``    estimated body Euler angles [rad] (IMU noise included)
        ``9:12``   noisy body rates [rad/s] (omitted when ``include_gyro`` is off)
        ``-3:-1``  normalized camera target error [-1, 1]
        ``-1``     current motor level [0, 1]

    Action (3,) in ``[-1, 1]``
        ``0`` motor power (mapped to 0..100 %), ``1`` pitch vane, ``2`` yaw vane
        (both mapped to +/- 30 deg of plume deflection).
    """

    metadata = {"render_modes": ["human", "direct", "rgb_array"], "render_fps": 200}

    def __init__(
        self,
        render_mode: str | None = None,
        config: DuctedTurbineConfig | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "direct", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', 'direct' or 'rgb_array'")
        self.cfg = config if config is not None else DuctedTurbineConfig()
        self.render_mode = render_mode
        self.metadata = {
            **DuctedTurbineEnv.metadata,
            "render_fps": self.cfg.physics_hz / self.cfg.control_decimation,
        }

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        position_bound = self.cfg.arena_radius + self.cfg.arena_ceiling
        bounds = [position_bound] * 3 + [30.0] * 3 + [math.pi] * 3
        if self.cfg.include_gyro:
            bounds += [self.cfg.max_spin_rate] * 3
        bounds += [1.0, 1.0, 1.0]
        high = np.array(bounds, dtype=np.float32)
        low = -high
        low[-1] = 0.0  # motor level is non-negative
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # --- PyBullet handles -------------------------------------------------
        self.client_id = -1
        self.plane_id = -1
        self.drone_id = -1
        self.target_id = -1
        self._debug_ids: dict[str, int] = {}

        # --- episode state ----------------------------------------------------
        self.target_position = np.zeros(3, dtype=np.float64)
        self.mass = self.cfg.mass
        self.motor_level = 0.0
        self.vision_error = np.zeros(2, dtype=np.float64)
        self.gyro_measurement = np.zeros(3, dtype=np.float64)
        self.wind = np.zeros(3, dtype=np.float64)

        self._thrust_scale = 1.0
        self._vane_scale = 1.0
        self._reference_pressure = max(
            self._slipstream_pressure(self.cfg.duct_factor * self.cfg.max_thrust), 1e-9
        )
        self._previous_action = np.zeros(3, dtype=np.float64)
        self._servo_queue: deque[np.ndarray] = deque()
        self._vision_frames: deque[np.ndarray] = deque(maxlen=2)
        self._vision_delay_frames = 0
        self._vision_timer = 0.0
        self._target_visible = False
        self._wind_timer = 0.0
        self._control_steps = 0
        self._physics_steps = 0
        self._vane_angle = np.zeros(2, dtype=np.float64)
        self._vane_force = np.zeros(3, dtype=np.float64)
        self._thrust = 0.0
        self._intercepted = False
        self._crashed = False
        self._tumbled = False
        self._out_of_bounds = False
        self._wall_clock = time.perf_counter()

        self._position = np.array(self.cfg.start_position, dtype=np.float64)
        self._orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._rotation = np.eye(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)
        self._euler = np.zeros(3, dtype=np.float64)
        self._attitude_estimate = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------ world
    def _build_world(self) -> None:
        """Connect to PyBullet and create the (persistent) scene bodies."""
        if self.client_id >= 0:
            return
        cfg = self.cfg
        self.client_id = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -GRAVITY, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=cfg.dt,
            numSolverIterations=20,
            numSubSteps=1,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client_id,
        )
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.6,
                cameraYaw=35,
                cameraPitch=-25,
                cameraTargetPosition=(0.0, 0.0, 0.8),
                physicsClientId=self.client_id,
            )

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.5 * cfg.ring_diameter,
            height=cfg.height,
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.5 * cfg.ring_diameter,
            length=cfg.height,
            rgbaColor=(0.16, 0.22, 0.28, 1.0),
            physicsClientId=self.client_id,
        )
        self.drone_id = p.createMultiBody(
            baseMass=cfg.mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=cfg.start_position,
            physicsClientId=self.client_id,
        )
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.045,
            rgbaColor=(1.0, 0.2, 0.05, 1.0),
            physicsClientId=self.client_id,
        )
        self.target_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=target_visual,
            basePosition=(0.0, 0.0, 1.0),
            physicsClientId=self.client_id,
        )

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.cfg
        self._build_world()
        rng = self.np_random

        # --- domain randomization -------------------------------------------
        if cfg.randomize:
            self.mass = cfg.mass * rng.uniform(1.0 - cfg.mass_jitter, 1.0 + cfg.mass_jitter)
            self._thrust_scale = rng.uniform(1.0 - cfg.thrust_jitter, 1.0 + cfg.thrust_jitter)
            self._vane_scale = rng.uniform(1.0 - cfg.vane_jitter, 1.0 + cfg.vane_jitter)
            self._vision_delay_frames = int(rng.integers(0, 2))
            self._wind_timer = float(rng.uniform(0.05, cfg.wind_interval_range[1]))
            start = np.array(cfg.start_position, dtype=np.float64) + rng.uniform(
                -cfg.start_position_jitter, cfg.start_position_jitter, size=3
            )
            tilt = rng.uniform(-cfg.start_tilt_jitter, cfg.start_tilt_jitter, size=2)
            target = np.array(
                [
                    rng.uniform(-cfg.target_xy_range, cfg.target_xy_range),
                    rng.uniform(-cfg.target_xy_range, cfg.target_xy_range),
                    rng.uniform(*cfg.target_z_range),
                ],
                dtype=np.float64,
            )
        else:
            self.mass = cfg.mass
            self._thrust_scale = 1.0
            self._vane_scale = 1.0
            self._vision_delay_frames = 0
            self._wind_timer = cfg.wind_interval_range[1]
            start = np.array(cfg.start_position, dtype=np.float64)
            tilt = np.zeros(2)
            target = np.array([0.0, 0.0, 0.5 * sum(cfg.target_z_range)], dtype=np.float64)

        self.target_position = target
        inertia_scale = self.mass / cfg.mass
        p.changeDynamics(
            self.drone_id,
            -1,
            mass=self.mass,
            localInertiaDiagonal=tuple(value * inertia_scale for value in cfg.inertia),
            linearDamping=0.0,  # drag is modelled analytically instead
            angularDamping=0.01,
            physicsClientId=self.client_id,
        )
        p.resetBasePositionAndOrientation(
            self.drone_id,
            start,
            p.getQuaternionFromEuler([tilt[0], tilt[1], 0.0]),
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=self.client_id
        )
        p.resetBasePositionAndOrientation(
            self.target_id,
            self.target_position,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

        # --- controller / sensor state ---------------------------------------
        self.motor_level = 0.0
        self.wind = np.zeros(3, dtype=np.float64)
        self.gyro_measurement = np.zeros(3, dtype=np.float64)
        self.vision_error = np.zeros(2, dtype=np.float64)
        self._previous_action = np.zeros(3, dtype=np.float64)
        self._servo_queue = deque(
            [np.zeros(2, dtype=np.float64) for _ in range(cfg.servo_delay_steps)]
        )
        self._vision_frames.clear()
        self._vision_timer = 0.0
        self._target_visible = False
        self._vane_angle = np.zeros(2, dtype=np.float64)
        self._vane_force = np.zeros(3, dtype=np.float64)
        self._thrust = 0.0
        self._control_steps = 0
        self._physics_steps = 0
        self._intercepted = False
        self._crashed = False
        self._tumbled = False
        self._out_of_bounds = False
        self._wall_clock = time.perf_counter()

        if self.render_mode == "human":
            p.removeAllUserDebugItems(physicsClientId=self.client_id)
            self._debug_ids.clear()

        self._sync_state()
        self._update_vision(force=True)
        info = {
            "mass": self.mass,
            "thrust_scale": self._thrust_scale,
            "vane_scale": self._vane_scale,
            "vision_delay_frames": self._vision_delay_frames,
            "target_position": self.target_position.copy(),
        }
        return self._observation(), info

    # -------------------------------------------------------------- dynamics
    def _sync_state(self) -> None:
        """Cache the rigid-body state and the noisy sensor estimates."""
        position, orientation = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self.client_id
        )
        velocity, angular_velocity = p.getBaseVelocity(
            self.drone_id, physicsClientId=self.client_id
        )
        self._position = np.asarray(position, dtype=np.float64)
        self._orientation = np.asarray(orientation, dtype=np.float64)
        self._rotation = np.asarray(
            p.getMatrixFromQuaternion(orientation), dtype=np.float64
        ).reshape(3, 3)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        self._euler = np.asarray(p.getEulerFromQuaternion(orientation), dtype=np.float64)

        noise_scale = 1.0 if self.cfg.randomize else 0.0
        self.gyro_measurement = self._angular_velocity + noise_scale * self.np_random.normal(
            0.0, self.cfg.gyro_noise_std, size=3
        )
        self._attitude_estimate = self._euler + noise_scale * self.np_random.normal(
            0.0, self.cfg.attitude_noise_std, size=3
        )

    def _slipstream_pressure(self, thrust: float) -> float:
        """Dynamic pressure of the duct exit jet for a given thrust."""
        exit_velocity = math.sqrt(max(0.0, 2.0 * thrust / (AIR_DENSITY * self.cfg.exit_area)))
        return 0.5 * AIR_DENSITY * exit_velocity**2

    def _apply_forces(self, action: np.ndarray, vane_command: np.ndarray) -> None:
        cfg = self.cfg

        # Motor: first-order spool-up toward the commanded level.
        motor_command = float(np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0))
        blend = cfg.dt / max(cfg.motor_time_constant, cfg.dt)
        self.motor_level += (motor_command - self.motor_level) * blend
        self.motor_level = float(np.clip(self.motor_level, 0.0, 1.0))
        thrust = cfg.duct_factor * cfg.max_thrust * self.motor_level * self._thrust_scale
        self._thrust = thrust

        # Ducted thrust acts along the body +Z axis through the center of mass.
        p.applyExternalForce(
            self.drone_id,
            -1,
            [0.0, 0.0, thrust],
            [0.0, 0.0, 0.0],
            p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

        # Servos slew toward the (already delayed) commanded angle.
        commanded_angle = np.clip(vane_command, -1.0, 1.0) * cfg.max_vane_angle
        max_slew = cfg.servo_rate_limit * cfg.dt
        self._vane_angle += np.clip(commanded_angle - self._vane_angle, -max_slew, max_slew)

        # Slipstream vanes: lift from the turned jet, applied at the exit plane
        # so that it also produces the steering moment about the center of mass.
        dynamic_pressure = self._slipstream_pressure(thrust)
        lift = (
            dynamic_pressure
            * cfg.vane_area
            * cfg.vane_cl_alpha
            * np.sin(self._vane_angle)
            * cfg.vane_efficiency
            * self._vane_scale
        )
        vane_force = np.array([lift[0], lift[1], 0.0], dtype=np.float64)
        self._vane_force = vane_force
        if np.any(vane_force):
            p.applyExternalForce(
                self.drone_id,
                -1,
                vane_force,
                [0.0, 0.0, -cfg.vane_arm],
                p.LINK_FRAME,
                physicsClientId=self.client_id,
            )

        # Net rotor reaction torque (what the stator does not recover) plus the
        # aerodynamic rate damping of the duct and its vanes. Damping grows with
        # jet dynamic pressure and never fully vanishes thanks to body drag.
        reaction = -(1.0 - cfg.stator_recovery) * cfg.reaction_torque_coefficient * thrust
        damping_scale = 0.25 + 0.75 * dynamic_pressure / self._reference_pressure
        body_rates = self._rotation.T @ self._angular_velocity
        torque = np.array(
            [
                -cfg.body_rate_damping * damping_scale * body_rates[0],
                -cfg.body_rate_damping * damping_scale * body_rates[1],
                reaction - cfg.yaw_damping * damping_scale * body_rates[2],
            ],
            dtype=np.float64,
        )
        p.applyExternalTorque(
            self.drone_id,
            -1,
            torque,
            p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

        # Bluff-body drag plus the current wind gust, both in world frame.
        speed = float(np.linalg.norm(self._velocity))
        drag = np.zeros(3, dtype=np.float64)
        if speed > 1e-6:
            drag = (
                -0.5
                * AIR_DENSITY
                * cfg.drag_coefficient
                * cfg.reference_area
                * speed
                * self._velocity
            )
        external = drag + self.wind
        if np.any(external):
            p.applyExternalForce(
                self.drone_id,
                -1,
                external,
                self._position,
                p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

    def _update_wind(self) -> None:
        cfg = self.cfg
        if not cfg.randomize:
            return
        self._wind_timer -= cfg.dt
        if self._wind_timer > 0.0:
            return
        direction = self.np_random.normal(size=3)
        direction[2] *= 0.2  # gusts are mostly horizontal
        norm = max(float(np.linalg.norm(direction)), 1e-9)
        magnitude = self.np_random.uniform(*cfg.wind_force_range)
        self.wind = direction / norm * magnitude
        self._wind_timer = float(self.np_random.uniform(*cfg.wind_interval_range))

    # ---------------------------------------------------------------- sensors
    def _update_vision(self, force: bool = False) -> None:
        """Refresh the normalized camera error at 30 Hz (zero-order hold)."""
        cfg = self.cfg
        if not force:
            self._vision_timer += cfg.dt
            if self._vision_timer < cfg.vision_period:
                return
            self._vision_timer -= cfg.vision_period

        relative = self._rotation.T @ (self.target_position - self._position)
        tan_half_fov = math.tan(0.5 * cfg.camera_fov)
        boresight = relative[2]  # camera looks along the body +Z (thrust) axis
        if boresight > 1e-3:
            measurement = np.array(
                [
                    relative[0] / (boresight * tan_half_fov),
                    relative[1] / (boresight * tan_half_fov),
                ],
                dtype=np.float64,
            )
            self._target_visible = bool(np.all(np.abs(measurement) <= 1.0))
        else:
            # Target is level with or behind the lens: the tracker saturates in
            # the direction it was last seen so the policy still knows the sign.
            lateral = relative[:2]
            norm = max(float(np.linalg.norm(lateral)), 1e-9)
            measurement = lateral / norm
            self._target_visible = False
        measurement = np.clip(measurement, -1.0, 1.0)

        self._vision_frames.append(measurement)
        if self._vision_delay_frames and len(self._vision_frames) == 2:
            self.vision_error = self._vision_frames[0].copy()
        else:
            self.vision_error = measurement.copy()

    def _observation(self) -> np.ndarray:
        offset = self.target_position - self._position
        parts = [offset, self._velocity, self._attitude_estimate]
        if self.cfg.include_gyro:
            parts.append(self.gyro_measurement)
        parts += [self.vision_error, [self.motor_level]]
        observation = np.concatenate(parts).astype(np.float32)
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------- step
    def _termination_flags(self) -> bool:
        cfg = self.cfg
        distance = float(np.linalg.norm(self.target_position - self._position))
        self._intercepted = distance < cfg.intercept_radius
        self._tumbled = float(np.linalg.norm(self._angular_velocity)) > cfg.max_spin_rate
        self._crashed = (
            abs(self._euler[0]) > cfg.max_tilt
            or abs(self._euler[1]) > cfg.max_tilt
            or self._position[2] <= 0.5 * cfg.height
            or self._tumbled
        )
        horizontal = float(np.linalg.norm(self._position[:2]))
        self._out_of_bounds = (
            horizontal > cfg.arena_radius or self._position[2] > cfg.arena_ceiling
        )
        return self._intercepted or self._crashed or self._out_of_bounds

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.cfg
        if self.client_id < 0:
            raise RuntimeError("step() called before reset()")
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)

        terminated = False
        for _ in range(cfg.control_decimation):
            self._servo_queue.append(action[1:3].copy())
            vane_command = self._servo_queue.popleft()
            self._update_wind()
            self._apply_forces(action, vane_command)
            p.stepSimulation(physicsClientId=self.client_id)
            self._physics_steps += 1
            self._sync_state()
            self._update_vision()
            if self._termination_flags():
                terminated = True
                break

        self._control_steps += 1
        offset = self.target_position - self._position
        distance = float(np.linalg.norm(offset))
        line_of_sight = offset / max(distance, 1e-9)
        omega = self._angular_velocity

        # Dense shaping terms, bounded so that a single tumbling step cannot
        # dominate the return (the spin penalty is quadratic in body rate).
        spin_scale = 1.0 / max(cfg.spin_reference_rate, 1e-9) ** 2
        shaping = {
            "distance": -cfg.w_distance * distance,
            "approach": cfg.w_approach * float(np.dot(self._velocity, line_of_sight)),
            "spin": spin_scale
            * (
                -cfg.w_spin_xy * float(np.sum(omega[:2] ** 2))
                - cfg.w_spin_z * float(omega[2] ** 2)
            ),
            "smoothness": -cfg.w_smooth * float(np.sum((action - self._previous_action) ** 2)),
        }
        shaped = float(np.clip(sum(shaping.values()), -cfg.reward_clip, cfg.reward_clip))
        terms = {
            **shaping,
            "shaping_clip": shaped - float(sum(shaping.values())),
            "intercept": cfg.intercept_bonus if self._intercepted else 0.0,
            "crash": -cfg.crash_penalty if self._crashed else 0.0,
            "out_of_bounds": -cfg.out_of_bounds_penalty if self._out_of_bounds else 0.0,
        }
        reward = float(sum(terms.values()))
        self._previous_action = action

        truncated = (not terminated) and self._control_steps >= cfg.max_episode_steps
        self._draw_debug()
        if self.render_mode == "human" and cfg.realtime_human:
            self._pace_realtime()

        info = {
            "distance": distance,
            "speed": float(np.linalg.norm(self._velocity)),
            "thrust": self._thrust,
            "vane_angle": self._vane_angle.copy(),
            "wind": self.wind.copy(),
            "gyro_measurement": self.gyro_measurement.copy(),
            "target_visible": self._target_visible,
            "intercepted": self._intercepted,
            "crashed": self._crashed,
            "tumbled": self._tumbled,
            "out_of_bounds": self._out_of_bounds,
            "is_success": self._intercepted,
            "reward_terms": terms,
        }
        return self._observation(), reward, terminated, truncated, info

    # -------------------------------------------------------------- rendering
    def _pace_realtime(self) -> None:
        target_period = self.cfg.dt * self.cfg.control_decimation
        elapsed = time.perf_counter() - self._wall_clock
        if elapsed < target_period:
            time.sleep(target_period - elapsed)
        self._wall_clock = time.perf_counter()

    def _line(
        self,
        key: str,
        start: np.ndarray,
        end: np.ndarray,
        color: list[float],
        width: float,
    ) -> None:
        """Draw (or update in place) a persistent debug line."""
        item_id = p.addUserDebugLine(
            start,
            end,
            lineColorRGB=color,
            lineWidth=width,
            lifeTime=0,
            replaceItemUniqueId=self._debug_ids.get(key, -1),
            physicsClientId=self.client_id,
        )
        self._debug_ids[key] = item_id

    def _draw_debug(self) -> None:
        if self.render_mode != "human":
            return
        cfg = self.cfg
        stride = max(1, int(round(cfg.physics_hz / max(cfg.debug_draw_hz, 1.0))))
        if self._physics_steps % stride:
            return
        start = self._position
        exit_plane = start + self._rotation @ np.array([0.0, 0.0, -cfg.vane_arm])
        thrust_vector = self._rotation @ np.array([0.0, 0.0, self.motor_level * 0.35])
        self._line("thrust", start, start + thrust_vector, [0.1, 1.0, 0.1], 3)
        self._line("wind", start, start + self.wind * 0.5, [0.2, 0.5, 1.0], 2)
        self._line("sight", start, self.target_position, [1.0, 0.35, 0.05], 1)
        self._line(
            "vane",
            exit_plane,
            exit_plane + self._rotation @ self._vane_force * 0.5,
            [1.0, 0.9, 0.1],
            3,
        )

    def render(self) -> np.ndarray | None:
        """Return the onboard camera image when ``render_mode='rgb_array'``."""
        if self.render_mode != "rgb_array":
            return None
        cfg = self.cfg
        width, height = cfg.camera_resolution
        eye = self._position + self._rotation @ np.array([0.0, 0.0, 0.5 * cfg.height])
        view = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=eye + self._rotation @ np.array([0.0, 0.0, 1.0]),
            cameraUpVector=self._rotation @ np.array([0.0, 1.0, 0.0]),
        )
        projection = p.computeProjectionMatrixFOV(
            fov=math.degrees(cfg.camera_fov), aspect=width / height, nearVal=0.01, farVal=20.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self) -> None:
        if self.client_id >= 0:
            with contextlib.suppress(p.error):  # may already be torn down
                p.disconnect(physicsClientId=self.client_id)
            self.client_id = -1
            self.plane_id = -1
            self.drone_id = -1
            self.target_id = -1
            self._debug_ids.clear()


# ---------------------------------------------------------------------- tools
REWARD_HORIZON = 2.0  # seconds of look-ahead the discount factor should cover


def ppo_kwargs(config: DuctedTurbineConfig) -> dict[str, Any]:
    """PPO hyper-parameters scaled to the control rate of ``config``.

    The discount factor is derived from the control period so that the agent
    always optimizes over the same ~2 s horizon, whatever the decimation.
    ``log_std_init`` is lowered because full-scale Gaussian noise resampled at
    the control rate slams the vanes and tumbles the airframe instantly.
    """
    control_dt = config.dt * config.control_decimation
    return {
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": math.exp(-control_dt / REWARD_HORIZON),
        "gae_lambda": 0.95,
        "learning_rate": 3.0e-4,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [128, 128], "log_std_init": -1.0},
    }


def _build_config(args: argparse.Namespace) -> DuctedTurbineConfig:
    return DuctedTurbineConfig(
        control_decimation=args.decimation,
        randomize=not args.no_randomize,
    )


def _stats_path(model_path: str) -> str:
    return f"{os.path.splitext(model_path)[0]}_vecnormalize.pkl"


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    config = _build_config(args)
    render_mode = "human" if args.render else None
    n_envs = 1 if args.render else args.n_envs
    vec_cls = SubprocVecEnv if args.subproc and n_envs > 1 else DummyVecEnv
    venv = make_vec_env(
        DuctedTurbineEnv,
        n_envs=n_envs,
        seed=args.seed,
        env_kwargs={"config": config, "render_mode": render_mode},
        vec_env_cls=vec_cls,
    )
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu", **ppo_kwargs(config)
    )
    try:
        model.learn(total_timesteps=args.timesteps)
    finally:
        model.save(args.model_path)
        venv.save(_stats_path(args.model_path))
        venv.close()
    print(f"saved policy to {args.model_path}.zip and stats to {_stats_path(args.model_path)}")


def play(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    config = _build_config(args)
    config.realtime_human = not args.fast
    render_mode = None if args.headless else "human"
    venv = DummyVecEnv([lambda: DuctedTurbineEnv(render_mode=render_mode, config=config)])
    stats = _stats_path(args.model_path)
    if os.path.exists(stats):
        venv = VecNormalize.load(stats, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(args.model_path, device="cpu")

    for episode in range(args.episodes):
        observation = venv.reset()
        total, steps, info = 0.0, 0, [{}]
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = venv.step(action)
            total += float(reward[0])
            steps += 1
            if done[0]:
                break
        outcome = (
            "intercept"
            if info[0].get("intercepted")
            else "crash"
            if info[0].get("crashed")
            else "out of bounds"
            if info[0].get("out_of_bounds")
            else "timeout"
        )
        print(
            f"episode {episode + 1}: {outcome} after {steps} steps, "
            f"return {total:.1f}, final distance {info[0].get('distance', float('nan')):.3f} m"
        )
    venv.close()


def bench(args: argparse.Namespace) -> None:
    config = _build_config(args)
    env = DuctedTurbineEnv(config=config)
    env.reset(seed=args.seed)
    actions = env.action_space.sample
    start = time.perf_counter()
    for _ in range(args.timesteps):
        _, _, terminated, truncated, _ = env.step(actions())
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    rate = args.timesteps / elapsed
    print(
        f"{args.timesteps} steps in {elapsed:.2f} s -> {rate:,.0f} control steps/s "
        f"({rate * config.control_decimation / config.physics_hz:.0f}x realtime)"
    )


def check(args: argparse.Namespace) -> None:
    from gymnasium.utils.env_checker import check_env

    env = DuctedTurbineEnv(config=_build_config(args))
    check_env(env.unwrapped, skip_render_check=True)
    env.close()
    print("gymnasium check_env passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", choices=("train", "play", "bench", "check"), default="train"
    )
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--subproc", action="store_true", help="use subprocess vec env")
    parser.add_argument("--decimation", type=int, default=1, help="physics steps per action")
    parser.add_argument("--no-randomize", action="store_true", help="disable domain randomization")
    parser.add_argument("--render", action="store_true", help="train in the PyBullet GUI (slow)")
    parser.add_argument("--headless", action="store_true", help="play without the GUI")
    parser.add_argument("--fast", action="store_true", help="play without realtime pacing")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default="ducted_turbine_ppo")
    args = parser.parse_args()
    {"train": train, "play": play, "bench": bench, "check": check}[args.mode](args)


if __name__ == "__main__":
    main()
