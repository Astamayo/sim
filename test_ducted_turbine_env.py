"""Standard-library test suite for :mod:`ducted_turbine_env`.

Run with ``python -m unittest -v test_ducted_turbine_env``.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from ducted_turbine_env import DuctedTurbineConfig, DuctedTurbineEnv

HOVER_ACTION = 2.0 * (0.083 * 9.81) / (1.18 * 5.0) - 1.0  # motor level for ~1 g


def deterministic_config(**overrides) -> DuctedTurbineConfig:
    base = DuctedTurbineConfig(randomize=False)
    return replace(base, **overrides) if overrides else base


class EnvContract(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DuctedTurbineEnv(config=deterministic_config())
        self.addCleanup(self.env.close)

    def test_spaces_and_dtypes(self) -> None:
        observation, info = self.env.reset(seed=0)
        self.assertEqual(observation.shape, (15,))  # 12 spec values + 3 gyro rates
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(self.env.action_space.shape, (3,))
        self.assertIn("mass", info)

    def test_strict_twelve_element_layout(self) -> None:
        env = DuctedTurbineEnv(config=deterministic_config(include_gyro=False))
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=0)
        self.assertEqual(observation.shape, (12,))
        np.testing.assert_allclose(
            observation[:9],
            np.concatenate([env.target_position - env._position, env._velocity, env._euler]),
            atol=1e-5,
        )
        self.assertAlmostEqual(float(observation[11]), env.motor_level, places=6)

    def test_gyro_is_observable_by_default(self) -> None:
        self.env.reset(seed=0)
        observation, _, _, _, _ = self.env.step([0.5, 0.1, 0.0])
        np.testing.assert_allclose(
            observation[9:12], self.env.gyro_measurement.astype(np.float32), atol=1e-5
        )

    def test_observations_stay_inside_declared_space(self) -> None:
        self.env.reset(seed=1)
        for _ in range(400):
            observation, _, terminated, truncated, _ = self.env.step(
                self.env.action_space.sample()
            )
            self.assertTrue(self.env.observation_space.contains(observation))
            if terminated or truncated:
                self.env.reset()

    def test_reward_equals_sum_of_terms(self) -> None:
        self.env.reset(seed=2)
        for _ in range(50):
            _, reward, terminated, truncated, info = self.env.step([0.2, 0.1, -0.1])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()), places=6)
            if terminated or truncated:
                break

    def test_spin_penalty_is_normalized_by_reference_rate(self) -> None:
        self.env.reset(seed=0)
        _, _, _, _, info = self.env.step([0.5, 0.3, 0.0])
        omega = self.env._angular_velocity
        cfg = self.env.cfg
        expected = (
            -(cfg.w_spin_xy * float(np.sum(omega[:2] ** 2)) + cfg.w_spin_z * float(omega[2] ** 2))
            / cfg.spin_reference_rate**2
        )
        self.assertAlmostEqual(info["reward_terms"]["spin"], expected, places=9)

    def test_literal_spec_weights_are_reachable(self) -> None:
        env = DuctedTurbineEnv(config=deterministic_config(spin_reference_rate=1.0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        _, _, _, _, info = env.step([0.5, 0.3, 0.0])
        omega = env._angular_velocity
        expected = -(0.2 * float(np.sum(omega[:2] ** 2)) + 0.5 * float(omega[2] ** 2))
        self.assertAlmostEqual(info["reward_terms"]["spin"], expected, places=9)

    def test_step_before_reset_raises(self) -> None:
        env = DuctedTurbineEnv(config=deterministic_config())
        self.addCleanup(env.close)
        with self.assertRaises(RuntimeError):
            env.step([0.0, 0.0, 0.0])

    def test_close_is_idempotent(self) -> None:
        self.env.reset(seed=0)
        self.env.close()
        self.env.close()

    def test_gymnasium_env_checker(self) -> None:
        from gymnasium.utils.env_checker import check_env

        env = DuctedTurbineEnv(config=deterministic_config())
        self.addCleanup(env.close)
        check_env(env, skip_render_check=True)


class Physics(unittest.TestCase):
    def make(self, **overrides) -> DuctedTurbineEnv:
        env = DuctedTurbineEnv(config=deterministic_config(**overrides))
        self.addCleanup(env.close)
        return env

    def test_full_thrust_climbs(self) -> None:
        env = self.make()
        env.reset(seed=0)
        for _ in range(40):
            env.step([1.0, 0.0, 0.0])
        self.assertGreater(env._velocity[2], 0.5)

    def test_idle_motor_falls(self) -> None:
        env = self.make()
        env.reset(seed=0)
        for _ in range(20):
            env.step([-1.0, 0.0, 0.0])
        self.assertLess(env._velocity[2], -0.5)

    def test_rotor_reaction_torque_yaws_negative(self) -> None:
        env = self.make()
        env.reset(seed=0)
        for _ in range(20):
            env.step([1.0, 0.0, 0.0])
        self.assertLess(env._angular_velocity[2], -1e-3)

    def test_stator_recovery_bounds_the_yaw_drift(self) -> None:
        env = self.make(arena_ceiling=1e6, arena_radius=1e6, episode_seconds=60.0)
        env.reset(seed=0)
        for _ in range(3000):
            env.step([HOVER_ACTION, 0.0, 0.0])
        self.assertLess(abs(env._angular_velocity[2]), 5.0)

    def test_yaw_damping_opposes_rotation(self) -> None:
        env = self.make(stator_recovery=1.0, arena_ceiling=1e6, episode_seconds=60.0)
        env.reset(seed=0)
        p_module = __import__("pybullet")
        p_module.resetBaseVelocity(
            env.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], physicsClientId=env.client_id
        )
        env._sync_state()
        for _ in range(100):
            env.step([1.0, 0.0, 0.0])
        self.assertLess(env._angular_velocity[2], 10.0)

    def test_pitch_vane_generates_pitch_moment(self) -> None:
        env = self.make()
        env.reset(seed=0)
        for _ in range(40):
            env.step([1.0, 1.0, 0.0])
        self.assertLess(env._angular_velocity[1], -1.0)  # +x force below the CoM
        self.assertLess(abs(env._angular_velocity[0]), abs(env._angular_velocity[1]))

    def test_yaw_vane_generates_roll_moment(self) -> None:
        env = self.make()
        env.reset(seed=0)
        for _ in range(40):
            env.step([1.0, 0.0, 1.0])
        self.assertGreater(env._angular_velocity[0], 1.0)
        self.assertLess(abs(env._angular_velocity[1]), abs(env._angular_velocity[0]))

    def test_vane_moment_is_antisymmetric(self) -> None:
        positive = self.make()
        positive.reset(seed=0)
        negative = self.make()
        negative.reset(seed=0)
        for _ in range(30):
            positive.step([1.0, 0.6, 0.0])
            negative.step([1.0, -0.6, 0.0])
        self.assertAlmostEqual(
            positive._angular_velocity[1], -negative._angular_velocity[1], places=3
        )

    def test_servo_delay_is_three_physics_steps(self) -> None:
        env = self.make()
        env.reset(seed=0)
        self.assertEqual(env.cfg.servo_delay_steps, 3)
        for _ in range(env.cfg.servo_delay_steps):
            env.step([1.0, 1.0, 0.0])
            self.assertLess(abs(env._angular_velocity[1]), 1e-6)
        env.step([1.0, 1.0, 0.0])
        self.assertGreater(abs(env._angular_velocity[1]), 1e-3)

    def test_servo_slew_rate_is_limited(self) -> None:
        env = self.make()
        env.reset(seed=0)
        steps_to_full = env.cfg.servo_delay_steps + math.ceil(
            env.cfg.max_vane_angle / (env.cfg.servo_rate_limit * env.cfg.dt)
        )
        for _ in range(env.cfg.servo_delay_steps + 1):
            env.step([1.0, 1.0, 0.0])
        self.assertLess(env._vane_angle[0], env.cfg.max_vane_angle)
        for _ in range(steps_to_full):
            env.step([1.0, 1.0, 0.0])
        self.assertAlmostEqual(env._vane_angle[0], env.cfg.max_vane_angle, places=6)

    def test_vane_authority_requires_thrust(self) -> None:
        env = self.make()
        env.reset(seed=0)
        for _ in range(30):
            env.step([-1.0, 1.0, 0.0])
        self.assertLess(abs(env._angular_velocity[1]), 1e-6)

    def test_slipstream_pressure_matches_momentum_theory(self) -> None:
        env = self.make()
        thrust = 5.9
        expected_velocity = math.sqrt(2.0 * thrust / (1.225 * env.cfg.exit_area))
        self.assertAlmostEqual(
            env._slipstream_pressure(thrust), 0.5 * 1.225 * expected_velocity**2, places=6
        )

    def test_drag_bounds_terminal_velocity(self) -> None:
        env = self.make(arena_ceiling=1e6, arena_radius=1e6, episode_seconds=60.0)
        env.reset(seed=0)
        for _ in range(4000):
            env.step([1.0, 0.0, 0.0])
        self.assertLess(env._velocity[2], 60.0)

    def test_inertia_scales_with_randomized_mass(self) -> None:
        env = DuctedTurbineEnv(config=DuctedTurbineConfig())
        self.addCleanup(env.close)
        _, info = env.reset(seed=3)
        self.assertLessEqual(abs(info["mass"] / 0.083 - 1.0), 0.08 + 1e-9)


class Termination(unittest.TestCase):
    def make(self, **overrides) -> DuctedTurbineEnv:
        env = DuctedTurbineEnv(config=deterministic_config(**overrides))
        self.addCleanup(env.close)
        return env

    def test_ground_contact_crashes(self) -> None:
        env = self.make()
        env.reset(seed=0)
        crashed = False
        for _ in range(400):
            _, reward, terminated, _, info = env.step([-1.0, 0.0, 0.0])
            if terminated:
                crashed = info["crashed"]
                self.assertLess(reward, 0.0)
                break
        self.assertTrue(crashed)

    def test_ceiling_breach_is_out_of_bounds(self) -> None:
        env = self.make(arena_ceiling=0.9)
        env.reset(seed=0)
        for _ in range(400):
            _, _, terminated, _, info = env.step([1.0, 0.0, 0.0])
            if terminated:
                self.assertTrue(info["out_of_bounds"])
                break
        else:  # pragma: no cover - would indicate a broken bounds check
            self.fail("fly-away was never terminated")

    def test_tumble_terminates_as_crash(self) -> None:
        env = self.make(
            max_tilt=math.radians(179.0), arena_ceiling=1e6, arena_radius=1e6, max_spin_rate=5.0
        )
        env.reset(seed=0)
        for _ in range(400):
            _, _, terminated, _, info = env.step([1.0, 1.0, 1.0])
            if terminated:
                self.assertTrue(info["tumbled"])
                self.assertTrue(info["crashed"])
                break
        else:  # pragma: no cover - would mean the spin guard never fired
            self.fail("uncontrolled spin was never terminated")

    def test_shaping_reward_is_bounded(self) -> None:
        env = self.make(max_tilt=math.radians(179.0), arena_ceiling=1e6, arena_radius=1e6)
        env.reset(seed=0)
        for _ in range(400):
            _, reward, terminated, truncated, info = env.step([1.0, 1.0, 1.0])
            shaping = sum(
                value
                for key, value in info["reward_terms"].items()
                if key in ("distance", "approach", "spin", "smoothness", "shaping_clip")
            )
            self.assertGreaterEqual(shaping, -env.cfg.reward_clip - 1e-6)
            self.assertLessEqual(shaping, env.cfg.reward_clip + 1e-6)
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()), places=6)
            if terminated or truncated:
                break

    def test_truncation_at_episode_limit(self) -> None:
        env = self.make(episode_seconds=0.05)
        env.reset(seed=0)
        self.assertEqual(env.cfg.max_episode_steps, 10)
        for step in range(10):
            _, _, terminated, truncated, _ = env.step([HOVER_ACTION, 0.0, 0.0])
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 9)

    def test_intercept_bonus_on_contact(self) -> None:
        env = self.make()
        env.reset(seed=0)
        env.target_position = env._position + np.array([0.0, 0.0, 0.02])
        _, reward, terminated, _, info = env.step([HOVER_ACTION, 0.0, 0.0])
        self.assertTrue(terminated)
        self.assertTrue(info["intercepted"])
        self.assertGreater(reward, 90.0)


class Sensors(unittest.TestCase):
    def make(self, **overrides) -> DuctedTurbineEnv:
        env = DuctedTurbineEnv(config=deterministic_config(**overrides))
        self.addCleanup(env.close)
        return env

    def test_vision_error_centered_when_target_overhead(self) -> None:
        env = self.make()
        env.reset(seed=0)
        env.target_position = env._position + np.array([0.0, 0.0, 1.0])
        env._update_vision(force=True)
        np.testing.assert_allclose(env.vision_error, [0.0, 0.0], atol=1e-6)
        self.assertTrue(env._target_visible)

    def test_vision_error_tracks_lateral_offset(self) -> None:
        env = self.make()
        env.reset(seed=0)
        env.target_position = env._position + np.array([0.3, -0.2, 1.0])
        env._update_vision(force=True)
        self.assertGreater(env.vision_error[0], 0.0)
        self.assertLess(env.vision_error[1], 0.0)
        self.assertTrue(np.all(np.abs(env.vision_error) <= 1.0))

    def test_target_outside_field_of_view_is_flagged(self) -> None:
        env = self.make()
        env.reset(seed=0)
        env.target_position = env._position + np.array([2.0, 0.0, 0.1])
        env._update_vision(force=True)
        self.assertFalse(env._target_visible)
        self.assertAlmostEqual(abs(env.vision_error[0]), 1.0, places=6)

    def test_vision_updates_at_thirty_hertz(self) -> None:
        env = self.make(arena_ceiling=1e6)
        env.reset(seed=0)
        env.target_position = env._position + np.array([0.5, 0.0, 1.0])
        updates = 0
        previous = env.vision_error.copy()
        for _ in range(200):  # one second of physics at 200 Hz
            env.step([1.0, 0.0, 0.0])  # climb so the camera error keeps moving
            if not np.allclose(previous, env.vision_error):
                updates += 1
                previous = env.vision_error.copy()
        self.assertGreaterEqual(updates, 28)
        self.assertLessEqual(updates, 31)

    def test_vision_delay_reports_the_previous_frame(self) -> None:
        env = self.make()
        env.reset(seed=0)
        env._vision_delay_frames = 1
        env.target_position = env._position + np.array([0.3, 0.0, 1.0])
        env._update_vision(force=True)
        env.target_position = env._position + np.array([-0.3, 0.0, 1.0])
        env._update_vision(force=True)
        self.assertGreater(env.vision_error[0], 0.0)  # still the older +x frame
        env._update_vision(force=True)
        self.assertLess(env.vision_error[0], 0.0)  # the -x frame finally arrives

    def test_gyro_noise_disabled_without_randomization(self) -> None:
        env = self.make()
        env.reset(seed=0)
        env.step([HOVER_ACTION, 0.0, 0.0])
        np.testing.assert_allclose(env.gyro_measurement, env._angular_velocity, atol=1e-12)

    def test_gyro_noise_matches_requested_sigma(self) -> None:
        env = DuctedTurbineEnv(config=DuctedTurbineConfig())
        self.addCleanup(env.close)
        env.reset(seed=0)
        errors = []
        for _ in range(500):
            env.step([HOVER_ACTION, 0.0, 0.0])
            errors.append(env.gyro_measurement - env._angular_velocity)
        sigma = float(np.std(np.asarray(errors)))
        self.assertAlmostEqual(sigma, 0.02, delta=0.005)


class Reproducibility(unittest.TestCase):
    def test_same_seed_gives_same_trajectory(self) -> None:
        actions = [[0.3, 0.2, -0.4], [0.9, -0.5, 0.1], [0.0, 0.0, 0.0]] * 20
        trajectories = []
        for _ in range(2):
            env = DuctedTurbineEnv(config=DuctedTurbineConfig())
            self.addCleanup(env.close)
            observation, _ = env.reset(seed=7)
            rollout = [observation]
            for action in actions:
                observation, _, terminated, truncated, _ = env.step(action)
                rollout.append(observation)
                if terminated or truncated:
                    break
            trajectories.append(np.asarray(rollout))
        np.testing.assert_allclose(trajectories[0], trajectories[1])

    def test_different_seeds_randomize_the_episode(self) -> None:
        env = DuctedTurbineEnv(config=DuctedTurbineConfig())
        self.addCleanup(env.close)
        _, first = env.reset(seed=1)
        _, second = env.reset(seed=2)
        self.assertNotAlmostEqual(first["mass"], second["mass"])
        self.assertFalse(
            np.allclose(first["target_position"], second["target_position"])
        )


class Rendering(unittest.TestCase):
    def test_onboard_camera_returns_image(self) -> None:
        env = DuctedTurbineEnv(render_mode="rgb_array", config=deterministic_config())
        self.addCleanup(env.close)
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.shape, (96, 96, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_render_returns_none_in_headless_mode(self) -> None:
        env = DuctedTurbineEnv(config=deterministic_config())
        self.addCleanup(env.close)
        env.reset(seed=0)
        self.assertIsNone(env.render())


if __name__ == "__main__":
    unittest.main()
