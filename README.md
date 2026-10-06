# Ducted micro-turbine interceptor simulation

A Gymnasium environment (`DuctedTurbineEnv`) for an 83 g ducted-fan interceptor
drone, built on PyBullet and trainable with Stable-Baselines3 PPO.

PyBullet integrates the rigid body at 200 Hz; every aerodynamic effect (ducted
thrust, slipstream vane forces, rotor reaction torque, body drag, rate damping,
wind gusts) is computed analytically and injected as an external force or
torque.

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Use

```powershell
# train a quick policy (headless, 4 parallel envs)
.\.venv\Scripts\python.exe ducted_turbine_env.py --mode train --timesteps 10000

# a policy that actually intercepts needs more samples and a 50 Hz control rate
.\.venv\Scripts\python.exe ducted_turbine_env.py --mode train --timesteps 600000 --n-envs 8 --decimation 4

# watch the trained policy in the PyBullet GUI, paced to real time
.\.venv\Scripts\python.exe ducted_turbine_env.py --mode play --decimation 4 --episodes 5

# sanity checks
.\.venv\Scripts\python.exe ducted_turbine_env.py --mode check     # gymnasium env checker
.\.venv\Scripts\python.exe ducted_turbine_env.py --mode bench     # throughput
.\.venv\Scripts\python.exe -m unittest -v test_ducted_turbine_env  # 41 unit tests
```

`--mode train` saves `ducted_turbine_ppo.zip` plus the matching
`ducted_turbine_ppo_vecnormalize.pkl` observation/reward statistics; `--mode
play` loads both. Use the same `--decimation` for playback as for training.

Headless throughput is roughly 6,500 control steps/s on one core (≈32× real
time); the GUI runs at a few hundred steps/s. A 600 k-step run on 8 parallel
environments takes about 7 minutes and reaches a ~64 % intercept rate.

## Environment

| | |
|---|---|
| Action | `Box(-1, 1, (3,))` → motor power `[0, 1]`, pitch vane ±30°, yaw vane ±30° |
| Observation | `Box((15,))` → target offset (3), world velocity (3), Euler angles (3), body rates (3), camera error (2), motor level (1) |
| Physics | 200 Hz fixed step; policy rate = 200 Hz / `control_decimation` |
| Episode | 10 s, or termination on intercept, crash, tumble, or fly-away |

Reward per step (spec terms):

```
-1.0 * distance                       intercept bonus  +100  (distance < 0.1 m)
+0.8 * (velocity . line_of_sight)     crash penalty     -50  (tilt > 75°, ground, tumble)
-0.2 * (wx^2 + wy^2) - 0.5 * wz^2     fly-away penalty  -50  (outside the arena)
-0.01 * ||a_t - a_{t-1}||^2
```

Body rates are divided by `spin_reference_rate` (10 rad/s) before the quadratic
spin penalty, and the four dense terms are clipped to ±20 per step
(`reward_clip`) so a single tumbling step cannot swamp the return. Terminal
bonuses are added afterwards and are never clipped. `info["reward_terms"]`
breaks the reward down term by term.

Domain randomization on `reset()`: mass ±8 %, thrust ±5 %, vane effectiveness
±10 %, gyro noise σ = 0.02 rad/s, attitude noise σ = 0.01 rad, wind gusts of
0.2–0.5 N at 0.25–1.0 s intervals, 0 or 1 frame of vision latency, plus jitter
on the start pose and target position. Set `randomize=False` for a
deterministic environment (used by the tests).

## Physics notes and deviations from the original specification

These are deliberate changes; each one is a configuration field, so the
original behaviour can be restored.

* **Vane forces act at the duct exit plane**, one half-height (38 mm) below the
  center of mass, which is what produces the steering moment. Applying them at
  the center of mass (as the first implementation did) gives a drone with zero
  attitude authority.
* **Vane lift** is `q · S · C_Lα · sin(δ) · η` with `δ = action · 30°` and
  `q = T / A_exit` from momentum theory, so the ±30° deflection limit is
  actually used.
* **Servos** have a 15 ms transport delay (3 physics steps) *and* a 600°/s slew
  rate, which is what a micro servo of this class can do.
* **Stator torque recovery (`stator_recovery = 0.95`).** The specified rotor
  reaction torque, 0.004 N·m per N of thrust, is 0.0236 N·m at full thrust.
  Against `I_zz = 5e-5 kg·m²` that is 470 rad/s² of yaw acceleration that two
  orthogonal plume vanes cannot trim out — the airframe tumbles within 0.1 s
  and no policy can learn anything. Real ducted fans recover most of the rotor
  torque in the duct stator, so only the remaining 5 % is applied. Set
  `stator_recovery=0.0` to reproduce the raw specification.
* **Aerodynamic rate damping** about all three axes, scaled with jet dynamic
  pressure, bounds the yaw drift to about 1.5 rad/s in hover.
* **Normalized spin penalty (`spin_reference_rate = 10 rad/s`).** Any useful
  maneuver on this airframe reaches 5–10 rad/s, where the literal
  `-0.2·(ωx²+ωy²) − 0.5·ωz²` is roughly 100× the distance term. The reward then
  ranks "do not rotate, fall out of the sky" above "intercept the target", and
  training plateaus at 0.4 s episodes with a 5 % hit rate. Normalizing by a
  reference rate keeps the terms in the order the specification intends; with
  it the same run reaches a 64 % hit rate. Set `spin_reference_rate=1.0` for the
  literal weights.
* **Body drag** (`Cd = 0.8` on the duct silhouette) replaces PyBullet's
  artificial linear damping, so velocities stay bounded without a fake force.
* **Body rates are part of the observation** (15 elements instead of 12). The
  specification simulates a gyro but never showed it to the policy; a
  memoryless MLP cannot damp an attitude-unstable vehicle without rate
  feedback. Set `include_gyro=False` for the strict 12-element layout.
* **Extra terminations**: tumble (|ω| > 40 rad/s) and fly-away (3 m horizontal
  or 4 m ceiling). Without them a diverging episode accumulates thousands of
  reward units and destabilizes training.
* **Mass randomization also scales the inertia tensor**, instead of leaving a
  randomized mass with nominal inertia.
* **The camera** is a pinhole model on the body +Z axis with a 90° field of
  view; the error saturates at ±1 and `info["target_visible"]` reports whether
  the target is actually in frame. `render_mode="rgb_array"` returns the onboard
  camera image.

## Files

| File | Purpose |
|---|---|
| `ducted_turbine_env.py` | environment, PPO recipe, and the `train`/`play`/`bench`/`check` CLI |
| `test_ducted_turbine_env.py` | 41 unit tests (stdlib `unittest`) covering physics signs, delays, sensors, terminations, determinism |
| `requirements.txt` | gymnasium, numpy, pybullet, stable-baselines3 |
