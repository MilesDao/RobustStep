"""Quick check of the G1 environment: Gymnasium API, contacts, control authority, random actions do not blow up the physics.

Note: with the G1's realistic PD gains (kp_scale=1), action = 0 can NOT stand by itself. That is correct physical behavior
(inverted pendulum, ankle kp 20 is too soft); the policy must learn to balance. So the standing test
uses a simple ankle feedback controller with kp_scale=4 to confirm the actuators have enough control authority.
"""
import time

import numpy as np
from stable_baselines3.common.env_checker import check_env

from biped_rl import terrains
from biped_rl.env import BipedEnv


def stand_with_ankle_feedback(kp_scale=4.0, seconds=8.0, k=10.0, kd=1.0):
    env = BipedEnv(randomize=False, on_reach="ignore", kp_scale=kp_scale)
    env.reset(seed=0)
    d = env.data
    for _ in range(int(seconds / env.dt)):
        R = d.xmat[env.torso_id].reshape(3, 3)
        a = np.zeros(env.nj)
        a[4] = a[10] = np.clip(k * R[0, 2] + kd * d.qvel[4], -1, 1)  # ankle_pitch of both legs
        _, _, term, _, _ = env.step(a)
        if term:
            return f"fell at t={d.time:.2f}s"
    return f"stood for {seconds:.0f}s"


def main():
    env = BipedEnv()
    check_env(env, warn=True)
    print("check_env: OK, obs_dim =", env.observation_space.shape[0], ", act_dim =", env.nj)
    print("G1 mass (kg):", round(env.total_mass, 2), "| policy dt:", env.dt, "s")

    env = BipedEnv(randomize=False, on_reach="ignore")
    env.reset(seed=0)
    print("after reset: both feet contact =", env._foot_contacts())
    for _ in range(10):
        env.step(np.zeros(env.nj))
    print("after 0.2s: both feet contact =", env._foot_contacts(), "(expected [1. 1.])")

    t = 0
    env.reset(seed=0)
    for i in range(500):
        _, _, term, _, _ = env.step(np.zeros(env.nj))
        if term:
            t = env.data.time
            break
    print(f"action = 0, kp_scale=1: {'fell at t=%.2fs (normal, see note at top of file)' % t if t else 'stood still'}")
    print("simple ankle feedback, kp_scale=4:", stand_with_ankle_feedback())

    rng = np.random.default_rng(0)
    env = BipedEnv(push_max=40.0, roughness=0.05)
    n, t0, nan_found, falls = 0, time.time(), False, 0
    for _ in range(5):
        env.reset()
        for _ in range(env.max_steps):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, env.nj))
            n += 1
            if not np.all(np.isfinite(obs)) or not np.isfinite(r):
                nan_found = True
            if term or trunc:
                falls += int(term)
                break
    sps = n / (time.time() - t0)
    print(f"random actions: {n} steps, {sps:.0f} steps/s (1 process), NaN={nan_found}, fell {falls}/5 episodes")
    course_check()


def course_check():
    """Builds every challenge at several difficulty levels, checks the terrain is valid and the robot is not counted as fallen right at the start."""
    env = BipedEnv(scene="course", height_scan=True, randomize=False)
    problems = 0
    for kind in terrains.KINDS + ["mixed"]:
        for lv in (0.0, 0.5, 1.0):
            env.reset(seed=1, options={"challenge": kind, "challenge_level": lv})
            c = env._course
            flat_start = float(np.ptp(env._ground_z_vec(np.linspace(c["start_x"] - 1, c["start_x"] + 1, 20), np.zeros(20))))
            ok = bool(c["segments"]) and flat_start < 1e-6 and np.all(np.isfinite(env._hf_array()))
            for _ in range(10):
                _, _, term, _, _ = env.step(np.zeros(env.nj))
                ok &= not term
            problems += int(not ok)
            if not ok:
                print(f"  problem at {kind} level {lv}: segments={c['segments']}, flat start ground={flat_start < 1e-6}")
    print(f"scene course: {len(terrains.KINDS) + 1} kinds x 3 difficulty levels, configs with problems = {problems}")


if __name__ == "__main__":
    main()
