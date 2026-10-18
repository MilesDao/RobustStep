import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from biped_env import BipedEnv, DEFAULT_REWARD_SCALES

def make(**kw):
    env = BipedEnv(randomize=False, **kw)
    env.reset(seed=0)
    return env

def test_default_target_speed_is_one():
    assert make().target_speed == 1.0

def test_obs_shape_unchanged():
    assert make().observation_space.shape == (50,)
    assert make(height_scan=True).observation_space.shape == (85,)

def test_set_difficulty_speed():
    env = make()
    env.set_difficulty(10.0, 0.02, 0.3)
    assert (env.push_max, env.roughness, env.target_speed) == (10.0, 0.02, 0.3)
    env.set_difficulty(0.0, 0.0)          # speed omitted -> unchanged
    assert env.target_speed == 0.3

def test_alive_default():
    assert DEFAULT_REWARD_SCALES["alive"] == 0.5

def test_air_time_rewards_long_steps_only_when_walking():
    env = make()
    env.target_speed = 1.0
    env._air_time = np.zeros(2)
    env._air_time_reward(np.array([1.0, 1.0]))            # both feet have touched ground once
    for _ in range(25):                                   # left foot airborne 0.5 s
        assert env._air_time_reward(np.array([0.0, 1.0])) == 0.0
    assert np.isclose(env._air_time_reward(np.array([1.0, 1.0])), 0.5 - 0.4)
    assert env._air_time_reward(np.array([1.0, 1.0])) == 0.0   # no new touchdown

def test_no_air_time_penalty_on_initial_landing():
    env = make()
    env.target_speed = 1.0
    assert env._air_time_reward(np.array([0.0, 0.0])) == 0.0
    assert env._air_time_reward(np.array([1.0, 1.0])) == 0.0

def test_air_time_zero_when_standing():
    env = make()
    env.target_speed = 0.0
    env._air_time = np.zeros(2)
    for _ in range(25):
        env._air_time_reward(np.array([0.0, 1.0]))
    assert env._air_time_reward(np.array([1.0, 1.0])) == 0.0

def test_slip_only_counts_feet_in_contact():
    env = make()
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert env._slip_penalty(np.array([1.0, 0.0]), v) == 1.0
    assert env._slip_penalty(np.array([1.0, 1.0]), v) == 5.0
    assert env._slip_penalty(np.array([0.0, 0.0]), v) == 0.0

def test_first_step_reward_finite_and_small_slip():
    env = make()
    for _ in range(3):                                    # episodes far apart in xy must not leak
        env.reset(seed=1)
        _, r, *_ , info = env.step(np.zeros(12))
        assert np.isfinite(r) and info["slip"] < 1.0

def test_new_scales_overridable():
    BipedEnv(randomize=False, reward_scales={"air_time": 0.0, "slip": 0.0})

def test_random_actions_stay_finite():
    env, rng = make(), np.random.default_rng(0)
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 12))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            env.reset()
