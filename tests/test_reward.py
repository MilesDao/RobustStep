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
