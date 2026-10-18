import numpy as np
from biped_rl.env import BipedEnv
from render_video import make_renderer, rollout_frames

def test_rollout_frames_fall_and_no_fall():
    env = BipedEnv(randomize=False)
    obs, _ = env.reset(seed=0)
    r, cam = make_renderer(env, 64, 48)
    frames, info = rollout_frames(env, lambda o: np.zeros(12), obs, r, cam, seconds=5.0, fps=10)
    assert info["fell"] and info["t"] < 5.0                  # zero action falls (~1.2 s)
    assert len(frames) >= 1 and frames[0][1].shape == (48, 64, 3)
    obs, _ = env.reset(seed=0)
    frames, info = rollout_frames(env, lambda o: np.zeros(12), obs, r, cam, seconds=0.2, fps=10)
    assert not info["fell"] and len(frames) == 3             # t=0 + 2 frames in 0.2 s
