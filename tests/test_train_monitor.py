import glob
import numpy as np
from train import make_env

def test_monitor_csv_written(tmp_path):
    env = make_env(0, 0, {"randomize": False}, log_dir=str(tmp_path))()
    env.reset(seed=0)
    for _ in range(2000):                       # zero action falls in ~1-2 s
        *_, term, trunc, _ = env.step(np.zeros(12))
        if term or trunc:
            break
    env.close()
    files = glob.glob(str(tmp_path / "monitor_0*.csv"))
    assert len(files) == 1
    lines = open(files[0]).read().splitlines()
    assert lines[1] == "r,l,t,fell,n_reached,path" and len(lines) == 3

def _one_episode(env):
    env.reset(seed=0)
    for _ in range(2000):
        *_, term, trunc, _ = env.step(np.zeros(12))
        if term or trunc:
            break

def test_monitor_appends_when_resuming(tmp_path):
    for append in (False, True):
        env = make_env(0, 0, {"randomize": False}, log_dir=str(tmp_path), append=append)()
        _one_episode(env)
        env.close()
    lines = open(glob.glob(str(tmp_path / "monitor_0*.csv"))[0]).read().splitlines()
    assert sum(l.startswith("r,l,t") for l in lines) == 1 and len(lines) == 4
