from torch.utils.tensorboard import SummaryWriter
from biped_rl.progress_data import list_checkpoints, attempts_at, scalar_at, format_steps

def test_list_checkpoints_sorted_numerically(tmp_path):
    for n in (999984, 249996, 1249980):
        (tmp_path / f"ckpt_{n}_steps.zip").write_bytes(b"")
        (tmp_path / f"ckpt_vecnormalize_{n}_steps.pkl").write_bytes(b"")
    assert [c.steps for c in list_checkpoints(str(tmp_path))] == [249996, 999984, 1249980]

def test_list_checkpoints_skips_missing_stats(tmp_path):
    (tmp_path / "ckpt_100_steps.zip").write_bytes(b"")
    assert list_checkpoints(str(tmp_path)) == []

def test_attempts_exact_from_monitor(tmp_path):
    for rank, ls in ((0, [10, 20, 30]), (1, [60])):
        rows = "\n".join(f"0,{l},0,1,0,0" for l in ls)
        (tmp_path / f"monitor_{rank}.monitor.csv").write_text('#{"t_start": 0}\nr,l,t,fell,n_reached,path\n' + rows + "\n")
    # 2 files, steps=60 -> 30 per env: env0 cumulative 10,30,60 -> 2 ; env1 cumulative 60 -> 0
    assert attempts_at(str(tmp_path), 60) == (2, True)

def test_attempts_falls_back_to_tensorboard(tmp_path):
    w = SummaryWriter(str(tmp_path / "tb" / "ppo_1"))
    w.add_scalar("rollout/ep_len_mean", 50.0, 1000); w.add_scalar("rollout/ep_len_mean", 100.0, 3000); w.close()
    assert attempts_at(str(tmp_path), 3000) == (40, False)    # 1000/50 + 2000/100

def test_scalar_at_last_point_before(tmp_path):
    w = SummaryWriter(str(tmp_path / "tb" / "ppo_1"))
    w.add_scalar("curriculum/level", 0, 100); w.add_scalar("curriculum/level", 2, 500); w.close()
    assert scalar_at(str(tmp_path), "curriculum/level", 499) == 0
    assert scalar_at(str(tmp_path), "curriculum/level", 500) == 2
    assert scalar_at(str(tmp_path), "curriculum/speed", 500) is None

def test_format_steps():
    assert [format_steps(s) for s in (249996, 1250000, 5000000)] == ["0.25M", "1.25M", "5M"]

def test_attempts_ignore_incomplete_monitor(tmp_path):
    # CSV overwritten by a resume: only 100 steps logged for a 3000-step checkpoint -> use TensorBoard
    (tmp_path / "monitor_0.monitor.csv").write_text('#{"t_start": 0}\nr,l,t,fell,n_reached,path\n0,100,0,1,0,0\n')
    w = SummaryWriter(str(tmp_path / "tb" / "ppo_1"))
    w.add_scalar("rollout/ep_len_mean", 50.0, 1000); w.add_scalar("rollout/ep_len_mean", 100.0, 3000); w.close()
    assert attempts_at(str(tmp_path), 3000) == (40, False)
