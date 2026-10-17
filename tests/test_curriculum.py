import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_sb3 import LEVELS_FIELD, LEVELS_COURSE, CurriculumCallback


class FakeVecEnv:
    def __init__(self): self.calls = []
    def env_method(self, name, *args): self.calls.append((name, *args))


def test_levels_are_triples():
    assert all(len(l) == 3 for l in LEVELS_FIELD + LEVELS_COURSE)


def test_field_starts_standing_and_speed_never_drops():
    speeds = [l[2] for l in LEVELS_FIELD]
    assert speeds[0] == 0.0 and speeds == sorted(speeds) and speeds[-1] == 1.0


def test_apply_passes_speed():
    cb = CurriculumCallback(LEVELS_FIELD, start_level=1, verbose=0)
    fake = FakeVecEnv()
    cb.model = types.SimpleNamespace(get_env=lambda: fake)
    cb._apply()
    assert fake.calls == [("set_difficulty", 0.0, 0.0, 0.3)]
