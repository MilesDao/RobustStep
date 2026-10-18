import numpy as np
import progress_video
from progress_video import draw_overlay, card, save_chart, stage_text, chain_offsets

def test_draw_overlay_keeps_shape_and_changes_pixels():
    img = np.zeros((72, 128, 3), np.uint8)
    out = draw_overlay(img, ["Attempt #1"], banner="FELL at 1.6 s", clock_red=True)
    assert out.shape == img.shape and out.dtype == np.uint8 and out.any() and not img.any()

def test_card_size():
    assert len(card("Hi", 128, 72, 1.0, 10)) == 10

def test_save_chart_writes_png(tmp_path):
    from PIL import Image
    out = tmp_path / "chart.png"
    save_chart([1, 10, 100], [1.2, 3.0, 20.0], str(out))
    assert Image.open(out).size == (1600, 900)

def test_stage_text():
    assert stage_text("field", 0, None) == "Stand still"
    assert stage_text("field", 1, None) == "Walk 0.3 m/s"
    assert stage_text("field", 5, None) == "Pushes up to 25 N·s, rough 2 cm"
    assert stage_text("course", 0, "stairs").startswith("Stairs: step height (cm) 4.0")
    assert stage_text("field", None, None) == "Stage: unknown"

def test_offsets_chain_runs(monkeypatch):
    monkeypatch.setattr(progress_video, "attempts_at", lambda run, steps: ({"A": 500}[run], True))
    assert chain_offsets(["A", "B"], [1_000_000, 300_000]) == [(0, 0), (500, 1_000_000)]

def test_run_bases_detects_resumed_counter():
    from biped_rl.progress_data import Checkpoint
    ck = lambda s: Checkpoint(s, "", "")
    parent, resumed, fresh = [ck(200), ck(1000)], [ck(1200), ck(1400)], [ck(200)]
    assert progress_video.run_bases([parent, resumed]) == [0, 1000]
    assert progress_video.run_bases([parent, fresh]) == [0, 0]
