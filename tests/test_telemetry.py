import time

import torch

from pretrain.telemetry import PhaseTimer, format_phases, peak_memory_gib


def test_phase_timer_attributes_spans_to_the_following_mark():
    timer = PhaseTimer(torch.device('cpu'))
    time.sleep(0.02)
    timer.mark('slow')
    time.sleep(0.001)
    timer.mark('fast')
    totals = timer.drain()
    assert totals['slow'] > totals['fast']
    assert totals['slow'] >= 15


def test_phase_timer_accumulates_repeats_within_a_window():
    timer = PhaseTimer(torch.device('cpu'))
    for _ in range(3):
        time.sleep(0.005)
        timer.mark('fwd')
    once = PhaseTimer(torch.device('cpu'))
    time.sleep(0.005)
    once.mark('fwd')
    assert timer.drain()['fwd'] > once.drain()['fwd'] * 2


def test_drain_resets_so_windows_do_not_double_count():
    timer = PhaseTimer(torch.device('cpu'))
    time.sleep(0.005)
    timer.mark('fwd')
    timer.drain()
    assert timer.drain() == {}


def test_format_phases_reports_per_step_means_longest_first():
    assert format_phases({'fwd': 100.0, 'bwd': 300.0}, 10) == "bwd 30 fwd 10"
    assert format_phases({}, 10) == ""
    assert format_phases({'fwd': 1.0}, 0) == ""


def test_peak_memory_is_none_off_cuda():
    assert peak_memory_gib(torch.device('cpu')) is None
