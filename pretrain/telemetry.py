"""Per-phase step timing for the training loop.

CUDA events keep the measurement off the critical path; totals are read once
per logging interval, so the one synchronize() that costs anything happens
every -printevery steps rather than every phase.
"""
import time

import torch


class PhaseTimer:
    """Attributes the time between marks to named phases.

    Call mark(name) at each phase boundary — the span since the previous mark
    is credited to `name` — then drain() at log time for the totals in ms.
    """

    def __init__(self, device):
        self.cuda = device.type == 'cuda'
        self.spans = []
        self.last = self._now()

    def _now(self):
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def mark(self, name):
        now = self._now()
        self.spans.append((name, self.last, now))
        self.last = now

    def drain(self):
        """Totals in ms since the last drain. Syncs once, then resets."""
        if self.cuda:
            torch.cuda.synchronize()
        totals = {}
        for name, start, end in self.spans:
            ms = start.elapsed_time(end) if self.cuda else (end - start) * 1000
            totals[name] = totals.get(name, 0.0) + ms
        self.spans.clear()
        self.last = self._now()
        return totals


def format_phases(totals, steps):
    """'fwd 210 bwd 340 red 8 opt 48' — per-step means, longest phase first."""
    if not totals or not steps:
        return ""
    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    return " ".join(f"{name} {ms / steps:.0f}" for name, ms in ordered)


def peak_memory_gib(device):
    if device.type != 'cuda':
        return None
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    torch.cuda.reset_peak_memory_stats(device)
    return peak
