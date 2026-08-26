"""Tests for the self-directed curriculum: arm shards and the director policy."""
import json
import os
import random
import struct
import sys
from unittest import mock

import numpy as np
import pytest
import torch

from core.data import load_bin
from core.model import load_checkpoint, model_from_config
from core.tokenizer import BPETokenizer
from selfdirect.director import Director
from selfdirect.domains import build_arm, discover_arms
from selfdirect.loop import JOURNAL_FILE, Arm, short_names, take_block
from selfdirect.loop import main as loop_main
from selfdirect.report import read_journal, summarize

CORPUS = ("the quick brown fox jumps over the lazy dog "
          "hello world hello there hello friends " * 30)


def _tokenizer():
    tok = BPETokenizer(special_tokens={'<|endoftext|>': 511})
    tok.train(CORPUS, vocab_size=400)
    return tok, 511


def _write_fetch(path, texts):
    with open(path, 'wb') as f:
        for t in texts:
            b = t.encode('utf-8')
            f.write(struct.pack('<Q', len(b)))
            f.write(b)


def test_discover_arms_finds_fetch_caches(tmp_path):
    _write_fetch(tmp_path / 'fetch_alpha.bin', ['a'])
    _write_fetch(tmp_path / 'fetch_beta.bin', ['b'])
    (tmp_path / 'notes.txt').write_text('ignore me')
    assert sorted(discover_arms(str(tmp_path))) == ['alpha', 'beta']


def test_build_arm_routes_every_nth_doc_to_the_probe(tmp_path):
    tok, eos = _tokenizer()
    texts = [f"hello world number {i}" for i in range(10)]
    _write_fetch(tmp_path / 'fetch_x.bin', texts)

    n_train, n_probe = build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'),
                                 tok, eos, probe_tokens=10**6, probe_period=2)

    total = sum(len(tok.encode_ordinary(t)) + 1 for t in texts)
    assert n_train + n_probe == total
    # Docs 0,2,4,6,8 are held out — half the docs, so roughly half the tokens.
    assert n_probe == sum(len(tok.encode_ordinary(t)) + 1 for t in texts[::2])
    assert len(load_bin(str(tmp_path / 'x' / 'probe.bin'))) == n_probe
    assert len(load_bin(str(tmp_path / 'x' / 'train.bin'))) == n_train


def test_build_arm_stops_holding_out_once_the_probe_budget_is_met(tmp_path):
    tok, eos = _tokenizer()
    texts = [f"hello world number {i}" for i in range(20)]
    _write_fetch(tmp_path / 'fetch_x.bin', texts)

    n_train, n_probe = build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'),
                                 tok, eos, probe_tokens=6, probe_period=1)

    # probe_period=1 would hold out everything; the budget cuts it off after
    # the first doc that crosses 6 tokens.
    assert 6 <= n_probe < 6 + len(tok.encode_ordinary(texts[0])) + 1
    assert n_train > n_probe


def test_build_arm_max_docs_caps_the_read(tmp_path):
    tok, eos = _tokenizer()
    _write_fetch(tmp_path / 'fetch_x.bin', [f"doc {i}" for i in range(50)])

    n_train, n_probe = build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'),
                                 tok, eos, probe_tokens=0, probe_period=25, max_docs=3)

    assert n_probe == 0
    assert n_train == sum(len(tok.encode_ordinary(f"doc {i}")) + 1 for i in range(3))


def test_arm_shards_are_uint16_token_ids(tmp_path):
    tok, eos = _tokenizer()
    _write_fetch(tmp_path / 'fetch_x.bin', ["hello world", "the quick brown fox"])
    build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'), tok, eos,
              probe_tokens=10**6, probe_period=2)

    train = load_bin(str(tmp_path / 'x' / 'train.bin'))
    assert train.dtype == np.uint16
    assert train[-1] == eos  # every doc is EOS-terminated


# --- director -------------------------------------------------------------

def _bandit(director, best_of, rounds, seed=0, noise=0.02, switch=None):
    """Drive a director with a synthetic reward: `best_of(round)` names the arm
    worth 0.10 and every other arm 0.02. Returns the final probabilities."""
    rng = random.Random(seed)
    for r in range(rounds):
        i = director.choose()
        reward = (0.10 if i == best_of(r) else 0.02) + rng.gauss(0, noise)
        director.update(i, reward)
    return director.probs()


def test_probs_are_a_distribution_with_an_exploration_floor():
    d = Director(list('abcde'), explore=0.1)
    d.log_w = [10.0, -10.0, -10.0, -10.0, -10.0]
    probs = d.probs()
    assert abs(sum(probs) - 1.0) < 1e-12
    assert min(probs) >= 0.1 / 5 - 1e-12   # no arm is ever starved to zero
    assert probs[0] == max(probs)


def test_director_concentrates_on_the_arm_that_pays():
    probs = _bandit(Director(list('abcde'), seed=1), lambda r: 0, 300)
    assert probs[0] > 0.7
    assert probs[0] > 5 * max(probs[1:])


def test_director_follows_a_moving_optimum():
    d = Director(list('abcde'), seed=1)
    _bandit(d, lambda r: 0, 200)
    assert d.probs()[0] > 0.7
    probs = _bandit(d, lambda r: 4, 300, seed=7)
    assert probs[4] > 0.7 and probs[0] < 0.2


def test_decay_bounds_drift_when_the_reward_is_pure_noise():
    """Without decay the log-weights are an unbiased random walk, so a long
    enough run drifts into a strong preference on no signal at all."""
    def noise_run(decay):
        d = Director(list('abcde'), decay=decay, seed=3)
        rng = random.Random(11)
        for _ in range(3000):
            d.update(d.choose(), rng.gauss(0, 0.01))
        return max(d.probs())

    assert noise_run(0.01) < noise_run(0.0)
    assert noise_run(0.01) < 0.5


def test_rescale_ranks_against_the_recent_window():
    d = Director(list('ab'))
    assert d.rescale(0.5) == 0.0          # empty window carries no information
    d.recent.extend([0.0, 1.0, 2.0, 3.0])
    assert d.rescale(9.0) == 1.0
    assert d.rescale(-9.0) == -1.0
    assert d.rescale(1.5) == 0.0          # exactly the median


def test_state_dict_round_trips_including_the_rng():
    a = Director(list('abcde'), seed=5)
    _bandit(a, lambda r: 2, 50)
    b = Director(list('abcde'), seed=999)
    b.load_state_dict(a.state_dict())
    assert b.probs() == a.probs()
    assert [b.choose() for _ in range(20)] == [a.choose() for _ in range(20)]


# --- loop -----------------------------------------------------------------

def _arm(tokens):
    return Arm('x', np.arange(tokens, dtype=np.uint16), np.zeros(4, dtype=np.uint16))


def test_take_block_reads_forward_across_rounds():
    arm = _arm(100)
    assert list(take_block(arm, 30)) == list(range(30))
    assert list(take_block(arm, 30)) == list(range(30, 60))
    assert arm.cursor == 60


def test_take_block_wraps_at_the_end_of_a_shard():
    arm = _arm(100)
    arm.cursor = 90
    assert list(take_block(arm, 20)) == list(range(90, 100)) + list(range(10))
    assert arm.cursor == 10


def test_take_block_handles_a_block_longer_than_the_shard():
    arm = _arm(10)
    assert list(take_block(arm, 25)) == (list(range(10)) * 3)[:25]
    assert arm.cursor == 5


def test_short_names_keeps_arms_distinguishable():
    assert short_names(['finemath', 'fineweb-edu', 'cosmopedia']) == \
        ['finem', 'finew', 'cosmo']
    assert short_names(['a', 'b']) == ['a', 'b']


# --- report ---------------------------------------------------------------

def _journal(tmp_path, lines):
    (tmp_path / JOURNAL_FILE).write_text(''.join(json.dumps(l) + '\n' for l in lines))
    return str(tmp_path)


def test_summarize_counts_choices_and_tracks_forgetting(tmp_path):
    probs = {'a': 0.7, 'b': 0.3}
    lines = [
        {'round': 1, 'step': 10, 'studied': 'a', 'seconds': 1.0, 'probs': probs,
         'probe_before': {'a': 4.0, 'b': 5.0}, 'probe_after': {'a': 3.0, 'b': 4.5}},
        {'round': 2, 'step': 20, 'studied': 'a', 'seconds': 1.0, 'probs': probs,
         'probe_before': {'a': 3.0, 'b': 4.5}, 'probe_after': {'a': 2.5, 'b': 4.9}},
    ]
    rows = {r['arm']: r for r in summarize(read_journal(_journal(tmp_path, lines)))}

    assert rows['a']['rounds'] == 2 and rows['b']['rounds'] == 0
    assert rows['a']['start'] == 4.0 and rows['a']['now'] == 2.5
    # b was never studied and its probe loss rose back off its best — forgetting.
    assert rows['b']['best'] == 4.5
    assert rows['b']['forgotten'] == pytest.approx(0.4)
    assert rows['a']['forgotten'] == 0.0


def test_report_rows_are_ordered_by_final_weight(tmp_path):
    lines = [{'round': 1, 'step': 1, 'studied': 'b', 'seconds': 1.0,
              'probs': {'a': 0.2, 'b': 0.5, 'c': 0.3},
              'probe_before': {'a': 1.0, 'b': 1.0, 'c': 1.0},
              'probe_after': {'a': 1.0, 'b': 1.0, 'c': 1.0}}]
    rows = summarize(read_journal(_journal(tmp_path, lines)))
    assert [r['arm'] for r in rows] == ['b', 'c', 'a']


def test_eta_zero_pins_the_director_to_a_uniform_mixture():
    """--eta 0 is the control for 'did the curriculum help', so it has to stay
    exactly uniform no matter what rewards arrive."""
    d = Director(list('abcde'), eta=0.0, seed=4)
    _bandit(d, lambda r: 0, 200)
    assert d.probs() == pytest.approx([0.2] * 5)


# --- end to end -----------------------------------------------------------

def _fake_arm_dir(tmp_path, names, vocab=64, tokens=4096, probe=512):
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        d = tmp_path / name
        d.mkdir(parents=True)
        # Each arm gets its own slice of the vocab, so the arms are genuinely
        # different distributions and their probe losses can diverge.
        lo = 1 + i * 8
        rng.integers(lo, lo + 8, tokens, dtype=np.uint16).tofile(d / 'train.bin')
        rng.integers(lo, lo + 8, probe, dtype=np.uint16).tofile(d / 'probe.bin')
    (tmp_path / 'arms.json').write_text(json.dumps({
        'vocab_size': vocab, 'eos_id': vocab - 1,
        'arms': [{'name': n, 'train_tokens': tokens, 'probe_tokens': probe}
                 for n in names]}))
    return str(tmp_path)


def _run_loop(data_dir, out, rounds, extra=()):
    argv = ['selfdirect.loop', '--data-dir', data_dir, '--out', out,
            '--rounds', str(rounds), '--block-steps', '2', '--batchsize', '2',
            '--seqlen', '32', '--probe-tokens', '256', '--warmup-steps', '2',
            '--save-every', '0', '--no-cuda', '--muon-impl', 'torch',
            '-d_model', '16', '-n_layers', '1', '-heads', '2', *extra]
    with mock.patch.object(sys, 'argv', argv):
        loop_main()


def test_loop_runs_end_to_end_and_journals_every_round(tmp_path):
    data = _fake_arm_dir(tmp_path / 'arms', ['alpha', 'beta'])
    out = str(tmp_path / 'run')
    _run_loop(data, out, rounds=3)

    rounds = read_journal(out)
    assert [r['round'] for r in rounds] == [1, 2, 3]
    assert all(r['studied'] in ('alpha', 'beta') for r in rounds)
    # Each round's "before" is the previous round's "after" — the probe is
    # measured once per round, not twice.
    assert rounds[1]['probe_before'] == rounds[0]['probe_after']
    assert os.path.exists(os.path.join(out, 'state.pt'))


def test_loop_resumes_where_it_stopped(tmp_path):
    data = _fake_arm_dir(tmp_path / 'arms', ['alpha', 'beta'])
    out = str(tmp_path / 'run')
    _run_loop(data, out, rounds=2)
    # No -d_model on the resume: the architecture comes back out of state.pt.
    argv = ['selfdirect.loop', '--data-dir', data, '--out', out, '--rounds', '2',
            '--block-steps', '2', '--batchsize', '2', '--seqlen', '32',
            '--probe-tokens', '256', '--warmup-steps', '2', '--save-every', '0',
            '--no-cuda', '--muon-impl', 'torch', '--resume']
    with mock.patch.object(sys, 'argv', argv):
        loop_main()

    rounds = read_journal(out)
    assert [r['round'] for r in rounds] == [1, 2, 3, 4]
    assert rounds[3]['step'] == 8          # 4 rounds x 2 steps, counted through


def test_loop_state_is_a_loadable_training_checkpoint(tmp_path):
    data = _fake_arm_dir(tmp_path / 'arms', ['alpha', 'beta'])
    out = str(tmp_path / 'run')
    _run_loop(data, out, rounds=1)

    ckpt = load_checkpoint(os.path.join(out, 'state.pt'))
    model = model_from_config(ckpt['config'], torch.device('cpu'))
    model.load_state_dict(ckpt['model'])   # what inference/sample.py does
    assert ckpt['config']['vocab_size'] == 64
