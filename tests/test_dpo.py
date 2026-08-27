import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.model import Transformer


def _tiny(vocab=64, d_model=32, n_layers=2, heads=4):
    torch.manual_seed(0)
    return Transformer(vocab, d_model, n_layers, heads, dropout=0.0)


class FakeTok:
    def encode_ordinary(self, text):
        return [min(ord(c), 250) for c in text]


IM_START, IM_END, EOS = 300, 301, 302
ROW = {
    'chosen': [{'role': 'user', 'content': 'hi'},
               {'role': 'assistant', 'content': 'good'}],
    'rejected': [{'role': 'user', 'content': 'hi'},
                 {'role': 'assistant', 'content': 'bad'}],
}


def test_split_prompt_completion():
    from dpo.dpo_prepare import split_prompt_completion
    prompt, completion = split_prompt_completion(ROW['chosen'])
    assert [m['role'] for m in prompt] == ['user']
    assert completion == 'good'
    # no trailing assistant message / empty completion -> unusable
    assert split_prompt_completion([{'role': 'user', 'content': 'hi'}]) is None
    assert split_prompt_completion(
        [{'role': 'user', 'content': 'hi'},
         {'role': 'assistant', 'content': '  '}]) is None


def test_encode_pair_shares_prompt_and_masks_completion():
    from dpo.dpo_prepare import encode_pair
    c_ids, c_mask, r_ids, r_mask = encode_pair(FakeTok(), ROW, IM_START, IM_END, EOS)

    # prompt prefix (system + user + assistant header) is identical on both sides
    first_diff = next(i for i, (a, b) in enumerate(zip(c_ids, r_ids)) if a != b)
    assert c_ids[first_diff] == ord('g') and r_ids[first_diff] == ord('b')
    assert all(m == 0 for m in c_mask[:first_diff])
    assert all(m == 0 for m in r_mask[:first_diff])

    # loss only on completion body + im_end + eos
    assert sum(c_mask) == len('good') + 2
    assert sum(r_mask) == len('bad') + 2
    assert c_ids[-1] == EOS and c_mask[-1] == 1
    assert r_ids[-1] == EOS and r_mask[-1] == 1

    # malformed rows are dropped
    bad = {'chosen': ROW['chosen'], 'rejected': [{'role': 'user', 'content': 'hi'}]}
    assert encode_pair(FakeTok(), bad, IM_START, IM_END, EOS) is None


def _write_flat(tmp_path, seqs):
    """seqs: list of (ids, mask). Returns (tokens, masks, pairs) for one pair."""
    tokens = np.concatenate([np.asarray(s, dtype=np.uint16) for s, _ in seqs])
    masks = np.concatenate([np.asarray(m, dtype=np.uint8) for _, m in seqs])
    c_len = len(seqs[0][0])
    pairs = np.array([[0, c_len, c_len, len(seqs[1][0])]], dtype=np.int32)
    return tokens, masks, pairs


def test_build_batch_shapes_and_padding(tmp_path):
    from dpo.dpo import build_batch
    chosen = (list(range(10, 18)), [0, 0, 0, 0, 1, 1, 1, 1])       # 8 tokens
    rejected = (list(range(20, 25)), [0, 0, 1, 1, 1])              # 5 tokens
    tokens, masks, pairs = _write_flat(tmp_path, [chosen, rejected])

    ids_in, targets, loss_mask, attn = build_batch(
        tokens, masks, pairs, [0], max_len=16, pad_id=99,
        device=torch.device('cpu'))

    # chosen first, rejected second; T = len(chosen) - 1 = 7
    assert ids_in.shape == (2, 7)
    assert ids_in[0].tolist() == list(range(10, 17))
    assert targets[0].tolist() == list(range(11, 18))
    assert ids_in[1, :4].tolist() == list(range(20, 24))
    assert ids_in[1, 4:].tolist() == [99, 99, 99]                  # right padding
    assert loss_mask[0].tolist() == [False]*3 + [True]*4
    assert loss_mask[1].tolist() == [False] + [True]*3 + [False]*3
    # attn: causal AND not-pad — padded keys are never attended
    assert attn.shape == (2, 7, 7)
    assert attn[1, 0].tolist() == [True] + [False]*6
    assert attn[1, 6].tolist() == [True]*4 + [False]*3
    assert not attn[1, :, 4:].any()


def test_sequence_logprobs_padding_invariant():
    from dpo.dpo import build_batch, sequence_logprobs
    model = _tiny().eval()
    device = torch.device('cpu')
    chosen = (list(range(10, 18)), [0, 0, 0, 0, 1, 1, 1, 1])
    rejected = (list(range(20, 25)), [0, 0, 1, 1, 1])
    tokens, masks, pairs = _write_flat(None, [chosen, rejected])
    batch = build_batch(tokens, masks, pairs, [0], max_len=16, pad_id=63,
                        device=device)
    with torch.no_grad():
        padded = sequence_logprobs(model, *batch, device)
    assert padded.shape == (2,)

    def single(seq):
        ids, msk = seq
        ids_in = torch.tensor([ids[:-1]])
        targets = torch.tensor([ids[1:]])
        loss_mask = torch.tensor([msk[1:]], dtype=torch.bool)
        T = ids_in.size(1)
        attn = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0)
        with torch.no_grad():
            return sequence_logprobs(model, ids_in, targets, loss_mask,
                                     attn, device)[0]

    # same function, same autocast path, no padding -> must match the padded batch
    assert torch.allclose(padded[0], single(chosen), atol=1e-3)
    assert torch.allclose(padded[1], single(rejected), atol=1e-3)


def test_sequence_logprobs_is_mean_not_sum():
    """sequence_logprobs must be length-normalized: sum / completion-token count."""
    from dpo.dpo import build_batch, sequence_logprobs
    model = _tiny().eval()
    device = torch.device('cpu')
    chosen = (list(range(10, 18)), [0, 0, 0, 0, 1, 1, 1, 1])
    rejected = (list(range(20, 25)), [0, 0, 1, 1, 1])
    tokens, masks, pairs = _write_flat(None, [chosen, rejected])
    ids_in, targets, loss_mask, attn = build_batch(
        tokens, masks, pairs, [0], max_len=16, pad_id=63, device=device)

    with torch.no_grad():
        out = sequence_logprobs(model, ids_in, targets, loss_mask, attn, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(ids_in, attn)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    summed = (tok_logp * loss_mask).sum(dim=-1)
    counts = loss_mask.sum(dim=-1)

    assert (counts > 1).all()  # mean and sum genuinely differ here
    assert torch.allclose(out, summed / counts, atol=1e-3)
    assert not torch.allclose(out, summed, atol=1e-3)


def test_make_dpo_optimizers_adamw_only_when_muon_disabled():
    from dpo.dpo import make_dpo_optimizers
    opts = make_dpo_optimizers(_tiny(), muon_lr=0.0, adamw_lr=1e-6)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.AdamW)
    assert opts[0].param_groups[0]['peak_lr'] == 1e-6
    assert len(make_dpo_optimizers(_tiny(), muon_lr=1e-4, adamw_lr=1e-6)) == 2


def test_dpo_loss_and_metrics_values():
    from dpo.dpo import dpo_loss_and_metrics
    beta = 0.5  # current default
    # chosen preferred over rejected -> loss < ln2, acc 1, margin > 0
    pi = torch.tensor([2.0, 1.0])    # [chosen, rejected]
    ref = torch.tensor([0.0, 0.0])
    loss, margin, acc = dpo_loss_and_metrics(pi, ref, beta)
    assert loss.item() < math.log(2)
    assert math.isclose(loss.item(), math.log1p(math.exp(-beta * 1.0)), rel_tol=1e-6)
    assert math.isclose(margin, beta * 1.0, rel_tol=1e-6) and acc == 1.0
    # no preference signal -> loss == ln2
    loss0, margin0, acc0 = dpo_loss_and_metrics(torch.zeros(2), torch.zeros(2), beta)
    assert math.isclose(loss0.item(), math.log(2), rel_tol=1e-6)
    assert margin0 == 0.0 and acc0 == 0.0


def _pairs(lengths):
    """lengths: list of (chosen_len, rejected_len). Offsets don't matter here."""
    return np.array([[0, c, 0, r] for c, r in lengths], dtype=np.int32)


@pytest.mark.parametrize("lengths", [
    [(5, 5)], [(1, 1)], [(0, 0)], [(1, 5)], [(5, 1)], [(2, 1)], [(1, 2)],
])
def test_viable_matches_build_batch(lengths):
    """viable() decides skips from lengths alone so every rank agrees. If it
    ever disagrees with build_batch, ranks run different step counts."""
    from dpo.dpo import build_batch, viable
    total = sum(c + r for c, r in lengths)
    tokens = np.arange(max(1, total), dtype=np.uint16)
    masks = np.ones_like(tokens, dtype=np.uint8)
    pairs = _pairs(lengths)
    ids = range(len(lengths))
    built = build_batch(tokens, masks, pairs, ids, max_len=16, pad_id=0,
                        device=torch.device('cpu'))
    assert viable(pairs, ids, max_len=16) == (built is not None)


def test_batch_indices_shards_equally_and_skips_degenerates():
    from dpo.dpo import batch_indices
    args = SimpleNamespace(batchsize=1, max_len=16)
    # batch 2 is degenerate, so 9 viable batches shard 4-way as 2 each.
    lengths = [(5, 5)] * 10
    lengths[2] = (1, 1)
    pairs = _pairs(lengths)
    assert batch_indices(pairs, args) == [0, 1, 3, 4, 5, 6, 7, 8, 9]
    shards = [batch_indices(pairs, args, rank=r, world=4) for r in range(4)]
    assert [len(s) for s in shards] == [2, 2, 2, 2]
    flat = [i for s in shards for i in s]
    assert len(flat) == len(set(flat))
    assert 2 not in flat
