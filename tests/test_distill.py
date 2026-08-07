"""Tests for distill_generate.py and the sft_prepare.py --input-jsonl path.

No API access needed: the teacher call itself is not exercised (parse + pack only).
"""
import json
import sys

import numpy as np

from chat_format import special_token_map
from data import BIN_DTYPE
from distill_generate import parse_synthetic_qa
from sft_prepare import encode_conversation
from tokenizer import BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog "
    "hello world hello there hello friends "
    "lorem ipsum dolor sit amet consectetur " * 30
)


def test_parse_synthetic_qa():
    assert parse_synthetic_qa(
        "Q: Why is the sky blue?\nA: Light scatters off air molecules.") == \
        ("Why is the sky blue?", "Light scatters off air molecules.")


def test_parse_synthetic_qa_multiline_answer():
    assert parse_synthetic_qa("Q: q?\nA: first line\nsecond line") == \
        ("q?", "first line\nsecond line")


def test_parse_synthetic_qa_rejects_garbage():
    assert parse_synthetic_qa("no markers here") is None
    assert parse_synthetic_qa("A: answer without question") is None
    assert parse_synthetic_qa("Q: \nA: ") is None


def _make_jsonl(tmp_path, convos):
    path = tmp_path / 'teacher.jsonl'
    with open(path, 'w') as f:
        for c in convos:
            f.write(json.dumps({'messages': c}) + '\n')
    return str(path)


def test_sft_prepare_input_jsonl_packs_like_hf_stream(tmp_path, monkeypatch):
    import sft_prepare

    specials = special_token_map(512)  # 509/510/511, outside the trained vocab
    tok = BPETokenizer(special_tokens=specials)
    tok.train(CORPUS, vocab_size=400)
    tok.save(str(tmp_path / 'tokenizer.json'))

    convos = [
        [{'role': 'system', 'content': 'You are a helpful assistant.'},
         {'role': 'user', 'content': 'hello world'},
         {'role': 'assistant', 'content': 'hello friends'}],
        [{'role': 'user', 'content': 'the quick brown fox'},
         {'role': 'assistant', 'content': 'jumps over the lazy dog'}],
        [{'role': 'user', 'content': 'lorem ipsum'},
         {'role': 'assistant', 'content': 'dolor sit amet'},
         {'role': 'user', 'content': 'consectetur'},
         {'role': 'assistant', 'content': 'adipiscing elit'}],
    ]
    jsonl = _make_jsonl(tmp_path, convos)

    monkeypatch.setattr(sys, 'argv', [
        'sft_prepare.py', '--output-dir', str(tmp_path), '--input-jsonl', jsonl,
        '--holdout-period', '100'])
    sft_prepare.main()

    # i % holdout_period == 0 routes to val, so conversation 0 is in val
    ids = np.fromfile(tmp_path / 'sft_train.bin', dtype=BIN_DTYPE).tolist() + \
        np.fromfile(tmp_path / 'sft_val.bin', dtype=BIN_DTYPE).tolist()
    mask = np.fromfile(tmp_path / 'sft_train_mask.bin', dtype=np.uint8).tolist() + \
        np.fromfile(tmp_path / 'sft_val_mask.bin', dtype=np.uint8).tolist()

    exp_ids, exp_mask = [], []
    for c in convos[1:] + convos[:1]:
        i, m = encode_conversation(tok, c, specials['<|im_start|>'],
                                   specials['<|im_end|>'], specials['<|endoftext|>'])
        exp_ids += i
        exp_mask += m

    assert ids == exp_ids
    assert mask == exp_mask
    # loss only on assistant spans: masked ids decode to assistant content
    # (+ im_end/eos specials), never to user/system text
    masked = tok.decode([i for i, m in zip(ids, mask) if m])
    assert 'hello friends' in masked and 'jumps over' in masked
    assert 'hello world' not in masked and 'quick brown' not in masked


def test_sft_prepare_input_jsonl_holdout_split(tmp_path, monkeypatch):
    import sft_prepare

    specials = special_token_map(512)
    tok = BPETokenizer(special_tokens=specials)
    tok.train(CORPUS, vocab_size=400)
    tok.save(str(tmp_path / 'tokenizer.json'))

    convos = [[{'role': 'user', 'content': f'question {i}'},
               {'role': 'assistant', 'content': f'answer {i}'}] for i in range(6)]
    jsonl = _make_jsonl(tmp_path, convos)

    monkeypatch.setattr(sys, 'argv', [
        'sft_prepare.py', '--output-dir', str(tmp_path), '--input-jsonl', jsonl,
        '--holdout-period', '2'])  # convos 0, 2, 4 -> val
    sft_prepare.main()

    train_ids = np.fromfile(tmp_path / 'sft_train.bin', dtype=BIN_DTYPE)
    val_ids = np.fromfile(tmp_path / 'sft_val.bin', dtype=BIN_DTYPE)
    assert len(train_ids) > 0 and len(val_ids) > 0
    total = len(train_ids) + len(val_ids)
    expected = sum(len(encode_conversation(tok, c, specials['<|im_start|>'],
                                           specials['<|im_end|>'],
                                           specials['<|endoftext|>'])[0])
                   for c in convos)
    assert total == expected
