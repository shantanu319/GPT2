"""Offline SFT data prep: tokenize a chat dataset into packed .bin shards.

Streams HuggingFaceTB/smol-smoltalk (built for ~100M-class models), renders
each conversation in ChatML, and writes:

  {output_dir}/sft_train.bin       uint16 token ids, conversations packed back-to-back
  {output_dir}/sft_train_mask.bin  uint8, 1 where loss should apply (assistant tokens)
  {output_dir}/sft_val.bin / sft_val_mask.bin

Loss is masked to assistant message bodies (+ <|im_end|> + final <|endoftext|>),
so the model learns to *respond*, not to imitate the user.

With --input-jsonl the HF stream is replaced by a local JSONL of
{"messages": [{role, content}, ...]} per line — e.g. teacher-distilled
conversations from distill_generate.py. Everything else (masking, holdout
split, shard layout) is identical, and since shards are headerless packed
arrays, smol-smoltalk and distilled bins can be concatenated directly.

Run after prepare.py (re-uses tokenizer.json; vocab must include
<|im_start|> / <|im_end|> — i.e. rebuilt with the current prepare.py).
"""
import argparse
import json
import os

import numpy as np
from datasets import load_dataset

from chat_format import EOS_TOKEN, IM_END, IM_START
from data import BIN_DTYPE
from tokenizer import BPETokenizer

DATASET_PATH = 'HuggingFaceTB/smol-smoltalk'


def encode_conversation(tokenizer, messages, im_start_id, im_end_id, eos_id):
    """Render a conversation to (ids, mask). mask=1 -> compute loss."""
    ids, mask = [], []
    for m in messages:
        role, content = m['role'], m['content']
        header = tokenizer.encode_ordinary(f"{role}\n")
        body = tokenizer.encode_ordinary(content)
        tail = tokenizer.encode_ordinary("\n")

        is_assistant = role == 'assistant'
        ids.append(im_start_id);   mask.append(0)
        ids.extend(header);        mask.extend([0] * len(header))
        ids.extend(body);          mask.extend([1 if is_assistant else 0] * len(body))
        ids.append(im_end_id);     mask.append(1 if is_assistant else 0)
        ids.extend(tail);          mask.extend([0] * len(tail))
    ids.append(eos_id)
    mask.append(1)
    return ids, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data_cache/cosmopedia')
    parser.add_argument('--input-jsonl', default=None,
                        help='Read conversations from a JSONL file (one '
                             '{"messages": [...]} object per line, e.g. from '
                             'distill_generate.py) instead of streaming smol-smoltalk')
    parser.add_argument('--max-conversations', type=int, default=None)
    parser.add_argument('--holdout-period', type=int, default=200,
                        help='1-in-N conversations go to the val shard')
    args = parser.parse_args()

    tok_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)

    for tok in (IM_START, IM_END, EOS_TOKEN):
        if tok not in tokenizer.special_tokens:
            raise ValueError(
                f"tokenizer at {tok_path} lacks {tok} — rerun prepare.py "
                f"(it now reserves chat specials)")
    im_start_id = tokenizer.special_tokens[IM_START]
    im_end_id = tokenizer.special_tokens[IM_END]
    eos_id = tokenizer.special_tokens[EOS_TOKEN]

    if args.input_jsonl:
        def ds():
            with open(args.input_jsonl) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        ds = ds()
    else:
        ds = load_dataset(DATASET_PATH, split='train', streaming=True)

    paths = {
        'train': (os.path.join(args.output_dir, 'sft_train.bin'),
                  os.path.join(args.output_dir, 'sft_train_mask.bin')),
        'val': (os.path.join(args.output_dir, 'sft_val.bin'),
                os.path.join(args.output_dir, 'sft_val_mask.bin')),
    }
    counts = {'train': 0, 'val': 0}
    files = {k: (open(t, 'wb'), open(m, 'wb')) for k, (t, m) in paths.items()}
    try:
        for i, row in enumerate(ds):
            if args.max_conversations is not None and i >= args.max_conversations:
                break
            ids, mask = encode_conversation(
                tokenizer, row['messages'], im_start_id, im_end_id, eos_id)
            arr = np.array(ids, dtype=BIN_DTYPE)
            if arr.size and arr.max() >= np.iinfo(BIN_DTYPE).max:
                raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
            split = 'val' if i % args.holdout_period == 0 else 'train'
            tf, mf = files[split]
            tf.write(arr.tobytes())
            mf.write(np.array(mask, dtype=np.uint8).tobytes())
            counts[split] += len(ids)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1} conversations, "
                      f"{counts['train'] + counts['val']:,} tokens")
    finally:
        for tf, mf in files.values():
            tf.close()
            mf.close()

    print(f"wrote {counts['train']:,} train tokens, {counts['val']:,} val tokens "
          f"to {args.output_dir}/sft_*.bin")


if __name__ == '__main__':
    main()
