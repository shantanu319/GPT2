"""Offline DPO data prep: tokenize preference pairs into packed .bin shards.

Streams HuggingFaceH4/ultrafeedback_binarized (61k GPT-4-ranked pairs),
renders each chosen/rejected completion as a ChatML conversation over the
shared prompt (same template + DEFAULT_SYSTEM as sft_prepare.py), and writes:

  {output_dir}/dpo_train.bin        uint16 token ids: chosen_0, rejected_0, chosen_1, ...
  {output_dir}/dpo_train_mask.bin   uint8, 1 on completion tokens (assistant body +
                                    <|im_end|> + final <|endoftext|>), 0 on the prompt
  {output_dir}/dpo_train_pairs.bin  int32, shape (n_pairs, 4):
                                    chosen_offset, chosen_len, rejected_offset, rejected_len
                                    (offsets/lengths in tokens into the flat bins above)
  {output_dir}/dpo_val*.bin         same layout for the holdout split

Run after prepare.py + sft_prepare.py (re-uses the same tokenizer.json; vocab
must include <|im_start|> / <|im_end|>).
"""
import argparse
import os

import numpy as np
from datasets import load_dataset

from chat_format import DEFAULT_SYSTEM, EOS_TOKEN, IM_END, IM_START
from data import BIN_DTYPE
from sft_prepare import encode_conversation
from tokenizer import BPETokenizer

DATASET_PATH = 'HuggingFaceH4/ultrafeedback_binarized'
DATASET_SPLIT = 'train_prefs'
PAIRS_DTYPE = np.int32


def split_prompt_completion(messages):
    """Split a chosen/rejected message list into (prompt_messages, completion).

    Returns None for malformed rows (no assistant reply at the end)."""
    if len(messages) < 2 or messages[-1].get('role') != 'assistant':
        return None
    completion = messages[-1].get('content') or ''
    if not completion.strip():
        return None
    return messages[:-1], completion


def encode_pair(tokenizer, row, im_start_id, im_end_id, eos_id):
    """Render one preference row to (chosen_ids, chosen_mask, rejected_ids, rejected_mask).

    Both sides share the same prompt prefix (system + prompt turns); only the
    final assistant completion differs. Returns None if the row is unusable."""
    chosen = split_prompt_completion(row['chosen'])
    rejected = split_prompt_completion(row['rejected'])
    if chosen is None or rejected is None:
        return None
    prompt_msgs, chosen_completion = chosen
    _, rejected_completion = rejected

    system = [{'role': 'system', 'content': DEFAULT_SYSTEM}]
    chosen_ids, chosen_mask = encode_conversation(
        tokenizer,
        system + prompt_msgs + [{'role': 'assistant', 'content': chosen_completion}],
        im_start_id, im_end_id, eos_id)
    rejected_ids, rejected_mask = encode_conversation(
        tokenizer,
        system + prompt_msgs + [{'role': 'assistant', 'content': rejected_completion}],
        im_start_id, im_end_id, eos_id)
    return chosen_ids, chosen_mask, rejected_ids, rejected_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data_cache/cosmopedia')
    parser.add_argument('--max-pairs', type=int, default=None)
    parser.add_argument('--holdout-period', type=int, default=200,
                        help='1-in-N pairs go to the val shard')
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

    ds = load_dataset(DATASET_PATH, split=DATASET_SPLIT, streaming=True)

    paths = {
        'train': (os.path.join(args.output_dir, 'dpo_train.bin'),
                  os.path.join(args.output_dir, 'dpo_train_mask.bin'),
                  os.path.join(args.output_dir, 'dpo_train_pairs.bin')),
        'val': (os.path.join(args.output_dir, 'dpo_val.bin'),
                os.path.join(args.output_dir, 'dpo_val_mask.bin'),
                os.path.join(args.output_dir, 'dpo_val_pairs.bin')),
    }
    counts = {'train': 0, 'val': 0}       # tokens (both sides)
    pairs = {'train': 0, 'val': 0}
    files = {k: (open(t, 'wb'), open(m, 'wb'), open(p, 'wb'))
             for k, (t, m, p) in paths.items()}
    skipped = 0
    try:
        for i, row in enumerate(ds):
            if args.max_pairs is not None and (pairs['train'] + pairs['val']) >= args.max_pairs:
                break
            encoded = encode_pair(tokenizer, row, im_start_id, im_end_id, eos_id)
            if encoded is None:
                skipped += 1
                continue
            c_ids, c_mask, r_ids, r_mask = encoded

            split = 'val' if i % args.holdout_period == 0 else 'train'
            tf, mf, pf = files[split]
            c_off = counts[split]
            for ids, mask in ((c_ids, c_mask), (r_ids, r_mask)):
                arr = np.array(ids, dtype=BIN_DTYPE)
                if arr.size and arr.max() >= np.iinfo(BIN_DTYPE).max:
                    raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
                tf.write(arr.tobytes())
                mf.write(np.array(mask, dtype=np.uint8).tobytes())
            pf.write(np.array([c_off, len(c_ids),
                               c_off + len(c_ids), len(r_ids)],
                              dtype=PAIRS_DTYPE).tobytes())
            counts[split] += len(c_ids) + len(r_ids)
            pairs[split] += 1
            if (pairs['train'] + pairs['val']) % 10000 == 0:
                print(f"  {pairs['train'] + pairs['val']} pairs, "
                      f"{counts['train'] + counts['val']:,} tokens")
    finally:
        for tf, mf, pf in files.values():
            tf.close()
            mf.close()
            pf.close()

    print(f"wrote {pairs['train']} train pairs ({counts['train']:,} tokens), "
          f"{pairs['val']} val pairs ({counts['val']:,} tokens) "
          f"to {args.output_dir}/dpo_*.bin ({skipped} rows skipped)")


if __name__ == '__main__':
    main()
