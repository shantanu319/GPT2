"""Per-domain arms for the self-directed curriculum.

pretrain/prepare.py interleaves every source into one train.bin, which erases
which corpus a token came from. The director needs that back: each fetch cache
under data_cache/<run>/fetch_cache becomes an arm with its own train shard and
a small fixed probe shard held out of it.

The probe is the control signal, so it is fixed at prep time and never
reshuffled -- evaluating the same tokens every round means a round-to-round
loss delta is entirely model change, with no sampling noise in it.

  python -m selfdirect.domains --output-dir data_cache/selfdirect
"""
import argparse
import json
import os

from core.chat_format import EOS_TOKEN, special_token_map
from core.data import BIN_DTYPE
from core.tokenizer import BPETokenizer
from pretrain.prepare import encode_text, read_docs

MANIFEST = 'arms.json'


def discover_arms(fetch_cache):
    """{name: path} for every fetch_<name>.bin in a prepare.py fetch cache."""
    return {f[len('fetch_'):-len('.bin')]: os.path.join(fetch_cache, f)
            for f in sorted(os.listdir(fetch_cache))
            if f.startswith('fetch_') and f.endswith('.bin')}


def build_arm(fetch_path, out_dir, tokenizer, eos_id, probe_tokens, probe_period,
              max_docs=None):
    """Tokenize one fetch cache into out_dir/{train,probe}.bin.

    Doc i is held out for the probe when i % probe_period == 0 and the probe is
    still under budget, so the probe is spread across the corpus rather than
    being its head. Returns (train_tokens, probe_tokens)."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'train.bin')
    probe_path = os.path.join(out_dir, 'probe.bin')
    n_train = n_probe = 0
    with open(train_path + '.tmp', 'wb') as trf, open(probe_path + '.tmp', 'wb') as pf:
        for i, text in enumerate(read_docs(fetch_path)):
            if max_docs is not None and i >= max_docs:
                break
            arr = encode_text(tokenizer, text, eos_id)
            if i % probe_period == 0 and n_probe < probe_tokens:
                pf.write(arr.tobytes())
                n_probe += len(arr)
            else:
                trf.write(arr.tobytes())
                n_train += len(arr)
    for path in (train_path, probe_path):
        os.replace(path + '.tmp', path)
    return n_train, n_probe


def load_manifest(data_dir):
    with open(os.path.join(data_dir, MANIFEST)) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-cache', default='data_cache/cosmopedia/fetch_cache')
    parser.add_argument('--tokenizer', default='data_cache/cosmopedia/tokenizer.json')
    parser.add_argument('--output-dir', default='data_cache/selfdirect')
    parser.add_argument('--probe-tokens', type=int, default=32768,
                        help='Held-out probe budget per arm. Bigger probes are '
                             'more representative but cost a forward pass per '
                             'arm every round.')
    parser.add_argument('--probe-period', type=int, default=25,
                        help='Consider every Nth doc for the probe, so the '
                             'budget is drawn from across the corpus')
    parser.add_argument('--max-docs', type=int, default=0,
                        help='Cap docs read per arm (0 = all)')
    args = parser.parse_args()

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    eos_id = special_token_map(tokenizer.vocab_size)[EOS_TOKEN]

    arms = discover_arms(args.fetch_cache)
    if not arms:
        raise SystemExit(f"no fetch_*.bin caches under {args.fetch_cache} — "
                         f"run `python -m pretrain.prepare --max-train-docs N` first")
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = []
    for name, fetch_path in arms.items():
        n_train, n_probe = build_arm(
            fetch_path, os.path.join(args.output_dir, name), tokenizer, eos_id,
            args.probe_tokens, args.probe_period, args.max_docs or None)
        manifest.append({'name': name, 'train_tokens': n_train,
                         'probe_tokens': n_probe})
        print(f"  {name:<16} train {n_train:>10,}  probe {n_probe:>8,}")

    with open(os.path.join(args.output_dir, MANIFEST), 'w') as f:
        json.dump({'vocab_size': tokenizer.vocab_size, 'eos_id': eos_id,
                   'arms': manifest}, f, indent=2)
    total = sum(a['train_tokens'] for a in manifest)
    print(f"wrote {len(manifest)} arms ({total:,} train tokens) to {args.output_dir}")


if __name__ == '__main__':
    main()
