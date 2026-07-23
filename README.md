Initially, this repository was an implementation of GPT2, from scratch, using PyTorch. It was trained on the WikiText dataset and then subsequently abandoned. I spent some time revamping the project, adding new features, and training it on the cosmopedia dataset. I also wrote a tokenizer for the model, using the Byte Pair Encoding algorithm. The main changes I made were to modernize it, add RoPE embeddings, SwiGLU activations, and RMSNorm layers. I also added a chat interface for the model, using clap + rustyline (essentially a rust wrapper around the python inference server).

## Architecture (end to end)

Five stages, glued together by artifacts in `data_cache/cosmopedia/` and checkpoints in `saved/<run>/`:

    HuggingFace streams ─ prepare.py ─► tokenizer.json + train/val/test.bin (uint16 tokens)
                                        (vocab reserves <|im_start|> <|im_end|> <|endoftext|>)
                      ─ train.py ─────► saved/<run>/ckpt_step*.pt, ckpt_final.pt   (pretrain)
    smol-smoltalk ─ sft_prepare.py ─► sft_{train,val}.bin + uint8 loss masks
                      ─ finetune.py ─► saved/<run>/sft_final.pt                    (chat SFT)
    ultrafeedback ─ dpo_prepare.py ─► dpo_{train,val}.bin + masks + pair index
                      ─ dpo.py ──────► saved/<run>/dpo_final.pt                    (DPO)
                                       └► sample.py | chat_server.py + chat/ REPL | evaluate.py

**1. Tokenizer + data (`tokenizer.py`, `prepare.py`, `data.py`).** Stdlib-only byte-level BPE with the GPT-2 pre-tokenization regex: 256 byte ids plus learned merges, vocab 32000 by default, with the 3 chat specials pinned to the top ids (`<|im_start|>`=vocab−3, `<|im_end|>`=vocab−2, `<|endoftext|>`=vocab−1) so SFT never has to resize the embedding. prepare.py streams a seeded (1337) weighted interleave — 55% cosmopedia-v2, 20% fineweb-edu-dedup, 15% FineMath-4+, 5% OpenMathInstruct-2, 5% CAMEL physics (sources that run dry are dropped and the weights renormalize) — trains the BPE on the first 10k mixed docs, then re-tokenizes the whole stream in a single pass, routing doc `i` to val if `i % 500 == 0`, to test if `== 1`, else to train (~0.4% held out; re-runs are byte-identical). The .bin shards are headerless little-endian uint16 with docs separated by a single `<|endoftext|>` id. data.py memmaps them and serves sequential (never shuffled) `(batch, seqlen)` windows, targets = inputs shifted by one.

**2. Model (`model.py`).** Pre-norm decoder-only transformer. Token embedding is tied to the output head, and logits are tanh-soft-capped at ±30. Each of the N layers is RMSNorm → attention → residual, then RMSNorm → SwiGLU → residual. Attention is fused SDPA with GQA (`-kv_heads` KV heads, each shared across a group of query heads), per-head QK RMSNorm, and partial RoPE (only the first ~50% of head dims rotate, base 10000). SwiGLU is bias-free with `d_ff = round64(8/3 · d_model)`. Init is Xavier on all 2D weights except the residual out-projections (attention out, FFN down), which are zeroed so every block starts as identity. `-loops N` optionally re-runs the layer stack N times (depth recurrence: N× depth at 1× params, with a separate KV cache per pass at inference).

**3. Pretraining (`train.py`).** Hybrid optimizer: `torch.optim.Muon` (weight decay 0.01) for every 2D matrix, AdamW (weight decay 0.1, betas 0.9/0.95) for the tied embedding, norms, and biases. Linear warmup then cosine decay to a 10%-of-peak floor, bf16 autocast, gradient clipping at norm 2.0, optional `-grad_accum`. Checkpoints are `{step, model, optimizers, config}` dicts — the embedded `config` (vocab, d_model, n_layers, heads, kv_heads, loops, dropout) is what finetune.py, sample.py, chat_server.py, and evaluate.py all use to rebuild the exact architecture, so inference never re-specifies the shape. (`muon.py` is a hand-rolled Muon kept as a reference; only the tests import it.)

**4. Chat SFT (`sft_prepare.py`, `finetune.py`, `chat_format.py`).** smol-smoltalk conversations are rendered as ChatML — `<|im_start|>role\ncontent<|im_end|>\n` per turn, conversation closed with `<|endoftext|>` — and packed into `sft_*.bin` with an element-aligned uint8 loss mask: loss lands only on assistant body tokens, their closing `<|im_end|>`, and the final EOS. finetune.py rebuilds the model from the pretrain checkpoint's `config`, runs masked cross-entropy (per-token CE × mask, normalized by the mask sum) at lr 3e-5 AdamW / 3e-3 Muon with grad clip 1.0, and saves the same checkpoint format, so every inference tool works on SFT weights unchanged.

**5. DPO (`dpo_prepare.py`, `dpo.py`).** Posttraining finishes with direct preference optimization. dpo_prepare.py streams HuggingFaceH4/ultrafeedback_binarized (61k GPT-4-ranked pairs), renders each chosen/rejected completion over the same ChatML prompt prefix (system + user, same template as SFT), and writes flat uint16 bins + masks plus an int32 pair index (`chosen_off, chosen_len, rejected_off, rejected_len` per pair) — no padding on disk. dpo.py loads the SFT checkpoint as the policy plus a frozen copy as the reference model, and minimizes `-log σ(β·[(π_c − ref_c) − (π_r − ref_r)])` where each term is the summed log-probability of completion tokens only. Batches are pairs (8/step default, padded in-memory with a causal ∧ not-pad attention mask), β = 0.1, lr 1e-6 AdamW / 1e-4 Muon, grad clip 1.0; logs the reward margin and preference accuracy alongside the loss. Same checkpoint format out, so the chat stack runs on DPO weights unchanged.

**6. Inference + eval (`sample.py`, `chat_server.py`, `chat/`, `evaluate.py`).** Sampling is temperature + top-p (defaults 0.8 / 0.9) with a KV cache; when the window fills (`max_context`, 512 default) the cache is dropped and the last `max_context − 1` tokens are re-prefilled. chat_server.py is a long-lived JSON-lines stdin/stdout process holding multi-turn ChatML state (system turn on the first turn only, generation stops at `<|im_end|>`); the Rust CLI in `chat/` (clap + rustyline) just spawns it as a child and runs the REPL (`/reset`, `/quit`). evaluate.py scores arc_easy / arc_challenge / hellaswag / piqa lm-eval-harness style: argmax over answer choices of summed log-likelihood (`acc`) and per-token log-likelihood (`acc_norm`), with `--chat` to wrap questions in the ChatML template when scoring SFT checkpoints.

The three tuned configurations that ship as defaults (vocab 32000 everywhere):

| | local (`run.sh`) | vast.ai `pipeline` | vast.ai `train` |
|---|---|---|---|
| d_model / layers / heads | 256 / 6 / 4 (full MHA) | 640 / 17 / 10 (5 KV) | 640 / 17 / 10 (5 KV) |
| seqlen / batch | 256 / 32 | 1024 / 64 × grad-accum 2 | 1024 / 128 |
| epochs | 1 | 1 | 1 |
| peak LR (Muon / AdamW) | 0.02 / 3e-4 | 0.03 / 3e-4 | 0.03 / 3e-4 |
| warmup / save / val cadence | 200 / 500 / epoch-end | 1000 / 1000 / 1000 | 1000 / 2000 / 2000 |
| parameters | **13.0M** (4.9M non-embedding) | **97.9M** (77.4M non-embedding) | **97.9M** (77.4M non-embedding) |

The two vast.ai columns share the same ~98M shape — `pipeline` (and therefore `watch_pipeline.sh`) passes the same model-shape flags as `train`; they differ only in effective batch (64 × grad-accum 2 vs 128) and checkpoint cadence. Every knob is overridable per-run, e.g. `python vast_train.py pipeline --n-layers 18 --save-every 2000`.

The model itself lives in model.py and is a pre-norm decoder stack: token embeddings, N decoder layers (attention + SwiGLU feed-forward, each wrapped in RMSNorm with residuals), a final RMSNorm, and an output projection whose weights are tied to the embedding table. Attention is fused SDPA with GQA (`-kv_heads` shares each KV head across several query heads, saving params and halving the KV cache), QK-norm on the per-head q/k (lets Muon run hotter), and partial RoPE (only half the head dims rotate — a parameter-golf leaderboard find). Residual out-projections are zero-initialized so every block starts as identity, and the tied head is tanh soft-capped at ±30. There's also opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params). Local default is ~13M at d_model=256; the vast.ai default is ~98M at d_model=640, 17 layers, 10 heads (5 KV) — see the config table above.

The tokenizer (tokenizer.py) is a minbpe-style byte-level BPE with the GPT-2 pre-tokenization regex bolted on. It's stdlib-only — no tiktoken or sentencepiece — so the merge loop is transparent and hackable. prepare.py streams a weighted mixture of HuggingFace datasets (SmolLM2-style recipe: 55% cosmopedia-v2 synthetic textbooks, 20% fineweb-edu-dedup real web, 15% FineMath-4+, 5% OpenMathInstruct-2 worked math solutions, 5% CAMEL physics Q/A — see SOURCES in prepare.py), trains BPE on the first N mixed docs (10k by default, so the vocab sees LaTeX/digits), then re-tokenizes the mixed stream into train.bin/val.bin/test.bin in a single pass via a deterministic 1-in-N holdout split. Small sources that run dry are dropped and weights renormalized; the interleave is seeded so a re-run is byte-identical. The .bin shards are raw uint16 token arrays separated by <|endoftext|>, which train.py mmaps for zero-copy batch sampling.

Training (train.py) uses a Muon + AdamW hybrid: Muon for the 2D+ weight matrices, AdamW for embeddings, norms, and biases. I originally hand-rolled Muon — muon.py is still in the repo as a reference artifact — but use torch.optim.Muon for the actual pipeline (muon.py was more of a learning exercise). Learning rate is warmup + cosine decay to 10% of peak, gradients are clipped to a max norm, and the forward pass runs under bfloat16 autocast. resolve_device picks CUDA, then MPS, then CPU, so the same script can run on a GPU cluster without any changes. Checkpoints are written every save_every steps with the full model config embedded in the payload, which is what the chat server later reads to rebuild the architecture.

The chat interface is split in two: a long-running Python inference server (chat_server.py) that loads a checkpoint and reads JSON-line prompts from stdin, and a Rust CLI in chat/ (clap + rustyline) that spawns the Python process as a child and pipes a REPL through it. Inference was kept in Python so I don't have to re-implement the transformer in Rust (low-aura move unfortunately). The Rust side just handles the user-facing loop, history, and process lifecycle. Sampling is top-p + temperature (sample.py), with the running token context capped at max_context so long sessions don't blow up the KV window.

to run the pipeline (train, test, and validate):
    ./run.sh                          # defaults: ~13M params, 1 epoch, full cosmopedia stream
    EPOCHS=3 D_MODEL=128 ./run.sh     # override any knob via env
    FORCE_PREPARE=1 ./run.sh          # rebuild BPE + .bin shards

Total Run:
D_MODEL=384 N_LAYERS=5 HEADS=6 \
SEQLEN=512 BATCHSIZE=16 \
EPOCHS=1 WARMUP_STEPS=300 \
./run.sh

Running on a vast.ai GPU:

vast_train.py runs the same prepare -> train -> sft_prepare -> sft -> dpo_prepare -> dpo chain on a rented vast.ai instance (plain ssh + rsync — no serverless glue). One-time setup:

    pip install vastai python-dotenv
    echo 'VAST_AI_API_KEY=...' >> .env.local   # from https://cloud.vast.ai/manage-keys/

Also: an SSH pubkey at ~/.ssh/id_ed25519.pub (attached to instances at create time — override the path with VAST_SSH_PUBKEY), and optionally HF_TOKEN in .env.local to raise HuggingFace streaming rate limits during prepare.

Sanity-check the whole loop first — it provisions a cheap GPU, runs a tiny train on it, pulls the checkpoint back, and destroys the instance (~6 min, under $0.01):

    python vast_train.py smoke

Then the usual flow. The current instance is tracked in .vast_instance.json so the commands chain; the meter runs until `destroy`, so pull artifacts first. Offers are filtered to GPUs torch 2.11 supports (compute_cap>=750):

    python vast_train.py create       # provision cheapest matching GPU (e.g. --query 'gpu_name=H100_SXM cuda_max_good>=12.8')
    python vast_train.py push         # rsync code + data_cache up
    python vast_train.py pipeline     # whole chain, detached on the instance (survives laptop sleep)

The recommended entrypoint for a full run is the watcher — it launches the chain detached (each stage skips itself if its artifact already exists), polls for dpo_final.pt, then pulls everything into ./vast_out:
    ./watch_pipeline.sh                               # launch + watch + auto-pull
    SKIP_LAUNCH=1 ./watch_pipeline.sh                 # just watch an existing run
    tail -f watch_pipeline.log                        # timestamped milestones

Piecemeal invocations:
    python vast_train.py prepare                      # just data prep (skips if train.bin exists on the instance)
    python vast_train.py train --epochs 2             # just pretrain (~98M defaults; add --detach)
    python vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt
    python vast_train.py dpo-prepare                  # just tokenize preference pairs
    python vast_train.py dpo --checkpoint saved/vast_run_sft/sft_final.pt
    python vast_train.py status                       # instance state + pipeline log tail
    python vast_train.py pull                         # rsync saved/ + tokenizer.json into ./vast_out
    python vast_train.py destroy                      # stop billing

Architecture upgrades (frontier small-model tricks, mostly from the nanoGPT speedrun and OpenAI's parameter-golf challenge): QK-norm on per-head q/k, zero-initialized residual out-projections, tanh logit soft-capping, GQA (`-kv_heads`, default 5 of 10 heads on vast.ai — saves params + halves the KV cache), partial RoPE (rotate half the head dims), and opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params — parameter-golf's best capacity trick). Training adds Muon weight decay, gradient accumulation (`-grad_accum`), and capped mid-epoch validation (`-val_every`).

Posttraining (SFT + DPO, target: chat-able under 100M params):
    prepare.py now reserves <|im_start|>/<|im_end|> chat specials in the vocab (rebuild with --force-prepare once),
    sft_prepare.py tokenizes HuggingFaceTB/smol-smoltalk into ChatML, packed into sft_*.bin with a uint8 loss mask,
    finetune.py loads a pretrain checkpoint and runs masked SFT (loss only on assistant tokens),
    dpo_prepare.py tokenizes HuggingFaceH4/ultrafeedback_binarized into chosen/rejected pairs (dpo_*.bin + pair index),
    dpo.py loads the SFT checkpoint as policy + frozen reference and runs DPO (β=0.1) on completion log-probs.

    python vast_train.py pipeline                                    # full chain: prepare -> pretrain -> sft_prepare -> sft -> dpo_prepare -> dpo
    python vast_train.py sft-prepare                                 # just tokenize chat data
    python vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt --dir-name sft_run
    python vast_train.py dpo-prepare                                 # just tokenize preference pairs
    python vast_train.py dpo --checkpoint saved/vast_run_sft/sft_final.pt --dir-name dpo_run

    # local chat with an SFT checkpoint (ChatML template auto-enabled when specials exist):
    python sample.py --checkpoint sft_final.pt --prompt "hi there" --chat
    python chat_server.py --checkpoint sft_final.pt --data-dir data_cache/cosmopedia   # add --raw for pretrain ckpts

Evaluation (evaluate.py): zero-shot multiple-choice in the lm-evaluation-harness style — each answer choice is scored by total log-likelihood (acc) and per-token log-likelihood (acc_norm), the standard for sub-100M models where generation evals are mostly noise. Supports arc_easy, arc_challenge, hellaswag, piqa; --chat wraps each question in the ChatML template so SFT checkpoints are scored in-distribution. Expect modest but above-chance numbers at this scale; compare base vs SFT vs DPO to check posttraining didn't cost capability.
    python evaluate.py --checkpoint vast_out/saved/vast_run_sft/sft_final.pt \
        --tokenizer vast_out/tokenizer.json --tasks arc_easy,hellaswag,piqa --limit 500

If you want to try it yourself, download the latest weights here: 
https://drive.google.com/file/d/1dS8MitkyJ7bBKZWqizLYizwkZ7WSJR_f/view?usp=sharing

Put them in the root directory of this project, then run the CLI by running this command in the terminal (also from the root dir):
cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint ckpt_step21500.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda

(This checkpoint is a pre-ChatML pretrain model, so the CLI falls back to raw mode. It needs a tokenizer.json in data_cache/cosmopedia — from any prepare run, or pull one from a vast.ai run: `python vast_train.py pull` drops one in vast_out/.)