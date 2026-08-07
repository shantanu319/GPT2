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

**1. Tokenizer + data (`tokenizer.py`, `prepare.py`, `data.py`).** Stdlib-only byte-level BPE with the GPT-4-style pre-tokenization regex (letters and digits never share a pre-token, digit runs chunk to ≤3 chars): 256 byte ids plus learned merges, vocab 32000 by default, with the 3 chat specials pinned to the top ids (`<|im_start|>`=vocab−3, `<|im_end|>`=vocab−2, `<|endoftext|>`=vocab−1) so SFT never has to resize the embedding. prepare.py streams a seeded (1337) weighted interleave — 42% fineweb-edu-dedup, 28% DCLM-baseline, 15% cosmopedia-v2, 10% Python code (codeparrot-clean), 5% FineMath-4+ (sources that run dry are dropped and the weights renormalize) — trains the BPE on the first 100k mixed docs, then re-tokenizes the whole stream in a single pass, routing doc `i` to val if `i % 500 == 0`, to test if `== 1`, else to train (~0.4% held out; re-runs are byte-identical). The .bin shards are headerless little-endian uint16 with docs separated by a single `<|endoftext|>` id. data.py memmaps them and serves `(batch, seqlen)` windows — sequential by default, or shuffled with pinned-memory prefetch under `-shuffle` — targets = inputs shifted by one.

**2. Model (`model.py`).** Pre-norm decoder-only transformer. Token embedding is tied to the output head, and logits are tanh-soft-capped at ±15. Each of the N layers is RMSNorm → attention → residual, then RMSNorm → SwiGLU → residual. Attention is flash SDPA (`is_causal` plus native GQA via `enable_gqa` in training) with `-kv_heads` KV heads each shared across a group of query heads, per-head QK RMSNorm, and partial RoPE (only the first ~50% of head dims rotate, base 10000). Two residual-stream upgrades on top: value residual learning (the layer-1 value tensor is mixed into later layers through per-layer learnable scalars) and U-net skip connections across the stack plus an embedding shortcut, each gated by a learnable scalar. SwiGLU is bias-free with `d_ff = round64(8/3 · d_model)`. Init is Xavier on all 2D weights except the residual out-projections (attention out, FFN down), which are zeroed so every block starts as identity. `-loops N` optionally re-runs the layer stack N times (depth recurrence: N× depth at 1× params, with a separate KV cache per pass at inference), and `-grad_ckpt` opts into activation checkpointing to trade compute for memory. `-attn_res B` opts into block Attention Residuals (arXiv:2603.15031): every B layers the residual stream is replaced by a per-token softmax-attention mix over all previous block outputs (learned query/key projections over depth), instead of the usual uniform accumulation.

**3. Pretraining (`train.py`).** Hybrid optimizer: Muon for every 2D matrix, AdamW for everything else. Muon defaults to the local Polar-Express implementation in muon.py (`-muon_impl local`; bf16 Newton-Schulz, decoupled weight decay) — what started as a learning exercise got promoted — with `torch.optim.Muon` still selectable. The AdamW groups are split by role (embedding lr 3e-3, learnable scalars 0.01, betas (0.8, 0.95), eps 1e-10) and Muon momentum warms up 0.85 → 0.95 over the first 300 steps. The default LR schedule is WSD (`-schedule wsd`): warmup → stable → 1−sqrt decay to 0 over the last 25% of steps; cosine is still available. bf16 autocast, gradient clipping at norm 2.0, optional `-grad_accum`, torch.compile on CUDA (`-no_compile` to disable), and a fused chunked cross-entropy so the logits never materialize as one giant tensor. The stopping condition is a fixed step budget (epochs × corpus ÷ batch), with the schedule annealed to land exactly on it — validation is informational only unless you enable `-early_stop N`: patience-based early stopping on periodic val loss (an eval counts as an improvement only if it beats the best by >0.5% relative, `-early_stop_delta`), each new best saved to `ckpt_best.pt`, and on trigger a short LR-to-0 cooldown (`-early_stop_cooldown`, 300 steps) so the final weights aren't left hot mid-schedule. Checkpoints are `{step, model, optimizers, config}` dicts — the embedded `config` (vocab, d_model, n_layers, heads, kv_heads, loops, dropout) is what finetune.py, sample.py, chat_server.py, and evaluate.py all use to rebuild the exact architecture, so inference never re-specifies the shape.

**4. Chat SFT (`sft_prepare.py`, `finetune.py`, `chat_format.py`).** smol-smoltalk conversations are rendered as ChatML — `<|im_start|>role\ncontent<|im_end|>\n` per turn, conversation closed with `<|endoftext|>` — and packed into `sft_*.bin` with an element-aligned uint8 loss mask: loss lands only on assistant body tokens, their closing `<|im_end|>`, and the final EOS. finetune.py rebuilds the model from the pretrain checkpoint's `config`, runs masked cross-entropy (per-token CE × mask, normalized by the mask sum) at lr 3e-4 AdamW / 3e-3 Muon with grad clip 1.0, and saves the same checkpoint format, so every inference tool works on SFT weights unchanged.

**5. DPO (`dpo_prepare.py`, `dpo.py`).** Posttraining finishes with direct preference optimization. dpo_prepare.py streams HuggingFaceH4/ultrafeedback_binarized (61k GPT-4-ranked pairs), renders each chosen/rejected completion over the same ChatML prompt prefix (system + user, same template as SFT), and writes flat uint16 bins + masks plus an int32 pair index (`chosen_off, chosen_len, rejected_off, rejected_len` per pair) — no padding on disk. dpo.py loads the SFT checkpoint as the policy plus a frozen copy as the reference model, and minimizes `-log σ(β·[(π_c − ref_c) − (π_r − ref_r)])` where each term is the mean log-probability over the completion tokens (length-normalized, SimPO-style: longer completions no longer accumulate extra negative reward). Batches are pairs (8/step default, padded in-memory with a causal ∧ not-pad attention mask), β = 0.5, 2 epochs, lr 1e-6 AdamW-only by default (`--muon-lr > 0` opts back into the Muon split), grad clip 1.0; logs the reward margin and preference accuracy alongside the loss. Same checkpoint format out, so the chat stack runs on DPO weights unchanged.

**6. Inference + eval (`sample.py`, `chat_server.py`, `chat/`, `evaluate.py`).** Sampling is temperature + top-p (defaults 0.5 / 0.9) with a KV cache; when the window fills (`max_context`, 512 default) the cache is dropped and the last `max_context − 1` tokens are re-prefilled. chat_server.py is a long-lived JSON-lines stdin/stdout process holding multi-turn ChatML state (system turn on the first turn only, generation stops at `<|im_end|>`); the Rust CLI in `chat/` (clap + rustyline) just spawns it as a child and runs the REPL (`/reset`, `/quit`). evaluate.py scores arc_easy / arc_challenge / hellaswag / piqa lm-eval-harness style: argmax over answer choices of summed log-likelihood (`acc`) and per-token log-likelihood (`acc_norm`), with `--chat` to wrap questions in the ChatML template when scoring SFT checkpoints.

The three tuned configurations that ship as defaults (vocab 32000 everywhere):

| | local (`run.sh`) | vast.ai `pipeline` | vast.ai `train` |
|---|---|---|---|
| d_model / layers / heads | 256 / 6 / 4 (full MHA) | 512 / 30 / 8 (2 KV) | 512 / 30 / 8 (2 KV) |
| seqlen / batch | 256 / 32 | 1024 / 128 (docs capped at ~8M) | 1024 / 128 |
| epochs | 1 | 1 | 1 |
| peak LR (Muon / AdamW) | 0.02 / 3e-4 | 0.03 / 3e-4 | 0.03 / 3e-4 |
| warmup / save / val cadence | 200 / 500 / epoch-end | 1000 / 4000 / 2000 | 1000 / 2000 / 2000 |
| parameters | **13.0M** (4.9M non-embedding) | **101.0M** (84.6M non-embedding) | **101.0M** (84.6M non-embedding) |

The two vast.ai columns share the same ~101M shape and batch — `pipeline` (and therefore `watch_pipeline.sh`) passes the same model-shape flags as `train`; they differ only in checkpoint cadence and the pretrain data cap (pipeline defaults to `--max-train-docs 8000000` ≈ 8B tokens, so the LR schedule is guaranteed to finish; `train`/`prepare` piecemeal stays uncapped unless you pass a cap). Every knob is overridable per-run, e.g. `python vast_train.py pipeline --n-layers 31 --max-train-docs 12000000`.

The model itself lives in model.py and is a pre-norm decoder stack: token embeddings, N decoder layers (attention + SwiGLU feed-forward, each wrapped in RMSNorm with residuals), a final RMSNorm, and an output projection whose weights are tied to the embedding table. Attention is flash SDPA with GQA (`-kv_heads` shares each KV head across several query heads, saving params and shrinking the KV cache; native `enable_gqa` in training), QK-norm on the per-head q/k (lets Muon run hotter), and partial RoPE (only half the head dims rotate — a parameter-golf leaderboard find). Value residual learning mixes the layer-1 value tensor into later layers via per-layer learnable scalars, and U-net skip connections plus an embedding shortcut (each with learnable scalar gates) wire early representations straight into late ones. Residual out-projections are zero-initialized so every block starts as identity, and the tied head is tanh soft-capped at ±15. There's also opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params) and opt-in activation checkpointing (`-grad_ckpt`). Local default is ~13M at d_model=256; the vast.ai default is ~101M at d_model=512, 30 layers, 8 heads (2 KV) — see the config table above.

The tokenizer (tokenizer.py) is a minbpe-style byte-level BPE with the GPT-4-style pre-tokenization regex bolted on (cl100k/o200k lineage, in stdlib re: letters and digits split into separate pre-tokens, digit runs chunk to ≤3 chars so rare numbers stop fragmenting — arXiv:2402.14903). It's stdlib-only — no tiktoken or sentencepiece — so the merge loop is transparent and hackable. prepare.py streams a weighted mixture of HuggingFace datasets (SmolLM2-style recipe: 42% fineweb-edu-dedup, 28% DCLM-baseline, 15% cosmopedia-v2 synthetic textbooks, 10% Python code — codeparrot-clean, since stack-edu proper ships SWH blob-ids only, no text column — 5% FineMath-4+; see SOURCES in prepare.py), trains BPE on the first N mixed docs (100k by default, so the vocab sees LaTeX/digits), then re-tokenizes the mixed stream into train.bin/val.bin/test.bin in a single pass via a deterministic 1-in-N holdout split. Small sources that run dry are dropped and weights renormalized; the interleave is seeded so a re-run is byte-identical. The .bin shards are raw uint16 token arrays separated by <|endoftext|>, which train.py mmaps for zero-copy batch sampling.

Training (train.py) uses a Muon + AdamW hybrid: Muon for the 2D+ weight matrices, AdamW for embeddings, norms, and scalars — the AdamW groups split by role (embedding lr 3e-3, learnable scalars 0.01, betas (0.8, 0.95), eps 1e-10), Muon momentum warming up 0.85 → 0.95 over the first 300 steps. Muon now defaults to the local Polar-Express implementation in muon.py (`-muon_impl local`; bf16 Newton-Schulz iterations, decoupled weight decay) — muon.py is no longer just a reference artifact — with torch.optim.Muon still selectable. The schedule is WSD by default (warmup → stable → 1−sqrt decay to 0 over the last 25%, `-schedule wsd`; cosine still available), gradients are clipped to a max norm, and the forward pass runs under bfloat16 autocast with a fused chunked cross-entropy (no more 50GB logits stack); CUDA runs are torch.compile'd (`-no_compile` to disable). resolve_device picks CUDA, then MPS, then CPU, so the same script can run on a GPU cluster without any changes. Checkpoints are written every save_every steps with the full model config embedded in the payload, which is what the chat server later reads to rebuild the architecture.

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

Time-boxed run (e.g. ~7h on an H200): the pipeline defaults are pre-sized for exactly
this. `--max-train-docs 8000000` caps pretrain at ~8M docs ≈ 8B tokens (~80 tokens/param),
batch 128×1024 = 131k tokens/step, so at the ~350–550k tokens/s an H200 sustains, the
WSD schedule actually completes inside the budget — you get an annealed ckpt_final.pt,
not a hot mid-schedule one. (H200 ≈ H100 compute with 1.7× memory bandwidth: expect a
modest speedup on this small model, not 2×.) Launch:

    python vast_train.py create --query 'gpu_name=H200 cuda_max_good>=12.8'   # disk defaults to 80GB now
    python vast_train.py push
    ./watch_pipeline.sh

Budget the day as: prepare ~1–2h (the Python BPE is the bottleneck, not the GPU),
pretrain ~4–6h, SFT + DPO ~1.5–2h. Checkpoints save every 4000 steps (~20 min) and
`pull` works at any time, so weights come back even if you destroy the instance early —
mid-schedule checkpoints are un-annealed but usable. If early `status` checks show
throughput well below 350k tokens/s, either plan to stop at a periodic checkpoint or
relaunch with a smaller --max-train-docs.

The watcher is itself time-boxed: if the chain hasn't produced dpo_final.pt within
MAX_HOURS (default 7), it pulls whatever checkpoints exist (periodic ckpt_step*,
ckpt_best, any stage finals — `pull` grabs all of saved/) and exits, logging the
newest remote checkpoints. DESTROY_ON_TIMEOUT=1 also stops the meter automatically:

    MAX_HOURS=8 DESTROY_ON_TIMEOUT=1 ./watch_pipeline.sh

Piecemeal invocations:
    python vast_train.py prepare                      # just data prep (skips if train.bin exists on the instance)
    python vast_train.py train --epochs 2             # just pretrain (~101M defaults; add --detach)
    python vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt
    python vast_train.py dpo-prepare                  # just tokenize preference pairs
    python vast_train.py dpo --checkpoint saved/vast_run_sft/sft_final.pt
    python vast_train.py status                       # instance state + pipeline log tail
    python vast_train.py pull                         # rsync saved/ + tokenizer.json into ./vast_out
    python vast_train.py destroy                      # stop billing

Architecture upgrades (frontier small-model tricks, mostly from the nanoGPT speedrun and OpenAI's parameter-golf challenge): QK-norm on per-head q/k, zero-initialized residual out-projections, tanh logit soft-capping at ±15, GQA (`-kv_heads`, default 2 of 8 heads on vast.ai — saves params + shrinks the KV cache), partial RoPE (rotate half the head dims), value residual learning (layer-1 V mixed into later layers via per-layer learnable scalars), U-net skip connections + an embedding shortcut with learnable scalar gates, opt-in block Attention Residuals (`-attn_res B` — softmax attention over previous block outputs replaces uniform residual accumulation, arXiv:2603.15031), and opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params — parameter-golf's best capacity trick). Training adds the local Polar-Express Muon as default (`-muon_impl local`), with opt-in per-head orthogonalization (`-muon_per_head` — attention projection updates are Newton-Schulz'd per head, in the style of Kimi K3's Per-Head Muon), a WSD schedule (`-schedule wsd`), Muon weight decay, torch.compile + fused chunked cross-entropy, shuffled pinned-prefetch data feeding (`-shuffle`), opt-in activation checkpointing (`-grad_ckpt`), gradient accumulation (`-grad_accum`), and capped mid-epoch validation (`-val_every`).

Posttraining (SFT + DPO, target: chat-able under 100M params):
    prepare.py now reserves <|im_start|>/<|im_end|> chat specials in the vocab (rebuild with --force-prepare once),
    sft_prepare.py tokenizes HuggingFaceTB/smol-smoltalk into ChatML, packed into sft_*.bin with a uint8 loss mask,
    finetune.py loads a pretrain checkpoint and runs masked SFT (loss only on assistant tokens, lr 3e-4),
    dpo_prepare.py tokenizes HuggingFaceH4/ultrafeedback_binarized into chosen/rejected pairs (dpo_*.bin + pair index),
    dpo.py loads the SFT checkpoint as policy + frozen reference and runs DPO (β=0.5, 2 epochs, AdamW-only) on length-normalized completion log-probs.

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
