Initially, this repository was an implementation of GPT2, from scratch, using PyTorch. It was trained on the WikiText dataset and then subsequently abandoned. I spent some time revamping the project, adding new features, and training it on the cosmopedia dataset. I also wrote a tokenizer for the model, using the Byte Pair Encoding algorithm. The main changes I made were to modernize it, add RoPE embeddings, SwiGLU activations, and RMSNorm layers. I also added a chat interface for the model, using clap + rustyline (essentially a rust wrapper around the python inference server). For a picture of the whole thing, `architecture_diagram.py` renders `architecture.pdf` / `architecture.png` — attention drawn as its full internal pipeline (projections → QK-RMSNorm → partial RoPE → value-residual mix → masked SDPA → zero-init out-projection), SwiGLU as real dataflow, with skip gates, tensor shapes, and init conventions annotated.

## Architecture (end to end)

Five stages, glued together by artifacts in `data_cache/cosmopedia/` and checkpoints in `saved/<run>/`:

    HuggingFace streams ─ prepare.py ─► tokenizer.json + train/val/test.bin (uint16 tokens)
                                        (vocab reserves <|im_start|> <|im_end|> <|endoftext|>)
                      ─ train.py ─────► saved/<run>/ckpt_step*.pt, ckpt_final.pt   (pretrain)
    smol-smoltalk ─ sft_prepare.py ─► sft_{train,val}.bin + uint8 loss masks
                      ─ finetune.py ─► saved/<run>/sft_final.pt                    (chat SFT)
    GPT teacher ─── distill_generate.py ─► teacher_*.jsonl ─ sft_prepare.py --input-jsonl
                                                                     ▲ same bins, distilled answers
    ultrafeedback ─ dpo_prepare.py ─► dpo_{train,val}.bin + masks + pair index
                      ─ dpo.py ──────► saved/<run>/dpo_final.pt                    (DPO)
                                       └► sample.py | chat_server.py + inference/chat/ REPL | evaluate.py

**1. Tokenizer + data (`tokenizer.py`, `prepare.py`, `data.py`).** Stdlib-only byte-level BPE with the GPT-4-style pre-tokenization regex (letters and digits never share a pre-token, digit runs chunk to ≤3 chars): 256 byte ids plus learned merges, vocab 32000 by default, with the 3 chat specials pinned to the top ids (`<|im_start|>`=vocab−3, `<|im_end|>`=vocab−2, `<|endoftext|>`=vocab−1) so SFT never has to resize the embedding. prepare.py streams a seeded (1337) weighted interleave — 42% fineweb-edu-dedup, 28% DCLM-baseline, 15% cosmopedia-v2, 10% Python code (codeparrot-clean), 5% FineMath-4+ (sources that run dry are dropped and the weights renormalize) — trains the BPE on the first 100k mixed docs, then re-tokenizes the whole stream in a single pass, routing doc `i` to val if `i % 500 == 0`, to test if `== 1`, else to train (~0.4% held out; re-runs are byte-identical). The .bin shards are headerless little-endian uint16 with docs separated by a single `<|endoftext|>` id. data.py memmaps them and serves `(batch, seqlen)` windows — sequential by default, or shuffled with pinned-memory prefetch under `-shuffle` — targets = inputs shifted by one. Packing is document-aware (`-doc_mask`, on by default in train.py and finetune.py): the feeders derive per-token segment ids from the EOS positions, attention is masked to the current segment (`causal ∧ same-segment`), RoPE positions restart at each boundary, and the KDA recurrent state is dropped at boundaries (exact per-segment equivalence in the chunked scan, tested) — so a packed window never blends an unrelated document or conversation into the context, at the cost of leaving the fused flash path for a masked SDPA kernel. `-doc_mask 0` / `--no-doc-mask` restores the old packed-causal behavior.

**2. Model (`model.py`).** Pre-norm decoder-only transformer. Token embedding is tied to the output head, and logits are tanh-soft-capped at ±15. Each of the N layers is RMSNorm → attention → residual, then RMSNorm → SwiGLU → residual. Attention is flash SDPA (`is_causal` plus native GQA via `enable_gqa` in training) with `-kv_heads` KV heads each shared across a group of query heads, per-head QK RMSNorm, and partial RoPE (only the first ~50% of head dims rotate, base 10000). Two residual-stream upgrades on top: value residual learning (the layer-1 value tensor is mixed into later layers through per-layer learnable scalars) and U-net skip connections across the stack plus an embedding shortcut, each gated by a learnable scalar. SwiGLU is bias-free with `d_ff = round64(8/3 · d_model)`. Init is Xavier on all 2D weights except the residual out-projections (attention out, FFN down), which are zeroed so every block starts as identity. `-loops N` optionally re-runs the layer stack N times (depth recurrence: N× depth at 1× params, with a separate KV cache per pass at inference), and `-grad_ckpt` opts into activation checkpointing to trade compute for memory. `-attn_res B` opts into block Attention Residuals (arXiv:2603.15031): every B layers the residual stream is replaced by a per-token softmax-attention mix over all previous block outputs (learned query/key projections over depth), instead of the usual uniform accumulation. `-kda N` swaps every layer but each Nth to Kimi Delta Attention (kda.py; Kimi Linear, arXiv:2510.26692): per-channel gated delta-rule linear attention — each head carries a constant-size `[head_dim × head_dim]` state updated as `S′ = Diag(exp(g_t)) S` then `S += β_t k_t (v_t − S′ᵀ k_t)ᵀ`, with a per-key-dim log-decay `g_t` from a low-rank projection (mamba-style dt bias init), a per-head sigmoid write strength β, L2-normalized q/k, and no RoPE (position rides the recurrence). The output passes a per-head RMSNorm gated by a low-rank sigmoid projection. KDA layers carry their state matrix at inference instead of a KV cache (constant memory per token); `kda=4` is the Kimi 3:1 hybrid recipe. Training uses the chunked DPLR form, cached inference the sequential scan, both fp32 — equivalence is pinned down in tests/test_kda.py. `-swa W` caps the layers that keep full attention at the last W tokens (Samba, arXiv:2406.07522), which is what takes a KDA stack from mostly-linear to linear: queries are cut into W-wide blocks, each block reads one 2W-key span starting a window earlier, and every query still sees exactly the W keys ending at itself, at O(T·W) instead of O(T²). Doc masking folds in by unfolding the segment ids the same way, so a fully windowed stack never materializes the dense (B, T, T) mask; cached decode reads at most W keys per step and needs no mask at all. Pair it with `-kda` — the recurrent layers carry the unbounded memory the window gives up, which is the whole point of the pairing.

**3. Pretraining (`train.py`).** Hybrid optimizer: Muon for every 2D matrix, AdamW for everything else. Muon defaults to the local Polar-Express implementation in muon.py (`-muon_impl local`; bf16 Newton-Schulz, decoupled weight decay) — what started as a learning exercise got promoted — with `torch.optim.Muon` still selectable. `-muon_per_head` (local impl only) orthogonalizes each attention head's slice independently — q/k/v row-split by head, the out-projection column-split — in the style of Kimi K3's Per-Head Muon. The AdamW groups are split by role (embedding lr 3e-3, learnable scalars 0.01, betas (0.8, 0.95), eps 1e-10) and Muon momentum warms up 0.85 → 0.95 over the first 300 steps. The default LR schedule is WSD (`-schedule wsd`): warmup → stable → 1−sqrt decay to 0 over the last 25% of steps; cosine is still available. bf16 autocast, gradient clipping at norm 2.0, optional `-grad_accum`, torch.compile on CUDA (`-no_compile` to disable), and a fused chunked cross-entropy so the logits never materialize as one giant tensor. The stopping condition is a fixed step budget (epochs × corpus ÷ batch), with the schedule annealed to land exactly on it — validation is informational only unless you enable `-early_stop N`: patience-based early stopping on periodic val loss (an eval counts as an improvement only if it beats the best by >0.5% relative, `-early_stop_delta`), each new best saved to `ckpt_best.pt`, and on trigger a short LR-to-0 cooldown (`-early_stop_cooldown`, 300 steps) so the final weights aren't left hot mid-schedule. Checkpoints are `{step, model, optimizers, config}` dicts — the embedded `config` (vocab, d_model, n_layers, heads, kv_heads, loops, dropout, value_residual, unet_skips, attn_res, grad_ckpt) is what finetune.py, sample.py, chat_server.py, evaluate.py, and dpo.py all use to rebuild the exact architecture, so inference never re-specifies the shape.

**4. Chat SFT (`sft_prepare.py`, `finetune.py`, `chat_format.py`).** smol-smoltalk conversations are rendered as ChatML — `<|im_start|>role\ncontent<|im_end|>\n` per turn, conversation closed with `<|endoftext|>` — and packed into `sft_*.bin` with an element-aligned uint8 loss mask: loss lands only on assistant body tokens, their closing `<|im_end|>`, and the final EOS. finetune.py rebuilds the model from the pretrain checkpoint's `config`, runs masked cross-entropy (per-token CE × mask, normalized by the mask sum) at lr 3e-4 AdamW / 3e-3 Muon with grad clip 1.0, and saves the same checkpoint format, so every inference tool works on SFT weights unchanged. Teacher distillation slots in here: `distill_generate.py` writes ChatML-ready JSONL conversations from an OpenAI teacher (`--source no_robots` re-answers human-written prompts; `--source synthetic` invents beginner-level QA from a seed topic list — answers kept short and plain so a ~100M student can imitate them), and `sft_prepare.py --input-jsonl` packs them through the identical masking path. Shards are headerless packed arrays, so distilled bins can be concatenated with smol-smoltalk bins to mix. The API key is read from `OPENAI_API_KEY` in `.env.local` or the environment — `.env.local` wins when they differ (an stderr note flags the shadowed shell key), and the key is never printed; teacher/model selectable via `--model` / `OPENAI_MODEL`. gpt-5.x parameter quirks (rejecting `max_completion_tokens`/`temperature`) are retried automatically, so older and newer teachers both work.

**5. DPO (`dpo_prepare.py`, `dpo.py`).** Posttraining finishes with direct preference optimization. dpo_prepare.py streams HuggingFaceH4/ultrafeedback_binarized (61k GPT-4-ranked pairs), renders each chosen/rejected completion over the same ChatML prompt prefix (system + user, same template as SFT), and writes flat uint16 bins + masks plus an int32 pair index (`chosen_off, chosen_len, rejected_off, rejected_len` per pair) — no padding on disk. dpo.py loads the SFT checkpoint as the policy plus a frozen copy as the reference model, and minimizes `-log σ(β·[(π_c − ref_c) − (π_r − ref_r)])` where each term is the mean log-probability over the completion tokens (length-normalized, SimPO-style: longer completions no longer accumulate extra negative reward). Batches are pairs (8/step default, padded in-memory with a causal ∧ not-pad attention mask), β = 0.5, 2 epochs, lr 1e-6 AdamW-only by default (`--muon-lr > 0` opts back into the Muon split), grad clip 1.0; logs the reward margin and preference accuracy alongside the loss. Same checkpoint format out, so the chat stack runs on DPO weights unchanged.

**6. Inference + eval (`sample.py`, `chat_server.py`, `inference/chat/`, `evaluate.py`).** Sampling is temperature + top-p (defaults 0.5 / 0.9) with a KV cache; when the window fills (`max_context`, 512 default) the cache is dropped and the last `max_context − 1` tokens are re-prefilled. chat_server.py is a long-lived JSON-lines stdin/stdout process holding multi-turn ChatML state (system turn on the first turn only, generation stops at `<|im_end|>`); the Rust CLI in `inference/chat/` (clap + rustyline) just spawns it as a child and runs the REPL (`/reset`, `/quit`). evaluate.py scores arc_easy / arc_challenge / hellaswag / piqa lm-eval-harness style: argmax over answer choices of summed log-likelihood (`acc`) and per-token log-likelihood (`acc_norm`), with `--chat` to wrap questions in the ChatML template when scoring SFT checkpoints.

**6b. MLX backend (`core/mlx_model.py`, `core/mlx_kda.py`, `inference/mlx_sample.py`).** `--backend mlx` runs the same checkpoint on Apple's MLX instead of torch/MPS. The MLX module tree mirrors model.py name for name — GQA with QK-norm and partial RoPE, SwiGLU, value residuals, U-net skips, attention residuals, depth loops, KDA's recurrence and chunked prefill scan — so the torch state dict loads with no key rewriting and the tests pin both backends' logits to each other. MLX does not yet implement sliding-window attention and rejects `-swa` checkpoints instead of silently running global attention. The decode loop itself is shared: `TorchBackend` / `MLXBackend` supply prefill, single-token step, sampling, and the token read-back, and everything above that (stop-token accounting, KV-window rebuild) is single-sourced in sample.py. `--backend mlx:4` / `mlx:8` quantize the weights on the way in. On an M3 with the 98M SFT checkpoint (64-token prompt, 128 tokens decoded, median of 11 interleaved runs — `python -m inference.bench_decode`):

| | decode | prefill | val NLL |
|---|---|---|---|
| torch / MPS, bf16 | 154 tok/s | 2.3k tok/s | 3.3991 |
| MLX, bf16 | **309 tok/s** (2.0×) | 7.4k tok/s (3.2×) | 3.3989 |
| MLX, 8-bit | **354 tok/s** (2.3×) | 6.9k tok/s | 3.3999 |
| MLX, 4-bit | 356 tok/s (2.3×) | 7.0k tok/s | 3.5016 |

Decode at this size is launch-overhead bound, not bandwidth bound, which is why MLX's lower per-op dispatch cost is worth 2× and why 4-bit buys almost nothing over 8-bit while costing 3% of the loss. bf16 MLX is numerically free; 8-bit is close enough to be the sensible default when you want the extra 15%.

The three tuned configurations that ship as defaults (vocab 32000 everywhere):

| | local (`run.sh`) | vast.ai `pipeline` | vast.ai `train` |
|---|---|---|---|
| d_model / layers / heads | 256 / 6 / 4 (full MHA) | 512 / 30 / 8 (2 KV) | 512 / 30 / 8 (2 KV) |
| seqlen / batch | 256 / 32 | 1024 / 128 (docs capped at ~8M) | 1024 / 128 |
| epochs | 1 | 1 | 1 |
| peak LR (Muon / AdamW) | 0.02 / 3e-4 | 0.03 / 3e-4 | 0.03 / 3e-4 |
| warmup / save / val cadence | 200 / 500 / epoch-end | 1000 / 4000 / 2000 | 1000 / 2000 / 2000 |
| parameters | **13.0M** (4.9M non-embedding) | **101.0M** (84.6M non-embedding) | **101.0M** (84.6M non-embedding) |

The two vast.ai columns share the same ~101M shape and batch — `pipeline` (and therefore `watch_pipeline.sh`) passes the same model-shape flags as `train`; they differ only in checkpoint cadence and the pretrain data cap (pipeline defaults to `--max-train-docs 8000000` ≈ 8B tokens, so the LR schedule is guaranteed to finish; `train`/`prepare` piecemeal stays uncapped unless you pass a cap). Every knob is overridable per-run, e.g. `python vast/vast_train.py pipeline --n-layers 31 --max-train-docs 12000000`.

The model itself lives in model.py and is a pre-norm decoder stack: token embeddings, N decoder layers (attention + SwiGLU feed-forward, each wrapped in RMSNorm with residuals), a final RMSNorm, and an output projection whose weights are tied to the embedding table. Attention is flash SDPA with GQA (`-kv_heads` shares each KV head across several query heads, saving params and shrinking the KV cache; native `enable_gqa` in training), QK-norm on the per-head q/k (lets Muon run hotter), and partial RoPE (only half the head dims rotate — a parameter-golf leaderboard find). Value residual learning mixes the layer-1 value tensor into later layers via per-layer learnable scalars, and U-net skip connections plus an embedding shortcut (each with learnable scalar gates) wire early representations straight into late ones. Residual out-projections are zero-initialized so every block starts as identity, and the tied head is tanh soft-capped at ±15. There's also opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params) and opt-in activation checkpointing (`-grad_ckpt`). Local default is ~13M at d_model=256; the vast.ai default is ~101M at d_model=512, 30 layers, 8 heads (2 KV) — see the config table above.

The tokenizer (tokenizer.py) is a minbpe-style byte-level BPE with the GPT-4-style pre-tokenization regex bolted on (cl100k/o200k lineage, in stdlib re: letters and digits split into separate pre-tokens, digit runs chunk to ≤3 chars so rare numbers stop fragmenting — arXiv:2402.14903). It's stdlib-only — no tiktoken or sentencepiece — so the merge loop is transparent and hackable. prepare.py streams a weighted mixture of HuggingFace datasets (SmolLM2-style recipe: 42% fineweb-edu-dedup, 28% DCLM-baseline, 15% cosmopedia-v2 synthetic textbooks, 10% Python code — codeparrot-clean, since stack-edu proper ships SWH blob-ids only, no text column — 5% FineMath-4+; see SOURCES in prepare.py), trains BPE on the first N mixed docs (100k by default, so the vocab sees LaTeX/digits), then re-tokenizes the mixed stream into train.bin/val.bin/test.bin in a single pass via a deterministic 1-in-N holdout split. Small sources that run dry are dropped and weights renormalized; the interleave is seeded so a re-run is byte-identical. The .bin shards are raw uint16 token arrays separated by <|endoftext|>, which train.py mmaps for zero-copy batch sampling.

Training (train.py) uses a Muon + AdamW hybrid: Muon for the 2D+ weight matrices, AdamW for embeddings, norms, and scalars — the AdamW groups split by role (embedding lr 3e-3, learnable scalars 0.01, betas (0.8, 0.95), eps 1e-10), Muon momentum warming up 0.85 → 0.95 over the first 300 steps. Muon now defaults to the local Polar-Express implementation in muon.py (`-muon_impl local`; bf16 Newton-Schulz iterations, decoupled weight decay) — muon.py is no longer just a reference artifact — with torch.optim.Muon still selectable. The schedule is WSD by default (warmup → stable → 1−sqrt decay to 0 over the last 25%, `-schedule wsd`; cosine still available), gradients are clipped to a max norm, and the forward pass runs under bfloat16 autocast with a fused chunked cross-entropy (no more 50GB logits stack); CUDA runs are torch.compile'd (`-no_compile` to disable). resolve_device picks CUDA, then MPS, then CPU, so the same script can run on a GPU cluster without any changes. Checkpoints are written every save_every steps with the full model config embedded in the payload, which is what the chat server later reads to rebuild the architecture.

The chat interface is split in two: a long-running Python inference server (chat_server.py) that loads a checkpoint and reads JSON-line prompts from stdin, and a Rust CLI in inference/chat/ (clap + rustyline) that spawns the Python process as a child and pipes a REPL through it. Inference was kept in Python so I don't have to re-implement the transformer in Rust (low-aura move unfortunately). The Rust side just handles the user-facing loop, history, and process lifecycle. Sampling is top-p + temperature (sample.py), with the running token context capped at max_context so long sessions don't blow up the KV window. `--backend mlx` on either the Rust CLI or chat_server.py swaps torch for MLX under the same protocol (`pip install mlx`; the torch path is unaffected if it's missing) — about 2× the decode rate on Apple silicon, see the table above.

to run the pipeline (train, test, and validate):
    ./scripts/run.sh                          # defaults: ~13M params, 1 epoch, full cosmopedia stream
    EPOCHS=3 D_MODEL=128 ./scripts/run.sh     # override any knob via env
    FORCE_PREPARE=1 ./scripts/run.sh          # rebuild BPE + .bin shards

Total Run:
D_MODEL=384 N_LAYERS=5 HEADS=6 \
SEQLEN=512 BATCHSIZE=16 \
EPOCHS=1 WARMUP_STEPS=300 \
./scripts/run.sh

Benchmarking the attention stack (`bench/`, needs `pip install mlx` on Apple Silicon):

    python3 -m bench.compare_mlx           # both benchmarks
    python3 -m bench.compare_mlx --attn    # just global vs sliding-window attention
    python3 -m bench.kda_kernel            # CUDA: fused KDA scan vs the Python loop
    python3 -m bench.kda_kernel --model    # ... measured as whole training steps
    python3 -m bench.cuda_attention        # CUDA: the whole stack, eager vs compiled

compare_mlx.py times the two kernels that dominate the stack — the KDA chunked path and windowed
attention — against MLX ports of the same math in bench/mlx_ops.py, which it checks against
core/kda.py and core/model.py before timing anything so a drifted port fails loudly instead of
reporting a fast wrong number. MLX is a measuring stick here, not a backend: it fuses the launch-
bound KDA scan (~2-3x on this hardware) and its SDPA runs sequence lengths where PyTorch's MPS
kernel runs out of memory, which makes it the practical way to check long-context behaviour
locally before renting a GPU.

kda_kernel.py is the CUDA counterpart and A/Bs core/kda_triton.py against the Python loop it
replaces, both in isolation and as whole kda_chunk calls — the second number is the one that
matters, since it bounds what any amount of tuning inside the kernel can buy. Run it under
autocast (it does this itself): without autocast kda_chunk's .float() puts the scan on the fp32
path, which is not how training runs and answers a different question.

Running on a vast.ai GPU:

vast_train.py runs the same prepare -> train -> sft_prepare -> sft -> dpo_prepare -> dpo chain on a rented vast.ai instance (plain ssh + rsync — no serverless glue). One-time setup:

    pip install vastai python-dotenv
    echo 'VAST_AI_API_KEY=...' >> .env.local   # from https://cloud.vast.ai/manage-keys/

Also: an SSH pubkey at ~/.ssh/id_ed25519.pub (attached to instances at create time — override the path with VAST_SSH_PUBKEY), and optionally HF_TOKEN in .env.local to raise HuggingFace streaming rate limits during prepare.

Sanity-check the whole loop first — it provisions a cheap GPU, runs a tiny train on it, pulls the checkpoint back, and destroys the instance (~6 min, under $0.01):

    python vast/vast_train.py smoke

Then the usual flow. The current instance is tracked in .vast_instance.json so the commands chain; the meter runs until `destroy`, so pull artifacts first. Offers are filtered to GPUs torch 2.11 supports (compute_cap>=750):

    python vast/vast_train.py create       # provision cheapest matching GPU (e.g. --query 'gpu_name=H100_SXM cuda_max_good>=12.8')
    python vast/vast_train.py push         # rsync code + data_cache up
    python vast/vast_train.py pipeline     # whole chain, detached on the instance (survives laptop sleep)

The recommended entrypoint for a full run is the watcher — it launches the chain detached (each stage skips itself if its artifact already exists), polls for dpo_final.pt, then pulls everything into ./vast_out:
    ./scripts/watch_pipeline.sh                               # launch + watch + auto-pull
    SKIP_LAUNCH=1 ./scripts/watch_pipeline.sh                 # just watch an existing run
    tail -f watch_pipeline.log                        # timestamped milestones

Time-boxed run (e.g. ~7h on an H200): the pipeline defaults are pre-sized for exactly
this. `--max-train-docs 8000000` caps pretrain at ~8M docs ≈ 8B tokens (~80 tokens/param),
batch 128×1024 = 131k tokens/step, so at the ~350–550k tokens/s an H200 sustains, the
WSD schedule actually completes inside the budget — you get an annealed ckpt_final.pt,
not a hot mid-schedule one. (H200 ≈ H100 compute with 1.7× memory bandwidth: expect a
modest speedup on this small model, not 2×.) Launch:

    python vast/vast_train.py create --query 'gpu_name=H200 cuda_max_good>=12.8'   # disk defaults to 80GB now
    python vast/vast_train.py push
    ./scripts/watch_pipeline.sh

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

    MAX_HOURS=8 DESTROY_ON_TIMEOUT=1 ./scripts/watch_pipeline.sh

Piecemeal invocations:
    python vast/vast_train.py prepare                      # just data prep (skips if train.bin exists on the instance)
    python vast/vast_train.py train --epochs 2             # just pretrain (~101M defaults; add --detach)
    python vast/vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt
    python vast/vast_train.py dpo-prepare                  # just tokenize preference pairs
    python vast/vast_train.py dpo --checkpoint saved/vast_run_sft/sft_final.pt
    python vast/vast_train.py status                       # instance state + pipeline log tail
    python vast/vast_train.py pull                         # rsync saved/ + tokenizer.json into ./vast_out
    python vast/vast_train.py destroy                      # stop billing

Running on a Vultr Cloud GPU:

`vultr_train.py` runs the same resumable prepare -> pretrain -> SFT -> DPO chain on
Vultr's on-demand Cloud GPU instances. It uses the public plan catalog for live
price/capacity discovery, provisions Ubuntu 24.04's GPU-enabled image, installs an
ephemeral SSH key when needed, and bootstraps a Python 3 virtual environment.

One-time setup:

    pip install python-dotenv
    echo 'VULTR_API_KEY=...' >> .env.local   # https://my.vultr.com/settings/#settingsapi

The API key needs plan-list, instance create/read/delete, and SSH-key
create/list/delete permissions. Vultr also requires the machine's public IP to be
allowed under the account's API settings. The default SSH key pair is
`~/.ssh/id_ed25519{,.pub}`; override both paths with CLI flags if necessary.

Run the smoke test first. It discovers the cheapest live GPU plan, creates synthetic
1K-token shards, trains a 32-wide one-layer model in eager mode, pulls
`ckpt_final.pt` into `vultr_out/smoke/`, and confirms destruction in a `finally`
block. If Cloud GPU access is not enabled on the account, it automatically falls
back to the cheapest viable shared-CPU plan; `--compute` selects that path directly.
The CPU path validates provisioning, SSH, training, pull, and teardown, but not CUDA:

    python3 vultr/vultr_train.py plans
    python3 vultr/vultr_train.py smoke

As of August 2026 the cheapest plan is `vcg-a16-2c-8g-2vram`: a fractional
2 GB NVIDIA A16 at $0.059/hour. Vultr has a one-hour minimum charge, so even a
short GPU smoke costs $0.059. The CPU fallback uses `vc2-1c-1gb` at $0.007/hour;
the verified smoke completed in 4.4 minutes and pulled a 16-step checkpoint. Stopped
instances still bill; only `destroy` stops billing.

For a real run, the CLI selects the cheapest currently available plan with at least
20 GB VRAM per GPU (override with `--plan`, `--region`, or `--min-vram`):

    python3 vultr/vultr_train.py create
    python3 vultr/vultr_train.py push
    python3 vultr/vultr_train.py pipeline
    PROVIDER=vultr ./scripts/watch_pipeline.sh
    python3 vultr/vultr_train.py status
    python3 vultr/vultr_train.py pull       # checkpoints and logs -> vultr_out/
    python3 vultr/vultr_train.py destroy    # also removes a pipeline-owned SSH key
    python3 vultr/vultr_train.py destroy --id INSTANCE_ID  # recovery without local state

The current instance lives in ignored `.vultr_instance.json`. The pipeline uses the
same 101M shape as Vast but a safer 1M-document default cap; the 8M-document H200
budget is a poor default for an A16. Benchmark throughput before raising
`--max-train-docs` or the watcher's seven-hour deadline. The watcher uses
`watch_vultr_pipeline.log`, pulls before optional timeout teardown, and accepts the
same `MAX_HOURS` and `DESTROY_ON_TIMEOUT` controls. See Vultr's
[GPU provisioning guide](https://docs.vultr.com/products/compute/instances/cloud-gpu/provisioning)
and [billing rules](https://docs.vultr.com/support/platform/billing/how-am-i-billed-for-my-servers).

Architecture upgrades (frontier small-model tricks, mostly from the nanoGPT speedrun and OpenAI's parameter-golf challenge): QK-norm on per-head q/k, zero-initialized residual out-projections, tanh logit soft-capping at ±15, GQA (`-kv_heads`, default 2 of 8 heads on vast.ai — saves params + shrinks the KV cache), partial RoPE (rotate half the head dims), value residual learning (layer-1 V mixed into later layers via per-layer learnable scalars), U-net skip connections + an embedding shortcut with learnable scalar gates, opt-in block Attention Residuals (`-attn_res B` — softmax attention over previous block outputs replaces uniform residual accumulation, arXiv:2603.15031), opt-in Kimi Delta Attention layers (`-kda N` — every layer but each Nth runs per-channel gated delta-rule linear attention against a constant-size recurrent state instead of a KV cache, Kimi Linear arXiv:2510.26692), opt-in sliding-window attention (`-swa W` — the layers that keep full attention see only the last W tokens, so a `-kda` hybrid has no O(T²) term left anywhere in the stack, Samba arXiv:2406.07522), and opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params — parameter-golf's best capacity trick). Training adds the local Polar-Express Muon as default (`-muon_impl local`), with opt-in per-head orthogonalization (`-muon_per_head` — attention projection updates are Newton-Schulz'd per head, in the style of Kimi K3's Per-Head Muon), a WSD schedule (`-schedule wsd`), Muon weight decay, torch.compile + fused chunked cross-entropy, shuffled pinned-prefetch data feeding (`-shuffle`), opt-in activation checkpointing (`-grad_ckpt`), gradient accumulation (`-grad_accum`), and capped mid-epoch validation (`-val_every`).

Posttraining (SFT + DPO, target: chat-able under 100M params):
    prepare.py now reserves <|im_start|>/<|im_end|> chat specials in the vocab (rebuild with --force-prepare once),
    sft_prepare.py tokenizes HuggingFaceTB/smol-smoltalk into ChatML, packed into sft_*.bin with a uint8 loss mask,
    finetune.py loads a pretrain checkpoint and runs masked SFT (loss only on assistant tokens, lr 3e-4),
    dpo_prepare.py tokenizes HuggingFaceH4/ultrafeedback_binarized into chosen/rejected pairs (dpo_*.bin + pair index),
    dpo.py loads the SFT checkpoint as policy + frozen reference and runs DPO (β=0.5, 2 epochs, AdamW-only) on length-normalized completion log-probs.

Teacher distillation (optional; needs OPENAI_API_KEY in .env.local — which takes precedence over any stale exported shell key):
    distill_generate.py answers prompts with a GPT teacher (default gpt-5.6-luna; override with --model / OPENAI_MODEL)
    and writes ChatML-ready JSONL: --source synthetic invents beginner-level QA from a seed topic list (targets the
    general-knowledge gap), --source no_robots re-answers ~9.5k human-written prompts (targets task/style variety).
    sft_prepare.py --input-jsonl packs the JSONL through the same masking path, so the bins drop straight into finetune.py:

    python3 -m sft.distill_generate --source synthetic --max-examples 5000   # add --resume to continue an interrupted run
    python3 -m sft.distill_generate --source no_robots --max-examples 5000
    cat data_cache/distill/teacher_*.jsonl > data_cache/distill/teacher_all.jsonl
    python3 -m sft.sft_prepare --input-jsonl data_cache/distill/teacher_all.jsonl --output-dir data_cache/distill
    python3 -m sft.finetune --checkpoint saved/<run>/ckpt_final.pt --data-dir data_cache/distill --dir-name sft_distill
    # to mix with smol-smoltalk instead of training on distilled data alone, cat the headerless bins
    # (tokens with tokens, masks with masks) into one directory and point --data-dir there.

    python vast/vast_train.py pipeline                                    # full chain: prepare -> pretrain -> sft_prepare -> sft -> dpo_prepare -> dpo
    python vast/vast_train.py sft-prepare                                 # just tokenize chat data
    python vast/vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt --dir-name sft_run
    python vast/vast_train.py dpo-prepare                                 # just tokenize preference pairs
    python vast/vast_train.py dpo --checkpoint saved/vast_run_sft/sft_final.pt --dir-name dpo_run

    # local chat with an SFT checkpoint (ChatML template auto-enabled when specials exist):
    python -m inference.sample --checkpoint sft_final.pt --prompt "hi there" --chat
    python -m inference.chat_server --checkpoint sft_final.pt --data-dir data_cache/cosmopedia   # add --raw for pretrain ckpts

Evaluation (evaluate.py): zero-shot multiple-choice in the lm-evaluation-harness style — each answer choice is scored by total log-likelihood (acc) and per-token log-likelihood (acc_norm), the standard for sub-100M models where generation evals are mostly noise. Supports arc_easy, arc_challenge, hellaswag, piqa; --chat wraps each question in the ChatML template so SFT checkpoints are scored in-distribution. Expect modest but above-chance numbers at this scale; compare base vs SFT vs DPO to check posttraining didn't cost capability.
    python -m inference.evaluate --checkpoint vast_out/saved/vast_run_sft/sft_final.pt \
        --tokenizer vast_out/tokenizer.json --tasks arc_easy,hellaswag,piqa --limit 500

The test suite runs with plain `pytest` (tests/ covers the tokenizer, data packing and masks, the model, Muon — including per-head orthogonalization — block AttnRes, distillation, sampling, and the train/DPO loops; network-dependent tests are marked `slow` and deselected by default).

If you want to try it yourself, download the latest weights here: 
https://drive.google.com/file/d/1dS8MitkyJ7bBKZWqizLYizwkZ7WSJR_f/view?usp=sharing

Put them in the root directory of this project, then run the CLI by running this command in the terminal (also from the root dir):
cargo run --manifest-path inference/chat/Cargo.toml --release -- \
  --checkpoint ckpt_step21500.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda

(This checkpoint is a pre-ChatML pretrain model, so the CLI falls back to raw mode. It needs a tokenizer.json in data_cache/cosmopedia — from any prepare run, or pull one from a vast.ai run: `python vast/vast_train.py pull` drops one in vast_out/.

## Self-directed continual training (`selfdirect/`)

The model picks what it studies next. Not by writing a curriculum in English —
at 98M parameters its prose is not worth reading — but from the one signal it
does produce reliably: **where it is currently learning fastest**.

    fetch_cache/fetch_<source>.bin ─ domains.py ─► data_cache/selfdirect/<arm>/{train,probe}.bin
                                   ─ loop.py ────► saved/<run>/{state.pt, journal.jsonl}
                                   ─ report.py ──► the curriculum, as a table and curriculum.png

Each round of `loop.py`:

1. probe every arm on a small fixed held-out slice of it
2. the **director** (`director.py`) samples one arm from its exponential weights
3. train a block of steps on that arm, resuming from that arm's own cursor
4. probe every arm again — **the reward is the mean probe-loss drop across all
   arms, not just the one that was studied**
5. update the director, journal the round, repeat

Step 4 is where the continual-learning question lives, and measuring it
falsified the obvious design. The default reward (`--reward global`) is the
mean drop over every arm including the studied one, which is the total-nats
objective — and it does **not** stop the model specializing. Over 41 rounds on
the 98M checkpoint the director drove camel-physics down 0.60 nats while
cosmopedia, finemath and fineweb-edu rose 0.26 between them, and by that
objective it was right to: −0.60 beats +0.26. It also beat the uniform control
(2.3151 vs 2.3333), so the director works; the objective it was given is what
lets specialization through.

`--reward transfer` leaves the studied arm out of its own reward, so an arm is
paid only for what it does to the other four and improving itself earns
nothing. `report.py` prints per-arm forgetting (how far each arm's probe loss
has drifted back off its own best), so which objective you picked is visible in
the output rather than taken on trust.

Design points worth stating, because each was a decision that could have gone
the other way:

- **The probe is fixed and never reshuffled.** The control signal is a
  round-to-round *delta*, so the same tokens have to be scored every time —
  otherwise sampling noise arrives on the same order as the progress being
  measured. It is also disjoint from the arm's train shard, so the loop can
  never train on its own reward signal. It is *not* held out of whatever the
  starting checkpoint was pretrained on — but that matters less than it
  sounds: the 98M checkpoint scores 2.62 on `prepare.py`'s train split against
  2.76 on its held-out val split, a 0.14-nat gap, while the five arm probes
  span 1.9 nats (openmath 1.35 to fineweb-edu 3.27). What separates the arms is
  domain difficulty, an order of magnitude above the memorization gap.
- **Probes run fp32 in eval mode.** Measured on MPS, a bf16 autocast probe was
  both less precise and *slower*: under `no_grad` autocast's cast cache is dead,
  so every weight is re-cast on every forward.
- **The probe can be small.** On the 98M checkpoint a 4k-token probe put the
  measured round delta within 3% of the full 33k one at a seventh of the cost
  (`--probe-tokens`).
- **No LR schedule past warmup.** WSD and cosine both anneal toward a horizon;
  a run that is meant never to end does not have one. Instead the loop watches
  the same mean the director reads: every `--lr-patience` rounds, if the global
  probe loss is no better than it was that many rounds ago, it halves the LR
  (floored at `--lr-floor`). That is the horizonless analogue of `train.py`'s
  early stop on a stalled validation loss.
- **Continuing a checkpoint wants a far lower LR than pretraining it.** At
  `train.py`'s `-muon_lr 0.03`, or even `0.01`, the 98M checkpoint diverges
  outright — mean probe loss 2.40 -> 2.93 over 40 steps, on both the local and
  the stdlib Muon alike, which is what prompted the backoff above. `selfdirect`
  defaults to `1e-3` Muon / `1e-4` embed, measured as the best of a sweep.
- **Arms are just named shards.** The director never learns what an arm *is*,
  so splitting a domain by difficulty into `finemath:easy` / `finemath:hard`
  needs no change to the policy at all.

Running it:

    python -m pretrain.prepare --max-train-docs 20000    # once, if fetch_cache/ is empty
    python -m selfdirect.domains --output-dir data_cache/selfdirect
    python -m selfdirect.loop --checkpoint saved/model/ckpt_final.pt \
        --out saved/selfdirect --rounds 200
    python -m selfdirect.report --out saved/selfdirect

`state.pt` holds weights, optimizers, per-arm cursors and director state in one
file, so `--resume` picks the run up mid-curriculum — and because it is also an
ordinary training checkpoint, `inference/sample.py` reads it directly.

`--eta 0` freezes the director at a uniform mixture, which is the control:
identical code path, identical seed, a curriculum chosen at random.
