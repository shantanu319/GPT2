Initially, this repository was an implementation of GPT2, from scratch, using PyTorch. It was trained on the WikiText dataset and then subsequently abandoned. I spent some time revamping the project, adding new features, and training it on the cosmopedia dataset. I also wrote a tokenizer for the model, using the Byte Pair Encoding algorithm. The main changes I made were to modernize it, add RoPE embeddings, SwiGLU activations, and RMSNorm layers. I also added a chat interface for the model, using clap + rustyline (essentially a rust wrapper around the python inference server).

The model itself lives in model.py and is a pre-norm decoder stack: token embeddings, N decoder layers (attention + SwiGLU feed-forward, each wrapped in RMSNorm with residuals), a final RMSNorm, and an output projection whose weights are tied to the embedding table. Attention is fused SDPA with GQA (`-kv_heads` shares each KV head across several query heads, saving params and halving the KV cache), QK-norm on the per-head q/k (lets Muon run hotter), and partial RoPE (only half the head dims rotate — a parameter-golf leaderboard find). Residual out-projections are zero-initialized so every block starts as identity, and the tied head is tanh soft-capped at ±30. There's also opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params). Local default is ~8M at d_model=256; the vast.ai default is ~84M at d_model=640, 14 layers, 10 heads (5 KV).

The tokenizer (tokenizer.py) is a minbpe-style byte-level BPE with the GPT-2 pre-tokenization regex bolted on. It's stdlib-only — no tiktoken or sentencepiece — so the merge loop is transparent and hackable. prepare.py streams a weighted mixture of HuggingFace datasets (SmolLM2-style recipe: 55% cosmopedia-v2 synthetic textbooks, 20% fineweb-edu-dedup real web, 15% FineMath-4+, 5% OpenMathInstruct-2 worked math solutions, 5% CAMEL physics Q/A — see SOURCES in prepare.py), trains BPE on the first N mixed docs (10k by default, so the vocab sees LaTeX/digits), then re-tokenizes the mixed stream into train.bin/val.bin/test.bin in a single pass via a deterministic 1-in-N holdout split. Small sources that run dry are dropped and weights renormalized; the interleave is seeded so a re-run is byte-identical. The .bin shards are raw uint16 token arrays separated by <|endoftext|>, which train.py mmaps for zero-copy batch sampling.

Training (train.py) uses a Muon + AdamW hybrid: Muon for the 2D+ weight matrices, AdamW for embeddings, norms, and biases. I originally hand-rolled Muon — muon.py is still in the repo as a reference artifact — but use torch.optim.Muon for the actual pipeline (muon.py was more of a learning exercise). Learning rate is warmup + cosine decay to 10% of peak, gradients are clipped to a max norm, and the forward pass runs under bfloat16 autocast. resolve_device picks CUDA, then MPS, then CPU, so the same script can run on a GPU cluster without any changes. Checkpoints are written every save_every steps with the full model config embedded in the payload, which is what the chat server later reads to rebuild the architecture.

The chat interface is split in two: a long-running Python inference server (chat_server.py) that loads a checkpoint and reads JSON-line prompts from stdin, and a Rust CLI in chat/ (clap + rustyline) that spawns the Python process as a child and pipes a REPL through it. Inference was kept in Python so I don't have to re-implement the transformer in Rust (low-aura move unfortunately). The Rust side just handles the user-facing loop, history, and process lifecycle. Sampling is top-p + temperature (sample.py), with the running token context capped at max_context so long sessions don't blow up the KV window.

to run the pipeline (train, test, and validate):
    ./run.sh                          # defaults: ~8M params, 1 epoch, full cosmopedia stream
    EPOCHS=3 D_MODEL=128 ./run.sh     # override any knob via env
    FORCE_PREPARE=1 ./run.sh          # rebuild BPE + .bin shards

Total Run:
D_MODEL=384 N_LAYERS=5 HEADS=6 \
SEQLEN=512 BATCHSIZE=16 \
EPOCHS=1 WARMUP_STEPS=300 \
./run.sh

Running on a vast.ai GPU:

vast_train.py runs the same prepare -> train -> sft_prepare -> sft chain on a rented vast.ai instance (plain ssh + rsync — no serverless glue). One-time setup:

    pip install vastai python-dotenv
    echo 'VAST_AI_API_KEY=...' >> .env.local   # from https://cloud.vast.ai/manage-keys/

Also: an SSH pubkey at ~/.ssh/id_ed25519.pub (attached to instances at create time — override the path with VAST_SSH_PUBKEY), and optionally HF_TOKEN in .env.local to raise HuggingFace streaming rate limits during prepare.

Sanity-check the whole loop first — it provisions a cheap GPU, runs a tiny train on it, pulls the checkpoint back, and destroys the instance (~6 min, under $0.01):

    python vast_train.py smoke

Then the usual flow. The current instance is tracked in .vast_instance.json so the commands chain; the meter runs until `destroy`, so pull artifacts first. Offers are filtered to GPUs torch 2.11 supports (compute_cap>=750):

    python vast_train.py create       # provision cheapest matching GPU (e.g. --query 'gpu_name=H100_SXM cuda_max_good>=12.8')
    python vast_train.py push         # rsync code + data_cache up
    python vast_train.py pipeline     # whole chain, detached on the instance (survives laptop sleep)

The recommended entrypoint for a full run is the watcher — it launches the chain detached (each stage skips itself if its artifact already exists), polls for sft_final.pt, then pulls everything into ./vast_out:
    ./watch_pipeline.sh                               # launch + watch + auto-pull
    SKIP_LAUNCH=1 ./watch_pipeline.sh                 # just watch an existing run
    tail -f watch_pipeline.log                        # timestamped milestones

Piecemeal invocations:
    python vast_train.py prepare                      # just data prep (skips if train.bin exists on the instance)
    python vast_train.py train --epochs 2             # just pretrain (~84M defaults; add --detach)
    python vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt
    python vast_train.py status                       # instance state + pipeline log tail
    python vast_train.py pull                         # rsync saved/ + tokenizer.json into ./vast_out
    python vast_train.py destroy                      # stop billing

Architecture upgrades (frontier small-model tricks, mostly from the nanoGPT speedrun and OpenAI's parameter-golf challenge): QK-norm on per-head q/k, zero-initialized residual out-projections, tanh logit soft-capping, GQA (`-kv_heads`, default 5 of 10 heads on vast.ai — saves params + halves the KV cache), partial RoPE (rotate half the head dims), and opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params — parameter-golf's best capacity trick). Training adds Muon weight decay, gradient accumulation (`-grad_accum`), and capped mid-epoch validation (`-val_every`).

Posttraining (chat SFT, target: chat-able under 100M params):
    prepare.py now reserves <|im_start|>/<|im_end|> chat specials in the vocab (rebuild with --force-prepare once),
    sft_prepare.py tokenizes HuggingFaceTB/smol-smoltalk into ChatML, packed into sft_*.bin with a uint8 loss mask,
    finetune.py loads a pretrain checkpoint and runs masked SFT (loss only on assistant tokens).

    python vast_train.py pipeline                                    # full chain: prepare -> pretrain -> sft_prepare -> sft
    python vast_train.py sft-prepare                                 # just tokenize chat data
    python vast_train.py sft --checkpoint saved/vast_run/ckpt_final.pt --dir-name sft_run

    # local chat with an SFT checkpoint (ChatML template auto-enabled when specials exist):
    python sample.py --checkpoint sft_final.pt --prompt "hi there" --chat
    python chat_server.py --checkpoint sft_final.pt --data-dir data_cache/cosmopedia   # add --raw for pretrain ckpts

Evaluation (evaluate.py): zero-shot multiple-choice in the lm-evaluation-harness style — each answer choice is scored by total log-likelihood (acc) and per-token log-likelihood (acc_norm), the standard for sub-100M models where generation evals are mostly noise. Supports arc_easy, arc_challenge, hellaswag, piqa; --chat wraps each question in the ChatML template so SFT checkpoints are scored in-distribution. Expect modest but above-chance numbers at this scale; compare base vs SFT to check posttraining didn't cost capability.
    python evaluate.py --checkpoint vast_out/saved/vast_run_sft/sft_final.pt \
        --tokenizer vast_out/tokenizer.json --tasks arc_easy,hellaswag,piqa --limit 500

If you want to try it yourself, download the latest weights here: 
https://drive.google.com/file/d/1dS8MitkyJ7bBKZWqizLYizwkZ7WSJR_f/view?usp=sharing

Put them in the root directory of this project, then run the CLI by running this command in the terminal (also from the root dir):
cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint ckpt_step21500.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda