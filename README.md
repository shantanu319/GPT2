Initially, this repository was an implementation of GPT2, from scratch, using PyTorch. It was trained on the WikiText dataset and then subsequently abandoned. I spent some time revamping the project, adding new features, and training it on the cosmopedia dataset. I also wrote a tokenizer for the model, using the Byte Pair Encoding algorithm. The main changes I made were to modernize it, add RoPE embeddings, SwiGLU activations, and RMSNorm layers. I also added a chat interface for the model, using clap + rustyline (essentially a rust wrapper around the python inference server).

The model itself lives in model.py and is a pretty standard pre-norm decoder stack: token embeddings, N decoder layers (attention + SwiGLU feed-forward, each wrapped in RMSNorm with residuals), a final RMSNorm, and an output projection whose weights are tied to the embedding table. RoPE is baked into each attention block — I precompute the cos/sin buffers once at construction time and apply them to the per-head q/k tensors. I stuck with a hand-rolled scaled-dot-product attention instead of the fused F.scaled_dot_product_attention so the mask + dropout plumbing stays visible. The default config is roughly 8M trainable params at d_model=256, 6 layers, 4 heads, seqlen=256 — small enough to fit an overnight run on an M3 Air (what I've been testing the pipeline on). Although I need a GPU cluster of some kind to perform a more complete training run and maybe scale up to 100M parameters.

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

Running on a Modal GPU:

modal_app.py packages the same prepare -> train flow onto Modal so I can burn through the cosmopedia stream on a cloud GPU instead of the M3 Air. One-time setup is `pip install modal && modal setup` (opens a browser to link your account). A named Modal Volume (`myowntransformer-data`) holds the tokenizer + .bin shards + checkpoints, so the data prep only runs once and re-training reuses it.

Common invocations:
    modal run modal_app.py                        # prepare (if needed) + train, defaults on an L4
    modal run modal_app.py --d-model 384 --n-layers 5 --heads 6 \
        --seqlen 512 --batchsize 16 --epochs 1 --warmup-steps 300
    modal run modal_app.py --force-prepare        # rebuild BPE + .bin shards on the volume
    modal run modal_app.py::prepare               # just prep, no training
    modal run modal_app.py::train --epochs 2      # just train, reuse existing volume data
    modal run modal_app.py --bpe-train-docs 500 --epochs 1   # cheap smoke run

Pulling artifacts back once a run finishes (replace `40M_run` with whatever `--dir-name` you used; default is `modal_run`):
    modal volume get myowntransformer-data /saved/40M_run ./modal_out/40M_run
    modal volume get myowntransformer-data /saved/40M_run_learning_curves.png ./modal_out/40M_run_learning_curves.png

    # or grab everything under /saved at once:
    modal volume get myowntransformer-data /saved ./modal_out
The checkpoints land at ./modal_out/<dir_name>/ckpt_*.pt and the plot at ./modal_out/<dir_name>_learning_curves.png.

GPU choice lives in modal_app.py (`gpu="L4"` — roughly $0.80/hr, plenty for the default 8M config). Bump it to `"L40S"` or `"A100"` if you scale the model up. Timeout is 8h; drop it if you want tighter cost guardrails.

Architecture upgrades (frontier small-model tricks, mostly from the nanoGPT speedrun and OpenAI's parameter-golf challenge): QK-norm on per-head q/k, zero-initialized residual out-projections, tanh logit soft-capping, GQA (`-kv_heads`, default 5 of 10 heads on Modal — saves params + halves the KV cache), partial RoPE (rotate half the head dims), and opt-in depth recurrence (`-loops N` runs the layer stack N times for N×depth at 1× params — parameter-golf's best capacity trick). Training adds Muon weight decay, gradient accumulation (`-grad_accum`), and capped mid-epoch validation (`-val_every`).

Posttraining (chat SFT, target: chat-able under 100M params):
    prepare.py now reserves <|im_start|>/<|im_end|> chat specials in the vocab (rebuild with --force-prepare once),
    sft_prepare.py tokenizes HuggingFaceTB/smol-smoltalk into ChatML, packed into sft_*.bin with a uint8 loss mask,
    finetune.py loads a pretrain checkpoint and runs masked SFT (loss only on assistant tokens).

    modal run modal_app.py --run-sft                         # full pipeline: prepare -> pretrain -> sft_prepare -> sft
    modal run modal_app.py::sft_prepare                      # just tokenize chat data
    modal run modal_app.py::sft --checkpoint modal_run/ckpt_final.pt --dir-name sft_run

    # local chat with an SFT checkpoint (ChatML template auto-enabled when specials exist):
    python sample.py --checkpoint sft_final.pt --prompt "hi there" --chat
    python chat_server.py --checkpoint sft_final.pt --data-dir data_cache/cosmopedia   # add --raw for pretrain ckpts

If you want to try it yourself, download the latest weights here: 
https://drive.google.com/file/d/1dS8MitkyJ7bBKZWqizLYizwkZ7WSJR_f/view?usp=sharing

Put them in the root directory of this project, then run the CLI by running this command in the terminal (also from the root dir):
cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint ckpt_step21500.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda