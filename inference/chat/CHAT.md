to run the chat (latest SFT weights from the 98M H200 run + regenerated tokenizer):

cargo run --manifest-path inference/chat/Cargo.toml --release -- \
  --checkpoint vast_out/saved/vast_run_sft/sft_step10000.pt \
  --data-dir data_cache/cosmopedia

(uses MPS on Apple silicon; add --no-cuda to force CPU. sft_final.pt doesn't
exist yet — SFT was cut at the deadline, sft_step10000.pt is the latest.
For the raw pretrain model instead: --checkpoint vast_out/saved/vast_run/ckpt_final.pt)

For a local pretrain checkpoint (run in raw mode, no chat template):

cargo run --manifest-path inference/chat/Cargo.toml --release -- \
  --checkpoint saved/<run>/ckpt_final.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda
