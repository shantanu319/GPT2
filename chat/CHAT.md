to run the chat (after `python vast_train.py pull`, which drops the checkpoints
and tokenizer.json into vast_out/):

cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint vast_out/saved/vast_run_sft/sft_final.pt \
  --data-dir vast_out \
  --no-cuda

For a local pretrain checkpoint (run in raw mode, no chat template):

cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint saved/<run>/ckpt_final.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda
