import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true',
                        help='Force CPU even if CUDA or MPS is available')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=30)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-kv_heads', type=int, default=2,
                        help='KV heads for GQA (0 = same as heads, i.e. full MHA; default 2)')
    parser.add_argument('-loops', type=int, default=1,
                        help='Depth recurrence: run the layer stack this many times')
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batchsize * grad_accum)')
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=float, default=3e-4, help='AdamW peak learning rate')
    parser.add_argument('-muon_lr', type=float, default=0.03, help='Muon peak learning rate')
    parser.add_argument('-embed_lr', type=float, default=3e-3,
                        help='AdamW peak LR for the tied embedding')
    parser.add_argument('-scalar_lr', type=float, default=0.01,
                        help='AdamW peak LR for 1D params (norm gains, biases, arch scalars)')
    parser.add_argument('-muon_impl', choices=['local', 'torch'], default='local',
                        help='Muon implementation: local muon.py (Polar Express) or torch.optim.Muon')
    parser.add_argument('-muon_per_head', type=int, default=0,
                        help='Per-head Muon (local impl only): orthogonalize attention '
                             'projection updates per head instead of as fused matrices')
    parser.add_argument('-schedule', choices=['wsd', 'cosine'], default='wsd',
                        help='LR schedule: warmup-stable-decay or warmup+cosine to a 10%% floor')
    parser.add_argument('-decay_frac', type=float, default=0.25,
                        help='WSD: fraction of total steps spent in the decay phase')
    parser.add_argument('-momentum_warmup', type=int, default=300,
                        help='Ramp Muon momentum 0.85 -> 0.95 over this many steps (0 disables)')
    parser.add_argument('-no_compile', action='store_true',
                        help='Disable torch.compile on the decoder trunk')
    parser.add_argument('-grad_ckpt', type=int, default=0,
                        help='Gradient checkpointing in the decoder (model.py)')
    parser.add_argument('-value_residual', type=int, default=1,
                        help='Value residual learning (model.py)')
    parser.add_argument('-unet_skips', type=int, default=1,
                        help='U-net skip connections across layers (model.py)')
    parser.add_argument('-attn_res', type=int, default=0,
                        help='Attention Residuals block size (model.py): every N layers, mix '
                             'all previous block outputs via softmax attention over depth '
                             '(0 = disabled)')
    parser.add_argument('-kda', type=int, default=0,
                        help='Kimi Delta Attention (kda.py): use KDA linear attention in '
                             'all but every Nth layer, which keeps full SDPA attention '
                             '(0 = disabled, 1 = every layer KDA, 4 = Kimi-style 3:1 hybrid)')
    parser.add_argument('-swa', type=int, default=0,
                        help='Sliding-window attention (model.py): full-attention '
                             'layers attend only the last N tokens, so with -kda '
                             'the whole stack is linear in sequence length '
                             '(0 = global attention; seqlen must be a multiple of N)')
    parser.add_argument('-shuffle', type=int, default=1,
                        help='Serve train windows in a seeded-permuted order per pass')
    parser.add_argument('-doc_mask', type=int, default=1,
                        help='Intra-document attention: block cross-document attention, '
                             'reset RoPE positions and KDA state at <|endoftext|> '
                             'boundaries (0 = old packed-causal behavior)')
    parser.add_argument('-ce_chunk', type=int, default=16384,
                        help='Rows per chunk in the fused cross-entropy (0 = old unfused path)')
    parser.add_argument('-warmup_steps', type=int, default=100)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-save_every', type=int, default=500,
                        help='Save a checkpoint every N training steps (0 disables periodic saves)')
    parser.add_argument('-val_every', type=int, default=0,
                        help='Run a capped validation pass every N steps (0 = only at epoch end)')
    parser.add_argument('-val_batches', type=int, default=50,
                        help='Max batches per mid-epoch validation pass')
    parser.add_argument('-early_stop', type=int, default=0,
                        help='Early-stop patience in val evaluations (0 = disabled)')
    parser.add_argument('-early_stop_delta', type=float, default=0.005,
                        help='Min relative val-loss improvement that resets patience')
    parser.add_argument('-early_stop_cooldown', type=int, default=300,
                        help='After an early stop, anneal LR to 0 over this many '
                             'steps (0 = stop immediately)')
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str, default='model')
    parser.add_argument('-norm', type=float, default=2.0)
    parser.add_argument('-data_dir', type=str, default='data_cache/cosmopedia',
                        help='Directory with tokenizer.json + train.bin + val.bin (run prepare.py)')

    opt, unknown = parser.parse_known_args()
    return opt
