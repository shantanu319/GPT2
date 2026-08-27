import os
import math
import random
import time

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from core.chat_format import EOS_TOKEN
from core.data import bin_exists, data_feeder, load_bin
from core import dist
from core.model import LOGIT_SOFTCAP, get_model, nopeak_mask
from core.tokenizer import BPETokenizer
from pretrain.config import parse_args
from pretrain.fused_ce import chunked_cross_entropy
from pretrain.muon import Muon as LocalMuon


def resolve_device(no_cuda):
    if not no_cuda:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{dist.local_rank()}")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def build_vocab_indices(vocab_size, device):
    return torch.arange(vocab_size, device=device)


def make_optimizers(model, muon_lr=0.02, embed_lr=3e-3, scalar_lr=0.01,
                    muon_impl='local', muon_per_head=False):
    """Three param groups: Muon on hidden 2D matrices, AdamW on the tied
    embedding (higher LR, decayed), AdamW on every ndim<2 scalar (no decay).

    muon_per_head tags attention projection weights with muon_head_split so the
    local Muon orthogonalizes each head's slice independently (q/k/v split along
    output rows, the out projection along input columns)."""
    embedding_weight = model.decoder.embed.embed.weight
    muon_params, scalar_params = [], []
    for p in model.parameters():
        if not p.requires_grad or p is embedding_weight:
            continue
        if p.ndim < 2:
            scalar_params.append(p)
        else:
            muon_params.append(p)
    if muon_per_head and muon_impl == 'local':
        attns = [layer.attn_1 for layer in model.decoder.layers]
        mha = next((a for a in attns if hasattr(a, 'h_kv')), None)
        n_heads = attns[0].h
        n_kv = mha.h_kv if mha is not None else n_heads
        n_tagged = 0
        for name, p in model.named_parameters():
            if name.endswith(('attn_1.q_linear.weight', 'attn_1.q_proj.weight')):
                p.muon_head_split = (n_heads, 0)
            elif name.endswith(('attn_1.k_linear.weight', 'attn_1.v_linear.weight')):
                p.muon_head_split = (n_kv, 0)
            elif name.endswith(('attn_1.k_proj.weight', 'attn_1.v_proj.weight')):
                p.muon_head_split = (n_heads, 0)
            elif name.endswith(('attn_1.out.weight', 'attn_1.o_proj.weight')):
                p.muon_head_split = (n_heads, 1)
            else:
                continue
            n_tagged += 1
        print(f'per-head Muon: tagged {n_tagged} attention matrices '
              f'({n_heads} q/out heads, {n_kv} kv heads)')
    elif muon_per_head:
        print('warning: -muon_per_head only affects the local Muon; ignoring')
    if muon_impl == 'local':
        muon = LocalMuon(muon_params, lr=muon_lr, weight_decay=0.01)
    else:
        muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=0.01)
    groups = [{'params': [embedding_weight], 'lr': embed_lr, 'weight_decay': 0.1}]
    if scalar_params:
        groups.append({'params': scalar_params, 'lr': scalar_lr, 'weight_decay': 0.0})
    adamw = torch.optim.AdamW(groups, lr=scalar_lr, weight_decay=0.0,
                              betas=(0.8, 0.95), eps=1e-10)
    for opt in (muon, adamw):
        for group in opt.param_groups:
            group['peak_lr'] = group['lr']
    return [muon, adamw]


def lr_factor(step, total_steps, warmup_steps=100, schedule='wsd',
              decay_frac=0.25, min_lr_ratio=0.1):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == 'cosine':
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    # WSD: constant peak until decay_start, then 1 - sqrt(p) down to ~0.
    decay_start = (1 - decay_frac) * total_steps
    if step < decay_start:
        return 1.0
    p = (step - decay_start) / max(1, total_steps - decay_start)
    return 1.0 - math.sqrt(min(1.0, p))


def apply_lr_schedule(optimizers, step, total_steps, warmup_steps,
                      schedule='wsd', decay_frac=0.25):
    factor = lr_factor(step, total_steps, warmup_steps=warmup_steps,
                       schedule=schedule, decay_frac=decay_frac)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['peak_lr'] * factor


class EarlyStopper:
    """Patience-based early stop on validation loss (HF/Lightning semantics).

    An eval counts as an improvement when val loss drops more than
    min_delta (relative) below the best seen; after `patience` consecutive
    non-improving evals the stop triggers."""
    def __init__(self, patience, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.bad_evals = 0
        self.triggered = False

    def check(self, val_loss):
        """Returns True when this eval improved on the best val loss."""
        if val_loss < self.best * (1 - self.min_delta):
            self.best = val_loss
            self.bad_evals = 0
            return True
        self.bad_evals += 1
        self.triggered = self.bad_evals >= self.patience
        return False


def save_checkpoint(model, optimizers, step, path, config=None, extra=None):
    """extra is merged in alongside the standard keys, so a caller with its own
    resume state can keep it in the one file sample.py already knows how to
    read."""
    if not dist.is_main():
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # Strip torch.compile's wrapper prefix so checkpoints load into eager models.
    state = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save({
        'step': step,
        'model': state,
        'optimizers': [o.state_dict() for o in optimizers],
        'config': config,
        **(extra or {}),
    }, path)


def _checkpoint_path(opt, tag):
    base = opt.savename or 'ckpt'
    return os.path.join(opt.dir_name, f'{base}_{tag}.pt')


def batch_loss(model, inX, out, opt, seg=None):
    """Fused chunked CE on CUDA (mask=None engages the causal flash path);
    otherwise the old soft-capped logits + F.cross_entropy route. With seg
    (document segment ids) the decoder builds the intra-document mask itself."""
    if opt.device.type == 'cuda' and getattr(opt, 'ce_chunk', 0) > 0:
        hidden = model.decoder(inX, None, seg_ids=seg)
        return chunked_cross_entropy(hidden.view(-1, hidden.size(-1)),
                                     model.out.weight, model.out.bias,
                                     out.reshape(-1), LOGIT_SOFTCAP, opt.ce_chunk)
    if seg is not None:
        pred = model(inX, None, seg_ids=seg)
    else:
        mask = nopeak_mask(inX.size(1), opt.device)
        pred = model(inX, mask)
    return F.cross_entropy(pred.view(-1, opt.vocab_size), out.reshape(-1))


def _feeder(opt, data, **kw):
    """data_feeder with the run's doc-masking setting applied (None = off;
    main() sets opt.eos_id from the tokenizer when -doc_mask is on)."""
    return data_feeder(data, opt.batchsize, opt.seqlen, opt.device,
                       eos_id=getattr(opt, 'eos_id', None),
                       rank=dist.rank(), world=dist.world_size(), **kw)


def _apply_momentum_warmup(optimizers, step, warmup):
    """Ramp Muon momentum 0.85 -> 0.95 over the first `warmup` steps."""
    if not warmup:
        return
    if step < warmup:
        m = 0.85 + 0.10 * (step + 1) / warmup
    elif step == warmup:
        m = 0.95
    else:
        return
    for o in optimizers:
        for g in o.param_groups:
            if 'momentum' in g:
                g['momentum'] = m


def run_lr_cooldown(model, opt, grad_accum, cooldown_steps):
    """After an early stop, anneal the LR linearly to 0 over a short tail so the
    final checkpoint isn't left mid-schedule (hot)."""
    dist.printr(f"cooldown: annealing LR to 0 over {cooldown_steps} steps")
    groups = [(g, g['lr']) for o in opt.optimizers for g in o.param_groups]
    for o in opt.optimizers:
        o.zero_grad()
    cd, micro = 0, 0
    for inX, out, *rest in _feeder(opt, opt.train):
        seg = rest[0] if rest else None
        frac = max(0.0, 1.0 - (cd + 1) / cooldown_steps)
        for g, lr0 in groups:
            g['lr'] = lr0 * frac
        with torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
            loss = batch_loss(model, inX, out, opt, seg=seg)
        (loss / grad_accum).backward()
        micro += 1
        if micro % grad_accum != 0:
            continue
        dist.average_grads(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.norm)
        for o in opt.optimizers:
            o.step()
            o.zero_grad()
        cd += 1
        if cd % opt.printevery == 0:
            dist.printr(f"cooldown | step {cd}/{cooldown_steps} | loss {loss.item():.4f}")
        if cd >= cooldown_steps:
            break


def train_model(model, opt):
    dist.printr("training model...")
    model.train()
    train_curve = []   # (step, batch loss) recorded at each print log
    val_curve = []     # (step, val loss) recorded at each eval

    grad_accum = max(1, getattr(opt, 'grad_accum', 1))
    val_every = getattr(opt, 'val_every', 0)
    val_batches = getattr(opt, 'val_batches', 50)

    early_stop = getattr(opt, 'early_stop', 0)
    stopper = (EarlyStopper(early_stop, getattr(opt, 'early_stop_delta', 0.005))
               if early_stop else None)
    cooldown_steps = getattr(opt, 'early_stop_cooldown', 300)
    if stopper is not None and not val_every:
        val_every = 1000
        print("early stop needs periodic validation — defaulting -val_every to 1000")

    step = 0
    micro = 0
    stop_training = False

    def eval_and_check(max_batches):
        nonlocal stop_training
        val_loss = validate_model(model, opt, max_batches=max_batches)
        model.train()
        if math.isfinite(val_loss):
            val_curve.append((step, val_loss))
        if stopper is not None:
            if stopper.check(val_loss):
                path = _checkpoint_path(opt, 'best')
                save_checkpoint(model, opt.optimizers, step, path,
                                config=opt.model_config)
                print(f"new best val loss {val_loss:.4f} — saved {path}")
            elif stopper.triggered:
                print(f"early stop: val loss stagnant for {stopper.bad_evals} "
                      f"evals (best {stopper.best:.4f})")
                stop_training = True
        return val_loss
    for epoch in range(opt.epochs):
        epoch_loss = torch.zeros((), device=opt.device)
        epoch_tokens = 0
        iter = 0

        for o in opt.optimizers:
            o.zero_grad()

        for inX, out, *rest in _feeder(opt, opt.train,
                                       shuffle=bool(getattr(opt, 'shuffle', 0)),
                                       seed=42 + epoch):
            seg = rest[0] if rest else None
            iter += 1
            apply_lr_schedule(opt.optimizers, step, opt.total_steps, opt.warmup_steps,
                              schedule=getattr(opt, 'schedule', 'wsd'),
                              decay_frac=getattr(opt, 'decay_frac', 0.25))
            _apply_momentum_warmup(opt.optimizers, step,
                                   getattr(opt, 'momentum_warmup', 0))

            with torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
                loss = batch_loss(model, inX, out, opt, seg=seg)

            epoch_loss += loss.detach() * out.numel()
            epoch_tokens += out.numel()

            (loss / grad_accum).backward()
            micro += 1
            if micro % grad_accum != 0:
                continue

            dist.average_grads(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.norm)
            for o in opt.optimizers:
                o.step()
                o.zero_grad()

            if iter % opt.printevery == 0:
                current_pplx = math.exp(loss.item())
                dist.printr(f"Epoch {epoch+1} | iter {iter} | step {step} | Loss: {loss.item():.4f} | pplx: {current_pplx:.2f}")
                train_curve.append((step, loss.item()))

            step += 1
            if opt.save_every and step % opt.save_every == 0:
                path = _checkpoint_path(opt, f'step{step}')
                save_checkpoint(model, opt.optimizers, step, path, config=opt.model_config)
                dist.printr(f"Saved checkpoint: {path}")
            if val_every and step % val_every == 0:
                eval_and_check(max_batches=val_batches)
                if stop_training:
                    break

        loss_sum, tok_sum = dist.sum_across(
            [epoch_loss.item(), epoch_tokens], opt.device)
        train_loss = loss_sum / tok_sum
        dist.printr(f"Epoch {epoch+1} finished: Train Loss = {train_loss:.4f}")

        # Validate at the end of each epoch:
        eval_and_check(max_batches=None)
        if stop_training:
            break

    if stop_training and cooldown_steps > 0:
        run_lr_cooldown(model, opt, grad_accum, cooldown_steps)

    final_path = _checkpoint_path(opt, 'final')
    save_checkpoint(model, opt.optimizers, step, final_path, config=opt.model_config)
    dist.printr(f"Saved final checkpoint: {final_path}")

    return train_curve, val_curve


def validate_model(model, opt, max_batches=None):
    dist.printr("validating model...")
    model.eval()  # Set to evaluation mode so dropout, etc. are disabled
    total_loss = torch.zeros((), device=opt.device)
    total_tokens = 0

    with torch.no_grad(), torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
        for i, (inX, out, *rest) in enumerate(_feeder(opt, opt.valid)):
            seg = rest[0] if rest else None
            if max_batches is not None and i >= max_batches:
                break
            loss = batch_loss(model, inX, out, opt, seg=seg)
            total_loss += loss * out.numel()
            total_tokens += out.numel()

    loss_sum, tok_sum = dist.sum_across(
        [total_loss.item(), total_tokens], opt.device)
    if tok_sum == 0:
        dist.printr("validation skipped: val set yields no full batches")
        return float('inf')
    avg_loss = loss_sum / tok_sum
    dist.printr(f"Validation Loss = {avg_loss:.4f}")
    return avg_loss


def plot_learning_curves(train_curve, val_curve, test_loss=None, path='learning_curves.png'):
    """train_curve / val_curve are (step, loss) lists recorded during training."""
    plt.figure(figsize=(10, 6))
    if train_curve:
        steps, losses = zip(*train_curve)
        plt.plot(steps, losses, color='tab:blue', alpha=0.3, linewidth=1,
                 label='Training (per log step)')
        window = max(2, len(losses) // 10)
        smoothed = [
            sum(losses[max(0, i - window + 1):i + 1]) / len(losses[max(0, i - window + 1):i + 1])
            for i in range(len(losses))
        ]
        plt.plot(steps, smoothed, color='tab:blue', linewidth=2,
                 label='Training (smoothed)')
    if val_curve:
        v_steps, v_losses = zip(*val_curve)
        plt.plot(v_steps, v_losses, marker='o', color='tab:orange', label='Validation')
    if test_loss is not None:
        plt.axhline(
            y=test_loss, linestyle='--', color='gray',
            label=f'Test (final) = {test_loss:.3f}',
        )
    plt.xlabel('Optimizer step')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.show()


def test_model(model, opt, epoch):
    dist.printr("testing model...")
    model.eval()
    total_loss = torch.zeros((), device=opt.device)
    total_tokens = 0

    with torch.no_grad(), torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
        for x_in, x_out, *rest in _feeder(opt, opt.test):
            seg = rest[0] if rest else None
            loss = batch_loss(model, x_in, x_out, opt, seg=seg)
            total_loss += loss * x_out.numel()
            total_tokens += x_out.numel()

    loss_sum, tok_sum = dist.sum_across(
        [total_loss.item(), total_tokens], opt.device)
    avg_loss = loss_sum / tok_sum
    pplx = math.exp(avg_loss)
    dist.printr(f"Epoch {epoch+1}: Test Loss = {avg_loss:.4f} | Perplexity = {pplx:.2f}")

    return avg_loss


def main():

    random.seed(42)

    opt = parse_args()
    opt.verbose = False

    opt.device = resolve_device(opt.no_cuda)
    if opt.device.type == 'cuda':
        torch.cuda.set_device(opt.device)
    dist.init(opt.device)

    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if dist.is_main():
        os.makedirs(dir_name, exist_ok=True)
    opt.dir_name = dir_name
    opt.log_file = dir_name + "log_file.txt"

    dist.printr(str(opt))

    tok_path = os.path.join(opt.data_dir, 'tokenizer.json')
    train_bin = os.path.join(opt.data_dir, 'train.bin')
    val_bin = os.path.join(opt.data_dir, 'val.bin')
    test_bin = os.path.join(opt.data_dir, 'test.bin')
    for p in (tok_path, train_bin, val_bin, test_bin):
        if not bin_exists(p):
            raise FileNotFoundError(
                f"missing {p} — run `python -m pretrain.prepare --output-dir {opt.data_dir}` first"
            )

    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)
    opt.tokenizer = tokenizer
    opt.vocab_size = tokenizer.vocab_size
    # Intra-document masking: feeders derive segment ids from EOS positions.
    opt.eos_id = tokenizer.special_tokens[EOS_TOKEN] if opt.doc_mask else None

    opt.train = load_bin(train_bin)
    opt.valid = load_bin(val_bin)
    opt.test = load_bin(test_bin)

    model = get_model(opt, opt.vocab_size)

    if opt.device.type == 'cuda' and not opt.no_compile:
        # Compile the trunk only; the fused CE stays eager (custom autograd).
        try:
            model.decoder = torch.compile(model.decoder)
        except Exception as e:
            print(f"torch.compile failed ({e}); falling back to eager")

    opt.model_config = {
        'vocab_size': opt.vocab_size,
        'd_model': opt.d_model,
        'n_layers': opt.n_layers,
        'heads': opt.heads,
        'kv_heads': opt.kv_heads or opt.heads,
        'loops': opt.loops,
        'dropout': opt.dropout,
        'value_residual': bool(getattr(opt, 'value_residual', 0)),
        'unet_skips': bool(getattr(opt, 'unet_skips', 0)),
        'attn_res': getattr(opt, 'attn_res', 0) or 0,
        'kda': getattr(opt, 'kda', 0) or 0,
        'swa': getattr(opt, 'swa', 0) or 0,
        'grad_ckpt': bool(getattr(opt, 'grad_ckpt', 0)),
    }

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dist.printr(f'total params: {params}')
    dist.printr(f'world size: {dist.world_size()} | '
                f'tokens per step: {opt.batchsize * opt.seqlen * opt.grad_accum * dist.world_size():,}')

    opt.optimizers = make_optimizers(model, muon_lr=opt.muon_lr,
                                     embed_lr=opt.embed_lr,
                                     scalar_lr=opt.scalar_lr,
                                     muon_impl=opt.muon_impl,
                                     muon_per_head=bool(getattr(opt, 'muon_per_head', 0)))
    batches_per_epoch = max(1, len(opt.train)
                            // (opt.batchsize * opt.seqlen * dist.world_size()))
    opt.total_steps = max(1, opt.epochs * batches_per_epoch // max(1, opt.grad_accum))

    train_curve, val_curve = train_model(model, opt)
    test_loss = test_model(model, opt, -1)
    if dist.is_main():
        plot_learning_curves(train_curve, val_curve, test_loss=test_loss)
    dist.shutdown()


if __name__ == "__main__":
    main()
