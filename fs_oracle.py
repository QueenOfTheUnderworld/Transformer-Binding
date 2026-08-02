"""Future-support oracle: ideal context distributions for arbitrary text.

THE ORIGINAL PROJECT IDEA, upgraded by the binding arc. Grade the whole
output distribution against an 'ideal' target: correct token on top, the
REST of the mass on tokens licensed by the situation ('water' -> cup,
lift, sip). The old graph oracle needed derivable structure (premises)
and HURT open narrative (TinyStories 0.176 -> 0.114); embedding-cosine
substitutes Goodharted five metrics.

The computable definition of 'licensed by the situation' for any text:
the CONTENT WORDS THAT ACTUALLY OCCUR IN THE DOCUMENT'S CONTINUATION.
    oracle_t = a*onehot(next) + b*normalize(future-content support, W tokens)
             + c*P_base
Discrete, bounded, anchored, mixture-not-product -- every constraint the
record earned. Scope-law reading: CE constrains 1 number per position and
leaves every argmax-preserving circuit alive; this constrains all V, so
situation-consistent circuits become the cheaper fit.

ARMS (methodology rules: null-intervention control mandatory)
  ce        plain CE
  distill   a*onehot + (b+c)*P_base  -- same weights, NO future term:
            isolates distillation from the situational signal
  oracle    the mixture above

EVAL (rule: never a coherence number without rank and entropy beside it)
  top1        next-token accuracy (must not degrade)
  fmass       prob. mass on the next-W future content words (the
              situational channel; the thing the oracle trains)
  fmass-CTRL  same mass measured on a RANDOM OTHER window's future set
              (stuffing detector: real situational mass raises fmass
              without raising fmass-CTRL)
  rank        mean rank of the true next token
  ent         mean output entropy (collapse detector)
"""

import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from common import set_seed
from transformer_torch import CausalTransformer
import cf_store as CF          # corpus loading + vocab (TinyStories)

PAD, UNK = 0, 1
W = 32                          # future window
N_FUNC = 100                    # top-N frequent tokens treated as function words


def build(words, size=8000):
    c = Counter(words)
    itos = ['<pad>', '<unk>'] + [w for w, _ in c.most_common(size - 2)]
    stoi = {w: i for i, w in enumerate(itos)}
    func = {stoi[w] for w, _ in c.most_common(N_FUNC) if w in stoi}
    func |= {PAD, UNK}
    return stoi, itos, func


def future_support(x, func_mask, V):
    """CORRECTED: support = the ESTABLISHED situation, not the
    future. x (B,T) -> (B,T,V) 0/1 support over content tokens in
    positions <= t (window W back). The ideal tail-of-distribution is
    grounded in what the discourse has already introduced -- in-scene
    alternatives, not upcoming tokens. This is copy-shaped (the words are
    visible in context), so unlike anticipation it is fully computable by
    the model; the anticipation version just inflated entropy (fmass
    0.069 vs CE 0.070, ent 2.25->4.23, fctrl +60%). Name kept for call
    compatibility."""
    B, T = x.shape
    sup = torch.zeros(B, T, V, device=x.device)
    is_content = ~func_mask[x]                       # (B,T) bool
    for t in range(1, T):
        lo = max(0, t + 1 - W)
        w = x[:, lo:t + 1]
        wc = is_content[:, lo:t + 1]
        rows = torch.arange(B, device=x.device).unsqueeze(1).expand_as(w)
        sup[rows[wc], t, w[wc]] = 1.0
    return sup


def train(ids, L, V, func_mask, mode, steps, seed, device,
          a=0.55, b=0.35, c=0.10, bs=32, base=None, lr=6e-4):
    set_seed(seed)
    m = CausalTransformer(V, 256, 4, n_layers=4, max_len=L + 1,
                          dropout=0.0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    if base is not None:
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
    rng = random.Random(seed)
    for st in range(steps):
        m.train()
        ix = [rng.randrange(0, len(ids) - L - 1) for _ in range(bs)]
        x = torch.from_numpy(np.stack([ids[i:i + L + 1] for i in ix])).to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        lg = m(inp)
        if mode == 'ce':
            loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1))
        else:
            with torch.no_grad():
                pb = F.softmax(base(inp), dim=-1)
                oh = torch.zeros_like(pb)
                oh.scatter_(-1, tgt.unsqueeze(-1), 1.0)
                if mode == 'distill':
                    orc = a * oh + (b + c) * pb
                else:
                    sup = future_support(x, func_mask, V)[:, :-1, :]
                    sup = sup / sup.sum(-1, keepdim=True).clamp(min=1e-6)
                    orc = a * oh + b * sup + c * pb
            lp = F.log_softmax(lg, dim=-1)
            loss = -(orc * lp).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(m, ids, L, V, func_mask, device, n=300, seed=7):
    m.eval()
    rng = random.Random(seed)
    ix = [rng.randrange(0, len(ids) - L - 1) for _ in range(n)]
    top1, fmass, fctrl, rank, ent = [], [], [], [], []
    for i0 in range(0, n, 32):
        chunk = ix[i0:i0 + 32]
        x = torch.from_numpy(np.stack([ids[i:i + L + 1] for i in chunk])).to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        p = F.softmax(m(inp), dim=-1)
        sup = future_support(x, func_mask, V)[:, :-1, :]
        # control: future sets from a shuffled pairing of the batch
        perm = torch.roll(torch.arange(x.shape[0]), 1)
        supc = sup[perm]
        T2 = inp.shape[1]
        sel = torch.arange(T2 // 4, 3 * T2 // 4, device=device)  # mid positions
        ps = p[:, sel, :]
        top1.append((ps.argmax(-1) == tgt[:, sel]).float().mean().item())
        fmass.append((ps * sup[:, sel, :]).sum(-1).mean().item())
        fctrl.append((ps * supc[:, sel, :]).sum(-1).mean().item())
        tp = torch.gather(ps, -1, tgt[:, sel].unsqueeze(-1)).squeeze(-1)
        rank.append(((ps > tp.unsqueeze(-1)).sum(-1) + 1).float().mean().item())
        ent.append((-(ps * (ps + 1e-9).log()).sum(-1)).mean().item())
    f = lambda z: float(np.mean(z))
    return f(top1), f(fmass), f(fctrl), f(rank), f(ent)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seeds = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 \
        else [0, 1]
    words = CF.load_words()
    stoi, itos, func = build(words)
    V = len(stoi)
    fm = torch.zeros(V, dtype=torch.bool)
    for t in func:
        fm[t] = True
    ids_all = np.array([stoi.get(w, UNK) for w in words], dtype=np.int64)
    n_tr = int(0.95 * len(ids_all))
    tr_ids, te_ids = ids_all[:n_tr], ids_all[n_tr:]
    L = 96
    fmd = fm.to(device)
    print(f'FS-ORACLE  V={V}  train-tokens={n_tr:,}  W={W}  steps={steps}',
          flush=True)
    print(f'{"arm":>8} {"seed":>4} {"top1":>7} {"fmass":>7} {"fctrl":>7} '
          f'{"rank":>7} {"ent":>6}', flush=True)

    for sd in seeds:
        base = train(tr_ids, L, V, fmd, 'ce', steps, sd, device)
        r = evaluate(base, te_ids, L, V, fmd, device)
        print(f'{"ce":>8} {sd:>4} ' + ' '.join(
            f'{v:7.3f}' for v in r), flush=True)
        for mode in ('distill', 'oracle'):
            m = train(tr_ids, L, V, fmd, mode, steps, sd, device, base=base)
            r = evaluate(m, te_ids, L, V, fmd, device)
            print(f'{mode:>8} {sd:>4} ' + ' '.join(
                f'{v:7.3f}' for v in r), flush=True)
