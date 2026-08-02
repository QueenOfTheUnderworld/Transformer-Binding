"""What is the register gain actually made of?

At n_entities=3, n_queries=2 the first run gave ans0: CE 0.635,
reg[queried] 0.595, reg[all] 1.000. The uninformative control BEAT the
informative registers, so the gain is not query-binding information.
Two candidate explanations, and one design suspicion:

  A  separability -- forcing each ANSWER CANDIDATE to have its own clean
     direction is what helps; the query then just selects among them.
  B  dense supervision -- any auxiliary readout of comparable density
     helps, regardless of what it tracks.
     -> separated by reg[names]: same structure, same density, tokens
        OUTSIDE the answer space. Helps under B, not under A.

  C  permutation invariance is the problem. reg[queried] reads out at
     0.974 yet scores BELOW plain CE. An unordered set of the two
     relevant properties cannot say which answers query 0, so it narrows
     3 candidates to 2 and coin-flips (0.595 ~ 0.333 + half the rest).
     -> tested by reg[queried] with an ORDERED loss. If that recovers,
        the scrambling -- added to stop positional memorisation -- is
        itself destroying the order the task needs.

3 seeds: the numbers above are n=1, and roughly one run in three in this
project fails to learn its base case.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

import wmem as W


def ordered_loss(reg_logits, reg_tgt, valid_mask):
    """Slot k must hold target k. No matching."""
    B, T, K, _ = reg_logits.shape
    lp = F.log_softmax(reg_logits, dim=-1)
    nll = -lp.gather(3, reg_tgt.unsqueeze(-1)).squeeze(-1)      # (B,T,K)
    return (nll.mean(-1) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)


def train_cond(items, L, K, cond, steps, seed, device):
    """cond: (use_reg, reg_src, perm_inv)"""
    use_reg, reg_src, perm_inv = cond
    if perm_inv or not use_reg:
        return W.train(items, L, K, mode='ce', use_reg=use_reg,
                       reg_src=reg_src, steps=steps, seed=seed, device=device)
    # ordered variant: same training loop, non-matching register loss
    import random
    from common import set_seed
    set_seed(seed)
    m = W.WMModel(K, max_len=L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    for _ in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(64)]
        x, rg = W.pad_batch(ch, L, device, K, reg_src)
        inp, tgt = x[:, :-1], x[:, 1:]
        lg, rlg = m.both(inp)
        vm = (tgt != W.PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, W.V), tgt.reshape(-1),
                               ignore_index=W.PAD)
        loss = loss + 0.5 * ordered_loss(rlg, rg[:, :-1, :], vm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


CONDS = [
    ('CE',                    (False, 'query', True)),
    ('reg[queried] perminv',  (True,  'query', True)),
    ('reg[queried] ORDERED',  (True,  'query', False)),
    ('reg[all] perminv',      (True,  'all',   True)),
    ('reg[names] perminv',    (True,  'names', True)),
]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    which = [int(i) for i in sys.argv[1].split(',')] if len(sys.argv) > 1 \
        else list(range(len(CONDS)))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    seeds = [0, 1, 2]
    ne, nq = 3, 2
    K = max(nq, ne) + 1

    tr, La = W.make_dataset(18000, seed=0, n_entities=ne, n_queries=nq)
    te, Lb = W.make_dataset(1500, seed=9000, n_entities=ne, n_queries=nq)
    L = max(La, Lb) + 1
    print(f'n_entities={ne}  n_queries={nq}  K={K}  steps={steps}  '
          f'seeds={seeds}   chance ans0 = {1.0/ne:.3f}', flush=True)

    for i in which:
        tag, cond = CONDS[i]
        a0, a1, rr = [], [], []
        for sd in seeds:
            m = train_cond(tr, L, K, cond, steps, sd, device)
            acc, sp, sc, sf = W.evaluate(m, te, L, K, device)
            a0.append(acc[0]); a1.append(acc[1]); rr.append(sp)
        print(f'  {tag:>22}  ans0 {np.mean(a0):.3f}+-{np.std(a0):.3f}   '
              f'ans1 {np.mean(a1):.3f}   reg {np.mean(rr):.3f}   '
              f'raw {[round(v,3) for v in a0]}', flush=True)
