"""Dissect the register scaffold: WHAT about it enables bind formation?

Established: Q=3 fusion, d=256, 5000 steps -- CE-only 0/4 flat; CE +
dense ordered-register supervision of the queried PROPERTIES 8/8. The
scaffold is read out at ~0.01 by solvers, so it acts during formation
only. Three hypotheses, one arm each (Q=3, d=256, 5000 steps, 3 seeds):

  prop    baseline scaffold (replication arm for these seeds)
  names   registers track the queried entities' NAMES -- identical
          density, structure, and clause-arrival timing, but the content
          is useless for the count (names are literals already in the
          query; no binding needed to know them). If names ~= prop, the
          scaffold works by dense gradient alone. If names ~= CE-flat,
          the CONTENT (bound values) is what matters.
  sparse  correct property targets but supervised ONLY at the answer
          prompt position, not densely. If sparse ~= prop, density is
          irrelevant and one extra supervised position suffices. If
          sparse ~= flat, the dense gradient path along the sequence is
          what forms the circuit.

Run the CE arm separately (mode 'ce') at 20000 steps, 2 seeds --
enabler vs accelerator. If CE solves late, the scaffold is a ~4x+
speedup; if flat at 20k, it is an enabler at this scale.
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from common import set_seed
import bind_fusion as B
from wmem_controls import ordered_loss


def names_register(items):
    """Rebuild register targets tracking the queried entities' NAMES.
    Same switch-on times as the property registers (the fact clause's
    final '.'), recovered by parsing each example's tokens."""
    out = []
    for it in items:
        ids, apos, ans, cands, reg, cnt = it
        T, Q = reg.shape[0], reg.shape[1]
        words = [B.ITOS[t] for t in ids]
        nreg = np.full((T, Q), B.NOTHING, dtype=np.int64)
        for j in range(Q):
            # first position where slot j leaves NOTHING
            t0 = next((t for t in range(T) if reg[t, j] != B.NOTHING), None)
            if t0 is None:
                continue
            prop = B.ITOS[reg[t0, j]]
            # the fact clause ending at t0 is  "<ent> has a <prop> <obj> ."
            ent_tok = None
            for k in range(len(words) - 3):
                if words[k + 1] == 'has' and words[k + 3] == prop \
                        and k + 5 < T and k + 5 >= t0 - 1:
                    ent_tok = B.STOI[words[k]]
                    break
            if ent_tok is None:
                for k in range(len(words) - 3):
                    if words[k + 1] == 'has' and words[k + 3] == prop:
                        ent_tok = B.STOI[words[k]]
                        break
            for t in range(t0, T):
                nreg[t, j] = ent_tok
        out.append((ids, apos, ans, cands, nreg, cnt))
    return out


def train(items, L, Q, d, steps, seed, device, mode, te=None, every=250):
    set_seed(seed)
    m = B.FModel(Q, d, L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    traj = []
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(64)]
        x, rg = B.pad_batch(ch, L, device, Q)
        inp, tgt = x[:, :-1], x[:, 1:]
        lg, rlg = m.both(inp)
        vm = (tgt != B.PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, B.V), tgt.reshape(-1),
                               ignore_index=B.PAD)
        if mode in ('prop', 'names'):
            loss = loss + 0.5 * ordered_loss(rlg, rg[:, :-1, :], vm)
        elif mode == 'sparse':
            smask = torch.zeros_like(vm)
            for r, it in enumerate(ch):
                p = it[1] - 1
                if p < smask.shape[1]:
                    smask[r, p] = 1.0
            loss = loss + 0.5 * ordered_loss(rlg, rg[:, :-1, :], smask)
        # mode 'ce': nothing added
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if te is not None and (st + 1) % every == 0:
            a, _ = B.evaluate(m, te, L, Q, device)
            traj.append((st + 1, round(a, 4)))
    return m, traj


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = sys.argv[1]
    if mode not in ('prop', 'names', 'sparse', 'ce'):
        sys.exit(f"unknown mode {mode!r}; expected one of "
                 f"prop | names | sparse | ce")
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 \
        else [0, 1, 2]
    Q, d = 3, 256

    tr, La = B.make_dataset(24000, 0, Q)
    te, Lb = B.make_dataset(1500, 9000, Q)
    L = max(La, Lb) + 1
    if mode == 'names':
        tr = names_register(tr)     # eval uses the unmodified `te`: the
                                    # metric is answer accuracy, which does
                                    # not depend on register targets
    print(f'DISSECT mode={mode}  Q={Q} d={d} steps={steps} seeds={seeds}  '
          f'majority={B.majority_baseline(te):.3f}', flush=True)

    results = {}
    for sd in seeds:
        _, traj = train(tr, L, Q, d, steps, sd, device, mode, te=te)
        results[sd] = traj
        solve = next((t for t, a in traj if a >= 0.95), None)
        line = ' '.join(f'{a:.2f}' for _, a in traj)
        print(f'  seed {sd}:  solve@{solve}  traj [{line}]', flush=True)
        with open(f'dissect_{mode}_{steps}.json', 'w') as f:
            json.dump({'mode': mode, 'steps': steps, 'results': results}, f)
