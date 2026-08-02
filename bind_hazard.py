"""First-passage measurement: WHY does the fusion circuit form or not?

Three formal models of circuit formation, distinguished by the
distribution of solve times across seeds (solve = held-out acc >= 0.95):

  A  basin volume    P(solved by t) = p_V * 1[t > t_min]
     decided at init; solvers cluster at one time; failures permanent.
  B  stochastic search    P(solved by t) = 1 - exp(-lambda (t - t0))
     memoryless plateau escape; solve times spread; EVERY seed solves
     eventually (grokking picture -- patience wins).
  C  race vs entrenchment    solves only before t_e, then never
     shortcut (majority class) fits first and kills the residual
     gradient; early solves only; failures permanent; longer runs buy
     nothing. Predicts the observed d=512 < d=256 (bigger fits the
     shortcut faster).

Protocol: Q=3, d=256 (the 1-of-3 cell), 8 seeds, 5000 steps, eval every
250. Trajectories also give the early-predictor test: are eventual
solvers distinguishable at step ~500 (A/C) or not (B)?
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

from common import set_seed
import bind_fusion as B
from wmem_controls import ordered_loss


def run_seed(tr, te, L, Q, d, steps, seed, device, every=250, use_reg=True):
    import random
    set_seed(seed)
    m = B.FModel(Q, d, L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    traj = []
    for st in range(steps):
        m.train()
        ch = [tr[rng.randrange(len(tr))] for _ in range(64)]
        x, rg = B.pad_batch(ch, L, device, Q)
        inp, tgt = x[:, :-1], x[:, 1:]
        lg, rlg = m.both(inp)
        vm = (tgt != B.PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, B.V), tgt.reshape(-1),
                               ignore_index=B.PAD)
        if use_reg:
            loss = loss + 0.5 * ordered_loss(rlg, rg[:, :-1, :], vm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if (st + 1) % every == 0:
            a, _ = B.evaluate(m, te, L, Q, device)
            traj.append((st + 1, round(a, 4)))
    return traj


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    seeds = [int(s) for s in sys.argv[4].split(',')] if len(sys.argv) > 4 \
        else list(range(8))
    use_reg = not (len(sys.argv) > 5 and sys.argv[5] == 'noreg')
    out_json = f'hazard_Q{Q}_d{d}' + ('' if use_reg else '_noreg') + '.json'

    tr, La = B.make_dataset(24000, 0, Q)
    te, Lb = B.make_dataset(1500, 9000, Q)
    L = max(La, Lb) + 1
    mb = B.majority_baseline(te)
    print(f'Q={Q}  d={d}  steps={steps}  seeds={seeds}  majority={mb:.3f}  '
          f'use_reg={use_reg}', flush=True)

    results = {}
    for sd in seeds:
        traj = run_seed(tr, te, L, Q, d, steps, sd, device, use_reg=use_reg)
        results[sd] = traj
        solve = next((t for t, a in traj if a >= 0.95), None)
        line = ' '.join(f'{a:.2f}' for _, a in traj)
        print(f'  seed {sd}:  solve@{solve}  traj [{line}]', flush=True)
        with open(out_json, 'w') as f:
            json.dump({'Q': Q, 'd': d, 'steps': steps, 'use_reg': use_reg,
                       'majority': mb, 'results': results}, f)

    solves = [next((t for t, a in tr_ if a >= 0.95), None)
              for tr_ in results.values()]
    ns = [s for s in solves if s is not None]
    print(f'\n  solved {len(ns)}/{len(solves)}   times {sorted(ns)}',
          flush=True)
