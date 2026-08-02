"""Single-position fusion: the regime where multiple binds MUST coexist.

THE MATH THIS TESTS. Sequentially-emitted answers give each binding its
own residual stream -- per-position load is constant in Q, so no capacity
wall exists there (construction: positional fetch + content fetch, 2
layers). Superposition only bites when ONE position's output depends on
ALL Q bindings jointly. Then the residual must hold Q bound pairs and
linear readout picks up crosstalk ~ sqrt(Q/d). Prediction: fusion
accuracy falls with Q at fixed d_model and recovers with d_model, while
sequential emission stays flat. This is the regime where the
d_model-sets-binding-capacity hypothesis is actually true, if it is.

TASK. ne = 2Q entities. Q are queried. Exactly Q of the 2Q properties
come from a fixed TARGET half of L0. Single-token answer: HOW MANY
queried objects have a target property ("zero".."four").

  ? oz cup . ? joe ball . <2Q fact clauses, shuffled> answer one .

Why this forces fusion:
  - the count is over the QUERIED subset; any one unresolved binding
    changes the achievable answer, so all Q bindings are needed;
  - total target-count over ALL entities is constant (= Q), so the
    bind-free "count target words in the sequence" shortcut carries zero
    information;
  - complement counting (over unqueried entities) also needs exactly Q
    bindings -- no cheaper path.

CAPABILITY PROBE, not an objective comparison: CE + ordered register
supervision on the queried properties (the best-case setup), scored by
the BEST seed. Baseline to beat is the majority count, not 1/n.
"""

import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

from common import set_seed
from wmem_controls import ordered_loss
from stress_test import NAMES, OBJECTS, L0

COUNTS = ['zero', 'one', 'two', 'three', 'four']
TARGET = set(L0[:4])          # red soft big wet
STRUCT = ['has', 'a', '.', '?', 'answer', '<pad>']
PAD = 0


def build_vocab():
    words = ['<pad>'] + STRUCT + NAMES + OBJECTS + L0 + COUNTS
    words = list(dict.fromkeys(words))
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, {i: w for w, i in stoi.items()}


STOI, ITOS = build_vocab()
V = len(STOI)
NOTHING = V
VR = V + 1


def make_example(rng, Q):
    ne = 2 * Q
    ents = rng.sample(NAMES, ne)
    objs = rng.sample(OBJECTS, ne)
    props = rng.sample(sorted(TARGET), Q) + \
        rng.sample(sorted(set(L0) - TARGET), Q)
    rng.shuffle(props)                       # total target-count == Q always
    qidx = rng.sample(range(ne), Q)

    toks = []
    for qi in qidx:
        toks += ['?', ents[qi], objs[qi], '.']
    clauses = [[ents[i], 'has', 'a', props[i], objs[i], '.']
               for i in range(ne)]
    order = list(range(ne))
    rng.shuffle(order)
    done = {}
    for i in order:
        toks += clauses[i]
        done[i] = len(toks) - 1

    count = sum(1 for qi in qidx if props[qi] in TARGET)
    toks += ['answer']
    apos = len(toks)
    toks += [COUNTS[count], '.']

    # ordered registers: slot j holds query j's property once its fact
    # clause has arrived. Best-case dense supervision for the bindings.
    T = len(toks)
    reg = np.full((T, Q), NOTHING, dtype=np.int64)
    for j, qi in enumerate(qidx):
        for k in range(done[qi], T):
            reg[k, j] = STOI[props[qi]]

    ids = [STOI[t] for t in toks]
    cands = [STOI[COUNTS[c]] for c in range(Q + 1)]
    return ids, apos, STOI[COUNTS[count]], cands, reg, count


def make_dataset(n, seed, Q):
    rng = random.Random(seed)
    out = [make_example(rng, Q) for _ in range(n)]
    return out, max(len(x[0]) for x in out)


def pad_batch(items, L, device, K):
    B = len(items)
    x = np.full((B, L), PAD, dtype=np.int64)
    reg = np.full((B, L, K), NOTHING, dtype=np.int64)
    for r, (ids, _, _, _, rg, _) in enumerate(items):
        n = min(len(ids), L)
        x[r, :n] = ids[:n]
        reg[r, :n, :rg.shape[1]] = rg[:n]
    return torch.from_numpy(x).to(device), torch.from_numpy(reg).to(device)


class FModel(torch.nn.Module):
    def __init__(self, K, d_model, max_len, n_layers=4, num_heads=4):
        super().__init__()
        from transformer_torch import CausalTransformer
        self.K = K
        self.core = CausalTransformer(V, d_model, num_heads,
                                      n_layers=n_layers, max_len=max_len,
                                      dropout=0.0)
        self.reg = torch.nn.Linear(d_model, K * VR)
        torch.nn.init.normal_(self.reg.weight, std=0.02)
        torch.nn.init.zeros_(self.reg.bias)

    def both(self, x):
        B, T = x.shape
        h = self.core.get_hidden(x)
        return self.core.output(h), self.reg(h).view(B, T, self.K, VR)


def train(items, L, K, d_model, steps, seed, device, use_reg=True):
    set_seed(seed)
    m = FModel(K, d_model, L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    for _ in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(64)]
        x, rg = pad_batch(ch, L, device, K)
        inp, tgt = x[:, :-1], x[:, 1:]
        lg, rlg = m.both(inp)
        vm = (tgt != PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1),
                               ignore_index=PAD)
        if use_reg:
            loss = loss + 0.5 * ordered_loss(rlg, rg[:, :-1, :], vm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(model, items, L, K, device):
    model.eval()
    hit, reghit = [], []
    for i in range(0, len(items), 128):
        ch = items[i:i + 128]
        x, rg = pad_batch(ch, L, device, K)
        lg, rlg = model.both(x)
        pred = rlg.argmax(-1)
        for r, (ids, apos, ans, cands, rgx, _) in enumerate(ch):
            sub = lg[r, apos - 1, cands]
            hit.append(int(cands[int(sub.argmax())] == ans))
            p0 = apos - 1
            reghit.append(int((pred[r, p0, :rgx.shape[1]] ==
                               rg[r, p0, :rgx.shape[1]]).all().item()))
    return float(np.mean(hit)), float(np.mean(reghit))


def majority_baseline(items):
    from collections import Counter
    c = Counter(it[5] for it in items)
    return c.most_common(1)[0][1] / len(items)


def audit(Q=3, n=4000, seed=3):
    """Numerical shortcut audit for the fusion domain. Prints; returns
    True if clean. Checks the three shortcuts that could substitute for
    binding: (1) counting target-property words without resolving which
    objects are queried -- dead by construction, the total is constant;
    (2) answering from the majority count; (3) position of the first
    target-property mention."""
    import random as _r
    rng = _r.Random(seed)
    exs = [make_example(rng, Q) for _ in range(n)]
    tot = {sum(1 for w in [ITOS[t] for t in e[0]] if w in TARGET)
           for e in exs}
    maj = majority_baseline(exs)
    firsts, counts = [], []
    for ids, apos, ans, cands, reg, c in exs:
        w = [ITOS[t] for t in ids]
        firsts.append(next((i for i, x in enumerate(w) if x in TARGET), -1))
        counts.append(c)
    corr = float(np.corrcoef(firsts, counts)[0, 1])
    ok = (len(tot) == 1) and abs(corr) < 0.05
    print(f'  AUDIT n={n}: total-target-words={sorted(tot)} '
          f'(must be one value)   majority={maj:.3f}   '
          f'corr(first-target-pos, answer)={corr:+.3f} (want ~0)')
    return ok


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Qs = [int(q) for q in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2, 3]
    dims = [int(d) for d in sys.argv[2].split(',')] if len(sys.argv) > 2 \
        else [32, 64, 128]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    seeds = [0, 1, 2]

    for Q in Qs:
        tr, La = make_dataset(24000, 0, Q)
        te, Lb = make_dataset(1500, 9000, Q)
        L = max(La, Lb) + 1
        mb = majority_baseline(te)
        print(f'\nQ={Q}  ne={2*Q}  L={L}  majority-baseline={mb:.3f}  '
              f'steps={steps}', flush=True)
        print('  sample: ' + ' '.join(ITOS[t] for t in tr[0][0]), flush=True)
        if not audit(Q):
            print('  AUDIT FAILED -- do not trust results below', flush=True)
        for d in dims:
            accs, regs = [], []
            for sd in seeds:
                m = train(tr, L, Q, d, steps, sd, device)
                a, rh = evaluate(m, te, L, Q, device)
                accs.append(a)
                regs.append(rh)
            print(f'  d={d:>4}  acc best {max(accs):.3f}  '
                  f'mean {np.mean(accs):.3f}  raw {[round(a,3) for a in accs]}'
                  f'  reg-exact best {max(regs):.3f}', flush=True)
