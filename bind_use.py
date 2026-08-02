"""USING the binds: retrieved values become keys for further computation.

The fusion result shows the model can hold Q bindings and count over
them. This task requires it to COMPUTE with them: query two entities,
retrieve each one's property (2 binds), then look up what that PAIR of
properties implies in an in-context rule table.

  ? oz joe .  oz has a wet cup .  sam has a red pen .  joe has a round
  ball .  wet round are friends .  wet red are rivals .  red round are
  strangers .  answer friends .

bind -> fuse -> RE-BIND: the pair of computed values acts as a literal
key for a third lookup. This is exactly the operation the depth-2 wall
was about (a lookup keyed by a computed value), except here the computed
key is a PAIR, and the whole thing sits in the regime where patience is
known to work.

Why there is no cheaper path (audited numerically in __main__):
  - the rule table is PER-EXAMPLE (pair->relation assignment reshuffled
    every example), so it cannot live in the weights;
  - every rule clause shares one property with the queried pair, so a
    single resolved binding narrows nothing: each queried property
    appears in exactly 2 rules with different answers;
  - relation words are balanced -> majority = 1/n_relations;
  - the queried pair's rule sits at a random position among the rules.

Entities: 3, props distinct; rules cover the 3 unordered prop pairs with
3 distinct relation words in shuffled assignment. Chance (candidate-
restricted over the 3 relation words present) = 1/3.
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from common import set_seed
from transformer_torch import CausalTransformer
from stress_test import NAMES, OBJECTS, L0

RELS = ['friends', 'rivals', 'strangers', 'partners']
STRUCT = ['has', 'a', 'are', '.', '?', 'answer', '<pad>']
PAD = 0


def build_vocab():
    words = ['<pad>'] + STRUCT + NAMES + OBJECTS + L0 + RELS
    words = list(dict.fromkeys(words))
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, {i: w for w, i in stoi.items()}


STOI, ITOS = build_vocab()
V = len(STOI)


NOTHING = V
VR = V + 1


def make_example(rng, ne=3):
    """ne entities, distinct props; rules for all C(ne,2) prop pairs with
    distinct relation words, assignment shuffled per example."""
    ents = rng.sample(NAMES, ne)
    objs = rng.sample(OBJECTS, ne)
    props = rng.sample(L0, ne)
    qa, qb = rng.sample(range(ne), 2)

    pairs = [(i, j) for i in range(ne) for j in range(i + 1, ne)]
    rels = rng.sample(RELS, len(pairs))
    table = {}
    for (i, j), rel in zip(pairs, rels):
        table[frozenset((i, j))] = rel

    toks = ['?', ents[qa], ents[qb], '.']
    facts = [[ents[i], 'has', 'a', props[i], objs[i], '.']
             for i in range(ne)]
    rng.shuffle(facts)
    for f in facts:
        toks += f
    rules = []
    for (i, j), rel in zip(pairs, rels):
        a, b = (props[i], props[j]) if rng.random() < 0.5 \
            else (props[j], props[i])
        rules.append([a, b, 'are', rel, '.'])
    rng.shuffle(rules)
    for r in rules:
        toks += r

    ans = table[frozenset((qa, qb))]
    toks += ['answer']
    apos = len(toks)
    toks += [ans, '.']

    # ordered registers: slot 0 = first queried entity's property, slot 1
    # = second's, from each fact clause onward. The CE-only ablation on the
    # count task (0/4 vs 8/8) showed this scaffold is the enabling
    # ingredient, so bind-use gets the same treatment.
    T = len(toks)
    reg = np.full((T, 2), NOTHING, dtype=np.int64)
    for slot, q in enumerate((qa, qb)):
        seen = None
        for k, w in enumerate(toks):
            if seen is None and w == ents[q] and k + 1 < T and toks[k + 1] == 'has':
                seen = k + 3
            if seen is not None and k >= seen:
                reg[k, slot] = STOI[props[q]]

    ids = [STOI[t] for t in toks]
    cands = [STOI[r] for r in rels]
    return ids, apos, STOI[ans], cands, reg


def make_dataset(n, seed, ne=3):
    rng = random.Random(seed)
    out = [make_example(rng, ne) for _ in range(n)]
    return out, max(len(x[0]) for x in out)


def pad_batch(items, L, device, K=2):
    x = np.full((len(items), L), PAD, dtype=np.int64)
    reg = np.full((len(items), L, K), NOTHING, dtype=np.int64)
    for r, it in enumerate(items):
        ids, rg = it[0], it[4]
        n = min(len(ids), L)
        x[r, :n] = ids[:n]
        reg[r, :n, :] = rg[:n]
    return torch.from_numpy(x).to(device), torch.from_numpy(reg).to(device)


def train(items, L, d, steps, seed, device, every=250, te=None,
          use_reg=False):
    from wmem_controls import ordered_loss
    set_seed(seed)
    if use_reg:
        import bind_fusion
        m = RModel(2, d, L + 1).to(device)
    else:
        m = CausalTransformer(V, d, 4, n_layers=4, max_len=L + 1,
                              dropout=0.0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    traj = []
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(64)]
        x, rg = pad_batch(ch, L, device)
        inp, tgt = x[:, :-1], x[:, 1:]
        if use_reg:
            lg, rlg = m.both(inp)
        else:
            lg = m(inp)
        loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1),
                               ignore_index=PAD)
        if use_reg:
            vm = (tgt != PAD).float()
            loss = loss + 0.5 * ordered_loss(rlg, rg[:, :-1, :], vm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if te is not None and (st + 1) % every == 0:
            traj.append((st + 1, round(evaluate(m, te, L, device), 4)))
    return m, traj


class RModel(torch.nn.Module):
    def __init__(self, K, d_model, max_len):
        super().__init__()
        self.K = K
        self.core = CausalTransformer(V, d_model, 4, n_layers=4,
                                      max_len=max_len, dropout=0.0)
        self.reg = torch.nn.Linear(d_model, K * VR)
        torch.nn.init.normal_(self.reg.weight, std=0.02)
        torch.nn.init.zeros_(self.reg.bias)

    def forward(self, x):
        return self.core(x)

    def both(self, x):
        B, T = x.shape
        h = self.core.get_hidden(x)
        return self.core.output(h), self.reg(h).view(B, T, self.K, VR)


@torch.no_grad()
def evaluate(model, items, L, device):
    model.eval()
    hit = []
    for i in range(0, len(items), 128):
        ch = items[i:i + 128]
        x, _ = pad_batch(ch, L, device)
        lg = model(x)
        for r, (ids, apos, ans, cands, _) in enumerate(ch):
            sub = lg[r, apos - 1, cands]
            hit.append(int(cands[int(sub.argmax())] == ans))
    return float(np.mean(hit))


def audit(n=4000, seed=3, ne=3):
    """Numerical shortcut audit. Prints; returns True if clean."""
    rng = random.Random(seed)
    exs = [make_example(rng, ne) for _ in range(n)]
    # 1. majority relation
    from collections import Counter
    c = Counter(ITOS[e[2]] for e in exs)
    maj = c.most_common(1)[0][1] / n
    # 2. single-prop determinism: given ONE queried prop and the visible
    # rules, how often is the answer determined? (should be ~1/2 among
    # that prop's 2 rules -> accuracy from one bind = 0.5, not 1.0)
    # measured as: pick the relation of a RANDOM rule containing the
    # first queried entity's prop
    one = 0
    for ids, apos, ans, cands, _ in exs:
        words = [ITOS[t] for t in ids]
        qa_prop = None
        # first queried entity name is words[1]; find its fact
        for k in range(len(words) - 3):
            if words[k] == words[1] and words[k + 1] == 'has':
                qa_prop = words[k + 3]
                break
        rels_with = [words[k + 3] for k in range(len(words) - 4)
                     if words[k + 2] == 'are'
                     and qa_prop in (words[k], words[k + 1])]
        pick = rels_with[rng.randrange(len(rels_with))]
        one += int(STOI[pick] == ans)
    # 3. rule position: is the answer the relation of the FIRST rule?
    first = 0
    for ids, apos, ans, cands, _ in exs:
        words = [ITOS[t] for t in ids]
        for k in range(len(words) - 4):
            if words[k + 2] == 'are':
                first += int(STOI[words[k + 3]] == ans)
                break
    print(f'  AUDIT n={n}: majority {maj:.3f} (want ~.333)   '
          f'one-bind {one/n:.3f} (want ~.5)   '
          f'first-rule {first/n:.3f} (want ~.333)')
    return maj < 0.4 and one / n < 0.6 and first / n < 0.4


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 \
        else [0, 1, 2]
    use_reg = len(sys.argv) > 4 and sys.argv[4] == 'reg'

    tr, La = make_dataset(24000, 0)
    te, Lb = make_dataset(1500, 9000)
    L = max(La, Lb) + 1
    print(f'BIND-USE  d={d}  steps={steps}  L={L}  chance=0.333  '
          f'use_reg={use_reg}', flush=True)
    print('  sample: ' + ' '.join(ITOS[t] for t in tr[0][0]), flush=True)
    if not audit():
        print('  AUDIT FAILED -- do not trust results below', flush=True)

    results = {}
    for sd in seeds:
        _, traj = train(tr, L, d, steps, sd, device, te=te, use_reg=use_reg)
        results[sd] = traj
        solve = next((t for t, a in traj if a >= 0.95), None)
        line = ' '.join(f'{a:.2f}' for _, a in traj)
        print(f'  seed {sd}:  solve@{solve}  traj [{line}]', flush=True)
        with open(f'binduse_d{d}' + ('_reg' if use_reg else '') + '.json', 'w') as f:
            json.dump({'d': d, 'steps': steps, 'results': results}, f)
