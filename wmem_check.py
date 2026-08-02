"""Sanity checks on the wmem domain before spending training time.

Checks, in order of how badly each would invalidate the run:
  1. register targets track clause arrival (NOTHING -> prop -> cons)
  2. the oracle support actually reaches the right consequence, and
     separates the two queries -- if it does not, "oracle" is noise
  3. the answer is not locally derivable: no shortcut from position or
     from the surface form of the answer clause
  4. permutation-invariant set loss is genuinely order-blind
"""

import random
import itertools
import torch
import numpy as np
import wmem as W

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = random.Random(0)

print('=' * 68)
print('1. EXAMPLE AND REGISTER TARGETS  (n_entities=2, n_queries=2)')
print('=' * 68)
ids, qk, ap, cands, valid, rg = W.make_example(rng, 2, 2)[:6]
print('  ' + ' '.join(W.ITOS[t] for t in ids))
print(f'  query keys {qk} -> ' + str([[W.ITOS[ids[p]] for p in k] for k in qk]))
print(f'  answer pos {ap} -> ' + str([W.ITOS[v] for v in valid]) +
      '   candidates ' + str([W.ITOS[c] for c in cands]))
print()


def nm(t):
    return '<nothing>' if t == W.NOTHING else W.ITOS[t]


for i in range(len(ids)):
    star = ' <-- answer' if i in [a - 1 for a in ap] else ''
    print(f'   {i:3d} {W.ITOS[ids[i]]:>8} | ' +
          '  '.join(f'{nm(t):>10}' for t in rg[i]) + star)

print()
print('=' * 68)
print('2. ORACLE SUPPORT AT EACH ANSWER POSITION')
print('=' * 68)
items, L = W.make_dataset(4, seed=1, n_entities=2, n_queries=2)
x, _ = W.pad_batch(items, L, device, 4)
for j in range(2):
    seedm = torch.zeros_like(x, dtype=torch.float)
    upto = torch.ones(len(items), dtype=torch.long, device=device)
    for r, it in enumerate(items):
        for kp in it[1][j]:
            seedm[r, kp] = 1.0
        upto[r] = it[2][j]
    sup = W.support_seeded(x, seedm, upto, device)
    it = items[0]
    top = torch.topk(sup[0], 6)
    want = W.ITOS[it[4][j]]
    other = [W.ITOS[c] for c in it[3] if c != it[4][j]]
    print(f'  query {j}: seed=' + str([W.ITOS[x[0, p].item()] for p in it[1][j]]) +
          f'  target={want}  rival={other}')
    print('     support: ' + '  '.join(
        f'{W.ITOS[i.item()]}({v.item():.3f})' for v, i in zip(*top)))
    ranks = {W.ITOS[c]: float(sup[0, c]) for c in it[3]}
    print(f'     -> target {ranks[want]:.3f}  vs rival ' +
          '  '.join(f'{k} {v:.3f}' for k, v in ranks.items() if k != want))

# how often is the correct consequence strictly top of the candidate set
ok = [0, 0]
big, Lb = W.make_dataset(400, seed=2, n_entities=2, n_queries=2)
xb, _ = W.pad_batch(big, Lb, device, 4)
for j in range(2):
    seedm = torch.zeros_like(xb, dtype=torch.float)
    upto = torch.ones(len(big), dtype=torch.long, device=device)
    for r, it in enumerate(big):
        for kp in it[1][j]:
            seedm[r, kp] = 1.0
        upto[r] = it[2][j]
    sup = W.support_seeded(xb, seedm, upto, device)
    for r, it in enumerate(big):
        sub = [float(sup[r, c]) for c in it[3]]
        ok[j] += int(it[3][int(np.argmax(sub))] == it[4][j])
print(f'\n  oracle picks the right consequence:  slot0 {ok[0]/len(big):.3f}'
      f'   slot1 {ok[1]/len(big):.3f}   (chance 0.500)')

print()
print('=' * 68)
print('3. SHORTCUT AUDIT')
print('=' * 68)
au, Lu = W.make_dataset(3000, seed=3, n_entities=2, n_queries=2)
# does answer-slot-0 correlate with position of the fact clause?
first_fact, cnt = 0, 0
pos_shortcut = [0, 0]
for it in au:
    ids2, qk2, ap2, cd, vl = it[:5]
    # "always answer the consequence that appears EARLIEST in the context"
    firsts = sorted(cd, key=lambda c: ids2.index(c))
    pos_shortcut[0] += int(firsts[0] == vl[0])
    pos_shortcut[1] += int(firsts[0] == vl[1])
n = len(au)
print(f'  earliest-mentioned consequence == answer0: {pos_shortcut[0]/n:.3f} (chance .500)')
print(f'  earliest-mentioned consequence == answer1: {pos_shortcut[1]/n:.3f} (chance .500)')
cg = [0, 0]
for it in au:
    ids2, _, _, cd, vl = it[:5]
    counts = {c: ids2.count(c) for c in cd}
    top = max(counts, key=counts.get)
    tie = len(set(counts.values())) == 1
    cg[0] += int((not tie) and top == vl[0])
print(f'  most-frequent consequence == answer0:      {cg[0]/n:.3f} (0 means no count cheat)')
qorder = sum(int(it[4][0] == it[3][0]) for it in au) / n
print(f'  answer0 == first-listed candidate:         {qorder:.3f} (chance .500)')

print()
print('=' * 68)
print('4. SET LOSS IS ORDER-BLIND')
print('=' * 68)
K = 4
perms = torch.tensor(list(itertools.permutations(range(K))), device=device)
torch.manual_seed(0)
lg = torch.randn(2, 5, K, W.VR, device=device)
tg = torch.randint(0, W.VR, (2, 5, K), device=device)
vmk = torch.ones(2, 5, device=device)
base = W.set_loss(lg, tg, vmk, perms)
shuf = W.set_loss(lg, tg[:, :, torch.randperm(K, device=device)], vmk, perms)
sw = W.set_loss(lg[:, :, torch.randperm(K, device=device)], tg, vmk, perms)
print(f'  loss                      {base:.6f}')
print(f'  targets permuted          {shuf:.6f}   delta {abs(base-shuf):.2e}')
print(f'  prediction heads permuted {sw:.6f}   delta {abs(base-sw):.2e}')
print('  (both deltas must be ~0: slot index carries no information)')
