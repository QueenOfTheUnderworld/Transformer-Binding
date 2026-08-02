"""
Stress test: find where the justification oracle breaks.

It works on two derivable domains (syllogism 0.502 -> 1.000, prose-surface
deduction 0.502 -> 0.840). Before building anything on top of that, find
the limits. Six axes:

  1. DEPTH          3/4/5-hop chains. Does the gain survive longer
                    derivations, or does propagation smear past 2 hops?
  2. DISTRACTORS    more competing entities. Does binding hold at 4-6?
  3. FILLER         irrelevant padding sentences. Graph propagation should
                    be robust; if it is not, the chains were being carried
                    by proximity rather than structure.
  4. SEEDS          everything so far is n=1.
  5. B-CLIFF        0.45 works, 0.60 collapses. Where exactly is the edge?
  6. TRUNCATED      give the task a 3-hop derivation but propagate only
     ORACLE         2 hops. Does a PARTIALLY CORRECT oracle still help, or
                    actively mislead? -- the sharpest test: does the model
                    need a COMPLETE justification or just a directional one?

Chain domain (depth-parameterised):
    lily has a red ball .  red things are bright .  bright things are nice .
    so lily 's ball is ___            depth=3:  red -> bright -> nice
Distractor entities own the excluded properties, so counting is dead;
the queried entity is random, so position is dead. Both verified.
"""

import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

from common import set_seed
from transformer_torch import CausalTransformer

NAMES = ['lily', 'ben', 'tom', 'mia', 'sam', 'ana', 'joe', 'eve',
         'kit', 'gus', 'ivy', 'oz']
OBJECTS = ['ball', 'box', 'cup', 'hat', 'toy', 'rock', 'pen', 'mug']
# chain vocabulary: level-0 properties, then successive consequences
L0 = ['red', 'soft', 'big', 'wet', 'old', 'blue', 'sharp', 'round']
L1 = ['bright', 'squishy', 'heavy', 'slippery', 'dusty', 'calm', 'risky', 'rollable']
L2 = ['nice', 'comfy', 'solid', 'tricky', 'faded', 'quiet', 'scary', 'fun']
L3 = ['loved', 'cozy', 'sturdy', 'messy', 'aged', 'still', 'wild', 'lively']
L4 = ['prized', 'warm', 'firm', 'muddy', 'worn', 'silent', 'fierce', 'merry']
LEVELS = [L0, L1, L2, L3, L4]
STRUCT = ['has', 'a', 'things', 'are', 'so', 'is', '.', "'s", 'the', 'day',
          'was', 'sunny', 'they', 'played', '<pad>']
PAD = 0

FILLER = [['the', 'day', 'was', 'sunny', '.'], ['they', 'played', '.']]


def build_vocab():
    words = ['<pad>'] + STRUCT + NAMES + OBJECTS
    for lv in LEVELS:
        words += lv
    words = list(dict.fromkeys(words))
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, {i: w for w, i in stoi.items()}


STOI, ITOS = build_vocab()
V = len(STOI)
FUNC = {STOI[w] for w in STRUCT if w in STOI}


def make_example(rng, depth=2, n_entities=2, n_filler=0):
    """depth = number of rule applications. depth=2 -> prop -> L1 -> L2.

    SHORTCUT REMOVED. The previous version used a single `slot` per entity
    and built the whole chain as LEVELS[d][slot], so the slot DETERMINED
    the answer: identify the property (one binding hop, readable straight
    off "joe has a round ball") and the conclusion follows by a fixed
    learned lookup. Depth never actually mattered -- which is why depths
    2, 3 and 4 all returned exactly 1.000. It was never multi-hop
    reasoning.

    Each rule now points to an INDEPENDENT random target at the next
    level, so no level determines any other and the chain must genuinely
    be traversed."""
    ents = rng.sample(NAMES, n_entities)
    objs = rng.sample(OBJECTS, n_entities)

    # independent slot per level per entity; distinct across entities at
    # each level so the distractor conclusion is always well defined
    chains, used = [], [set() for _ in range(depth + 1)]
    for _ in range(n_entities):
        ch = []
        for d in range(depth + 1):
            choices = [i for i in range(len(LEVELS[d])) if i not in used[d]]
            pick = rng.choice(choices)
            used[d].add(pick)
            ch.append(pick)
        chains.append(ch)

    toks, rules = [], []
    for e, o, ch in zip(ents, objs, chains):
        toks += [e, 'has', 'a', LEVELS[0][ch[0]], o, '.']
        for d in range(depth):
            rules.append([LEVELS[d][ch[d]], 'things', 'are',
                          LEVELS[d + 1][ch[d + 1]], '.'])

    for f in range(n_filler):
        toks += FILLER[f % len(FILLER)]

    rng.shuffle(rules)
    for r in rules:
        toks += r

    q = rng.randrange(n_entities)
    toks += ['so', ents[q], "'s", objs[q], 'is']
    valid = [STOI[LEVELS[depth][chains[q][depth]]]]
    invalid = [STOI[LEVELS[depth][chains[j][depth]]]
               for j in range(n_entities) if j != q]
    apos = len(toks)
    toks += [ITOS[valid[0]], '.']
    return [STOI[t] for t in toks], valid, invalid, apos


def make_dataset(n, seed=0, depth=2, n_entities=2, n_filler=0):
    rng = random.Random(seed)
    out = [make_example(rng, depth, n_entities, n_filler) for _ in range(n)]
    return out, max(len(x[0]) for x in out)


def pad_batch(items, L, device):
    x = np.full((len(items), L), PAD, dtype=np.int64)
    for r, (ids, _, _, _) in enumerate(items):
        x[r, :min(len(ids), L)] = ids[:L]
    return torch.from_numpy(x).to(device)


def graph_support(batch, device, hops=6, decay=0.6, func_weight=None,
                  local_damp=None, freq_power=None):
    """RESTORED to the formula that produced 0.502 -> 1.000 on the
    syllogism and 1.000 at depth 1 here.

    Everything I added later was removed: row-normalisation of A,
    query-identity damping, distal subtraction, per-hop normalisation,
    mean aggregation, frequency-scaled identity edges. Each was chasing a
    magnitude problem that row-normalisation itself introduced -- without
    it, mass is not divided by branching and the chain survives.

    Extra kwargs are accepted and ignored so existing call sites keep
    working.
    """
    B, T = batch.shape
    dot = STOI['.']
    is_dot = (batch == dot).float()
    clause = torch.cumsum(is_dot, 1) - is_dot
    onehot = F.one_hot(batch.clamp(min=0), V).float()
    valid = (batch != PAD).float()
    is_func = torch.zeros_like(valid)
    for f in FUNC:
        is_func = torch.maximum(is_func, (batch == f).float())
    is_content = valid * (1.0 - is_func)

    same_clause = (clause.unsqueeze(2) == clause.unsqueeze(1)).float()
    same_token = (batch.unsqueeze(2) == batch.unsqueeze(1)).float()
    eye = torch.eye(T, device=device).unsqueeze(0)

    out = torch.zeros(B, T, V, device=device)
    for t in range(1, T):
        avail = torch.zeros(1, T, device=device)
        avail[0, :t] = 1.0
        act = is_content * avail
        A = torch.clamp(same_clause + same_token, max=1.0)
        A = A * act.unsqueeze(1) * act.unsqueeze(2) * (1.0 - eye)

        cur = (clause == clause[:, t - 1:t]).float() * act
        denom = cur.sum(dim=1, keepdim=True)
        cur = torch.where(denom > 0, cur, act)
        s = cur / cur.sum(dim=1, keepdim=True).clamp(min=1e-6)

        p, acc = s, s.clone()
        for _ in range(hops):
            p = decay * torch.bmm(p.unsqueeze(1), A).squeeze(1)
            acc = acc + p
        sup = torch.bmm(acc.unsqueeze(1), onehot).squeeze(1)
        out[:, t, :] = sup / sup.amax(-1, keepdim=True).clamp(min=1e-6)
    return out


def train(items, L, mode='ce', a=0.45, b=0.45, c=0.10, steps=2500, bs=64,
          d_model=128, n_layers=4, num_heads=4, lr=1e-3, device='cuda',
          seed=0, base=None, hops=6, quiet=True):
    set_seed(seed)
    model = CausalTransformer(V, d_model, num_heads, n_layers=n_layers,
                              max_len=L + 1, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    if base is not None:
        base.eval()
        for p_ in base.parameters():
            p_.requires_grad_(False)
    rng = random.Random(seed)
    for step in range(steps):
        model.train()
        chunk = [items[rng.randrange(len(items))] for _ in range(bs)]
        x = pad_batch(chunk, L, device)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        if mode == 'ce':
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                   ignore_index=PAD)
        else:
            with torch.no_grad():
                p_base = F.softmax(base(inp), dim=-1)
                sup = graph_support(inp, device, hops=hops)
                sup = sup / sup.sum(-1, keepdim=True).clamp(min=1e-6)
                oh = torch.zeros_like(p_base)
                oh.scatter_(-1, tgt.clamp(min=0).unsqueeze(-1), 1.0)
                oracle = a * oh + b * sup + c * p_base
            logp = F.log_softmax(logits, dim=-1)
            kl = -(oracle * logp).sum(-1)
            vm = (tgt != PAD).float()
            loss = (kl * vm).sum() / vm.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def evaluate(model, items, L, device):
    model.eval()
    vt, vm, im = [], [], []
    for i in range(0, len(items), 128):
        chunk = items[i:i + 128]
        x = pad_batch(chunk, L, device)
        logits = model(x)
        for r, (ids, valid, invalid, apos) in enumerate(chunk):
            if apos - 1 >= x.shape[1]:
                continue
            probs = F.softmax(logits[r, apos - 1], dim=-1)
            vt.append(int(probs.argmax().item() in valid))
            vm.append(float(sum(probs[t] for t in valid)))
            im.append(float(sum(probs[t] for t in invalid)))
    return {'valid_top1': float(np.mean(vt)), 'valid_mass': float(np.mean(vm)),
            'invalid_mass': float(np.mean(im))}


def run(depth, n_ent, n_fill, b, seed, steps, device, hops=6):
    tr, L1 = make_dataset(16000, seed=seed, depth=depth,
                          n_entities=n_ent, n_filler=n_fill)
    te, L2 = make_dataset(1500, seed=9000 + seed, depth=depth,
                          n_entities=n_ent, n_filler=n_fill)
    L = max(L1, L2)
    base = train(tr, L, mode='ce', steps=steps, device=device, seed=seed)
    rb = evaluate(base, te, L, device)
    stu = train(tr, L, mode='justif', b=b, a=0.90 - b, c=0.10, steps=steps,
                device=device, seed=seed, base=base, hops=hops)
    rs = evaluate(stu, te, L, device)
    return rb, rs, 1.0 / n_ent


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    which = sys.argv[2] if len(sys.argv) > 2 else 'all'
    print(f"Device: {device}  steps={steps}  vocab={V}  axis={which}")

    ids, val, inv, ap = make_dataset(2, seed=0, depth=3)[0][0]
    print("  depth-3 example: " + " ".join(ITOS[i] for i in ids))
    print(f"    valid={[ITOS[t] for t in val]} invalid={[ITOS[t] for t in inv]}")

    results = []

    if which in ('all', 'depth'):
        print("\n### AXIS 1: DEPTH ###")
        for d in [2, 3, 4]:
            rb, rs, ch = run(d, 2, 0, 0.45, 0, steps, device)
            results.append((f'depth={d}', ch, rb, rs))
            print(f"  depth={d}: CE {rb['valid_top1']:.3f} -> "
                  f"justif {rs['valid_top1']:.3f}  (chance {ch:.3f})")

    if which in ('all', 'distract'):
        print("\n### AXIS 2: DISTRACTOR ENTITIES ###")
        for ne in [2, 4, 6]:
            rb, rs, ch = run(2, ne, 0, 0.45, 0, steps, device)
            results.append((f'entities={ne}', ch, rb, rs))
            print(f"  entities={ne}: CE {rb['valid_top1']:.3f} -> "
                  f"justif {rs['valid_top1']:.3f}  (chance {ch:.3f})")

    if which in ('all', 'filler'):
        print("\n### AXIS 3: IRRELEVANT FILLER ###")
        for nf in [0, 2]:
            rb, rs, ch = run(2, 2, nf, 0.45, 0, steps, device)
            results.append((f'filler={nf}', ch, rb, rs))
            print(f"  filler={nf}: CE {rb['valid_top1']:.3f} -> "
                  f"justif {rs['valid_top1']:.3f}  (chance {ch:.3f})")

    if which in ('all', 'seeds'):
        print("\n### AXIS 4: SEEDS ###")
        for sd in [0, 1, 2]:
            rb, rs, ch = run(2, 2, 0, 0.45, sd, steps, device)
            results.append((f'seed={sd}', ch, rb, rs))
            print(f"  seed={sd}: CE {rb['valid_top1']:.3f} -> "
                  f"justif {rs['valid_top1']:.3f}  (chance {ch:.3f})")

    if which in ('all', 'bcliff'):
        print("\n### AXIS 5: B CLIFF ###")
        for b in [0.30, 0.45, 0.55, 0.65]:
            rb, rs, ch = run(2, 2, 0, b, 0, steps, device)
            results.append((f'b={b}', ch, rb, rs))
            print(f"  b={b}: CE {rb['valid_top1']:.3f} -> "
                  f"justif {rs['valid_top1']:.3f}  (chance {ch:.3f})")

    if which in ('all', 'truncated'):
        print("\n### AXIS 6: TRUNCATED ORACLE (3-hop task, fewer propagation hops) ###")
        for h in [2, 4, 8]:
            rb, rs, ch = run(3, 2, 0, 0.45, 0, steps, device, hops=h)
            results.append((f'depth3_hops={h}', ch, rb, rs))
            print(f"  hops={h}: CE {rb['valid_top1']:.3f} -> "
                  f"justif {rs['valid_top1']:.3f}  (chance {ch:.3f})")

    print("\n" + "=" * 88)
    print("STRESS TEST SUMMARY")
    print("=" * 88)
    print(f"\n  {'config':>18} | {'chance':>7} | {'CE':>7} | {'justif':>7} | "
          f"{'gain':>7} | {'inv_mass':>9}")
    print("  " + "-" * 70)
    for lab, ch, rb, rs in results:
        print(f"  {lab:>18} | {ch:>7.3f} | {rb['valid_top1']:>7.3f} | "
              f"{rs['valid_top1']:>7.3f} | "
              f"{rs['valid_top1']-rb['valid_top1']:>+7.3f} | "
              f"{rs['invalid_mass']:>9.4f}")

    print("\nDONE")
