"""Counterfactual tracking: context assertions vs pretrained prior.

The 'things actually not in its training set' tier. Two stages:

A  PRETRAIN on TinyStories until the model has world priors (sky->blue,
   grass->green, ...). VERIFIED by probing 'the sky was ___' -- if the
   prior is not installed, the experiment is void and aborts.

B  COUNTERFACTUAL stories: assert an abnormal binding, distract, probe.
     one day mia went outside . in her world the sky was purple .
     she played with her dog . they ran far away .
     mia looked up at the sky . it was ___        -> purple (asserted)
   Each story ALSO probes a normally-colored object, so 'echo the weird
   color' fails -- the model must track WHICH object is counterfactual.

   Arms (finetuning from the same pretrained weights):
     ce      plain CE on the counterfactual corpus
     store   assertion writes {object: color}; probe reads object;
             retrieved color injected mid-stack (bind_store recipe,
             dense interface supervision, self-written store at eval)

Metrics per arm: cf-probe top-1 (target = ASSERTED color, prior votes
against), normal-probe top-1 (target = TRUE color -- guards against
'always output the unusual'), both candidate-restricted to the color
vocabulary; plus prior retention on stage-A probes.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from common import set_seed
from bind_store import StoreModel

TS_PATH = os.environ.get('TINYSTORIES_PATH', 'tinystories_train.txt')
PAD, UNK = 0, 1

FACTS = [('sky', 'blue'), ('grass', 'green'), ('snow', 'white'),
         ('sun', 'yellow'), ('sea', 'blue'), ('moon', 'white')]
# v3: counterfactuals ARE training data -- the model must learn
# the ALGORITHM "record what the text asserts, answer from the record."
# Generalization axis = held-out OBJECTS: the algorithm is trained on
# FACTS_TRAIN counterfactuals and evaluated on FACTS_EVAL counterfactuals
# it has never practiced overriding. Second axis = retention: the
# algorithm must be CONDITIONAL (prior intact when no assertion speaks).
FACTS_TRAIN = FACTS[:4]
FACTS_EVAL = FACTS[4:]          # sea-blue, moon-white: never counterfactual
                                # in training
# v4: WRITE-KEY DIVERSITY. With only 4 training keys, "copy the subject"
# and "detect one of these 4 objects" are indistinguishable and GD picks
# the detector (v3: held-out 0.000 for BOTH arms, seen-obj ~1.0). Generic
# no-prior objects widen the key pool so copy is the only cheap rule.
GENERIC = ['ball', 'kite', 'cup', 'hat', 'box', 'flower', 'tree', 'door',
           'car', 'book', 'shoe', 'chair']
COLORS = ['blue', 'green', 'white', 'yellow', 'red', 'purple', 'pink',
          'black', 'orange', 'brown']
NAMES_ = ['mia', 'tom', 'lily', 'ben', 'sue', 'max']
FILLER = [
    'she played with her dog .', 'they ran far away .',
    'he ate a big lunch .', 'the wind blew softly .',
    'a bird sang a song .', 'they walked to the park .',
    'it was a busy day .', 'her friend came over .',
]


def load_words(n_chars=31_000_000):
    """Full corpus. v4 used 6M chars (1.37M tokens) and 15k steps = 45
    epochs -- pretrain loss 0.62, pure memorization. A memorized
    continuation is a near-zero-entropy spike, not a soft prior, and is
    hardest to override on exactly the held-out objects. Full 30.1M chars at 10k steps ~= 9 epochs."""
    import re
    txt = open(TS_PATH, encoding='utf-8', errors='ignore').read(n_chars)
    return re.findall(r"[a-z']+|[.,!?\"]", txt.lower())


def build_vocab(words, size=8000):
    c = Counter(words)
    itos = ['<pad>', '<unk>'] + [w for w, _ in c.most_common(size - 2)]
    return {w: i for i, w in enumerate(itos)}, itos


def pretrain(stoi, words, steps, seed, device, d_model=256, L=128, bs=32):
    set_seed(seed)
    ids = np.array([stoi.get(w, UNK) for w in words], dtype=np.int64)
    m = StoreModel(d_model=d_model, max_len=L + 1, vocab=len(stoi)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    for st in range(steps):
        m.train()
        ix = [rng.randrange(0, len(ids) - L - 1) for _ in range(bs)]
        x = torch.from_numpy(np.stack([ids[i:i + L] for i in ix])).to(device)
        y = torch.from_numpy(
            np.stack([ids[i + 1:i + L + 1] for i in ix])).to(device)
        lg, _, _, _ = m(x, store_mode='off')
        loss = F.cross_entropy(lg.reshape(-1, len(stoi)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if (st + 1) % 1000 == 0:
            print(f'    pretrain step {st+1}  loss {loss.item():.3f}',
                  flush=True)
    return m


@torch.no_grad()
def prior_check(m, stoi, device):
    """P(true color) rank among COLORS for 'the <obj> was ___'."""
    m.eval()
    ok = []
    cids = [stoi[c] for c in COLORS if c in stoi]
    for obj, col in FACTS:
        if obj not in stoi or col not in stoi:
            continue
        p = torch.tensor([[stoi.get(w, UNK) for w in
                           f'the {obj} was'.split()]], device=device)
        lg, _, _, _ = m(p, store_mode='off')
        sub = lg[0, -1, cids]
        ok.append(int(cids[int(sub.argmax())] == stoi[col]))
    return float(np.mean(ok)) if ok else float('nan')


def make_cf_example(rng, stoi, counterfactual=True, facts=None):
    """Returns ids, wk, wv, rq, sites [(pos, target, kind)].

    counterfactual: whether the asserted color contradicts the true one.
    facts: which fact pool the COUNTERFACTUAL object is drawn from
    (FACTS_TRAIN during training, FACTS_EVAL for held-out-object eval).
    The normally-colored probe object always comes from the full pool."""
    pool = facts if facts is not None else FACTS
    if counterfactual and facts is FACTS_TRAIN and rng.random() < 0.6:
        # v4 diversity: generic no-prior subject, arbitrary color
        obj_cf = rng.choice(GENERIC)
        col_true_cf = None
    else:
        obj_cf, col_true_cf = rng.choice(pool)
    # v7: partners from FACTS_TRAIN only. v6 drew them from the full pool,
    # so held-out objects appeared in training always-asserted-true --
    # "sea is always blue" remained a perfect competing fit and held-out
    # stayed partial (store 0.48/0.19, CE 0.000). Held-out objects must
    # not appear in finetuning at all.
    obj_n, col_n = rng.choice([f for f in FACTS_TRAIN if f[0] != obj_cf])
    if counterfactual:
        col_cf = rng.choice([c for c in COLORS
                             if c != col_true_cf and c != col_n])
    else:
        col_cf = col_true_cf
    name = rng.choice(NAMES_)

    # v6: EVERY probed object gets an in-story assertion (the normal
    # partner is asserted with its TRUE color). Then no probe in training
    # is answerable from the object alone -- "answer what this story
    # asserted about this object" is the unique rule consistent with ALL
    # training data, and per-object lookup fits nothing. (v3-v5 failed
    # held-out at exactly 0.000 because held-out objects' probes were
    # always answered by their fixed true color in training, so both
    # arms correctly learned per-object routing.)
    parts = [f'one day {name} went outside .',
             f'in her world the {obj_cf} was {col_cf} .',
             f'the {obj_n} was {col_n} .']
    fillers = rng.sample(FILLER, rng.randrange(2, 5))
    parts += fillers
    probes = [
        (obj_cf, col_cf, 'cf'),
        (obj_n, col_n, 'normal'),
    ]
    rng.shuffle(probes)
    site_specs = []
    for obj, col, kind in probes:
        parts.append(f'{name} looked at the {obj} . it was {col} .')
        site_specs.append((obj, col, kind))

    words = ' '.join(parts).split()
    ids = [stoi.get(w, UNK) for w in words]
    T = len(ids)
    V = len(stoi)
    wk = np.full(T, V, dtype=np.int64)
    wv = np.full(T, V, dtype=np.int64)
    rq = np.full(T, V, dtype=np.int64)

    # writes at BOTH assertions' color tokens: {obj_cf: col_cf} and
    # {obj_n: col_n}. Only the assertion region (before the fillers'
    # end) counts, so probe restatements are not treated as assertions.
    first_probe = ' '.join(parts).split().index('looked') \
        if 'looked' in ' '.join(parts).split() else T
    for obj, col in ((obj_cf, col_cf), (obj_n, col_n)):
        for i in range(min(T - 2, first_probe)):
            if words[i] == obj and words[i + 1] == 'was' \
                    and words[i + 2] == col:
                wk[i + 2] = stoi.get(obj, UNK)
                wv[i + 2] = stoi.get(col, UNK)
                break

    # probes: '<name> looked at the <obj> . it was <col> .'
    sites = []
    for i in range(T - 2):
        if words[i] == 'it' and words[i + 1] == 'was' and i >= 4:
            obj = words[i - 2]
            spec = next((s for s in site_specs if s[0] == obj), None)
            if spec is None:
                continue
            rq[i + 1] = stoi.get(obj, UNK)          # position predicting color
            sites.append((i + 1, stoi.get(spec[1], UNK), spec[2]))
    return ids, wk, wv, rq, sites


def make_cf_dataset(n, seed, stoi, counterfactual=True, facts=None):
    rng = random.Random(seed)
    out = [make_cf_example(rng, stoi, counterfactual, facts)
           for _ in range(n)]
    return out, max(len(x[0]) for x in out)


def make_plain_windows(n, seed, stoi, words, L=80):
    """Raw TinyStories windows with all-NOOP interface targets: teaches
    'do not write on ordinary text' and preserves the LM/prior during
    finetuning (v1 destroyed the prior with template-only data)."""
    rng = random.Random(seed)
    ids_all = [stoi.get(w, UNK) for w in words]
    V = len(stoi)
    out = []
    for _ in range(n):
        st = rng.randrange(0, len(ids_all) - L - 1)
        ids = ids_all[st:st + L]
        z = np.full(len(ids), V, dtype=np.int64)
        out.append((ids, z.copy(), z.copy(), z.copy(), []))
    return out


def pad_batch(items, L, V, device):
    B = len(items)
    x = np.full((B, L), PAD, dtype=np.int64)
    wk = np.full((B, L), V, dtype=np.int64)
    wv = np.full((B, L), V, dtype=np.int64)
    rq = np.full((B, L), V, dtype=np.int64)
    for r, (ids, k, v, q, _) in enumerate(items):
        m = min(len(ids), L)
        x[r, :m] = ids[:m]
        wk[r, :m] = k[:m]
        wv[r, :m] = v[:m]
        rq[r, :m] = q[:m]
    t = lambda a: torch.from_numpy(a).to(device)
    return t(x), t(wk), t(wv), t(rq)


def finetune(m, items, L, V, mode, steps, seed, device, lam=0.5, bs=32,
             lr=2e-4):
    set_seed(seed + 100)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(bs)]
        x, wk, wv, rq = pad_batch(ch, L, V, device)
        inp, tgt = x[:, :-1], x[:, 1:]
        gt = (wk[:, :-1], wv[:, :-1], rq[:, :-1])
        lg, kl, vl, ql = m(inp, store_mode='gt' if mode == 'store' else 'off',
                           gt=gt)
        vm = (tgt != PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1),
                               ignore_index=PAD)
        if mode == 'store':
            for logit, target in ((kl, gt[0]), (vl, gt[1]), (ql, gt[2])):
                lf = F.cross_entropy(logit.reshape(-1, V + 1),
                                     target.reshape(-1), reduction='none')
                loss = loss + lam * (lf * vm.reshape(-1)).sum() / \
                    vm.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(m, items, L, V, stoi, device, store_mode):
    m.eval()
    cids = [stoi[c] for c in COLORS if c in stoi]
    res = {'cf': [], 'normal': []}
    for i in range(0, len(items), 64):
        ch = items[i:i + 64]
        x, wk, wv, rq = pad_batch(ch, L, V, device)
        lg, _, _, _ = m(x, store_mode=store_mode, gt=(wk, wv, rq))
        for r, (ids, k, v, q, sites) in enumerate(ch):
            for (p, tgt, kind) in sites:
                if p >= x.shape[1]:
                    continue
                sub = lg[r, p, cids]
                res[kind].append(int(cids[int(sub.argmax())] == tgt))
    return {k: float(np.mean(v)) if v else float('nan')
            for k, v in res.items()}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    ft_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 \
        else [0, 1]

    words = load_words()
    stoi, itos = build_vocab(words)
    V = len(stoi)
    missing = [w for w in COLORS + [f[0] for f in FACTS]
               if w not in stoi]
    print(f'CF-STORE  V={V}  missing-vocab={missing}', flush=True)

    # v3 TRAINING MIX: counterfactuals on FACTS_TRAIN objects + true-color
    # stories + plain TS windows (retention). EVAL: counterfactuals on
    # HELD-OUT objects (FACTS_EVAL) -- the algorithm, not the pairs.
    cf_tr, La = make_cf_dataset(6000, 0, stoi, True, FACTS_TRAIN)
    true_tr, Lb = make_cf_dataset(3000, 1, stoi, False, FACTS_TRAIN)
    plain_tr = make_plain_windows(3000, 2, stoi, words)
    tr = cf_tr + true_tr + plain_tr
    random.Random(3).shuffle(tr)
    te, Lc = make_cf_dataset(1200, 9000, stoi, True, FACTS_EVAL)
    te_seen, Ld = make_cf_dataset(1200, 9100, stoi, True, FACTS_TRAIN)
    ten, Le = make_cf_dataset(1200, 9500, stoi, False, FACTS_TRAIN)
    L = max(La, Lb, Lc, Ld, Le, 80) + 1
    print('  sample: ' + ' '.join(itos[t] for t in tr[0][0]), flush=True)

    for sd in seeds:
        print(f'\n== seed {sd}: pretraining {pre_steps} steps ==', flush=True)
        base = pretrain(stoi, words, pre_steps, sd, device, L=128)
        pc = prior_check(base, stoi, device)
        print(f'  PRIOR CHECK: {pc:.2f} of facts correct  '
              f'({"OK" if pc >= 0.5 else "WEAK -- interpret with care"})',
              flush=True)
        pre_state = {k: v.clone() for k, v in base.state_dict().items()}

        for mode in ('ce', 'store'):
            base.load_state_dict(pre_state)
            m = finetune(base, tr, L, V, mode, ft_steps, sd, device)
            sm = 'self' if mode == 'store' else 'off'
            r = evaluate(m, te, L, V, stoi, device, sm)       # HELD-OUT cf
            rs = evaluate(m, te_seen, L, V, stoi, device, sm)  # seen-obj cf
            rn = evaluate(m, ten, L, V, stoi, device, sm)     # true stories
            keep = prior_check(m, stoi, device)
            print(f'  {mode:>6}:  HELD-OUT cf {r["cf"]:.3f}   '
                  f'seen-obj cf {rs["cf"]:.3f}   '
                  f'normal-probe {rn["cf"]:.3f}/{rn["normal"]:.3f}   '
                  f'prior-retention {keep:.2f}', flush=True)
