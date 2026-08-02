"""Working-memory slots: register readout as extra output distributions.

THE PROPOSAL. Superposition is unavoidable -- every layer mixes. The
problem is the model holding a binding intact the whole way through, like
working memory. So add slots to the output distribution: special heads
trained to report what is currently held, <nothing> when empty, with the
slot ordering scrambled so position cannot be memorised.

WHY THIS IS NOT PAUSE TOKENS OR SCRATCHPAD (both of which failed)
  Those put the intermediate into the SEQUENCE, where the model reads it
  back by attention on the next step -- the state lives in the input, not
  in the model. Here the register contents appear ONLY in the output
  distribution and are never fed back, so there is nowhere to keep them
  but the residual stream.

HONEST FRAMING. A full-attention transformer never HAS to hold anything:
the context is readable at every position, so "recompute on demand" is
always available and literal RNN-style holding cannot be forced by any
output-side loss. What this actually does is force the intermediate to be
computed and present DENSELY, at every position, instead of being
synthesised in the last two layers at the last position. That is still
the right intervention for the superposition worry -- it makes the
intermediate a first-class feature with a direct gradient path.

DOMAIN CHANGES REQUIRED (the obvious version would have been vacuous)
  1. QUERY FIRST. In stress_test the query is last, so bind and hop both
     happen in the final few positions and nothing needs holding at all.
  2. ANSWER DOES NOT RESTATE THE QUERY. "answer heavy risky ." rather
     than "so oz 's cup is ___", otherwise the answer position just
     re-binds locally and bypasses the registers entirely.
  3. ENTITY COUNT HELD AT 2. Vary only Q, the number of simultaneous
     queries. Varying entity count would confound multi-bind with the
     competition effect already measured (3+ entities fails on its own).

CONSTRAINT: heads read the FINAL layer only. A supervised readout on an
intermediate layer would be training correction on the hidden state.

FALSIFIER, stated before running: the readout heads may learn to compute
the intermediate themselves off the final state while the answer head
goes on ignoring it. Two heads on one state need not share a mechanism.
A register that reads correctly while the answer stays at chance is a
NEGATIVE result, not a partial win.
"""

import sys
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import set_seed
from transformer_torch import CausalTransformer
from stress_test import NAMES, OBJECTS, L0, L1

STRUCT = ['has', 'a', 'things', 'are', '.', '?', 'answer', '<pad>']
PAD = 0


def build_vocab():
    words = ['<pad>'] + STRUCT + NAMES + OBJECTS + L0 + L1
    words = list(dict.fromkeys(words))
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, {i: w for w, i in stoi.items()}


STOI, ITOS = build_vocab()
V = len(STOI)
NOTHING = V              # register-only symbol
VR = V + 1
FUNC = {STOI[w] for w in STRUCT if w in STOI}
DOT = STOI['.']


def make_example(rng, n_entities=2, n_queries=1, n_hops=0):
    """? oz cup .  <fact (and rule) clauses, shuffled>  answer wet .

    n_hops=0 -- PURE BIND. No rule clauses; the answer is the queried
      object's property. Each query costs 1 bind + 1 hop, the depth that
      already reaches 1.000, so the only thing that scales with Q is the
      number of simultaneous bindings. This is the configuration that
      isolates multi-bind capacity.
    n_hops=1 -- bind then rule application. Measured first and it sits at
      chance for every objective: retrieving the query from the start of
      the sequence is itself a lookup, so this costs 1 bind + 2 hops,
      which is the depth known to fail under every intervention tried.

    Register state for query q evolves NOTHING -> property (-> consequence).
    Clause order is shuffled, so WHEN each becomes available is
    unpredictable, and a rule may precede its fact (in which case the
    state stays NOTHING -- the rule is not yet known to be relevant).
    """
    ents = rng.sample(NAMES, n_entities)
    objs = rng.sample(OBJECTS, n_entities)
    props = rng.sample(L0, n_entities)
    cons = rng.sample(L1, n_entities)
    qidx = rng.sample(range(n_entities), n_queries)

    # each query gets its OWN clause. Sharing one clause across queries
    # wires their keys together through the clause edge, so the support
    # seeded from query 0 reaches query 1's entity at 0.991 and the two
    # chains become indistinguishable (verified in wmem_check).
    toks, qkeys = [], []
    for qi in qidx:
        qkeys.append([len(toks) + 1, len(toks) + 2])   # entity, object
        toks += ['?', ents[qi], objs[qi], '.']

    clauses = []
    for i in range(n_entities):
        clauses.append(('fact', i, [ents[i], 'has', 'a', props[i], objs[i], '.']))
        if n_hops:
            clauses.append(('rule', i, [props[i], 'things', 'are', cons[i], '.']))
    rng.shuffle(clauses)
    done = {}
    for kind, i, ws in clauses:
        toks += ws
        done[(kind, i)] = len(toks) - 1               # index of the '.'

    ans = cons if n_hops else props
    toks += ['answer']
    apos = []
    for qi in qidx:
        apos.append(len(toks))
        toks += [ans[qi]]
    toks += ['.']

    T = len(toks)
    reg = np.full((T, n_queries), NOTHING, dtype=np.int64)
    for j, qi in enumerate(qidx):
        f = done[('fact', qi)]
        r = done[('rule', qi)] if n_hops else f
        for i in range(T):
            if i >= f and i >= r:
                reg[i, j] = STOI[ans[qi]]
            elif i >= f:
                reg[i, j] = STOI[props[qi]]

    # stage positions per query: where the register FIRST holds the
    # property (needs 1 bind + 1 hop -- the depth that reaches 1.000 in
    # the query-last format) and where it first holds the consequence
    # (1 bind + 2 hops -- the depth that fails everywhere). Scoring these
    # separately is what distinguishes "the register mechanism does not
    # work" from "it works exactly as far as the depth limit allows".
    # CONTROL register set: every entity's property, regardless of which
    # was queried. Equally dense, equally learnable, same head count and
    # same clause-arrival structure -- but carries no information about
    # WHICH entity was asked for. If this helps as much as the queried
    # registers, the gain is from extra gradient signal, not from binding.
    reg_all = np.full((T, n_entities), NOTHING, dtype=np.int64)
    for i in range(n_entities):
        f = done[('fact', i)]
        for k in range(f, T):
            reg_all[k, i] = STOI[props[i]]

    # SECOND control: same structure and density, but the tracked tokens
    # are entity NAMES -- outside the answer space. Separates "the answer
    # candidates must be separably represented" from "any dense auxiliary
    # supervision helps".
    reg_names = np.full((T, n_entities), NOTHING, dtype=np.int64)
    for i in range(n_entities):
        f = done[('fact', i)]
        for k in range(f, T):
            reg_names[k, i] = STOI[ents[i]]

    stages = []
    for qi in qidx:
        f = done[('fact', qi)]
        r = done[('rule', qi)] if n_hops else f
        stages.append((f, max(f, r), STOI[props[qi]], STOI[ans[qi]]))

    ids = [STOI[t] for t in toks]
    cands = [STOI[c] for c in ans]
    valid = [STOI[ans[qi]] for qi in qidx]
    return ids, qkeys, apos, cands, valid, reg, stages, reg_all, reg_names


def make_dataset(n, seed=0, n_entities=2, n_queries=1, n_hops=0):
    rng = random.Random(seed)
    out = [make_example(rng, n_entities, n_queries, n_hops) for _ in range(n)]
    return out, max(len(x[0]) for x in out)


def pad_batch(items, L, device, K, reg_src='query'):
    B = len(items)
    x = np.full((B, L), PAD, dtype=np.int64)
    reg = np.full((B, L, K), NOTHING, dtype=np.int64)
    col = {'query': 5, 'all': 7, 'names': 8}[reg_src]
    for r, it in enumerate(items):
        ids, rg = it[0], it[col]
        n = min(len(ids), L)
        x[r, :n] = ids[:n]
        reg[r, :n, :rg.shape[1]] = rg[:n]
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(reg).to(device))


def support_seeded(batch, seed, upto, device, hops=10, decay=0.8):
    """Graph support propagated from an EXPLICIT seed set of token
    positions, using only tokens strictly before `upto` (per example).

    Explicit seeding is needed because the answer clause here contains no
    keys ("answer heavy risky ."), so the clause-seeded version used
    elsewhere has nothing to start from -- and because that version is
    order-blind and so could not distinguish answer slot 1 from slot 2.
    Formula is otherwise the restored one (no row-normalisation, no
    damping) that gave 0.502 -> 1.000.
    """
    B, T = batch.shape
    is_dot = (batch == DOT).float()
    clause = torch.cumsum(is_dot, 1) - is_dot
    onehot = F.one_hot(batch.clamp(min=0), V).float()
    ok = (batch != PAD).float()
    is_func = torch.zeros_like(ok)
    for f in FUNC:
        is_func = torch.maximum(is_func, (batch == f).float())
    ar = torch.arange(T, device=device).unsqueeze(0)
    act = ok * (1.0 - is_func) * (ar < upto.unsqueeze(1)).float()

    same_clause = (clause.unsqueeze(2) == clause.unsqueeze(1)).float()
    same_token = (batch.unsqueeze(2) == batch.unsqueeze(1)).float()
    eye = torch.eye(T, device=device).unsqueeze(0)
    A = torch.clamp(same_clause + same_token, max=1.0)
    A = A * act.unsqueeze(1) * act.unsqueeze(2) * (1.0 - eye)

    s = seed * act
    s = s / s.sum(1, keepdim=True).clamp(min=1e-6)
    p, acc = s, s.clone()
    for _ in range(hops):
        p = decay * torch.bmm(p.unsqueeze(1), A).squeeze(1)
        acc = acc + p
    sup = torch.bmm(acc.unsqueeze(1), onehot).squeeze(1)
    return sup / sup.amax(-1, keepdim=True).clamp(min=1e-6)


class WMModel(nn.Module):
    def __init__(self, K, d_model=128, n_layers=4, num_heads=4, max_len=128):
        super().__init__()
        self.K = K
        self.core = CausalTransformer(V, d_model, num_heads, n_layers=n_layers,
                                      max_len=max_len, dropout=0.0)
        self.reg = nn.Linear(d_model, K * VR)
        nn.init.normal_(self.reg.weight, std=0.02)
        nn.init.zeros_(self.reg.bias)

    def forward(self, x):
        return self.core(x)

    def both(self, x):
        B, T = x.shape
        h = self.core.get_hidden(x)
        return self.core.output(h), self.reg(h).view(B, T, self.K, VR)


def set_loss(reg_logits, reg_tgt, valid_mask, perms):
    """Permutation-invariant matching over slots. With optimal matching
    the slot index carries no gradient signal about identity, which is
    what makes position unmemorisable -- the loss scores the SET held,
    not an ordered list."""
    B, T, K, _ = reg_logits.shape
    lp = F.log_softmax(reg_logits, dim=-1)
    # C[b,t,k,j] = -log p_k( target_j )
    C = -lp.gather(3, reg_tgt.unsqueeze(2).expand(B, T, K, K))
    ar = torch.arange(K, device=C.device)
    Cp = C[:, :, ar.unsqueeze(0).expand(perms.shape[0], K), perms]  # (B,T,P,K)
    cost = Cp.sum(-1).min(-1).values                                # (B,T)
    return (cost * valid_mask).sum() / valid_mask.sum().clamp(min=1.0) / K


def train(items, L, K, mode='ce', use_reg=False, lam=0.5, reg_src='query',
          a=0.55, b=0.35, c=0.10, steps=1500, bs=64, seed=0, base=None,
          d_model=128, n_layers=4, device='cuda'):
    set_seed(seed)
    m = WMModel(K, d_model=d_model, n_layers=n_layers, max_len=L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    perms = torch.tensor(list(itertools.permutations(range(K))), device=device)
    if base is not None:
        base.eval()
        for p_ in base.parameters():
            p_.requires_grad_(False)
    rng = random.Random(seed)
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(bs)]
        x, rg = pad_batch(ch, L, device, K, reg_src)
        inp, tgt = x[:, :-1], x[:, 1:]
        rgt = rg[:, :-1, :]
        lg, rlg = m.both(inp)
        vm = (tgt != PAD).float()

        if mode == 'ce':
            loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1),
                                   ignore_index=PAD)
        else:
            with torch.no_grad():
                pb = F.softmax(base(inp), dim=-1)
                oracle = pb.clone()
                nq = len(ch[0][2])
                for j in range(nq):
                    seedm = torch.zeros_like(inp, dtype=torch.float)
                    upto = torch.ones(len(ch), dtype=torch.long, device=device)
                    rows = []
                    for r, it in enumerate(ch):
                        ap = it[2][j]
                        if ap - 1 >= inp.shape[1]:
                            continue
                        for kp in it[1][j]:
                            if kp < inp.shape[1]:
                                seedm[r, kp] = 1.0
                        upto[r] = ap
                        rows.append((r, ap))
                    sup = support_seeded(inp, seedm, upto, device)
                    sup = sup / sup.sum(-1, keepdim=True).clamp(min=1e-9)
                    for r, ap in rows:
                        oh = torch.zeros(V, device=device)
                        oh[tgt[r, ap - 1]] = 1.0
                        oracle[r, ap - 1] = a * oh + b * sup[r] + c * pb[r, ap - 1]
            lp = F.log_softmax(lg, dim=-1)
            kl = -(oracle * lp).sum(-1)
            loss = (kl * vm).sum() / vm.sum().clamp(min=1.0)

        if use_reg:
            loss = loss + lam * set_loss(rlg, rgt, vm, perms)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(model, items, L, K, device='cuda'):
    """Answer accuracy per query slot, restricted to the consequence set
    (chance 1/n_entities for slot 0).

    Register readout scored at two STAGES, because they need different
    amounts of computation and lumping them hides the whole result:
      prop  -- register must hold the property, right after the fact
               clause. 1 bind + 1 hop.
      cons  -- register must hold the consequence, after both clauses.
               1 bind + 2 hops.
    A slot counts as correct if ANY head puts its argmax on the required
    token, which is the right criterion for a permutation-invariant set.
    """
    model.eval()
    nq = len(items[0][2])
    hit = [[] for _ in range(nq)]
    sprop, scons, sfull = [], [], []
    for i in range(0, len(items), 128):
        ch = items[i:i + 128]
        x, rg = pad_batch(ch, L, device, K)
        lg, rlg = model.both(x)
        pred = rlg.argmax(-1)
        for r, it in enumerate(ch):
            apos, cands, valid, stages = it[2], it[3], it[4], it[6]
            for j, ap in enumerate(apos):
                if ap - 1 >= x.shape[1]:
                    continue
                sub = lg[r, ap - 1, cands]
                hit[j].append(int(cands[int(sub.argmax())] == valid[j]))
            for (fp, cp, pid, cid) in stages:
                if fp < x.shape[1]:
                    sprop.append(int(pid in pred[r, fp].tolist()))
                if cp < x.shape[1]:
                    scons.append(int(cid in pred[r, cp].tolist()))
            # p0 is the 'answer' prompt, distant from every clause, so
            # unlike the stage positions it admits no copy-the-last-token
            # shortcut. This is the honest register number.
            p0 = apos[0] - 1
            if p0 < x.shape[1]:
                sfull.append(int(sorted(pred[r, p0].tolist()) ==
                                 sorted(rg[r, p0].tolist())))
    f = lambda v: float(np.mean(v)) if v else float('nan')
    return ([f(h) for h in hit], f(sprop), f(scons), f(sfull))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    nh = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    # same K for both register variants so the control is matched on head
    # count; +1 guarantees at least one always-empty slot
    K = max(nq, ne) + 1

    tr, La = make_dataset(18000, seed=0, n_entities=ne, n_queries=nq, n_hops=nh)
    te, Lb = make_dataset(1500, seed=9000, n_entities=ne, n_queries=nq, n_hops=nh)
    L = max(La, Lb) + 1
    print(f'n_entities={ne}  n_queries={nq}  n_hops={nh}  K={K}  '
          f'steps={steps}  L={L}')
    print('  sample: ' + ' '.join(ITOS[t] for t in tr[0][0]))
    print(f'  chance per slot = {1.0 / ne:.3f}   register vocab {VR}')

    rows = []
    base = train(tr, L, K, mode='ce', use_reg=False, steps=steps, device=device)
    rows.append(('CE', base))
    rows.append(('CE + reg[queried]',
                 train(tr, L, K, mode='ce', use_reg=True, steps=steps,
                       device=device)))
    rows.append(('CE + reg[all] CONTROL',
                 train(tr, L, K, mode='ce', use_reg=True, reg_src='all',
                       steps=steps, device=device)))
    rows.append(('oracle',
                 train(tr, L, K, mode='justif', use_reg=False, steps=steps,
                       base=base, device=device)))
    rows.append(('oracle + reg[queried]',
                 train(tr, L, K, mode='justif', use_reg=True, steps=steps,
                       base=base, device=device)))

    print('\n  ' + ' ' * 20 + '  ' + '  '.join(f'ans{j}' for j in range(nq)) +
          '   reg:prop  reg:cons  reg:exact')
    print('  ' + ' ' * 20 + '  ' + ' ' * (6 * nq) +
          '   1b+1hop   1b+2hop   all slots')
    for tag, m in rows:
        acc, sp, sc, sf = evaluate(m, te, L, K, device)
        print(f'  {tag:>22}  ' + '  '.join(f'{a:.3f}' for a in acc) +
              f'     {sp:.3f}     {sc:.3f}     {sf:.3f}', flush=True)
