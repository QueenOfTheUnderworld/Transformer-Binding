"""External binding store: dict-backed query-lookup with SUPERVISED
interface heads. The scaffold trick applied to tool use.

THE IDEA: don't make the model hold bindings in the residual and
throw them away -- give it a store it can upload bindings to and look
them up from, mid-forward. No differentiable-memory machinery: the store
is a plain exact dict, reads are hard argmax lookups, NO gradient flows
through the store. The interface is installed the way the registers
were -- dense supervision -- which the dissection says needs BOTH dense
coverage and task-relevant content (prop-dense 3/3; names-dense 1/3
slow; prop-sparse 0/3; CE 0/4).

WIRING (mid-stack):
  emb -> block0 -> block1 -> [write head | read head] -> inject -> block2 -> block3 -> logits
  write head: per position, key logits (V+1: token or NO-WRITE) and
              value logits. Target: at each fact clause's '.', key =
              entity, value = property; NO-WRITE elsewhere (dense).
  read head:  per position, query logits (V+1: token or NO-READ).
              Target: at 'answer' and each answer token, query = the
              corresponding queried entity; NO-READ elsewhere.
  store:      dict {key token -> value token}. TRAIN: built from ground
              truth (teacher-forced tool). EVAL: reported both ways --
              GT store and SELF-written store (model's own write-head
              argmax) -- the self-written number is the honest one.
  inject:     retrieved value v -> x[:, t] += W_inj(emb[v]) before
              block2. Gradients flow through W_inj from the CE loss;
              the lookup itself is discrete.

WHY THIS CAN BREAK THE COMPETITION WALL: in-residual binding died at 3
entities (sequential regime) and cost width in fusion. Store rows do not
interfere with each other -- ne=40 should cost the same as ne=4. The
read query at answer slot j is a POSITIONAL copy of query clause j's
entity (no binding needed); the store performs the bind. That is the
externalization.

Scale: generated vocab (64 entities/objects/properties), ne up to 40,
Q=3 queries, sequential answers.
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import set_seed
from transformer_torch import TransformerBlock, RoPE

NE_MAX = 64
ENTS = [f'e{i}' for i in range(NE_MAX)]
OBJS = [f'o{i}' for i in range(NE_MAX)]
PROPS = [f'p{i}' for i in range(NE_MAX)]
STRUCT = ['has', 'a', '.', '?', 'answer', '<pad>']
PAD = 0


def build_vocab():
    words = list(dict.fromkeys(['<pad>'] + STRUCT + ENTS + OBJS + PROPS))
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, {i: w for w, i in stoi.items()}


STOI, ITOS = build_vocab()
V = len(STOI)
NOOP = V                    # shared no-write / no-read class
VI = V + 1


def make_example(rng, ne, Q=3):
    ents = rng.sample(ENTS, ne)
    objs = rng.sample(OBJS, ne)
    props = rng.sample(PROPS, ne)
    qidx = rng.sample(range(ne), Q)

    toks = []
    for qi in qidx:
        toks += ['?', ents[qi], objs[qi], '.']
    facts = [[ents[i], 'has', 'a', props[i], objs[i], '.'] for i in range(ne)]
    rng.shuffle(facts)
    fact_end = {}                        # position of '.' -> (entity, prop)
    for f in facts:
        toks += f
        fact_end[len(toks) - 1] = (STOI[f[0]], STOI[f[3]])

    toks += ['answer']
    apos = []
    for qi in qidx:
        apos.append(len(toks))
        toks += [props[qi]]
    toks += ['.']

    T = len(toks)
    wkey = np.full(T, NOOP, dtype=np.int64)
    wval = np.full(T, NOOP, dtype=np.int64)
    for p, (e, pr) in fact_end.items():
        wkey[p] = e
        wval[p] = pr
    rquery = np.full(T, NOOP, dtype=np.int64)
    for j, qi in enumerate(qidx):
        rquery[apos[j] - 1] = STOI[ents[qi]]    # position predicting answer j

    ids = [STOI[t] for t in toks]
    cands = [STOI[p] for p in props]
    valid = [STOI[props[qi]] for qi in qidx]
    return ids, apos, cands, valid, wkey, wval, rquery


def make_dataset(n, seed, ne, Q=3):
    rng = random.Random(seed)
    out = [make_example(rng, ne, Q) for _ in range(n)]
    return out, max(len(x[0]) for x in out)


def pad_batch(items, L, device):
    B = len(items)
    x = np.full((B, L), PAD, dtype=np.int64)
    wk = np.full((B, L), NOOP, dtype=np.int64)
    wv = np.full((B, L), NOOP, dtype=np.int64)
    rq = np.full((B, L), NOOP, dtype=np.int64)
    for r, (ids, _, _, _, k, v, q) in enumerate(items):
        n = min(len(ids), L)
        x[r, :n] = ids[:n]
        wk[r, :n] = k[:n]
        wv[r, :n] = v[:n]
        rq[r, :n] = q[:n]
    t = lambda a: torch.from_numpy(a).to(device)
    return t(x), t(wk), t(wv), t(rq)


class StoreModel(nn.Module):
    """4 blocks; write/read heads + store injection between blocks 1|2.
    vocab: defaults to this module's synthetic V. Passing it explicitly
    lets the module be reused with another tokenizer, but NOTE that the
    train()/evaluate() helpers below are specific to this file's domain
    (they assume its item tuples and vocabulary) -- an external caller
    should bring its own training loop."""

    def __init__(self, d_model=256, num_heads=4, max_len=300, split=2,
                 vocab=None):
        super().__init__()
        if vocab is not None:
            self.V, self.VI, self.NOOP = vocab, vocab + 1, vocab
        else:
            self.V, self.VI, self.NOOP = V, VI, NOOP
        self.d = d_model
        self.split = split
        self.rope = RoPE(d_model // num_heads, max_len)
        self.embedding = nn.Embedding(self.V, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, self.rope)
             for _ in range(4)])
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, self.V)
        self.wkey = nn.Linear(d_model, self.VI)
        self.wval = nn.Linear(d_model, self.VI)
        self.rq = nn.Linear(d_model, self.VI)
        self.inj = nn.Linear(d_model, d_model)
        for m in (self.wkey, self.wval, self.rq, self.inj):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x, store_mode='gt', gt=None):
        """store_mode: 'gt' (teacher-forced store+queries via gt tuple),
        'self' (model's own writes and queries), 'off' (no injection)."""
        B, T = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        h = self.embedding(x)
        for b in self.blocks[:self.split]:
            h = b(h, mask)
        kl, vl, ql = self.wkey(h), self.wval(h), self.rq(h)

        if store_mode != 'off':
            if store_mode == 'gt':
                wk, wv, rq = gt
            else:
                wk, wv, rq = kl.argmax(-1), vl.argmax(-1), ql.argmax(-1)
            # exact dict lookup, batched: key -> value per example
            table = torch.full((B, self.VI), -1, dtype=torch.long,
                               device=x.device)
            wmask = wk != self.NOOP
            # later writes overwrite earlier (scatter in position order)
            table.scatter_(1, wk.clamp(max=self.VI - 1) * wmask
                           + self.NOOP * (~wmask),
                           wv * wmask - (~wmask).long())
            table[:, self.NOOP] = -1
            got = table.gather(1, rq.clamp(max=self.VI - 1))     # (B,T)
            hit = (rq != self.NOOP) & (got >= 0)
            vemb = self.embedding(got.clamp(min=0).clamp(max=self.V - 1))
            h = h + self.inj(vemb) * hit.unsqueeze(-1).float()

        for b in self.blocks[self.split:]:
            h = b(h, mask)
        return self.output(self.ln_f(h)), kl, vl, ql


def train(items, L, mode, steps, seed, device, d_model=256, lam=0.5,
          bs=32, te=None, every=500):
    """mode: 'ce' | 'store'"""
    set_seed(seed)
    m = StoreModel(d_model=d_model, max_len=L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    traj = []
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(bs)]
        x, wk, wv, rq = pad_batch(ch, L, device)
        inp, tgt = x[:, :-1], x[:, 1:]
        gt = (wk[:, :-1], wv[:, :-1], rq[:, :-1])
        sm = 'gt' if mode == 'store' else 'off'
        lg, kl, vl, ql = m(inp, store_mode=sm, gt=gt)
        vm = (tgt != PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1),
                               ignore_index=PAD)
        if mode == 'store':
            # DENSE interface supervision (dissection: density + content
            # both required). NOOP is a real target, not ignored.
            for logit, target in ((kl, gt[0]), (vl, gt[1]), (ql, gt[2])):
                lf = F.cross_entropy(logit.reshape(-1, VI),
                                     target.reshape(-1), reduction='none')
                loss = loss + lam * (lf * vm.reshape(-1)).sum() / \
                    vm.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if te is not None and (st + 1) % every == 0:
            a = evaluate(m, te, L, device,
                         'self' if mode == 'store' else 'off')[0]
            traj.append((st + 1, round(a, 4)))
    return m, traj


@torch.no_grad()
def evaluate(model, items, L, device, store_mode):
    """Answer top-1 (candidate-restricted, all Q slots pooled) + interface
    accuracies (write events, read events) under the given store mode."""
    model.eval()
    hit, wacc, racc = [], [], []
    for i in range(0, len(items), 64):
        ch = items[i:i + 64]
        x, wk, wv, rq = pad_batch(ch, L, device)
        gt = (wk, wv, rq)
        lg, kl, vl, ql = model(x, store_mode=store_mode, gt=gt)
        for r, (ids, apos, cands, valid, k, v, q) in enumerate(ch):
            for j, ap in enumerate(apos):
                if ap - 1 >= x.shape[1]:
                    continue
                sub = lg[r, ap - 1, cands]
                hit.append(int(cands[int(sub.argmax())] == valid[j]))
            wp = [p for p in range(len(ids)) if k[p] != NOOP]
            if wp:
                wacc.append(float(np.mean(
                    [int(kl[r, p].argmax().item() == k[p]
                         and vl[r, p].argmax().item() == v[p])
                     for p in wp])))
            rp = [p for p in range(len(ids)) if q[p] != NOOP]
            if rp:
                racc.append(float(np.mean(
                    [int(ql[r, p].argmax().item() == q[p]) for p in rp])))
    f = lambda z: float(np.mean(z)) if z else float('nan')
    return f(hit), f(wacc), f(racc)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nes = [int(n) for n in sys.argv[1].split(',')] if len(sys.argv) > 1 \
        else [4, 10, 20, 40]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 \
        else [0, 1]
    Q = 3

    for ne in nes:
        tr, La = make_dataset(16000, 0, ne, Q)
        te, Lb = make_dataset(1000, 9000, ne, Q)
        L = max(La, Lb) + 1
        print(f'\nne={ne}  Q={Q}  L={L}  chance={1.0/ne:.3f}  steps={steps}',
              flush=True)
        if ne == nes[0]:
            print('  sample: ' + ' '.join(ITOS[t] for t in tr[0][0][:40]) +
                  ' ...', flush=True)
        for mode in ('ce', 'store'):
            for sd in seeds:
                m, traj = train(tr, L, mode, steps, sd, device, te=te)
                sm = 'self' if mode == 'store' else 'off'
                a, wa, ra = evaluate(m, te, L, device, sm)
                extra = ''
                if mode == 'store':
                    ag, _, _ = evaluate(m, te, L, device, 'gt')
                    extra = (f'  [gt-store {ag:.3f}  write {wa:.3f}  '
                             f'read {ra:.3f}]')
                print(f'  {mode:>6} seed {sd}:  acc {a:.3f}{extra}  '
                      f'traj {[v for _, v in traj]}', flush=True)
