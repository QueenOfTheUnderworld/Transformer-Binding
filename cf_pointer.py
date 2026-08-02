"""Pointer-structured store interface: the fix for v7's inversion.

v7 (clean data, algorithm the unique fit): CE generalizes counterfactual-
following to held-out objects at 0.985/0.994; the classifier-headed store
collapses to 0.162/0.096, because a Linear(d, V+1) write head has an
untrained logit for a never-trained key token ('sea') -- it structurally
cannot name unseen keys -- and the model has learned to rely on injection.

FIX UNDER TEST: write-key / write-value / read-query heads become POINTER
heads -- causal attention over context positions with a learned no-op
class. The emitted token is the input token at the selected position, so
key identity never passes through per-class weights. Prediction: held-out
cf rises toward CE's 0.99 (the pointer op 'select the subject two tokens
back' is position/content-structured, identical for seen and unseen
objects).

Supervision: position targets derived from the existing token targets --
for a target token at site t, the pointer target is the LAST occurrence
of that token at position <= t (assertion subject for keys, the color
token itself for values, the probe's own object mention for queries).
Dense: no-op is the supervised class everywhere else (per the dissection
result that density matters).

Same v7 data (held-out objects absent from finetuning), same pretrain
recipe, 2 seeds. Reference points (v7): CE 0.985/0.994, classifier-store
0.162/0.096.
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import set_seed
import cf_store as CF
from bind_store import StoreModel

PAD, UNK = 0, 1


class PointerHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.noop = nn.Linear(d, 1)
        self.scale = d ** -0.5
        for m in (self.q, self.k):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.noop.weight)
        nn.init.zeros_(self.noop.bias)

    def forward(self, h):
        """h (B,T,d) -> logits (B,T,T+1): positions 0..T-1, class T = noop.
        Causal: position s > t is masked."""
        B, T, _ = h.shape
        sc = torch.einsum('btd,bsd->bts', self.q(h), self.k(h)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=h.device), 1).bool()
        sc = sc.masked_fill(mask.unsqueeze(0), float('-inf'))
        return torch.cat([sc, self.noop(h)], dim=-1)


class PStoreModel(StoreModel):
    """StoreModel with pointer interface heads. Classifier heads from the
    parent are unused; pointer heads emit (position|noop), and the token
    at the selected position becomes the key/value/query."""

    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.d
        self.pk = PointerHead(d)
        self.pv = PointerHead(d)
        self.pq = PointerHead(d)

    def forward(self, x, store_mode='off', gt=None):
        B, T = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        h = self.embedding(x)
        for b in self.blocks[:self.split]:
            h = b(h, mask)
        kl, vl, ql = self.pk(h), self.pv(h), self.pq(h)   # (B,T,T+1)

        if store_mode != 'off':
            if store_mode == 'gt':
                wk, wv, rq = gt                            # token-level GT
            else:
                def sel_tok(logits):
                    s = logits.argmax(-1)                  # (B,T) in [0,T]
                    isn = s == T
                    tok = torch.gather(x, 1, s.clamp(max=T - 1))
                    return torch.where(isn, torch.full_like(tok, self.NOOP),
                                       tok)
                wk, wv, rq = sel_tok(kl), sel_tok(vl), sel_tok(ql)
            table = torch.full((B, self.VI), -1, dtype=torch.long,
                               device=x.device)
            wmask = wk != self.NOOP
            table.scatter_(1, wk.clamp(max=self.VI - 1) * wmask
                           + self.NOOP * (~wmask),
                           wv * wmask - (~wmask).long())
            table[:, self.NOOP] = -1
            got = table.gather(1, rq.clamp(max=self.VI - 1))
            hit = (rq != self.NOOP) & (got >= 0)
            vemb = self.embedding(got.clamp(min=0).clamp(max=self.V - 1))
            h = h + self.inj(vemb) * hit.unsqueeze(-1).float()

        for b in self.blocks[self.split:]:
            h = b(h, mask)
        return self.output(self.ln_f(h)), kl, vl, ql


def make_pos(items, V):
    """For each example, convert (wk, wv, rq) token targets into pointer
    position targets (noop class = L-1 index space handled at loss)."""
    out = []
    for ids, wk, wv, rq, sites in items:
        T = len(ids)
        pk = np.full(T, -1, dtype=np.int64)
        pv = np.full(T, -1, dtype=np.int64)
        pq = np.full(T, -1, dtype=np.int64)
        last = {}
        for t in range(T):
            last[ids[t]] = t
            for tokarr, posarr in ((wk, pk), (wv, pv), (rq, pq)):
                if tokarr[t] != V and tokarr[t] in last:
                    posarr[t] = last[tokarr[t]]
        out.append((ids, wk, wv, rq, sites, pk, pv, pq))
    return out


def pad_batch(items, L, V, device):
    B = len(items)
    x = np.full((B, L), PAD, dtype=np.int64)
    wk = np.full((B, L), V, dtype=np.int64)
    wv = np.full((B, L), V, dtype=np.int64)
    rq = np.full((B, L), V, dtype=np.int64)
    pk = np.full((B, L), -1, dtype=np.int64)
    pv = np.full((B, L), -1, dtype=np.int64)
    pq = np.full((B, L), -1, dtype=np.int64)
    for r, it in enumerate(items):
        ids = it[0]
        n = min(len(ids), L)
        x[r, :n] = ids[:n]
        for src, dst in zip(it[1:4], (wk, wv, rq)):
            dst[r, :n] = src[:n]
        for src, dst in zip(it[5:8], (pk, pv, pq)):
            dst[r, :n] = src[:n]
    t = lambda a: torch.from_numpy(a).to(device)
    return (t(x), t(wk), t(wv), t(rq), t(pk), t(pv), t(pq))


def pointer_loss(logits, pos_tgt, valid):
    """logits (B,T',C) with C = full_T + 1; pos_tgt -1 = noop; noop class
    index = C - 1 (position classes are indexed by the FULL sequence
    length even when the time axis is sliced)."""
    C = logits.shape[-1]
    tgt = pos_tgt.clone()
    tgt[tgt < 0] = C - 1
    lf = F.cross_entropy(logits.reshape(-1, C), tgt.reshape(-1),
                         reduction='none')
    return (lf * valid.reshape(-1)).sum() / valid.sum().clamp(min=1.0)


def finetune(m, items, L, V, steps, seed, device, lam=0.5, bs=32, lr=2e-4):
    set_seed(seed + 100)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(bs)]
        x, wk, wv, rq, pk, pv, pq = pad_batch(ch, L, V, device)
        inp, tgt = x[:, :-1], x[:, 1:]
        gt = (wk[:, :-1], wv[:, :-1], rq[:, :-1])
        lg, kl, vl, ql = m(inp, store_mode='gt', gt=gt)
        vm = (tgt != PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1),
                               ignore_index=PAD)
        for logit, ptgt in ((kl, pk[:, :-1]), (vl, pv[:, :-1]),
                            (ql, pq[:, :-1])):
            loss = loss + lam * pointer_loss(logit, ptgt, vm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(m, items, L, V, stoi, device, store_mode):
    m.eval()
    cids = [stoi[c] for c in CF.COLORS if c in stoi]
    res = {'cf': [], 'normal': []}
    for i in range(0, len(items), 64):
        ch = items[i:i + 64]
        x, wk, wv, rq, _, _, _ = pad_batch(ch, L, V, device)
        lg, _, _, _ = m(x, store_mode=store_mode, gt=(wk, wv, rq))
        for r, it in enumerate(ch):
            for (p, tgt, kind) in it[4]:
                if p >= x.shape[1]:
                    continue
                sub = lg[r, p, cids]
                res[kind].append(int(cids[int(sub.argmax())] == tgt))
    return {k: float(np.mean(v)) if v else float('nan')
            for k, v in res.items()}


def pretrain(stoi, words, steps, seed, device, d_model=256, L=128, bs=32):
    set_seed(seed)
    ids = np.array([stoi.get(w, UNK) for w in words], dtype=np.int64)
    m = PStoreModel(d_model=d_model, max_len=200,
                    vocab=len(stoi)).to(device)
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
        if (st + 1) % 2000 == 0:
            print(f'    pretrain step {st+1}  loss {loss.item():.3f}',
                  flush=True)
    return m


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    ft_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 \
        else [0, 1]

    words = CF.load_words()
    stoi, itos = CF.build_vocab(words)
    V = len(stoi)

    tr_raw, La = CF.make_cf_dataset(6000, 0, stoi, True, CF.FACTS_TRAIN)
    true_raw, Lb = CF.make_cf_dataset(3000, 1, stoi, False, CF.FACTS_TRAIN)
    plain_raw = CF.make_plain_windows(3000, 2, stoi, words)
    tr = make_pos(tr_raw + true_raw + plain_raw, V)
    random.Random(3).shuffle(tr)
    te_raw, Lc = CF.make_cf_dataset(1200, 9000, stoi, True, CF.FACTS_EVAL)
    seen_raw, Ld = CF.make_cf_dataset(1200, 9100, stoi, True, CF.FACTS_TRAIN)
    te = make_pos(te_raw, V)
    te_seen = make_pos(seen_raw, V)
    # eval lengths MUST enter the budget: pad_batch would silently
    # truncate a longer held-out example and evaluate() would silently
    # drop its probe sites
    L = max(La, Lb, Lc, Ld, 80) + 1
    print(f'CF-POINTER  V={V}  L={L}  (v7 data; refs: CE .985/.994, '
          f'classifier-store .162/.096)', flush=True)

    for sd in seeds:
        print(f'\n== seed {sd}: pretraining {pre_steps} steps ==',
              flush=True)
        m = pretrain(stoi, words, pre_steps, sd, device)
        pc = CF.prior_check(m, stoi, device)
        print(f'  PRIOR CHECK: {pc:.2f}', flush=True)
        m = finetune(m, tr, L, V, ft_steps, sd, device)
        r = evaluate(m, te, L, V, stoi, device, 'self')
        rs = evaluate(m, te_seen, L, V, stoi, device, 'self')
        keep = CF.prior_check(m, stoi, device)
        print(f'  pointer-store:  HELD-OUT cf {r["cf"]:.3f}   '
              f'seen-obj cf {rs["cf"]:.3f}   '
              f'normal {r["normal"]:.3f}   prior-retention {keep:.2f}',
              flush=True)
