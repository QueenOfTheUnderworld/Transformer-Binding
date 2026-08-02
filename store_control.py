"""Two gap-closing measurements for the release.

A. MATCHED-SUPERVISION CONTROL for finding 5.
   The store arm gets dense ground-truth interface supervision that the
   CE arm does not, so "the store beats CE at ne=40" confounds
   EXTERNALIZATION with SUPERVISION QUANTITY. This adds the missing third
   arm: the same entity->property mapping supervised just as densely, via
   finding 2's register scaffold, but with NO store and NO injection
   (store_mode='off'). Both informed arms are told the same thing; only
   one has retrieval machinery.
     store ~ reg  -> the machinery is unnecessary; dense supervision is
                     the whole effect, and finding 5 must be restated.
     store >> reg -> externalization is doing real work.

B. PER-SLOT REGISTER READOUT at the 5,000-step solvers (finding 2).
   Only an all-K-slots-exact metric was ever logged, and only on a
   1,500-step grid; 0.010 exact-match is consistent with ~0.22 per-slot,
   so "the heads encode nothing" was never established. This trains
   Q=3/d=256 for 5,000 steps and reports per-slot accuracy alongside
   exact-match at the answer prompt.
   (Note: the register heads are a pure side-output -- verified that
   zeroing them leaves answer logits bit-identical -- so they cannot be
   load-bearing at inference by construction. The open question is what
   they encode, which is what this measures.)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import set_seed
import bind_store as BS
import bind_fusion as BF


# ---------------------------------------------------------------- A ----

class RegModel(BS.StoreModel):
    """StoreModel + Q register heads; used with store_mode='off'."""

    def __init__(self, Q, **kw):
        super().__init__(**kw)
        self.Q = Q
        self.regh = nn.Linear(self.d, Q * self.VI)
        nn.init.normal_(self.regh.weight, std=0.02)
        nn.init.zeros_(self.regh.bias)

    def with_reg(self, x):
        B, T = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        h = self.embedding(x)
        for b in self.blocks:
            h = b(h, mask)
        hf = self.ln_f(h)
        return self.output(hf), self.regh(h).view(B, T, self.Q, self.VI)


def reg_targets(items, L, Q, NOOP):
    """Slot j holds query j's property from its fact clause onward.
    Derived from the existing interface targets: the entity for query j
    is rquery at that query's answer position; its fact clause is where
    wkey equals that entity."""
    out = np.full((len(items), L, Q), NOOP, dtype=np.int64)
    for r, (ids, apos, cands, valid, wk, wv, rq) in enumerate(items):
        T = min(len(ids), L)
        for j, ap in enumerate(apos):
            if ap - 1 >= T:
                continue
            ent = rq[ap - 1]
            fp = next((p for p in range(T) if wk[p] == ent), None)
            if fp is None:
                continue
            out[r, fp:T, j] = wv[fp]
    return out


def train_reg(items, L, Q, steps, seed, device, lam=0.5, bs=32,
              d_model=256):
    set_seed(seed)
    m = RegModel(Q, d_model=d_model, max_len=L + 1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    import random
    rng = random.Random(seed)
    for st in range(steps):
        m.train()
        ch = [items[rng.randrange(len(items))] for _ in range(bs)]
        x, wk, wv, rq = BS.pad_batch(ch, L, device)
        rg = torch.from_numpy(reg_targets(ch, L, Q, BS.NOOP)).to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        lg, rlg = m.with_reg(inp)
        vm = (tgt != BS.PAD).float()
        loss = F.cross_entropy(lg.reshape(-1, BS.V), tgt.reshape(-1),
                               ignore_index=BS.PAD)
        lf = F.cross_entropy(rlg.reshape(-1, BS.VI),
                             rg[:, :-1, :].reshape(-1), reduction='none')
        loss = loss + lam * (lf.view(vm.shape[0], vm.shape[1], Q).mean(-1)
                             * vm).sum() / vm.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def eval_reg(m, items, L, device):
    m.eval()
    hit = []
    for i in range(0, len(items), 64):
        ch = items[i:i + 64]
        x, wk, wv, rq = BS.pad_batch(ch, L, device)
        lg, _ = m.with_reg(x)
        for r, (ids, apos, cands, valid, _, _, _) in enumerate(ch):
            for j, ap in enumerate(apos):
                if ap - 1 >= x.shape[1]:
                    continue
                sub = lg[r, ap - 1, cands]
                hit.append(int(cands[int(sub.argmax())] == valid[j]))
    return float(np.mean(hit)) if hit else float('nan')


def run_A(device, steps=3000, seeds=(0, 1), nes=(10, 20, 40), Q=3):
    print('=' * 66)
    print('A. MATCHED-SUPERVISION CONTROL (finding 5)')
    print('   register scaffold, same mapping supervised, NO store')
    print('=' * 66, flush=True)
    for ne in nes:
        tr, La = BS.make_dataset(16000, 0, ne, Q)
        te, Lb = BS.make_dataset(1000, 9000, ne, Q)
        L = max(La, Lb) + 1
        accs = []
        for sd in seeds:
            m = train_reg(tr, L, Q, steps, sd, device)
            accs.append(eval_reg(m, te, L, device))
        print(f'  ne={ne:>3}  chance {1.0/ne:.3f}   reg-scaffold '
              + ' / '.join(f'{a:.3f}' for a in accs)
              + '   [store was 0.99, CE was ~chance]', flush=True)


# ---------------------------------------------------------------- B ----

@torch.no_grad()
def readout(m, items, L, K, device):
    """Per-slot and all-slots-exact register accuracy at the answer
    prompt (the position furthest from any clause)."""
    m.eval()
    per, exact = [], []
    for i in range(0, len(items), 128):
        ch = items[i:i + 128]
        x, rg = BF.pad_batch(ch, L, device, K)
        _, rlg = m.both(x)
        pred = rlg.argmax(-1)
        for r, it in enumerate(ch):
            p0 = it[1] - 1
            if p0 >= x.shape[1]:
                continue
            pv, tv = pred[r, p0], rg[r, p0]
            per.append((pv == tv).float().mean().item())
            exact.append(int(torch.equal(pv, tv)))
    return float(np.mean(per)), float(np.mean(exact))


def run_B(device, steps=5000, seeds=(0, 1), Q=3, d=256):
    print()
    print('=' * 66)
    print('B. PER-SLOT REGISTER READOUT at 5,000-step solvers (finding 2)')
    print('=' * 66, flush=True)
    tr, La = BF.make_dataset(24000, 0, Q)
    te, Lb = BF.make_dataset(1500, 9000, Q)
    L = max(La, Lb) + 1
    for sd in seeds:
        m = BF.train(tr, L, Q, d, steps, sd, device, use_reg=True)
        acc, _ = BF.evaluate(m, te, L, Q, device)
        ps, ex = readout(m, te, L, Q, device)
        print(f'  seed {sd}:  answer {acc:.3f}   per-slot readout {ps:.3f}'
              f'   all-slots-exact {ex:.3f}', flush=True)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    which = sys.argv[1] if len(sys.argv) > 1 else 'AB'
    if 'A' in which:
        run_A(device)
    if 'B' in which:
        run_B(device)
