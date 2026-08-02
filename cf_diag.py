"""Why does the store arm fail on held-out counterfactuals? (finding 6c)

The README asserts two mechanisms:
  (i)  the write head is a vocabulary classifier and structurally cannot
       emit a key token it was never trained to emit ('sea', 'moon');
  (ii) the model has learned to RELY on injection, so an empty store is
       worse than never having had one.

Neither was measured. A mundane alternative to (ii) is plain
interference: the three extra interface losses degrade the LM path's
ability to learn the copy rule, and injection has nothing to do with it.
These predict opposite results when the store is switched off at eval:

  RELIANCE      store off -> held-out RECOVERS toward CE's ~0.99
                (the LM path knows the rule; injection was overriding it)
  INTERFERENCE  store off -> held-out stays ~0.1-0.2
                (the LM path never learned the rule)

and (i) is measured directly: on held-out examples, what does the write
head actually emit at the assertion, and how often is it the correct key?

One seed (the mechanism is qualitative; the 0.162/0.096 result it
explains is already replicated in cfstore7.log).
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

import cf_store as CF

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    ft_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    sd = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    words = CF.load_words()
    stoi, itos = CF.build_vocab(words)
    V = len(stoi)
    tr_cf, La = CF.make_cf_dataset(6000, 0, stoi, True, CF.FACTS_TRAIN)
    tr_true, Lb = CF.make_cf_dataset(3000, 1, stoi, False, CF.FACTS_TRAIN)
    tr_plain = CF.make_plain_windows(3000, 2, stoi, words)
    tr = tr_cf + tr_true + tr_plain
    import random
    random.Random(3).shuffle(tr)
    te, Lc = CF.make_cf_dataset(1200, 9000, stoi, True, CF.FACTS_EVAL)
    te_seen, Ld = CF.make_cf_dataset(1200, 9100, stoi, True, CF.FACTS_TRAIN)
    L = max(La, Lb, Lc, Ld, 80) + 1

    print(f'CF-DIAG  seed={sd}  L={L}  (explains cfstore7 store arm '
          f'0.162/0.096 vs CE 0.985/0.994)', flush=True)
    m = CF.pretrain(stoi, words, pre_steps, sd, device)
    print(f'  prior check {CF.prior_check(m, stoi, device):.2f}', flush=True)
    m = CF.finetune(m, tr, L, V, 'store', ft_steps, sd, device)

    for tag, items in (('HELD-OUT', te), ('seen-obj', te_seen)):
        on = CF.evaluate(m, items, L, V, stoi, device, 'self')
        off = CF.evaluate(m, items, L, V, stoi, device, 'off')
        print(f'  {tag:>9}: store ON  cf {on["cf"]:.3f}  normal {on["normal"]:.3f}'
              f'   |  store OFF cf {off["cf"]:.3f}  normal {off["normal"]:.3f}',
              flush=True)

    # (i) what does the write head emit, SPLIT BY KEY TYPE?
    # Each held-out example contains two writes: the counterfactual
    # subject (a key never written in training) and the normal partner
    # (a key written throughout training). Pooling them averages ~1.0
    # and ~0.0 into an uninformative 0.5, which is what the first run
    # reported.
    m.eval()
    heldtok = {stoi[o] for o, _ in CF.FACTS_EVAL}
    stat = {'never-trained key': [0, 0, {}], 'trained key': [0, 0, {}]}
    with torch.no_grad():
        for i in range(0, 600, 64):
            ch = te[i:i + 64]
            x, wk, wv, rq = CF.pad_batch(ch, L, V, device)
            _, kl, vl, ql = m(x, store_mode='gt', gt=(wk, wv, rq))
            pk = kl.argmax(-1)
            for r, (ids, k, v, q, sites) in enumerate(ch):
                for p in range(min(len(ids), L)):
                    if k[p] == V:
                        continue
                    st = stat['never-trained key' if int(k[p]) in heldtok
                              else 'trained key']
                    st[1] += 1
                    got = int(pk[r, p])
                    st[0] += int(got == k[p])
                    nm = itos[got] if got < V else '<no-write>'
                    st[2][nm] = st[2].get(nm, 0) + 1
    print('\n  write head at held-out-example assertions, by key type:',
          flush=True)
    for name, (ok, tot, em) in stat.items():
        top = sorted(em.items(), key=lambda z: -z[1])[:5]
        print(f'    {name:>18}: correct {ok / max(tot, 1):.3f} (n={tot})'
              '   emits: ' + ', '.join(f'{a} {b}' for a, b in top),
              flush=True)
