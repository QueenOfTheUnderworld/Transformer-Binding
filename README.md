# Binding in Small Transformers: Formation Cost, Scaffolds, and External Stores

Research code for a study of **variable binding** in small (4-layer, ~3–13M
parameter) transformers: when binding circuits form under gradient descent,
what auxiliary training signals change that, and what happens when binding is
externalized to a non-differentiable store.

All experiments are on **synthetic diagnostic domains** with exact ground
truth and numerically audited shortcut controls. This is not a benchmark
suite and makes **no claims about large language models** beyond stated
hypotheses.

> **Status: archived.** This is a complete record of a finished
> investigation, not an active project — it is not maintained and issues
> or pull requests will not be answered. Everything measured is reported,
> including the failures. Raw stdout is committed under
> [`results/`](results/) for most numbers, with the gaps listed in
> [`results/README.md`](results/README.md) — some logs predate small
> changes to the scripts, and two of finding 6's three regimes are
> earlier data designs that the released `cf_store.py` no longer
> produces.

## Summary

1. Cross-entropy alone does not form multi-binding circuits at these
   budgets: 0/4 seeds solve at 5,000 steps, 0/2 at 20,000.
2. Adding dense auxiliary readout heads, one per binding (the register
   scaffold), takes the same task to 8/8 seeds.
3. Formation is plateau escape, not a capacity wall. Solve times spread
   from 500 to 2,250 steps and every seed eventually solves, so short
   negative runs mean little in this regime.
4. Bound values can key a further in-context lookup (2/3 seeds).
5. A non-differentiable dict can be driven end-to-end by supervised
   write/read heads, and the model uses it correctly from its own writes
   (0.99 at up to 40 entities). But in this domain the dict returns the
   answer itself and the model is at chance without it, so this measures
   tool-interface installation, not binding.
6. What generalizes is set by what the training data uniquely
   determines. Where the store loses to CE, the cause is measured: its
   write head never emits a key it was not trained on (0.000 vs 1.000
   for trained keys), and the model has delegated so completely that
   there is no fallback (0.076 with the store off, on trained objects).

Findings 4, 5 and 6 each carry a defect or confound found in late
adversarial review; each is stated inline where the finding is. Limits
are also collected in "What is NOT claimed".

## The idea behind it

> If you want the model to descend to the solution you want, that
> solution has to be the lowest-energy way for it to live.

That is the perspective the work was done from, not a result. It reads
as obvious and is easy to violate. Gradient descent does not find the
solution you had in mind; it finds the cheapest thing that fits the data
in front of it, and "cheapest" is set jointly by the loss and the
training distribution. Most of what is written up here is a case of that
going one way or the other.

Where the desired solution was not the cheapest option, it did not get
learned. Cross-entropy alone never formed the multi-binding circuit
(finding 1). Interfaces settled on whatever narrow rule the data still
permitted — a four-way key detector, a fixed per-object answer — and
transfer to held-out cases was exactly zero (finding 6). Twice the
training data admitted a shortcut we had not thought to audit, and the
model took it (findings 4 and 6).

Where the desired solution was made cheap, it appeared. A dense
auxiliary loss on per-binding readouts lowered the cost of the circuit
enough that every seed found it (finding 2). Handing the binding to an
external dict made the remaining work positional copying, and the model
did it immediately — so completely that it learned nothing else, which
is the same principle producing an unwanted result (finding 5). And when
the data was arranged so the general algorithm was the *only* rule
consistent with it, plain cross-entropy learned that algorithm and
carried it to objects it had never seen (finding 6, row 3).

The corollary is the part worth keeping: an objective change and a data
change are the same kind of move. Both are edits to the energy
landscape, and the second is often the cheaper edit.

## Findings (each with the script that produces it)

Model throughout: 4-layer causal transformer, RoPE, d_model 16–512
across sweeps (headline results at 256), trained with AdamW. "Solve" =
≥0.95 held-out accuracy, candidate-restricted.

**1. Cross-entropy alone does not form multi-binding circuits at these
budgets** (`bind_hazard.py` with `noreg`; ablations in
`bind_dissect.py`). On a task requiring Q=3 simultaneous bindings fused
into a single answer token (count of queried objects with a target
property; total target-count is constant by construction, so the
bind-free "count target words" shortcut carries zero information):
CE-only training stays at the majority baseline: 0/4 seeds solve at
5,000 steps, 0/2 at 20,000. The best seed reaches 0.57 against a 0.45
baseline, which is a crawl rather than a solve.

**Schedule caveat, and it applies to every negative in this repo.** All
runs use a cosine LR schedule with `T_max` set to the run's own step
count, so any arm declared a failure was annealed to LR≈0 at exactly the
step where it was scored. This repo contains a demonstration that the
practice manufactures negatives: bind-use at 5,000 steps is 3/3 flat
(`results/bind_queue.log`), and the same domain and scaffold at 10,000
steps solves on 2/3 seeds — at steps 5,000 and 6,250
(`results/binduse_reg10k.log`). Read every 0/N in this document with
that in mind. The partial defence for finding 1 specifically is the
20,000-step arm: it ran at full LR through step ~10,000, four times the
scaffold's slowest solve (2,250), and still did not solve. Constant-LR
and warm-restart controls were not run.

**2. A dense auxiliary loss on per-binding readout heads enables it**
(the **register scaffold**; `bind_fusion.py`,
`bind_hazard.py`). Adding Q output-side readout heads (one per queried object) trained to report
each queried object's bound property at every position (ordered targets,
`<nothing>` when unbound) takes the same task to **8/8 seeds solved** at
d=256 within 5,000 steps, and Q=4 to 3/3 seeds (6,000-step run, solves at
1,500–3,000). Ablations (`bind_dissect.py`): the effect requires **both**
dense positional coverage (supervision only at the answer position: 0/3)
**and** task-relevant content (same-density supervision on query-entity
*names*: 1/3, slow).

The heads are a **pure side-output**: answer logits and register logits
are two separate linear maps of the same post-`ln_f` hidden state, and
the register map feeds nothing downstream, so zeroing it leaves answer
logits bit-identical (verified, max |Δ| = 0).
Whatever the heads contribute has to happen through the shared trunk
during training. At
convergence the register heads themselves read out perfectly (per-slot
1.000, both seeds), though that is a circular measurement — see "Two
late controls".

**3. Formation happens after a variable delay, not at a capacity wall**
(`bind_hazard.py`). First-passage times across 8 seeds (Q=3, d=256):
[500, 750, 1000, 1000, 1000, 1500, 2250, 2250] steps, all reaching
1.000. Runs sit near the majority baseline for a long stretch, then
climb over roughly 1,000–1,250 steps. The script pre-registers three
formation models (basin-volume, memoryless search, race-against-
entrenchment) and the original writeup called this a win for the
memoryless one; that model selection was never actually performed, and
with 8/8 solving and solve times quantized to a 250-step grid it cannot
be. What the data supports is only the weaker statement in this
heading. Negative results
from short runs are therefore unreliable in this regime. Several earlier
"walls" in this project turned out to be truncated plateaus. The
1,500-step observation that d=512 underperformed d=256 (best 0.643 vs.
1.000) is probably another one; it was never retested at 5,000 steps.
The width sweep is also unscaled — initialization is N(0, 0.02) and lr
is 1e-3 at every d_model — so any width comparison here has a mundane
init/effective-LR explanation available before any capacity story. Median first-passage time grows with Q (~1,000 at Q=3 → ~2,250 at Q=4,
under the scaffold), so "patience wins" is established only for Q≤4 —
and that comparison crosses different schedule lengths (5,000 vs 6,000
steps) and different seed counts (8 vs 3), so the growth rate is
indicative at best. If plateau length grows quickly in Q there is an
effective wall; it was not measured.

**4. Bound values can key further computation** (`bind_use.py`). Task:
retrieve two queried objects' properties, then answer with the relation
that *pair* maps to in an in-context rule table (reshuffled per example, so
not weight-memorizable). With the register scaffold: 2/3 seeds reach
~1.000 (solve at 5,000–6,250 steps; third seed still on plateau at
10,000). Trajectories show two-stage formation: chance → a ~0.68 shelf →
1.000.

**Known defect in this domain, found in late review.** The committed
audit tests three shortcuts (majority 0.264, one-resolved-binding 0.499,
first-rule 0.341) and misses a fourth that works perfectly. With three
entities and rules covering all three property pairs, the *unqueried*
entity is identifiable from names alone, and the one rule not containing
its property is the answer: measured accuracy **1.0000** over 4,000
examples. So the intended reading — that the model must resolve two
bindings and use the resulting *pair* as a key — is not forced by the
data; resolving one binding and taking a set complement suffices. The
claim that survives is the weaker one: a bound value can key a further
in-context lookup. Blocking the shortcut needs ne≥4 or distractor rules
beyond the pair cover, and was not run.

**5. A non-differentiable key–value store can be driven end-to-end by
supervised interface heads — but in this domain the store, not the
model, does the binding** (`bind_store.py`). The store is a plain exact
dict. Supervised write and read heads sit mid-stack, lookups are hard
argmax, **no gradient flows through the store**, and the retrieved value
is injected into the residual stream. Evaluated with the model's own
writes and queries:

| entities | chance | CE (2 seeds) | store, self-written (2 seeds) |
|---|---|---|---|
| 4  | 0.250 | 0.317 / 0.350 | 0.912 / 0.975 |
| 10 | 0.100 | 0.100 / 0.153 | 0.991 / 0.992 |
| 20 | 0.050 | 0.055 / 0.052 | 0.994 / 0.992 |
| 40 | 0.025 | 0.029 / 0.028 | 0.993 / 0.991 |

**Read this table narrowly.** Three facts, each verified, bound what it
can mean:

1. In this domain the answer *is* the stored value, and the read query
   fires at the position where the answer is scored. The ground-truth
   lookup at that position returns the label itself in **1500/1500**
   cases. The store is handing the model its answer.
2. All three interface targets are fixed-offset positional copies: the
   write key is the token at offset −5 (3000/3000), the write value at
   −2 (3000/3000), the read query at absolute position 4j+1 for slot j
   (300/300). No operation in the model performs an association; the
   dict's lookup does.
3. Switching injection off at eval collapses the trained store model to
   chance — **0.098 at ne=10 (chance 0.100), 0.030 at ne=40 (chance
   0.025)**. The model has learned none of the task itself.

So the flat curve from 4 to 40 entities is a property of a Python dict,
not a measurement of model capacity, and the earlier framing of this
result as "removes the capacity ceiling" was wrong. Note also that the
CE arm never solved *any* entity count, including ne=4, so this table
contains no ceiling to remove.

What the result does support is narrower and still worth recording: a
**non-differentiable tool interface can be installed by dense output-side
supervision alone** — hard argmax reads and writes, no gradient through
the lookup — and the model then drives it correctly at inference from its
own writes (interface accuracy 1.000, one cell 0.999), immediately
(≥0.90 by step 500) and without degrading as the table grows. The
matched-supervision control below asks whether the same information
delivered without the machinery would do as well; it would not, though
see that section for why that comparison is weaker than it looks.

Further caveats: the store is teacher-forced during training; keys are
unique by construction, so collisions are untested; n=2 seeds per cell.

**6. Models learn exactly what the training distribution uniquely
determines — and interface shape decides what generalizes**
(`cf_store.py`). Counterfactual-tracking domain: pretrained TinyStories
model, finetuned on *templated* stories (not natural text) to answer
probes about objects whose colors are asserted in-context; evaluation on
held-out objects never seen in finetuning, asserted counterfactually.
The pretrain prior is only partially installed: the check reports 0.33–0.50
of the six facts top-1 and warns rather than aborting, and it is an
average over all six — whether the prior actually holds for the two
*evaluated* objects was never checked, so "the prior votes against" is
unverified for exactly the cells that matter. Note also that held-out
objects are absent only from the templated stories; the finetuning mix
includes ~240k tokens of raw TinyStories, where they occur as ordinary
words. Three regimes, 2 seeds
each (held-out cf accuracy):

| finetuning data admits | CE | store interface |
|---|---|---|
| a fixed per-object answer for held-out objects | 0.000 / 0.000 | 0.000 / 0.000 |
| the answer only via in-story assertion, but held-out objects still appear with fixed true colors | 0.000 / 0.000 | 0.480 / 0.190 |
| held-out objects absent from the templated finetuning data (algorithm is the unique global fit) | **0.985 / 0.994** | 0.162 / 0.096 |

**Row 1.** When a narrow per-slice rule fits the training data, both
regimes learn that rule and held-out transfer is exactly zero. Four
successive designs reproduced this before the cause was isolated. Three
of the four were stopped after one seed once the pattern was identical;
only the last ran both seeds.

**Row 2.** When the data contains a competing fixed association, the
store interface partially resists it and CE does not: 0.480 vs. 0.190.
At n=2 seeds the direction replicates but the magnitude does not.

**Row 3.** When the data makes the general algorithm the unique fit,
plain CE generalizes it to held-out objects (0.985 / 0.994). **The right
comparison here is not chance.** A rule needing no object tracking at
all — "answer whichever asserted color is not one of the four
fact-colors" — scores **0.883** on this eval set (0.866 on the normal
probe). So the measured gain over a bind-free heuristic is 0.985 vs
0.883, not 0.985 vs 0.1. The store
arm *underperforms* CE here. Two measurements explain why (`cf_diag.py`,
`results/cfdiag.log`). First, the write head is a vocabulary classifier,
and on held-out examples it emits the correct key **1.000** of the time
for keys it was trained on and **0.000** of the time for keys it was
not — it substitutes a training key instead. Second, the model delegates
the whole task to the store: switching the store off at eval drops
accuracy to **0.076 even on trained objects** (from 1.000), while the
normal-color probe stays at 1.000. So the answer path never learns
assertion-following at all. An unseen key produces no write, no
injection, and nothing behind it.

A pointer-structured interface was the predicted fix for row 3: heads
select context *positions* by causal attention instead of vocabulary
classes (`cf_pointer.py`). It scored held-out 0.051 / 0.044 over 2 seeds — no better than the
classifier interface. **But this test has a bug and should be treated as
inconclusive rather than as a falsification.** `PointerHead` sizes its
class space from the current sequence length, and training runs on
`inp = x[:, :-1]` while eval runs on the full `x`; the no-op class
therefore sits at a different index at eval than in training, and an
extra position column appears that never existed during training. The
comparison classifier has a fixed vocabulary-sized output and no such
shift. The intended reading — that a learnable pointer scorer keys on
token identity as readily as a classifier, so changing the output space
does not force structural selection — remains plausible and untested.

Generalization scope is set by what the data uniquely determines, and
interface architecture alone does not buy more of it. The obvious
remaining lever was not tested: hundreds rather than sixteen distinct
training keys, so that memorizing the key set costs more than learning
the structural rule.

## Two late controls: is the store just extra supervision, and what do the register heads encode?

Run late, specifically to close the two weakest points above
(`store_control.py`; raw output in `results/store_control.log`).

**A. Matched supervision for finding 5.** Third arm: the same
entity→property mapping supervised just as densely via finding 2's
register scaffold, but with **no store and no injection**. Both informed
arms are told the same thing; only one has retrieval machinery.

| entities | chance | CE | **register scaffold, no store** | store |
|---|---|---|---|---|
| 10 | 0.100 | 0.100 / 0.153 | **0.095 / 0.103** | 0.991 / 0.992 |
| 20 | 0.050 | 0.055 / 0.052 | **0.050 / 0.055** | 0.994 / 0.992 |
| 40 | 0.025 | 0.029 / 0.028 | **0.025 / 0.027** | 0.993 / 0.991 |

The scaffold arm is at chance at every entity count. Dense supervision
of the identical mapping does **not** substitute for the store, so
finding 5's gap is not explained by supervision quantity — the retrieval
machinery is doing the work.

This also cuts the other way: the same scaffold that *enables* Q=3
fusion in finding 2 is useless here. In finding 2 the answer is a count,
so the registers supply an intermediate the model must still combine.
Here the answer *is* the register contents, and supervising a
side-output that already holds the answer teaches the answer path
nothing. The scaffold helps when it supplies a step, not when it
supplies the result.

**Three implementation caveats weaken this control**, all found in late
review. The store arm carries three auxiliary CE terms at λ=0.5 each
while the scaffold arm carries one, so "told the same thing" is true of
the information but not of the loss weight. `RegModel` reads the
pre-`ln_f` hidden state whereas finding 2's `FModel` reads post-`ln_f`,
so this is not bit-for-bit finding 2's scaffold. And `StoreModel` never
calls `_init_weights`, so its embedding std is ~1.0 against
`CausalTransformer`'s 0.02 — the scaffold was tested at an
initialization scale where finding 2 never demonstrated it. Given the
store-off result in finding 5 (the store model is at chance without
injection), the honest summary is that neither arm learns the task in
the model: one is handed the answer by a dict, the other is handed it in
a side-output it cannot route into the answer path.

**B. What the register heads encode (finding 2).** Per-slot readout at
the answer prompt after 5,000 steps at Q=3/d=256. This metric was
missing from the earlier runs: what had been logged was an
all-slots-exact figure on a 1,500-step grid, where it read 0.010. That
number is consistent with per-slot accuracy anywhere up to about 0.22,
so it never showed the heads encode nothing.

Both seeds: answer 1.000, **per-slot readout 1.000, all-slots-exact
1.000**. The earlier 0.010 came from a 1,500-step cell whose model had
not solved the task, and says nothing about converged behavior.

So the mechanism is: the scaffold makes the bindings **fully linearly
decodable from the shared trunk**, and the answer head reads that same
trunk. Combined with the side-output fact above, "installs a
representation, then is architecturally incapable of being leaned on at
inference" is the accurate statement — not "is discarded". What is still
untested is the counterfactual that would make this causal: whether a
model that reaches the same answer accuracy *without* the scaffold also
has linearly decodable bindings. Finding 1 makes that hard to test here,
since without the scaffold no model reaches that accuracy.

## If you have compute

This was run on one consumer GPU at ~3M parameters. The results most
worth someone else's time, in order:

**1. Does the register scaffold survive scale?** Finding 2 is the only
result here that would matter if it holds at 100M–1B parameters: a dense
auxiliary loss on per-binding readouts installed a circuit that
cross-entropy did not form at 4× the budget. It costs one extra linear
head and one loss term. If it still helps at scale it is a cheap
mid-training intervention; if it stops helping, that is worth knowing
too, and would suggest the effect is a small-model optimization artifact.
Nothing else here is as portable.

**2. Does it help where the answer is not the register contents?**
Control A found the scaffold useless in a domain where the registers
already held the answer, and finding 2 found it decisive where they held
an intermediate. If that distinction is real, it predicts where this
technique applies: tasks with a genuine intermediate quantity. It is a
one-domain observation and needs a second.

**3. Kill or confirm the shortcuts.** Finding 4's domain has a
complement shortcut scoring 1.000; the fix is ne≥4 or distractor rules.
Finding 6's row 3 has a 0.883 bind-free baseline; a harder colour
assignment closes that. Both are small data changes and both would
sharpen claims currently stated with caveats.

**4. Redo the negatives without the schedule confound.** Every 0/N here
was annealed to LR≈0 at its scoring step. Constant LR, or `T_max` well
beyond the horizon, would tell you which negatives are real.

The store line (finding 5) is not worth scaling as built — the dict
returns the answer. A version where the retrieved value is an
intermediate rather than the label would be a different and more
interesting experiment.

## What is NOT claimed

- No claims about models beyond ~13M parameters or natural-language
  corpora. The headline models for findings 1–5 are ~3–3.5M parameters;
  only finding 6's 8k-vocab models reach ~13M.
- Finding 5's interface targets come free from synthetic structure;
  deciding *what to write* on real text is unsolved here. A real-code
  variant was attempted and was statistically underpowered (36 eval
  sites) — for the record, in that underpowered run the store arm
  trailed the CE arm on both seeds.
- CE-only might solve these tasks at budgets beyond those tested; claims
  are matched-budget, not impossibility.
- The register scaffold's benefit is established for these binding
  tasks only; it has no non-synthetic validation.
- No claim is made that any model here learned to *bind* internally
  through the store (finding 5) — the opposite is measured.
- Finding 4's domain admits a one-binding shortcut scoring 1.000, so it
  does not establish that a computed *pair* keys the lookup.
- The pointer-interface test (finding 6) has a train/eval class-space
  bug and is inconclusive, not a falsification.
- This project began as a broader study of distribution-shaping training
  objectives; that line is summarized in the appendix below and is not
  part of the binding claims.

## Reproduce

Requires: Python ≥3.10, PyTorch ≥2.0 (CUDA strongly recommended), NumPy.
Approximate runtimes come from the committed logs, on a single consumer
RTX-class GPU and will vary.

```bash
pip install torch numpy

# Findings 1+2+3: registers enable (8/8 + first-passage times), ~15 min
python bind_hazard.py 3 256 5000 0,1,2,3,4,5,6,7
# CE-only arm (finding 1), ~8 min
python bind_hazard.py 3 256 5000 0,1,2,3 noreg

# Width/Q grid at 1,500 steps (context for findings 2-3), ~20 min
python bind_fusion.py 2,3 32,64,128 1500

# Finding 2 ablations: names / sparse / ce arms, ~25 min total
python bind_dissect.py names 5000 0,1,2
python bind_dissect.py sparse 5000 0,1,2
python bind_dissect.py ce 20000 0,1

# Finding 4: computing with bound values, ~10 min
python bind_use.py 256 10000 0,1,2 reg

# Finding 5: external store, entity sweep, ~20 min
python bind_store.py 4,10,20,40 3000 0,1

# The two gap-closing controls, ~30 min
python store_control.py AB

# Finding 6: requires TinyStories as plain text, ~1 h per seed pair
TINYSTORIES_PATH=/path/to/tinystories_train.txt python cf_store.py 10000 3000 0,1
# Pointer-interface falsification
TINYSTORIES_PATH=/path/to/tinystories_train.txt python cf_pointer.py 10000 3000 0,1
# Why the store arm fails on held-out keys, ~35 min
TINYSTORIES_PATH=/path/to/tinystories_train.txt python cf_diag.py 10000 3000 0

# Appendix (distribution grounding on natural text), ~1 h
TINYSTORIES_PATH=/path/to/tinystories_train.txt python fs_oracle.py 6000 0,1
```

TinyStories is not redistributed here; any plain-text dump works
(`TINYSTORIES_PATH` points at it). Finding 6 and the appendix are the
only experiments that need it — findings 1–5 are fully synthetic and
need no data files.

Shortcut auditing: `bind_fusion.py` and `bind_use.py` each run a
numerical audit at startup and print the result. For the fusion domain:
the total count of target-property words is constant across examples
(so bind-free counting carries zero information), plus the majority
baseline and the correlation between the first target mention's position
and the answer (−0.016 at Q=2, −0.002 at Q=3). For the rule-lookup
domain: majority 0.264, one-resolved-binding 0.499, first-rule 0.341 —
but see finding 4, where late review found a fourth shortcut this audit
does not test and which scores 1.000. An audit only rules out the
shortcuts it enumerates. `wmem_check.py` is the check script for the register
domains (register-target correctness, oracle support separation,
shortcut audit, and a proof that the permutation-invariant loss is
order-blind). `bind_store.py` and `cf_store.py` have no automated audit;
their shortcut arguments are structural and stated in their docstrings.
Treat any new domain as shortcut-suspect until audited — two of this
project's own early domains contained shortcuts found only this way.

Seeds: findings 1–4 rest on ≥3 seeds or 8-seed solve counts; findings 5
and 6 are 2 seeds per cell (per-seed numbers shown, no averaging over
hidden seeds). `bind_fusion.py` prints best-of-seeds. That is an
existence protocol, chosen because runs in this regime are bimodal
(finding 3): mean accuracy at a fixed step count understates capability.
Prefer the first-passage and solve-count readouts.

## Layout

| file | role |
|---|---|
| `transformer_torch.py` | causal transformer, RoPE. Has softmax, linear and phase attention variants; only softmax is used here. `n_layers` defaults to 1, but every experiment passes `n_layers=4` explicitly. |
| `common.py` | seeding |
| `stress_test.py` | chain-derivation domain + graph-support oracle (shared vocab/eval helpers) |
| `wmem.py`, `wmem_controls.py`, `wmem_check.py` | register readouts, controls, and domain checks; note this line's own headline result is a *negative* — permutation-invariant ("scrambled slot") register losses cap performance by destroying order information |
| `bind_fusion.py` | Q-binding fusion domain (findings 1–2) |
| `bind_hazard.py` | first-passage measurement + CE-only arm (findings 1, 3) |
| `bind_dissect.py` | scaffold ablations (finding 2) |
| `bind_use.py` | pair-keyed rule lookup (finding 4) |
| `bind_store.py` | external binding store (finding 5) |
| `cf_store.py` | counterfactual tracking (finding 6) |
| `cf_pointer.py` | pointer-interface variant; result inconclusive, see finding 6 |
| `cf_diag.py` | why the store arm fails on held-out keys (finding 6, row 3) |
| `store_control.py` | the two gap-closing controls |
| `fs_oracle.py` | appendix: distribution grounding on natural text |
| `results/` | committed stdout of every run behind the numbers above |

## Appendix — the objective this project started from

Before the binding work, the question was whether an "ideal" target
distribution — correct token on top, remaining mass on situationally
licensed words — trains a model toward a more consistent internal world
model than cross-entropy. That line is included because it produced a
clean measurement, not because it succeeded.

`fs_oracle.py` tests it on TinyStories with a mixture target
`a·onehot + b·normalize(support) + c·P_base`, against two controls: a
**distillation control** (same weights, support term removed — isolates
smoothing) and a **mismatch control** (`fctrl`: the same mass metric
scored against a *different* window's support — catches generic
content-word stuffing). Reported with rank and entropy, because five
earlier metrics in this project were maxed out by degenerate models.
Two definitions of "situationally licensed", 2 seeds each:

| support = | top1 | mass on support | mismatch ctrl | rank | entropy |
|---|---|---|---|---|---|
| — (CE baseline) | 0.494 | 0.037 | 0.004 | 33 | 2.25 |
| — (distill control) | 0.498 | 0.038 | 0.004 | 31 | 2.24 |
| *future* content words (W=32 ahead) | 0.483 | 0.069 | 0.008 | 48 | 4.23 |
| *established* content words (W=32 back) | 0.431/0.430 | 0.244/0.241 | 0.008 | 56 | 3.53 |

The `mass on support` column is not one metric. The future-support row
is scored against future support; every other row against established
support. Compare each arm only with a CE baseline measured the same way.
Against future support: 0.069 (oracle) vs 0.070 (CE) — no change.
Against established support: 0.244 (oracle) vs 0.037 (CE) — a 6.6x
rise.

Anticipating upcoming words does nothing except raise entropy (that row
is a single seed — the second seed's oracle arm was never run).
Grounding in the *already-established* scene moves the trained quantity:
mass on the scene's own content words rises 6.6×. The mismatch control
is not flat, though — it doubles, 0.004 → 0.008 — so the "selectivity
improves ~9× to ~30×" framing is a ratio that hides the control firing.
The costs are real: next-token accuracy falls 0.494 → 0.430 and the mean
rank of the true token degrades 33 → 56, proportionally the larger harm
of the two.

Three limits. First, the mass metric is partly circular: it is the
trained target, so its rise mainly shows the objective is learnable.
Second, the mismatch control cannot separate scene grounding from a
plain recency-copy bias — the support is by construction words visible
in the current context, and the control window's words are by
construction not, so a head that simply favours recently-seen content
tokens reproduces the whole signature. No copy or n-gram baseline was
run. Third, nothing here shows the grounding is *useful*: top1 was the
only utility measure, and it went down. Whether a lower `b` buys grounding without the accuracy
cost, and whether a downstream situational-consistency measure would
show a gain that top1 cannot, were not tested.

## Notes

Does Nicholas Plouffe own a display?

## License

MIT — see `LICENSE`.
