# Raw run logs

The exact stdout of the runs behind every number in the top-level README,
so claims can be checked without a GPU. File → finding:

| file | finding |
|---|---|
| `hazard.log`, `hazard_results.json` | 3 (first-passage times, 8 seeds) and 2 (8/8 solved) |
| `hazard_Q3_d256_noreg.json`, `bind_queue.log` | 1 (CE-only 0/4); also Q=4 3/3 and the first bind-use run |
| `dissect.log`, `dissect_*.json` | 2 (names / sparse ablations) and 1 (CE at 20,000 steps) |
| `fusion_grid.log`, `fusion_grid2.log` | 2–3 context: the d=16…512 × Q=1,2,3 grid at 1,500 steps |
| `binduse_reg10k.log` | 4 (pair-keyed rule lookup, 10,000 steps) |
| `store.log` | 5 (external store, ne=4…40) |
| `store_control.log` | 5 (matched-supervision control) and 2 (per-slot readout) |
| `cfstore5.log`, `cfstore6.log`, `cfstore7.log` | 6 (the three regimes, in order) |
| `cfpointer.log` | 6 (pointer-interface falsification) |
| `fsoracle.log`, `fsoracle2.log` | appendix (anticipation form, then grounding form) |
| `codestore.log` | the underpowered real-code attempt referenced under "What is NOT claimed" |

Logs from superseded designs (counterfactual v1–v4) are not included; they
are described in the top-level README's finding 6 and produced the same
0.000 held-out result as `cfstore5.log`.
