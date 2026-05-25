# Phase 3 Stage C Report

Gate status: PASS

This stage is a corruption/trend gate only, not final science.

## Targets

| Environment | Expected completed tasks per framework/seed |
| --- | ---: |
| ALFWorld valid_unseen | 134 |
| SQL test | 200 |
| ScienceWorld test categories | 30 |

## Gate Checks

- Errors: 0
- Warnings: 0

### Notes

- SQL react_cr task sql_288 emitted recovered bare code-fence actions in seeds 0, 1, 2; the environment rejected them as invalid SQL and the final task records still succeeded, so this is tracked as non-fatal model-formatting noise rather than metric corruption.

## Run Summary

| Environment | Framework | Seed | Tasks | Successes | Avg Reward | Context Memories | Help Calls | Retrieval Records |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alfworld | react | 0 | 134 | 91 |  | 0 | 0 | 0 |
| alfworld | react | 1 | 134 | 91 |  | 0 | 0 | 0 |
| alfworld | react | 2 | 134 | 91 |  | 0 | 0 | 0 |
| alfworld | react_cr | 0 | 134 | 88 | 0.6567 | 543 | 0 | 268 |
| alfworld | react_cr | 1 | 134 | 84 | 0.6269 | 543 | 0 | 268 |
| alfworld | react_cr | 2 | 134 | 85 | 0.6343 | 543 | 0 | 268 |
| alfworld | react_tr | 0 | 134 | 101 | 0.7537 | 0 | 302 | 0 |
| alfworld | react_tr | 1 | 134 | 101 | 0.7537 | 0 | 302 | 0 |
| alfworld | react_tr | 2 | 134 | 101 | 0.7537 | 0 | 303 | 0 |
| alfworld | react_cr_tr | 0 | 134 | 111 | 0.8284 | 543 | 206 | 268 |
| alfworld | react_cr_tr | 1 | 134 | 111 | 0.8284 | 543 | 206 | 268 |
| alfworld | react_cr_tr | 2 | 134 | 111 | 0.8284 | 543 | 206 | 268 |
| alfworld | react_hard_neg_cr_tr | 0 | 134 | 75 | 0.5597 | 626 | 353 | 268 |
| alfworld | react_hard_neg_cr_tr | 1 | 134 | 74 | 0.5522 | 626 | 364 | 268 |
| alfworld | react_hard_neg_cr_tr | 2 | 134 | 73 | 0.5448 | 626 | 364 | 268 |
| sql | react | 0 | 200 | 183 | 0.9203 | 0 | 0 | 0 |
| sql | react | 1 | 200 | 183 | 0.9203 | 0 | 0 | 0 |
| sql | react | 2 | 200 | 183 | 0.9203 | 0 | 0 | 0 |
| sql | react_cr | 0 | 200 | 177 | 0.8888 | 410 | 0 | 400 |
| sql | react_cr | 1 | 200 | 177 | 0.8897 | 410 | 0 | 400 |
| sql | react_cr | 2 | 200 | 177 | 0.8897 | 410 | 0 | 400 |
| sql | react_tr | 0 | 200 | 182 | 0.9189 | 0 | 23 | 0 |
| sql | react_tr | 1 | 200 | 182 | 0.9189 | 0 | 23 | 0 |
| sql | react_tr | 2 | 200 | 181 | 0.9139 | 0 | 23 | 0 |
| sql | react_cr_tr | 0 | 200 | 184 | 0.9218 | 410 | 20 | 400 |
| sql | react_cr_tr | 1 | 200 | 184 | 0.9218 | 410 | 20 | 400 |
| sql | react_cr_tr | 2 | 200 | 184 | 0.9218 | 410 | 20 | 400 |
| sql | react_hard_neg_cr_tr | 0 | 200 | 179 | 0.9059 | 336 | 24 | 400 |
| sql | react_hard_neg_cr_tr | 1 | 200 | 179 | 0.9059 | 334 | 24 | 398 |
| sql | react_hard_neg_cr_tr | 2 | 200 | 179 | 0.9059 | 336 | 24 | 400 |
| scienceworld | react | 0 | 30 | 10 | 0.3673 | 0 | 0 | 0 |
| scienceworld | react | 1 | 30 | 11 | 0.3383 | 0 | 0 | 0 |
| scienceworld | react | 2 | 30 | 6 | -0.0630 | 0 | 0 | 0 |
| scienceworld | react_cr | 0 | 30 | 8 | 0.1057 | 132 | 0 | 60 |
| scienceworld | react_cr | 1 | 30 | 7 | 0.1923 | 132 | 0 | 60 |
| scienceworld | react_cr | 2 | 30 | 9 | 0.2150 | 132 | 0 | 60 |
| scienceworld | react_tr | 0 | 30 | 9 | 0.3477 | 0 | 42 | 0 |
| scienceworld | react_tr | 1 | 30 | 8 | 0.1770 | 0 | 30 | 0 |
| scienceworld | react_tr | 2 | 30 | 7 | 0.2433 | 0 | 40 | 0 |
| scienceworld | react_cr_tr | 0 | 30 | 7 | 0.2137 | 132 | 25 | 60 |
| scienceworld | react_cr_tr | 1 | 30 | 11 | 0.3450 | 132 | 19 | 60 |
| scienceworld | react_cr_tr | 2 | 30 | 7 | 0.0780 | 132 | 18 | 60 |
| scienceworld | react_hard_neg_cr_tr | 0 | 30 | 7 | 0.0880 | 125 | 33 | 60 |
| scienceworld | react_hard_neg_cr_tr | 1 | 30 | 8 | 0.3310 | 125 | 29 | 60 |
| scienceworld | react_hard_neg_cr_tr | 2 | 30 | 11 | 0.3603 | 125 | 32 | 60 |

## Continuation Decision

The stage passed the hard gates and is safe to continue to the next stage.
