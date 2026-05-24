# Phase 3 Stage A Report

Gate status: PASS

This stage is a corruption/trend gate only, not final science.

## Targets

| Environment | Expected completed tasks per framework/seed |
| --- | ---: |
| ALFWorld valid_unseen | 34 |
| SQL test | 50 |
| ScienceWorld test categories | 8 |

## Gate Checks

- Errors: 0
- Warnings: 3

### Warnings

- scienceworld/react_tr/seed_0 made zero help calls
- scienceworld/react_tr/seed_1 made zero help calls
- scienceworld/react_tr/seed_2 made zero help calls

## Run Summary

| Environment | Framework | Seed | Tasks | Successes | Avg Reward | Context Memories | Help Calls | Retrieval Records |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alfworld | react | 0 | 34 | 15 |  | 0 | 0 | 0 |
| alfworld | react | 1 | 34 | 15 |  | 0 | 0 | 0 |
| alfworld | react | 2 | 34 | 15 |  | 0 | 0 | 0 |
| alfworld | react_cr | 0 | 34 | 22 | 0.6471 | 126 | 0 | 68 |
| alfworld | react_cr | 1 | 34 | 21 | 0.6176 | 126 | 0 | 68 |
| alfworld | react_cr | 2 | 34 | 21 | 0.6176 | 126 | 0 | 68 |
| alfworld | react_tr | 0 | 34 | 26 | 0.7647 | 0 | 84 | 0 |
| alfworld | react_tr | 1 | 34 | 26 | 0.7647 | 0 | 84 | 0 |
| alfworld | react_tr | 2 | 34 | 26 | 0.7647 | 0 | 84 | 0 |
| alfworld | react_cr_tr | 0 | 34 | 31 | 0.9118 | 126 | 63 | 68 |
| alfworld | react_cr_tr | 1 | 34 | 31 | 0.9118 | 126 | 63 | 68 |
| alfworld | react_cr_tr | 2 | 34 | 31 | 0.9118 | 126 | 63 | 68 |
| alfworld | react_hard_neg_cr_tr | 0 | 34 | 18 | 0.5294 | 155 | 69 | 68 |
| alfworld | react_hard_neg_cr_tr | 1 | 34 | 18 | 0.5294 | 155 | 68 | 68 |
| alfworld | react_hard_neg_cr_tr | 2 | 34 | 17 | 0.5000 | 155 | 69 | 68 |
| sql | react | 0 | 50 | 46 | 0.9200 | 0 | 0 | 0 |
| sql | react | 1 | 50 | 46 | 0.9200 | 0 | 0 | 0 |
| sql | react | 2 | 50 | 46 | 0.9200 | 0 | 0 | 0 |
| sql | react_cr | 0 | 50 | 46 | 0.9300 | 101 | 0 | 100 |
| sql | react_cr | 1 | 50 | 46 | 0.9300 | 101 | 0 | 100 |
| sql | react_cr | 2 | 50 | 46 | 0.9300 | 101 | 0 | 100 |
| sql | react_tr | 0 | 50 | 45 | 0.9170 | 0 | 9 | 0 |
| sql | react_tr | 1 | 50 | 45 | 0.9170 | 0 | 9 | 0 |
| sql | react_tr | 2 | 50 | 45 | 0.9170 | 0 | 9 | 0 |
| sql | react_cr_tr | 0 | 50 | 47 | 0.9400 | 101 | 5 | 100 |
| sql | react_cr_tr | 1 | 50 | 47 | 0.9400 | 101 | 5 | 100 |
| sql | react_cr_tr | 2 | 50 | 47 | 0.9400 | 101 | 5 | 100 |
| sql | react_hard_neg_cr_tr | 0 | 50 | 44 | 0.9033 | 86 | 4 | 100 |
| sql | react_hard_neg_cr_tr | 1 | 50 | 44 | 0.9033 | 86 | 4 | 100 |
| sql | react_hard_neg_cr_tr | 2 | 50 | 44 | 0.9033 | 86 | 4 | 100 |
| scienceworld | react | 0 | 8 | 1 | 0.3362 | 0 | 0 | 0 |
| scienceworld | react | 1 | 8 | 1 | -0.0212 | 0 | 0 | 0 |
| scienceworld | react | 2 | 8 | 2 | 0.2737 | 0 | 0 | 0 |
| scienceworld | react_cr | 0 | 8 | 1 | -0.0388 | 0 | 0 | 16 |
| scienceworld | react_cr | 1 | 8 | 2 | 0.3613 | 0 | 0 | 16 |
| scienceworld | react_cr | 2 | 8 | 2 | -0.0938 | 0 | 0 | 16 |
| scienceworld | react_tr | 0 | 8 | 2 | 0.2913 | 0 | 0 | 0 |
| scienceworld | react_tr | 1 | 8 | 2 | 0.1913 | 0 | 0 | 0 |
| scienceworld | react_tr | 2 | 8 | 0 | 0.2075 | 0 | 0 | 0 |
| scienceworld | react_cr_tr | 0 | 8 | 1 | 0.0950 | 0 | 0 | 16 |
| scienceworld | react_cr_tr | 1 | 8 | 2 | 0.3200 | 0 | 0 | 16 |
| scienceworld | react_cr_tr | 2 | 8 | 3 | 0.2888 | 0 | 0 | 16 |
| scienceworld | react_hard_neg_cr_tr | 0 | 8 | 2 | 0.1187 | 0 | 0 | 16 |
| scienceworld | react_hard_neg_cr_tr | 1 | 8 | 2 | 0.1925 | 0 | 0 | 16 |
| scienceworld | react_hard_neg_cr_tr | 2 | 8 | 3 | 0.3100 | 0 | 0 | 16 |

## Continuation Decision

The stage passed the hard gates and is safe to continue to the next stage.

