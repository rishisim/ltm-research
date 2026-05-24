# Phase 3 Stage B Report

Gate status: FAIL

This stage is a corruption/trend gate only, not final science.

## Targets

| Environment | Expected completed tasks per framework/seed |
| --- | ---: |
| ALFWorld valid_unseen | 67 |
| SQL test | 100 |
| ScienceWorld test categories | 15 |

## Gate Checks

- Errors: 3
- Warnings: 3

### Errors

- scienceworld/react_tr/seed_2 has 4 suspected LLM/provider failure records: ['scienceworld_grow-plant_var_93', 'scienceworld_identify-life-stages-1_var_9', 'scienceworld_identify-life-stages-2_var_6', 'scienceworld_inclined-plane-determine-angle_var_126']
- scienceworld/react_cr_tr/seed_2 has 7 suspected LLM/provider failure records: ['scienceworld_find-plant_var_225', 'scienceworld_freeze_var_21', 'scienceworld_grow-fruit_var_93', 'scienceworld_grow-plant_var_93', 'scienceworld_identify-life-stages-1_var_9', 'scienceworld_identify-life-stages-2_var_6', 'scienceworld_inclined-plane-determine-angle_var_126']
- scienceworld/react_hard_neg_cr_tr/seed_2 has 7 suspected LLM/provider failure records: ['scienceworld_find-plant_var_225', 'scienceworld_freeze_var_21', 'scienceworld_grow-fruit_var_93', 'scienceworld_grow-plant_var_93', 'scienceworld_identify-life-stages-1_var_9', 'scienceworld_identify-life-stages-2_var_6', 'scienceworld_inclined-plane-determine-angle_var_126']

### Warnings

- scienceworld/react_tr/seed_0 made zero help calls
- scienceworld/react_tr/seed_1 made zero help calls
- scienceworld/react_tr/seed_2 made zero help calls

## Run Summary

| Environment | Framework | Seed | Tasks | Successes | Avg Reward | Context Memories | Help Calls | Retrieval Records |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alfworld | react | 0 | 67 | 47 |  | 0 | 0 | 0 |
| alfworld | react | 1 | 67 | 47 |  | 0 | 0 | 0 |
| alfworld | react | 2 | 67 | 47 |  | 0 | 0 | 0 |
| alfworld | react_cr | 0 | 67 | 45 | 0.6716 | 267 | 0 | 134 |
| alfworld | react_cr | 1 | 67 | 43 | 0.6418 | 267 | 0 | 134 |
| alfworld | react_cr | 2 | 67 | 44 | 0.6567 | 267 | 0 | 134 |
| alfworld | react_tr | 0 | 67 | 51 | 0.7612 | 0 | 147 | 0 |
| alfworld | react_tr | 1 | 67 | 51 | 0.7612 | 0 | 147 | 0 |
| alfworld | react_tr | 2 | 67 | 51 | 0.7612 | 0 | 148 | 0 |
| alfworld | react_cr_tr | 0 | 67 | 58 | 0.8657 | 267 | 105 | 134 |
| alfworld | react_cr_tr | 1 | 67 | 58 | 0.8657 | 267 | 105 | 134 |
| alfworld | react_cr_tr | 2 | 67 | 58 | 0.8657 | 267 | 105 | 134 |
| alfworld | react_hard_neg_cr_tr | 0 | 67 | 33 | 0.4925 | 316 | 173 | 134 |
| alfworld | react_hard_neg_cr_tr | 1 | 67 | 33 | 0.4925 | 316 | 172 | 134 |
| alfworld | react_hard_neg_cr_tr | 2 | 67 | 32 | 0.4776 | 316 | 172 | 134 |
| sql | react | 0 | 100 | 92 | 0.9227 | 0 | 0 | 0 |
| sql | react | 1 | 100 | 92 | 0.9227 | 0 | 0 | 0 |
| sql | react | 2 | 100 | 92 | 0.9227 | 0 | 0 | 0 |
| sql | react_cr | 0 | 100 | 90 | 0.9057 | 194 | 0 | 200 |
| sql | react_cr | 1 | 100 | 90 | 0.9057 | 194 | 0 | 200 |
| sql | react_cr | 2 | 100 | 90 | 0.9057 | 194 | 0 | 200 |
| sql | react_tr | 0 | 100 | 92 | 0.9312 | 0 | 14 | 0 |
| sql | react_tr | 1 | 100 | 92 | 0.9312 | 0 | 14 | 0 |
| sql | react_tr | 2 | 100 | 91 | 0.9212 | 0 | 14 | 0 |
| sql | react_cr_tr | 0 | 100 | 93 | 0.9300 | 194 | 9 | 200 |
| sql | react_cr_tr | 1 | 100 | 93 | 0.9300 | 194 | 9 | 200 |
| sql | react_cr_tr | 2 | 100 | 93 | 0.9300 | 194 | 9 | 200 |
| sql | react_hard_neg_cr_tr | 0 | 100 | 91 | 0.9225 | 173 | 12 | 200 |
| sql | react_hard_neg_cr_tr | 1 | 100 | 91 | 0.9225 | 173 | 12 | 200 |
| sql | react_hard_neg_cr_tr | 2 | 100 | 91 | 0.9225 | 173 | 12 | 200 |
| scienceworld | react | 0 | 15 | 3 | 0.3173 | 0 | 0 | 0 |
| scienceworld | react | 1 | 15 | 3 | 0.0060 | 0 | 0 | 0 |
| scienceworld | react | 2 | 15 | 3 | 0.1047 | 0 | 0 | 0 |
| scienceworld | react_cr | 0 | 15 | 2 | -0.0387 | 0 | 0 | 30 |
| scienceworld | react_cr | 1 | 15 | 3 | 0.2000 | 0 | 0 | 30 |
| scienceworld | react_cr | 2 | 15 | 4 | 0.0653 | 0 | 0 | 30 |
| scienceworld | react_tr | 0 | 15 | 4 | 0.2740 | 0 | 0 | 0 |
| scienceworld | react_tr | 1 | 15 | 4 | 0.1193 | 0 | 0 | 0 |
| scienceworld | react_tr | 2 | 15 | 1 | 0.2253 | 0 | 0 | 0 |
| scienceworld | react_cr_tr | 0 | 15 | 3 | 0.1933 | 0 | 0 | 30 |
| scienceworld | react_cr_tr | 1 | 15 | 4 | 0.3080 | 0 | 0 | 30 |
| scienceworld | react_cr_tr | 2 | 15 | 3 | 0.1707 | 0 | 0 | 30 |
| scienceworld | react_hard_neg_cr_tr | 0 | 15 | 3 | -0.0480 | 0 | 0 | 30 |
| scienceworld | react_hard_neg_cr_tr | 1 | 15 | 3 | 0.2940 | 0 | 0 | 30 |
| scienceworld | react_hard_neg_cr_tr | 2 | 15 | 3 | 0.1820 | 0 | 0 | 30 |

## Continuation Decision

Continue to the next stage only after this report passes all hard gates.

