# WebShop Summary (dev)

| Framework | Success / Total | Accuracy | Avg Reward | Avg Steps/Trial | Avg Steps/Task | Gain from ReAct |
|---|---:|---:|---:|---:|---:|---:|
| ReAct | 112 / 200 | 0.5600 | 0.3681 | 24.4350 | 24.4350 | +0.0000 |
| ReAct + Context Retrieval (CR) | 54 / 200 | 0.2700 | 0.1816 | 29.3850 | 29.3850 | -0.2900 |
| ReAct + Tool Retrieval (TR) | 69 / 200 | 0.3450 | 0.2501 | 23.5050 | 23.5050 | -0.2150 |
| react + CR + TR | 51 / 200 | 0.2550 | 0.1701 | 30.5150 | 30.5150 | -0.3050 |
| react + hard_neg CR + TR | 0 / 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.5600 |
| React + Reflexion (final trials, max 7) | 134 / 200 | 0.6700 | 0.1424 | 24.6382 | 85.1250 | +0.1100 |
