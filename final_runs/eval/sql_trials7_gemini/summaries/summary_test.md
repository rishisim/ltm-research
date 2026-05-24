# Summary (test)

| Framework | Success / Total | Accuracy | Avg Reward | Avg Steps Per Trial | Avg Steps per Task | Gain from Base ReAct |
|---|---:|---:|---:|---:|---:|---:|
| ReAct | 46 / 50 | 0.9200 | 0.9200 | 9.3000 | 9.3000 | +0.0000 |
| ReAct + Context Retrieval (CR) | 46 / 50 | 0.9200 | 0.9300 | 9.1800 | 9.1800 | +0.0000 |
| ReAct + Tool Retrieval (TR) | 45 / 50 | 0.9000 | 0.9170 | 9.9400 | 9.9400 | -0.0200 |
| react + CR + TR | 47 / 50 | 0.9400 | 0.9400 | 9.1800 | 9.1800 | +0.0200 |
| react + hard_neg CR + TR | 44 / 50 | 0.8800 | 0.9033 | 9.2400 | 9.2400 | -0.0400 |
