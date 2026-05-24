# Summary (valid_unseen)

| Framework | Success / Total | Accuracy | Avg Steps Per Trial | Avg Steps per Task | Gain from Base ReAct |
|---|---:|---:|---:|---:|---:|
| ReAct | 15 / 34 | 0.4412 | 20.8824 | 20.8824 | +0.0000 |
| ReAct + Context Retrieval (CR) | 21 / 34 | 0.6176 | 25.9118 | 25.9118 | +0.1765 |
| ReAct + Tool Retrieval (TR) | 26 / 34 | 0.7647 | 24.8529 | 24.8529 | +0.3235 |
| react + CR + TR | 31 / 34 | 0.9118 | 24.9706 | 24.9706 | +0.4706 |
| react + hard_neg CR + TR | 17 / 34 | 0.5000 | 33.3529 | 33.3529 | +0.0588 |
