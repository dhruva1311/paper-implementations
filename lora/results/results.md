FashionMNIST test accuracy after 3 epochs, mean +/- std over 3 seeds (no adaptation is deterministic, 1 run).

| Method | Trainable params | % of base model | Test accuracy | Train time / run |
|---|---:|---:|---:|---:|
| no adaptation | 0 | 0.00% | 8.44% | - |
| head-only | 5,130 | 0.77% | 77.03% +/- 0.21 | 19s |
| full | 669,706 | 100.00% | 87.11% +/- 0.14 | 72s |
| LoRA r=1 | 2,842 | 0.42% | 78.58% +/- 0.21 | 38s |
| LoRA r=2 | 5,684 | 0.85% | 82.34% +/- 0.02 | 40s |
| LoRA r=4 | 11,368 | 1.70% | 84.15% +/- 0.42 | 40s |
| LoRA r=8 | 22,736 | 3.39% | 84.83% +/- 0.57 | 39s |
