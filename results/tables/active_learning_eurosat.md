### EuroSAT: simulated active learning (linear probe)

Start: the 1% label-budget subset (162 images); 10 rounds of +1% of the train pool (final 1,782 labels). Test Accuracy after the last round and the mean area under the curve over rounds (mean ± std over 3 seeds). Labels to match random = labels at which the strategy's mean curve first reaches random's final mean (interpolated); savings = 1 − that / the labels random itself first needs.

| Model | Strategy | Final Accuracy | Mean over rounds | Labels to match random | Savings |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | random | 0.941 ± 0.005 | 0.924 ± 0.001 | 1,782 | +0% |
| DINOv2 ViT-B/14 | least confidence | 0.959 ± 0.001 | 0.941 ± 0.001 | 623 | +65% |
| DINOv2 ViT-B/14 | entropy | 0.959 ± 0.000 | 0.939 ± 0.001 | 756 | +58% |
| DINOv2 ViT-B/14 | conformal set size | 0.941 ± 0.001 | 0.928 ± 0.001 | not reached | – |
| DINOv2 ViT-B/14 | k-center greedy | 0.930 ± 0.002 | 0.917 ± 0.004 | not reached | – |
| Phikon | random | 0.903 ± 0.003 | 0.874 ± 0.002 | 1,782 | +0% |
| Phikon | least confidence | 0.940 ± 0.002 | 0.900 ± 0.000 | 746 | +58% |
| Phikon | entropy | 0.934 ± 0.002 | 0.895 ± 0.001 | 804 | +55% |
| Phikon | conformal set size | 0.925 ± 0.001 | 0.897 ± 0.001 | 675 | +62% |
| Phikon | k-center greedy | 0.917 ± 0.002 | 0.878 ± 0.001 | 1,389 | +22% |
