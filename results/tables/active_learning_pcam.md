### PatchCamelyon: simulated active learning (linear probe)

Start: the 1% label-budget subset (2,622 images); 10 rounds of +1% of the train pool (final 28,832 labels). Test AUROC after the last round and the mean area under the curve over rounds (mean ± std over 3 seeds). Labels to match random = labels at which the strategy's mean curve first reaches random's final mean (interpolated); savings = 1 − that / the labels random itself first needs.

| Model | Strategy | Final AUROC | Mean over rounds | Labels to match random | Savings |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | random | 0.937 ± 0.003 | 0.933 ± 0.001 | 22,915 | +0% |
| DINOv2 ViT-B/14 | least confidence | 0.935 ± 0.001 | 0.933 ± 0.000 | not reached | – |
| DINOv2 ViT-B/14 | entropy | 0.935 ± 0.000 | 0.933 ± 0.000 | not reached | – |
| DINOv2 ViT-B/14 | conformal set size | 0.935 ± 0.001 | 0.934 ± 0.001 | 26,128 | -14% |
| DINOv2 ViT-B/14 | k-center greedy | 0.932 ± 0.004 | 0.927 ± 0.003 | not reached | – |
| Phikon | random | 0.982 ± 0.000 | 0.980 ± 0.001 | 10,386 | +0% |
| Phikon | least confidence | 0.975 ± 0.003 | 0.974 ± 0.002 | not reached | – |
| Phikon | entropy | 0.977 ± 0.003 | 0.974 ± 0.002 | not reached | – |
| Phikon | conformal set size | 0.982 ± 0.000 | 0.980 ± 0.001 | 22,380 | -115% |
| Phikon | k-center greedy | 0.965 ± 0.007 | 0.968 ± 0.004 | not reached | – |
