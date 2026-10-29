### EuroSAT: split conformal prediction and triage on the test split

Calibrated on 1,350 calibration images (temperature-scaled probabilities), evaluated on 5,400 test images. Auto-labeled = prediction set of exactly one class (the share of images the expert does not review); auto-label error = error rate among auto-labeled images; system error = wrong auto-labels as a share of all images (≤ α by the coverage guarantee if the expert is always right). Percentages; mean ± std over 3 seeds at 1% and 10%.

Coverage check: 144 of 144 (run, method, α) results lie within 3 sd of the coverage expected under exchangeability (coverage minus expected: -1.2 to +1.5 points). Exchangeable control (calibrate on a random half of test, evaluate on the other half): 144 of 144 within 3 sd.

**Target coverage 90%** (α = 0.1; expected test coverage 90.01% ± 0.91%)

| Model | Method | Labels | Score | Coverage | Set size | Auto-labeled | Auto-label error | System error | Min class coverage |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | APS | 91.1 ± 0.2 | 1.42 ± 0.03 | 65.2 ± 1.1 | 4.4 ± 0.3 | 2.9 ± 0.2 | 81.5 ± 3.3 |
| ResNet-50 | linear probe | 1% | LAC | 90.3 ± 0.6 | 1.09 ± 0.02 | 91.1 ± 1.4 | 9.5 ± 0.4 | 8.6 ± 0.5 | 74.6 ± 5.0 |
| ResNet-50 | linear probe | 10% | APS | 91.0 ± 0.2 | 1.09 ± 0.01 | 81.2 ± 0.9 | 2.6 ± 0.2 | 2.1 ± 0.1 | 87.7 ± 0.8 |
| ResNet-50 | linear probe | 10% | LAC | 91.0 ± 0.4 | 0.95 ± 0.01 | 94.6 ± 0.5 | 3.8 ± 0.2 | 3.6 ± 0.2 | 82.6 ± 1.1 |
| ResNet-50 | linear probe | 100% | APS | 90.3 | 1.00 | 85.1 | 1.8 | 1.6 | 88.6 |
| ResNet-50 | linear probe | 100% | LAC | 91.0 | 0.92 | 92.0 | 1.1 | 1.0 | 84.2 |
| ResNet-50 | fine-tuned | 1% | APS | 90.8 ± 0.7 | 1.59 ± 0.08 | 62.5 ± 1.2 | 4.3 ± 0.3 | 2.7 ± 0.1 | 84.0 ± 2.0 |
| ResNet-50 | fine-tuned | 1% | LAC | 90.5 ± 0.4 | 1.12 ± 0.03 | 88.1 ± 1.9 | 8.9 ± 0.6 | 7.8 ± 0.7 | 76.2 ± 4.5 |
| ResNet-50 | fine-tuned | 10% | APS | 90.5 ± 0.4 | 0.99 ± 0.01 | 87.2 ± 0.3 | 1.6 ± 0.2 | 1.4 ± 0.2 | 87.7 ± 0.9 |
| ResNet-50 | fine-tuned | 10% | LAC | 89.4 ± 0.0 | 0.90 ± 0.00 | 89.9 ± 0.2 | 0.6 ± 0.2 | 0.5 ± 0.2 | 79.3 ± 2.9 |
| ResNet-50 | fine-tuned | 100% | APS | 90.0 | 0.94 | 88.6 | 0.9 | 0.8 | 87.6 |
| ResNet-50 | fine-tuned | 100% | LAC | 90.3 | 0.90 | 90.5 | 0.2 | 0.2 | 77.4 |
| DINOv2 ViT-B/14 | linear probe | 1% | APS | 90.7 ± 0.2 | 1.37 ± 0.07 | 68.0 ± 3.0 | 4.3 ± 0.3 | 3.0 ± 0.3 | 82.6 ± 3.1 |
| DINOv2 ViT-B/14 | linear probe | 1% | LAC | 90.3 ± 0.5 | 1.06 ± 0.05 | 93.0 ± 3.6 | 9.3 ± 0.5 | 8.6 ± 0.6 | 75.9 ± 5.0 |
| DINOv2 ViT-B/14 | linear probe | 10% | APS | 90.5 ± 0.5 | 1.08 ± 0.02 | 81.1 ± 0.6 | 2.7 ± 0.1 | 2.2 ± 0.1 | 87.5 ± 0.5 |
| DINOv2 ViT-B/14 | linear probe | 10% | LAC | 90.5 ± 0.3 | 0.94 ± 0.00 | 93.9 ± 0.2 | 3.6 ± 0.3 | 3.4 ± 0.3 | 82.5 ± 2.4 |
| DINOv2 ViT-B/14 | linear probe | 100% | APS | 90.1 | 1.00 | 85.3 | 1.8 | 1.5 | 87.6 |
| DINOv2 ViT-B/14 | linear probe | 100% | LAC | 89.6 | 0.91 | 90.7 | 1.2 | 1.1 | 81.8 |
| CLIP ViT-B/16 | linear probe | 1% | APS | 90.8 ± 0.7 | 1.44 ± 0.06 | 65.7 ± 2.2 | 4.5 ± 0.6 | 3.0 ± 0.5 | 83.5 ± 3.6 |
| CLIP ViT-B/16 | linear probe | 1% | LAC | 89.9 ± 0.7 | 1.11 ± 0.04 | 88.6 ± 3.5 | 9.6 ± 1.1 | 8.5 ± 1.3 | 76.8 ± 4.0 |
| CLIP ViT-B/16 | linear probe | 10% | APS | 91.1 ± 0.3 | 1.13 ± 0.01 | 79.8 ± 0.2 | 2.7 ± 0.3 | 2.1 ± 0.2 | 87.0 ± 1.6 |
| CLIP ViT-B/16 | linear probe | 10% | LAC | 89.7 ± 0.1 | 0.93 ± 0.01 | 93.5 ± 0.6 | 4.0 ± 0.5 | 3.7 ± 0.5 | 81.1 ± 0.1 |
| CLIP ViT-B/16 | linear probe | 100% | APS | 90.9 | 1.03 | 84.3 | 1.6 | 1.3 | 88.8 |
| CLIP ViT-B/16 | linear probe | 100% | LAC | 89.4 | 0.90 | 90.3 | 1.0 | 0.9 | 81.0 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | APS | 91.2 | 4.38 | 12.6 | 6.2 | 0.8 | 54.7 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | LAC | 91.0 | 4.11 | 10.2 | 6.2 | 0.6 | 49.8 |
| Phikon | linear probe | 1% | APS | 90.1 ± 0.4 | 1.77 ± 0.01 | 51.0 ± 0.3 | 5.7 ± 0.5 | 2.9 ± 0.3 | 81.5 ± 1.7 |
| Phikon | linear probe | 1% | LAC | 89.8 ± 0.7 | 1.42 ± 0.03 | 65.7 ± 1.4 | 8.3 ± 0.9 | 5.5 ± 0.7 | 75.2 ± 5.8 |
| Phikon | linear probe | 10% | APS | 90.5 ± 0.8 | 1.22 ± 0.02 | 73.9 ± 0.8 | 3.5 ± 0.3 | 2.6 ± 0.2 | 87.7 ± 0.9 |
| Phikon | linear probe | 10% | LAC | 89.3 ± 0.7 | 0.97 ± 0.01 | 97.0 ± 0.5 | 8.2 ± 0.3 | 7.9 ± 0.3 | 73.9 ± 3.3 |
| Phikon | linear probe | 100% | APS | 90.6 | 1.05 | 83.4 | 2.3 | 1.9 | 88.6 |
| Phikon | linear probe | 100% | LAC | 90.4 | 0.93 | 92.6 | 2.3 | 2.1 | 79.0 |

**Target coverage 95%** (α = 0.05; expected test coverage 95.04% ± 0.66%)

| Model | Method | Labels | Score | Coverage | Set size | Auto-labeled | Auto-label error | System error | Min class coverage |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | APS | 95.3 ± 0.2 | 1.79 ± 0.10 | 55.3 ± 3.5 | 2.8 ± 0.4 | 1.6 ± 0.1 | 86.5 ± 2.7 |
| ResNet-50 | linear probe | 1% | LAC | 95.6 ± 0.3 | 1.41 ± 0.05 | 69.2 ± 2.9 | 3.9 ± 0.3 | 2.7 ± 0.3 | 83.4 ± 4.7 |
| ResNet-50 | linear probe | 10% | APS | 95.8 ± 0.3 | 1.25 ± 0.04 | 76.9 ± 1.7 | 1.3 ± 0.1 | 1.0 ± 0.1 | 93.3 ± 1.0 |
| ResNet-50 | linear probe | 10% | LAC | 95.6 ± 0.4 | 1.05 ± 0.01 | 95.4 ± 1.1 | 4.3 ± 0.4 | 4.1 ± 0.4 | 89.7 ± 0.6 |
| ResNet-50 | linear probe | 100% | APS | 95.0 | 1.10 | 84.8 | 1.1 | 1.0 | 93.3 |
| ResNet-50 | linear probe | 100% | LAC | 95.8 | 0.98 | 98.5 | 2.7 | 2.7 | 90.2 |
| ResNet-50 | fine-tuned | 1% | APS | 94.9 ± 0.5 | 2.06 ± 0.18 | 55.2 ± 2.4 | 2.8 ± 0.3 | 1.6 ± 0.1 | 90.3 ± 1.4 |
| ResNet-50 | fine-tuned | 1% | LAC | 95.2 ± 0.5 | 1.44 ± 0.08 | 69.8 ± 2.7 | 4.1 ± 0.3 | 2.9 ± 0.3 | 85.6 ± 2.7 |
| ResNet-50 | fine-tuned | 10% | APS | 95.0 ± 0.6 | 1.08 ± 0.02 | 88.1 ± 0.3 | 0.9 ± 0.3 | 0.8 ± 0.3 | 93.1 ± 1.6 |
| ResNet-50 | fine-tuned | 10% | LAC | 94.6 ± 0.7 | 0.96 ± 0.01 | 96.1 ± 1.1 | 1.6 ± 0.4 | 1.5 ± 0.4 | 87.3 ± 0.8 |
| ResNet-50 | fine-tuned | 100% | APS | 94.8 | 1.00 | 91.2 | 0.5 | 0.5 | 93.8 |
| ResNet-50 | fine-tuned | 100% | LAC | 95.1 | 0.96 | 95.6 | 0.5 | 0.5 | 86.0 |
| DINOv2 ViT-B/14 | linear probe | 1% | APS | 95.4 ± 0.2 | 1.75 ± 0.14 | 57.4 ± 4.5 | 2.6 ± 0.3 | 1.5 ± 0.2 | 88.5 ± 3.2 |
| DINOv2 ViT-B/14 | linear probe | 1% | LAC | 95.6 ± 0.4 | 1.39 ± 0.15 | 71.5 ± 7.5 | 3.7 ± 0.6 | 2.7 ± 0.7 | 85.5 ± 4.4 |
| DINOv2 ViT-B/14 | linear probe | 10% | APS | 94.9 ± 0.4 | 1.24 ± 0.03 | 78.1 ± 1.4 | 1.8 ± 0.1 | 1.4 ± 0.1 | 92.5 ± 0.5 |
| DINOv2 ViT-B/14 | linear probe | 10% | LAC | 95.7 ± 0.3 | 1.04 ± 0.01 | 95.5 ± 1.2 | 4.2 ± 0.3 | 4.0 ± 0.3 | 91.0 ± 0.8 |
| DINOv2 ViT-B/14 | linear probe | 100% | APS | 94.8 | 1.10 | 85.7 | 1.1 | 1.0 | 93.2 |
| DINOv2 ViT-B/14 | linear probe | 100% | LAC | 94.5 | 0.97 | 96.6 | 2.2 | 2.1 | 89.4 |
| CLIP ViT-B/16 | linear probe | 1% | APS | 95.6 ± 0.6 | 1.87 ± 0.14 | 53.1 ± 3.8 | 2.4 ± 0.7 | 1.3 ± 0.5 | 89.5 ± 2.1 |
| CLIP ViT-B/16 | linear probe | 1% | LAC | 95.1 ± 0.2 | 1.47 ± 0.08 | 66.7 ± 4.0 | 4.1 ± 0.5 | 2.7 ± 0.5 | 86.1 ± 3.2 |
| CLIP ViT-B/16 | linear probe | 10% | APS | 95.9 ± 0.3 | 1.32 ± 0.02 | 75.0 ± 0.7 | 1.4 ± 0.2 | 1.1 ± 0.2 | 93.3 ± 1.9 |
| CLIP ViT-B/16 | linear probe | 10% | LAC | 94.7 ± 0.4 | 1.04 ± 0.02 | 95.5 ± 1.7 | 5.1 ± 0.4 | 4.9 ± 0.5 | 88.7 ± 1.0 |
| CLIP ViT-B/16 | linear probe | 100% | APS | 95.5 | 1.15 | 83.3 | 1.0 | 0.8 | 93.8 |
| CLIP ViT-B/16 | linear probe | 100% | LAC | 95.5 | 0.99 | 99.1 | 3.7 | 3.6 | 91.6 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | APS | 94.9 | 5.24 | 7.5 | 5.7 | 0.4 | 73.7 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | LAC | 95.8 | 5.18 | 6.4 | 4.1 | 0.3 | 76.7 |
| Phikon | linear probe | 1% | APS | 94.5 ± 0.2 | 2.24 ± 0.06 | 40.4 ± 0.4 | 3.2 ± 0.3 | 1.3 ± 0.1 | 88.6 ± 2.4 |
| Phikon | linear probe | 1% | LAC | 95.0 ± 0.2 | 1.95 ± 0.01 | 46.5 ± 0.5 | 3.7 ± 0.5 | 1.7 ± 0.2 | 87.9 ± 3.4 |
| Phikon | linear probe | 10% | APS | 94.6 ± 0.2 | 1.41 ± 0.02 | 67.9 ± 0.8 | 2.2 ± 0.1 | 1.5 ± 0.1 | 92.0 ± 0.9 |
| Phikon | linear probe | 10% | LAC | 95.5 ± 0.6 | 1.18 ± 0.03 | 84.1 ± 2.3 | 3.9 ± 0.6 | 3.3 ± 0.6 | 88.7 ± 0.9 |
| Phikon | linear probe | 100% | APS | 95.4 | 1.17 | 81.3 | 1.2 | 1.0 | 93.5 |
| Phikon | linear probe | 100% | LAC | 95.4 | 1.00 | 98.8 | 4.2 | 4.1 | 89.0 |
