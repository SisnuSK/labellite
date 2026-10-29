### PatchCamelyon: split conformal prediction and triage on the test split

Calibrated on 8,192 calibration images (temperature-scaled probabilities), evaluated on 32,768 test images. Auto-labeled = prediction set of exactly one class (the share of images the expert does not review); auto-label error = error rate among auto-labeled images; system error = wrong auto-labels as a share of all images (≤ α by the coverage guarantee if the expert is always right). Percentages; mean ± std over 3 seeds at 1% and 10%.

Coverage check: 94 of 144 (run, method, α) results lie within 3 sd of the coverage expected under exchangeability (coverage minus expected: -4.7 to +1.3 points). Exchangeable control (calibrate on a random half of test, evaluate on the other half): 144 of 144 within 3 sd.

**Target coverage 90%** (α = 0.1; expected test coverage 90.00% ± 0.37%)

| Model | Method | Labels | Score | Coverage | Set size | Auto-labeled | Auto-label error | System error | Min class coverage |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | APS | 89.9 ± 0.5 | 1.29 ± 0.01 | 65.8 ± 0.8 | 11.6 ± 0.5 | 7.7 ± 0.4 | 86.1 ± 0.9 |
| ResNet-50 | linear probe | 1% | LAC | 88.9 ± 0.3 | 1.16 ± 0.01 | 83.7 ± 1.4 | 13.3 ± 0.2 | 11.1 ± 0.3 | 83.0 ± 0.4 |
| ResNet-50 | linear probe | 10% | APS | 89.7 ± 0.5 | 1.27 ± 0.01 | 67.3 ± 0.4 | 11.4 ± 0.6 | 7.7 ± 0.4 | 86.0 ± 1.1 |
| ResNet-50 | linear probe | 10% | LAC | 88.3 ± 0.7 | 1.14 ± 0.01 | 86.0 ± 0.5 | 13.6 ± 0.7 | 11.7 ± 0.7 | 82.1 ± 1.7 |
| ResNet-50 | linear probe | 100% | APS | 88.9 | 1.26 | 68.5 | 12.0 | 8.2 | 84.5 |
| ResNet-50 | linear probe | 100% | LAC | 87.8 | 1.13 | 86.6 | 14.1 | 12.2 | 80.6 |
| ResNet-50 | fine-tuned | 1% | APS | 88.5 ± 0.5 | 1.19 ± 0.02 | 75.3 ± 1.3 | 11.2 ± 0.9 | 8.5 ± 0.5 | 83.4 ± 1.2 |
| ResNet-50 | fine-tuned | 1% | LAC | 85.7 ± 0.4 | 1.03 ± 0.01 | 96.7 ± 1.5 | 14.8 ± 0.6 | 14.3 ± 0.4 | 76.9 ± 1.3 |
| ResNet-50 | fine-tuned | 10% | APS | 90.1 ± 0.3 | 1.17 ± 0.03 | 74.9 ± 2.3 | 8.1 ± 1.0 | 6.1 ± 0.6 | 86.7 ± 0.7 |
| ResNet-50 | fine-tuned | 10% | LAC | 88.1 ± 1.2 | 1.02 ± 0.01 | 98.3 ± 1.2 | 12.1 ± 1.3 | 11.9 ± 1.2 | 80.8 ± 2.7 |
| ResNet-50 | fine-tuned | 100% | APS | 90.2 | 1.15 | 76.2 | 7.4 | 5.6 | 86.4 |
| ResNet-50 | fine-tuned | 100% | LAC | 87.9 | 0.99 | 98.5 | 10.7 | 10.6 | 78.5 |
| DINOv2 ViT-B/14 | linear probe | 1% | APS | 90.5 ± 0.2 | 1.25 ± 0.01 | 68.3 ± 0.8 | 9.3 ± 0.3 | 6.4 ± 0.2 | 89.6 ± 0.8 |
| DINOv2 ViT-B/14 | linear probe | 1% | LAC | 90.3 ± 0.3 | 1.12 ± 0.01 | 87.7 ± 0.7 | 11.1 ± 0.3 | 9.7 ± 0.3 | 88.7 ± 1.7 |
| DINOv2 ViT-B/14 | linear probe | 10% | APS | 90.5 ± 0.2 | 1.23 ± 0.01 | 70.0 ± 0.7 | 8.6 ± 0.1 | 6.0 ± 0.1 | 89.4 ± 0.1 |
| DINOv2 ViT-B/14 | linear probe | 10% | LAC | 89.7 ± 0.3 | 1.08 ± 0.01 | 91.6 ± 0.9 | 11.2 ± 0.2 | 10.3 ± 0.3 | 87.5 ± 0.6 |
| DINOv2 ViT-B/14 | linear probe | 100% | APS | 90.7 | 1.22 | 70.8 | 8.1 | 5.7 | 89.7 |
| DINOv2 ViT-B/14 | linear probe | 100% | LAC | 89.6 | 1.07 | 93.3 | 11.2 | 10.4 | 87.4 |
| CLIP ViT-B/16 | linear probe | 1% | APS | 89.6 ± 0.4 | 1.30 ± 0.02 | 64.7 ± 1.2 | 12.4 ± 0.2 | 8.0 ± 0.3 | 86.7 ± 0.7 |
| CLIP ViT-B/16 | linear probe | 1% | LAC | 88.8 ± 0.4 | 1.18 ± 0.02 | 82.4 ± 1.7 | 13.6 ± 0.3 | 11.2 ± 0.4 | 84.0 ± 0.9 |
| CLIP ViT-B/16 | linear probe | 10% | APS | 89.3 ± 0.3 | 1.28 ± 0.01 | 66.5 ± 0.8 | 12.0 ± 0.1 | 8.0 ± 0.1 | 86.4 ± 0.1 |
| CLIP ViT-B/16 | linear probe | 10% | LAC | 88.4 ± 0.1 | 1.15 ± 0.00 | 85.0 ± 0.2 | 13.7 ± 0.1 | 11.6 ± 0.1 | 83.5 ± 0.2 |
| CLIP ViT-B/16 | linear probe | 100% | APS | 89.7 | 1.28 | 66.2 | 11.5 | 7.6 | 86.6 |
| CLIP ViT-B/16 | linear probe | 100% | LAC | 88.5 | 1.14 | 85.6 | 13.5 | 11.5 | 83.1 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | APS | 90.7 | 1.81 | 18.7 | 49.9 | 9.3 | 86.0 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | LAC | 87.0 | 1.71 | 29.3 | 44.6 | 13.0 | 74.8 |
| Phikon | linear probe | 1% | APS | 91.1 ± 0.2 | 1.08 ± 0.01 | 81.5 ± 0.6 | 4.5 ± 0.2 | 3.7 ± 0.1 | 88.5 ± 0.1 |
| Phikon | linear probe | 1% | LAC | 90.9 ± 0.5 | 0.97 ± 0.00 | 97.0 ± 0.5 | 6.3 ± 0.3 | 6.1 ± 0.3 | 85.0 ± 1.2 |
| Phikon | linear probe | 10% | APS | 90.7 ± 0.6 | 1.08 ± 0.00 | 81.4 ± 0.4 | 5.0 ± 0.7 | 4.0 ± 0.5 | 87.8 ± 1.1 |
| Phikon | linear probe | 10% | LAC | 90.4 ± 0.8 | 0.97 ± 0.00 | 96.8 ± 0.4 | 6.6 ± 0.5 | 6.4 ± 0.4 | 83.6 ± 1.6 |
| Phikon | linear probe | 100% | APS | 91.0 | 1.08 | 81.7 | 4.6 | 3.8 | 88.3 |
| Phikon | linear probe | 100% | LAC | 90.8 | 0.97 | 97.0 | 6.4 | 6.2 | 84.4 |

**Target coverage 95%** (α = 0.05; expected test coverage 95.01% ± 0.27%)

| Model | Method | Labels | Score | Coverage | Set size | Auto-labeled | Auto-label error | System error | Min class coverage |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | APS | 95.0 ± 0.3 | 1.49 ± 0.01 | 49.2 ± 1.0 | 8.9 ± 0.5 | 4.4 ± 0.3 | 92.6 ± 0.6 |
| ResNet-50 | linear probe | 1% | LAC | 94.6 ± 0.9 | 1.35 ± 0.02 | 64.5 ± 1.9 | 8.3 ± 1.2 | 5.4 ± 0.9 | 91.4 ± 1.7 |
| ResNet-50 | linear probe | 10% | APS | 94.7 ± 0.2 | 1.46 ± 0.00 | 52.1 ± 0.3 | 8.6 ± 0.4 | 4.5 ± 0.2 | 92.3 ± 0.5 |
| ResNet-50 | linear probe | 10% | LAC | 94.3 ± 0.7 | 1.33 ± 0.02 | 67.2 ± 1.7 | 8.4 ± 0.9 | 5.7 ± 0.7 | 90.9 ± 1.4 |
| ResNet-50 | linear probe | 100% | APS | 94.4 | 1.45 | 53.1 | 9.0 | 4.8 | 91.6 |
| ResNet-50 | linear probe | 100% | LAC | 93.5 | 1.31 | 69.4 | 9.4 | 6.5 | 89.1 |
| ResNet-50 | fine-tuned | 1% | APS | 94.0 ± 0.5 | 1.39 ± 0.03 | 59.6 ± 2.9 | 9.0 ± 0.9 | 5.3 ± 0.5 | 90.4 ± 1.3 |
| ResNet-50 | fine-tuned | 1% | LAC | 92.6 ± 0.6 | 1.22 ± 0.03 | 77.7 ± 2.7 | 9.6 ± 0.8 | 7.4 ± 0.6 | 87.3 ± 1.6 |
| ResNet-50 | fine-tuned | 10% | APS | 95.2 ± 0.3 | 1.36 ± 0.04 | 61.9 ± 3.5 | 5.8 ± 0.7 | 3.6 ± 0.3 | 93.0 ± 0.6 |
| ResNet-50 | fine-tuned | 10% | LAC | 94.4 ± 0.5 | 1.19 ± 0.04 | 80.9 ± 3.8 | 6.9 ± 0.8 | 5.6 ± 0.5 | 90.5 ± 1.2 |
| ResNet-50 | fine-tuned | 100% | APS | 95.4 | 1.34 | 63.6 | 5.2 | 3.3 | 92.9 |
| ResNet-50 | fine-tuned | 100% | LAC | 95.2 | 1.19 | 81.5 | 5.9 | 4.8 | 90.9 |
| DINOv2 ViT-B/14 | linear probe | 1% | APS | 95.0 ± 0.1 | 1.43 ± 0.02 | 54.8 ± 1.7 | 7.1 ± 0.2 | 3.9 ± 0.0 | 94.3 ± 0.4 |
| DINOv2 ViT-B/14 | linear probe | 1% | LAC | 95.4 ± 0.2 | 1.30 ± 0.02 | 69.6 ± 1.9 | 6.6 ± 0.2 | 4.6 ± 0.2 | 94.2 ± 0.9 |
| DINOv2 ViT-B/14 | linear probe | 10% | APS | 95.0 ± 0.2 | 1.40 ± 0.01 | 57.5 ± 1.1 | 6.4 ± 0.2 | 3.7 ± 0.2 | 94.1 ± 0.1 |
| DINOv2 ViT-B/14 | linear probe | 10% | LAC | 95.3 ± 0.2 | 1.27 ± 0.01 | 73.2 ± 1.1 | 6.4 ± 0.2 | 4.7 ± 0.2 | 94.0 ± 0.2 |
| DINOv2 ViT-B/14 | linear probe | 100% | APS | 95.2 | 1.39 | 58.5 | 5.9 | 3.4 | 94.2 |
| DINOv2 ViT-B/14 | linear probe | 100% | LAC | 95.4 | 1.25 | 74.6 | 6.1 | 4.6 | 94.2 |
| CLIP ViT-B/16 | linear probe | 1% | APS | 94.6 ± 0.2 | 1.50 ± 0.01 | 48.9 ± 0.8 | 9.4 ± 0.3 | 4.6 ± 0.2 | 92.8 ± 0.5 |
| CLIP ViT-B/16 | linear probe | 1% | LAC | 94.4 ± 0.4 | 1.37 ± 0.01 | 63.1 ± 1.3 | 8.9 ± 0.4 | 5.6 ± 0.4 | 91.3 ± 0.7 |
| CLIP ViT-B/16 | linear probe | 10% | APS | 94.7 ± 0.1 | 1.48 ± 0.01 | 50.5 ± 1.0 | 8.8 ± 0.0 | 4.4 ± 0.1 | 92.8 ± 0.0 |
| CLIP ViT-B/16 | linear probe | 10% | LAC | 94.1 ± 0.1 | 1.34 ± 0.00 | 66.1 ± 0.3 | 8.9 ± 0.1 | 5.9 ± 0.1 | 91.0 ± 0.2 |
| CLIP ViT-B/16 | linear probe | 100% | APS | 94.9 | 1.48 | 50.2 | 8.4 | 4.2 | 93.0 |
| CLIP ViT-B/16 | linear probe | 100% | LAC | 94.3 | 1.34 | 66.1 | 8.7 | 5.7 | 90.9 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | APS | 95.4 | 1.91 | 9.3 | 49.7 | 4.6 | 93.1 |
| CLIP ViT-B/16 | zero-shot | 0 (zero-shot) | LAC | 91.2 | 1.80 | 20.4 | 43.0 | 8.8 | 82.6 |
| Phikon | linear probe | 1% | APS | 96.0 ± 0.3 | 1.23 ± 0.01 | 73.3 ± 1.1 | 3.0 ± 0.2 | 2.2 ± 0.2 | 94.1 ± 0.4 |
| Phikon | linear probe | 1% | LAC | 96.2 ± 0.1 | 1.10 ± 0.00 | 90.4 ± 0.4 | 4.2 ± 0.2 | 3.8 ± 0.1 | 93.2 ± 0.3 |
| Phikon | linear probe | 10% | APS | 95.7 ± 0.4 | 1.23 ± 0.01 | 73.3 ± 0.9 | 3.4 ± 0.6 | 2.5 ± 0.4 | 93.7 ± 0.8 |
| Phikon | linear probe | 10% | LAC | 95.8 ± 0.7 | 1.10 ± 0.00 | 90.1 ± 0.4 | 4.7 ± 0.7 | 4.2 ± 0.7 | 92.4 ± 1.3 |
| Phikon | linear probe | 100% | APS | 96.1 | 1.23 | 73.5 | 3.0 | 2.2 | 94.2 |
| Phikon | linear probe | 100% | LAC | 96.2 | 1.10 | 90.2 | 4.2 | 3.8 | 93.1 |
