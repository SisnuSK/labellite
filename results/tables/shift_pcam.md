### PatchCamelyon: synthetic stain shift of the test split

At severity s, hematoxylin is scaled by (1 − 0.15·s) and eosin by (1 + 0.25·s) in Ruifrok–Johnston stain space; severity 0 is the clean test split. Models, temperatures and conformal thresholds all come from clean images. Mean ± std over 3 seeds at 1% and 10% labels.

**Accuracy**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 0.818 ± 0.003 | 0.809 ± 0.005 | 0.796 ± 0.009 | 0.773 ± 0.021 |
| ResNet-50 | linear probe | 10% | 0.822 ± 0.004 | 0.812 ± 0.006 | 0.796 ± 0.009 | 0.772 ± 0.014 |
| ResNet-50 | linear probe | 100% | 0.821 | 0.809 | 0.792 | 0.766 |
| ResNet-50 | fine-tuned | 1% | 0.841 ± 0.010 | 0.841 ± 0.006 | 0.834 ± 0.008 | 0.818 ± 0.012 |
| ResNet-50 | fine-tuned | 10% | 0.873 ± 0.016 | 0.870 ± 0.023 | 0.839 ± 0.021 | 0.783 ± 0.035 |
| ResNet-50 | fine-tuned | 100% | 0.886 | 0.889 | 0.865 | 0.818 |
| DINOv2 ViT-B/14 | linear probe | 1% | 0.850 ± 0.003 | 0.849 ± 0.006 | 0.844 ± 0.007 | 0.837 ± 0.009 |
| DINOv2 ViT-B/14 | linear probe | 10% | 0.859 ± 0.002 | 0.857 ± 0.002 | 0.853 ± 0.002 | 0.848 ± 0.003 |
| DINOv2 ViT-B/14 | linear probe | 100% | 0.865 | 0.863 | 0.858 | 0.854 |
| CLIP ViT-B/16 | linear probe | 1% | 0.811 ± 0.003 | 0.802 ± 0.004 | 0.781 ± 0.011 | 0.758 ± 0.018 |
| CLIP ViT-B/16 | linear probe | 10% | 0.818 ± 0.000 | 0.811 ± 0.002 | 0.793 ± 0.002 | 0.765 ± 0.010 |
| CLIP ViT-B/16 | linear probe | 100% | 0.822 | 0.812 | 0.794 | 0.774 |
| Phikon | linear probe | 1% | 0.925 ± 0.003 | 0.923 ± 0.005 | 0.913 ± 0.012 | 0.898 ± 0.017 |
| Phikon | linear probe | 10% | 0.921 ± 0.006 | 0.919 ± 0.004 | 0.908 ± 0.003 | 0.894 ± 0.003 |
| Phikon | linear probe | 100% | 0.924 | 0.920 | 0.906 | 0.891 |

**AUROC**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 0.907 ± 0.004 | 0.899 ± 0.007 | 0.882 ± 0.014 | 0.856 ± 0.024 |
| ResNet-50 | linear probe | 10% | 0.914 ± 0.002 | 0.904 ± 0.005 | 0.884 ± 0.009 | 0.858 ± 0.015 |
| ResNet-50 | linear probe | 100% | 0.913 | 0.902 | 0.879 | 0.849 |
| ResNet-50 | fine-tuned | 1% | 0.922 ± 0.006 | 0.921 ± 0.003 | 0.914 ± 0.004 | 0.903 ± 0.007 |
| ResNet-50 | fine-tuned | 10% | 0.953 ± 0.004 | 0.945 ± 0.008 | 0.920 ± 0.008 | 0.875 ± 0.027 |
| ResNet-50 | fine-tuned | 100% | 0.966 | 0.959 | 0.941 | 0.905 |
| DINOv2 ViT-B/14 | linear probe | 1% | 0.927 ± 0.003 | 0.927 ± 0.003 | 0.926 ± 0.003 | 0.921 ± 0.003 |
| DINOv2 ViT-B/14 | linear probe | 10% | 0.936 ± 0.001 | 0.935 ± 0.001 | 0.933 ± 0.001 | 0.928 ± 0.002 |
| DINOv2 ViT-B/14 | linear probe | 100% | 0.941 | 0.940 | 0.938 | 0.933 |
| CLIP ViT-B/16 | linear probe | 1% | 0.900 ± 0.002 | 0.894 ± 0.002 | 0.885 ± 0.003 | 0.873 ± 0.005 |
| CLIP ViT-B/16 | linear probe | 10% | 0.906 ± 0.001 | 0.901 ± 0.001 | 0.891 ± 0.004 | 0.876 ± 0.009 |
| CLIP ViT-B/16 | linear probe | 100% | 0.910 | 0.902 | 0.889 | 0.873 |
| Phikon | linear probe | 1% | 0.980 ± 0.001 | 0.980 ± 0.002 | 0.976 ± 0.003 | 0.967 ± 0.005 |
| Phikon | linear probe | 10% | 0.979 ± 0.005 | 0.979 ± 0.004 | 0.976 ± 0.003 | 0.967 ± 0.003 |
| Phikon | linear probe | 100% | 0.982 | 0.981 | 0.977 | 0.969 |

**ECE after temperature scaling (T fitted on clean images)**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 0.012 ± 0.003 | 0.014 ± 0.002 | 0.015 ± 0.008 | 0.025 ± 0.017 |
| ResNet-50 | linear probe | 10% | 0.015 ± 0.005 | 0.017 ± 0.007 | 0.022 ± 0.012 | 0.035 ± 0.017 |
| ResNet-50 | linear probe | 100% | 0.019 | 0.023 | 0.032 | 0.048 |
| ResNet-50 | fine-tuned | 1% | 0.028 ± 0.005 | 0.022 ± 0.002 | 0.014 ± 0.009 | 0.013 ± 0.009 |
| ResNet-50 | fine-tuned | 10% | 0.008 ± 0.001 | 0.012 ± 0.005 | 0.022 ± 0.009 | 0.058 ± 0.029 |
| ResNet-50 | fine-tuned | 100% | 0.010 | 0.010 | 0.011 | 0.040 |
| DINOv2 ViT-B/14 | linear probe | 1% | 0.014 ± 0.003 | 0.014 ± 0.002 | 0.014 ± 0.003 | 0.013 ± 0.004 |
| DINOv2 ViT-B/14 | linear probe | 10% | 0.009 ± 0.001 | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.002 |
| DINOv2 ViT-B/14 | linear probe | 100% | 0.013 | 0.010 | 0.009 | 0.013 |
| CLIP ViT-B/16 | linear probe | 1% | 0.012 ± 0.003 | 0.023 ± 0.007 | 0.043 ± 0.020 | 0.058 ± 0.033 |
| CLIP ViT-B/16 | linear probe | 10% | 0.014 ± 0.003 | 0.026 ± 0.001 | 0.044 ± 0.000 | 0.064 ± 0.011 |
| CLIP ViT-B/16 | linear probe | 100% | 0.012 | 0.029 | 0.045 | 0.059 |
| Phikon | linear probe | 1% | 0.013 ± 0.002 | 0.014 ± 0.007 | 0.015 ± 0.002 | 0.018 ± 0.010 |
| Phikon | linear probe | 10% | 0.010 ± 0.004 | 0.011 ± 0.003 | 0.010 ± 0.003 | 0.010 ± 0.005 |
| Phikon | linear probe | 100% | 0.014 | 0.013 | 0.010 | 0.013 |

**LAC coverage (%) at 90% target**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 88.9 ± 0.3 | 88.3 ± 0.4 | 87.5 ± 0.9 | 86.2 ± 1.7 |
| ResNet-50 | linear probe | 10% | 88.3 ± 0.7 | 87.6 ± 0.6 | 86.6 ± 1.1 | 85.1 ± 1.5 |
| ResNet-50 | linear probe | 100% | 87.8 | 87.0 | 85.7 | 83.8 |
| ResNet-50 | fine-tuned | 1% | 85.7 ± 0.4 | 85.9 ± 0.3 | 85.4 ± 0.9 | 84.4 ± 1.2 |
| ResNet-50 | fine-tuned | 10% | 88.1 ± 1.2 | 87.9 ± 1.7 | 84.8 ± 1.4 | 79.4 ± 3.3 |
| ResNet-50 | fine-tuned | 100% | 87.9 | 88.1 | 85.8 | 80.7 |
| DINOv2 ViT-B/14 | linear probe | 1% | 90.3 ± 0.3 | 90.1 ± 0.4 | 89.8 ± 0.5 | 89.2 ± 0.8 |
| DINOv2 ViT-B/14 | linear probe | 10% | 89.7 ± 0.3 | 89.5 ± 0.1 | 89.1 ± 0.2 | 88.7 ± 0.1 |
| DINOv2 ViT-B/14 | linear probe | 100% | 89.6 | 89.4 | 89.0 | 88.5 |
| CLIP ViT-B/16 | linear probe | 1% | 88.8 ± 0.4 | 87.8 ± 0.7 | 86.1 ± 1.6 | 84.5 ± 2.7 |
| CLIP ViT-B/16 | linear probe | 10% | 88.4 ± 0.1 | 87.6 ± 0.2 | 85.8 ± 0.3 | 83.7 ± 0.9 |
| CLIP ViT-B/16 | linear probe | 100% | 88.5 | 87.4 | 85.8 | 84.4 |
| Phikon | linear probe | 1% | 90.9 ± 0.5 | 90.7 ± 0.8 | 89.5 ± 1.4 | 87.8 ± 1.7 |
| Phikon | linear probe | 10% | 90.4 ± 0.8 | 90.1 ± 0.6 | 89.0 ± 0.4 | 87.2 ± 0.5 |
| Phikon | linear probe | 100% | 90.8 | 90.4 | 89.0 | 87.0 |

**APS coverage (%) at 90% target**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 89.9 ± 0.5 | 89.6 ± 0.6 | 89.1 ± 0.9 | 88.3 ± 1.4 |
| ResNet-50 | linear probe | 10% | 89.7 ± 0.5 | 89.2 ± 0.6 | 88.7 ± 0.8 | 87.9 ± 1.2 |
| ResNet-50 | linear probe | 100% | 88.9 | 88.2 | 87.6 | 86.5 |
| ResNet-50 | fine-tuned | 1% | 88.5 ± 0.5 | 89.1 ± 0.3 | 89.6 ± 0.3 | 90.0 ± 0.6 |
| ResNet-50 | fine-tuned | 10% | 90.1 ± 0.3 | 90.0 ± 0.2 | 88.6 ± 0.5 | 86.0 ± 2.4 |
| ResNet-50 | fine-tuned | 100% | 90.2 | 90.0 | 88.8 | 86.6 |
| DINOv2 ViT-B/14 | linear probe | 1% | 90.5 ± 0.2 | 90.4 ± 0.3 | 90.1 ± 0.3 | 89.7 ± 0.6 |
| DINOv2 ViT-B/14 | linear probe | 10% | 90.5 ± 0.2 | 90.3 ± 0.2 | 90.1 ± 0.2 | 90.0 ± 0.2 |
| DINOv2 ViT-B/14 | linear probe | 100% | 90.7 | 90.5 | 90.2 | 90.2 |
| CLIP ViT-B/16 | linear probe | 1% | 89.6 ± 0.4 | 88.7 ± 0.7 | 87.0 ± 1.7 | 86.0 ± 2.5 |
| CLIP ViT-B/16 | linear probe | 10% | 89.3 ± 0.3 | 88.4 ± 0.2 | 86.8 ± 0.3 | 85.3 ± 0.9 |
| CLIP ViT-B/16 | linear probe | 100% | 89.7 | 88.6 | 86.8 | 85.9 |
| Phikon | linear probe | 1% | 91.1 ± 0.2 | 91.0 ± 0.6 | 90.5 ± 1.0 | 89.9 ± 1.3 |
| Phikon | linear probe | 10% | 90.7 ± 0.6 | 90.6 ± 0.3 | 90.3 ± 0.1 | 90.0 ± 0.2 |
| Phikon | linear probe | 100% | 91.0 | 90.7 | 90.1 | 89.7 |

**Auto-labeled (%), LAC**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 83.7 ± 1.4 | 83.2 ± 1.6 | 81.5 ± 2.0 | 79.5 ± 2.5 |
| ResNet-50 | linear probe | 10% | 86.0 ± 0.5 | 85.5 ± 0.3 | 84.2 ± 0.6 | 82.5 ± 0.7 |
| ResNet-50 | linear probe | 100% | 86.6 | 86.2 | 85.4 | 83.7 |
| ResNet-50 | fine-tuned | 1% | 96.7 ± 1.5 | 96.4 ± 1.5 | 95.8 ± 2.1 | 94.6 ± 2.6 |
| ResNet-50 | fine-tuned | 10% | 98.3 ± 1.2 | 98.3 ± 1.2 | 98.1 ± 1.4 | 97.7 ± 1.7 |
| ResNet-50 | fine-tuned | 100% | 98.5 | 98.4 | 98.5 | 97.9 |
| DINOv2 ViT-B/14 | linear probe | 1% | 87.7 ± 0.7 | 87.9 ± 0.8 | 87.8 ± 0.9 | 87.5 ± 0.8 |
| DINOv2 ViT-B/14 | linear probe | 10% | 91.6 ± 0.9 | 91.5 ± 0.8 | 91.3 ± 0.9 | 91.1 ± 1.0 |
| DINOv2 ViT-B/14 | linear probe | 100% | 93.3 | 93.2 | 93.1 | 92.9 |
| CLIP ViT-B/16 | linear probe | 1% | 82.4 ± 1.7 | 82.5 ± 1.6 | 82.0 ± 2.0 | 80.7 ± 2.8 |
| CLIP ViT-B/16 | linear probe | 10% | 85.0 ± 0.2 | 85.4 ± 0.1 | 85.2 ± 0.3 | 84.2 ± 0.4 |
| CLIP ViT-B/16 | linear probe | 100% | 85.6 | 85.8 | 85.5 | 84.3 |
| Phikon | linear probe | 1% | 97.0 ± 0.5 | 97.0 ± 0.4 | 96.6 ± 0.4 | 96.3 ± 0.3 |
| Phikon | linear probe | 10% | 96.8 ± 0.4 | 96.9 ± 0.5 | 96.5 ± 0.4 | 95.9 ± 0.7 |
| Phikon | linear probe | 100% | 97.0 | 97.1 | 96.7 | 96.2 |

**System error (%): wrong auto-labels / all images, LAC**

| Model | Method | Labels | s = 0 | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|---|---|
| ResNet-50 | linear probe | 1% | 11.1 ± 0.3 | 11.7 ± 0.4 | 12.5 ± 0.9 | 13.8 ± 1.7 |
| ResNet-50 | linear probe | 10% | 11.7 ± 0.7 | 12.4 ± 0.6 | 13.4 ± 1.1 | 14.9 ± 1.5 |
| ResNet-50 | linear probe | 100% | 12.2 | 13.0 | 14.3 | 16.2 |
| ResNet-50 | fine-tuned | 1% | 14.3 ± 0.4 | 14.1 ± 0.3 | 14.6 ± 0.9 | 15.6 ± 1.2 |
| ResNet-50 | fine-tuned | 10% | 11.9 ± 1.2 | 12.1 ± 1.7 | 15.2 ± 1.4 | 20.6 ± 3.3 |
| ResNet-50 | fine-tuned | 100% | 10.6 | 10.3 | 12.7 | 17.2 |
| DINOv2 ViT-B/14 | linear probe | 1% | 9.7 ± 0.3 | 9.9 ± 0.4 | 10.2 ± 0.5 | 10.8 ± 0.8 |
| DINOv2 ViT-B/14 | linear probe | 10% | 10.3 ± 0.3 | 10.5 ± 0.1 | 10.9 ± 0.2 | 11.3 ± 0.1 |
| DINOv2 ViT-B/14 | linear probe | 100% | 10.4 | 10.6 | 11.0 | 11.5 |
| CLIP ViT-B/16 | linear probe | 1% | 11.2 ± 0.4 | 12.2 ± 0.7 | 13.9 ± 1.6 | 15.5 ± 2.7 |
| CLIP ViT-B/16 | linear probe | 10% | 11.6 ± 0.1 | 12.4 ± 0.2 | 14.2 ± 0.3 | 16.3 ± 0.9 |
| CLIP ViT-B/16 | linear probe | 100% | 11.5 | 12.6 | 14.2 | 15.6 |
| Phikon | linear probe | 1% | 6.1 ± 0.3 | 6.3 ± 0.4 | 7.1 ± 1.0 | 8.5 ± 1.5 |
| Phikon | linear probe | 10% | 6.4 ± 0.4 | 6.7 ± 0.1 | 7.5 ± 0.3 | 8.7 ± 0.5 |
| Phikon | linear probe | 100% | 6.2 | 6.8 | 7.7 | 9.2 |
