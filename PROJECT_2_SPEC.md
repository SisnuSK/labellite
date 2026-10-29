# LabelLite: Label-Efficient, Uncertainty-Aware Image Analysis with Foundation Models

> This is the original project specification, written before any code. The scope changes below override it where they conflict.

---

## Scope changes (2026-10-27)

These override the rest of this spec:

- **Datasets:** EuroSAT (RGB) and PatchCamelyon only. **Camelyon17-WILDS is dropped**: its only
  official download host (CodaLab) was serving an expired TLS certificate.
- **RQ4 (distribution shift):** uses the fallback in Section 2, a synthetic stain/color
  perturbation of PCam at increasing severity, instead of the cross-hospital shift. Results are
  reported per perturbation severity instead of per hospital (Section 9, item 6).
- **Pathology model:** `owkin/phikon` (open) replaces the gated UNI / CONCH (Section 3), whose
  access requires an institutional email.
- **Web app dropped:** Section 6, Phase 6, the `app/` directory and `docker-compose.yml` are out
  of scope. Error galleries (Section 5.6) are still produced as static files.
- **Final deliverables:** the Python package and CLI, results tables and figures, the README, and
  the 4-page report, published as a GitHub repository.

---

## 1. Goal and research questions

Labs in pathology, radiology, ecology, and remote sensing have lots of images but very few expert labels, because labeling needs a pathologist or domain expert. They also need to know **when a model's prediction can't be trusted**. This project builds a reproducible pipeline and tool that answers:

- **RQ1 (Label efficiency):** Can general-purpose and domain-specific **vision foundation models**, used as frozen feature extractors with a lightweight classifier, match a fully fine-tuned CNN when only 1%, 10%, or 100% of training labels are available?
- **RQ2 (Calibration & trust):** Are these models' confidence scores calibrated? Which post-hoc fixes (temperature scaling) help?
- **RQ3 (Conformal triage):** Using split conformal prediction, can we auto-label confident images and route only uncertain ones to an expert, with a **guaranteed error rate**? How much expert effort does that save?
- **RQ4 (Distribution shift):** How do accuracy, calibration, and conformal coverage hold up when test images come from a **different hospital / acquisition site** than the training images?
- **RQ5 (Active learning):** If an expert labels images chosen by uncertainty, how quickly does performance improve versus labeling random images?

**Deliverables:**
1. A Python package + CLI that runs the full pipeline from config files.
2. A results directory with tables and figures.
3. A web app where an expert can review uncertain predictions with heatmap explanations, correct labels, and trigger an active-learning round.
4. A 4-page paper-style report (LaTeX → PDF).
5. A polished README with results table, figures, and demo screenshots/GIF.

---

## 2. Datasets (two image domains + one shift test)

**Before using any dataset, verify:** (a) download source, (b) license/terms of use, (c) exact size, image resolution, label set, and official splits. Record all of this in `data/DATASETS.md`. If a dataset is unavailable or its license forbids this use, drop it and record why.

### Domain A: Histopathology
- **Primary:** **PatchCamelyon (PCam)**. Small H&E lymph-node tissue patches, binary label (metastatic tumor tissue present or not). Derived from the Camelyon16 challenge.

### Domain B: Satellite imagery
- **Primary:** **EuroSAT (RGB version)**. Sentinel-2 satellite image patches, 10 land-use/land-cover classes. No official split in the original release, so create a fixed stratified train/val/calibration/test split with a seed and save the IDs.

### Shift test: Camelyon17-WILDS
- **Camelyon17-WILDS** (from the WILDS benchmark): pathology patches labeled with the **hospital they came from**, with official splits where test hospitals differ from training hospitals. Use it for RQ4: train/calibrate on in-distribution hospitals, evaluate on held-out hospitals.
- If time is short, a fallback shift test is synthetic stain/color perturbation on PCam. Prefer the real WILDS shift.

### Splits and sampling
- For every dataset: **train / val / calibration / test**. The calibration split is used **only** for temperature scaling and conformal prediction, never for training or model selection.
- Label-budget subsets: **1%, 10%, 100%** of the training split, stratified by class. For 1% and 10%, run **3 random seeds** and report mean ± std.
- If full test sets are too large for available compute, evaluate on a fixed stratified subset (configurable) and save the IDs to `data/splits/`.

---

## 3. Models

Configurable via YAML. Use pretrained weights from `timm`, `open_clip`, or Hugging Face.

| Role | Model | Notes |
|---|---|---|
| Supervised baseline | ResNet-50 (ImageNet-pretrained), **fully fine-tuned** | The "classic" approach |
| General foundation model | **DINOv2** (ViT-B/14; ViT-S/14 if compute is tight) | Self-supervised, strong frozen features |
| Vision-language model | **CLIP / OpenCLIP** (ViT-B/16 or ViT-B/32) | Also enables **zero-shot** classification from text prompts |
| Pathology foundation model | A pathology-specific model such as **UNI** or **CONCH** | Often **gated** on Hugging Face (requires an access request and agreeing to terms). If access is not granted, substitute an openly available pathology model and document the substitution. |

### How each model is used
1. **Frozen features + linear probe** (logistic regression) — the main comparison.
2. **Frozen features + k-NN** (k configurable) — a training-free check.
3. **CLIP zero-shot** — text prompts per class, no labels at all (a "0%" point on the label-efficiency curve).
4. **Full fine-tuning** — ResNet-50 always; DINOv2 fine-tuning only if compute allows (optional).

### Compute plan
- **Extract features once per (model, dataset) and cache them to disk** (`.npy` or `.safetensors` + an index file). All probes, conformal, and active learning then run on CPU in seconds. This is the key to keeping the project feasible on one GPU (university HPC, a lab GPU, or a cloud notebook).
- Log GPU type, batch size, and extraction time in the run manifest.

---

## 4. Training details

- **Linear probe:** scikit-learn logistic regression on L2-normalized features; tune the regularization strength on the val split.
- **Fine-tuning (ResNet-50):** PyTorch (Lightning optional), AdamW, cosine schedule, early stopping on val, standard augmentation (flips, rotations; for pathology also mild color jitter). Mixed precision.
- **Fixed seeds** everywhere; deterministic flags where practical.
- Every trained model/probe is saved with its config and the hash of the split it was trained on.

---

## 5. Metrics and analysis

### 5.1 Label efficiency (RQ1)
- Accuracy, macro-F1, **AUROC** (binary datasets), balanced accuracy.
- **95% bootstrap confidence intervals** (1,000 resamples) on test metrics.
- **Label-efficiency curves:** metric vs. % labels (0% CLIP zero-shot, 1%, 10%, 100%) for every model.
- Paired comparisons between models on the same test items (McNemar's test or paired bootstrap), with Holm–Bonferroni correction.

### 5.2 Calibration (RQ2)
- Expected Calibration Error (ECE, 15 bins), Brier score, negative log-likelihood.
- **Reliability diagrams** before and after **temperature scaling** (fitted on the calibration split).

### 5.3 Conformal triage (RQ3)
- **Split conformal prediction** on the calibration split, implementing **LAC** and **APS** (directly, or via a library such as MAPIE or TorchCP; verify the library API before using it).
- Target coverage levels: 90%, 95%.
- Report empirical coverage on test, **average prediction-set size**, and **class-conditional coverage** (coverage per class, since overall coverage can hide a class that is under-covered).
- **Triage policy:** auto-label images whose prediction set has exactly one label; route the rest to an expert. Report % routed, error rate on auto-labeled images, and expert effort saved.

### 5.4 Distribution shift (RQ4) — Camelyon17-WILDS
- Train and calibrate on in-distribution hospitals; evaluate on out-of-distribution hospitals.
- Report the drop in accuracy/AUROC, the change in ECE, and whether conformal coverage **breaks** under shift (it is only guaranteed when calibration and test data are exchangeable).
- Show per-hospital results.

### 5.5 Active learning (RQ5)
- Simulated loop using the hidden true labels as the "expert": start with 1% labels; each round, query a batch (e.g., 1% of the pool) and retrain the linear probe on cached features.
- Strategies: **random**, **least-confidence / entropy**, **conformal set size** (query images with the largest prediction sets), and optionally **diversity-based** (k-center greedy on features).
- Plot performance vs. number of labels; run 3 seeds.

### 5.6 Explanations and error analysis
- **Grad-CAM** for ResNet-50; **attention maps or Grad-CAM-style maps** for ViT models (e.g., attention rollout or a library such as `pytorch-grad-cam`; verify it supports the chosen ViT).
- Automatically save galleries of confident errors and uncertain cases per model to `results/error_gallery/`.
- In the report, write a short qualitative analysis of failure modes from a manually reviewed sample (~100 images). The review tooling was planned as part of the web app.
- **Do not claim explanation maps are proof of what the model "looks at."** Treat them as a qualitative aid and say so.

---

## 6. Web app: expert review tool

A full-stack app so a domain expert (pathologist, ecologist) could use the system without code.

**Stack:** FastAPI backend + React (Vite + TypeScript) frontend + SQLite. Tailwind is fine. Images served from disk by the backend.

**Features:**
1. **Dashboard:** headline metrics per model/dataset/label budget, label-efficiency curves, triage savings.
2. **Triage queue:** images routed to the expert (prediction set size > 1) or confident errors. Filters: dataset, model, predicted class, confidence range, hospital (for WILDS).
3. **Image view:** image with a toggleable **heatmap overlay** (opacity slider), predicted label, confidence, prediction set, nearest-neighbor training images in feature space (helps the expert see why the model decided).
4. **Label / tag:** expert confirms or corrects the label and tags a failure type (configurable taxonomy, e.g., artifact, blur, ambiguous, out-of-focus, cloud cover). Saved to SQLite.
5. **Active-learning round:** a button that retrains the linear probe on cached features plus the expert's new labels, then refreshes the queue and metrics. This should complete in seconds because features are cached.
6. **Export:** expert labels and tags as CSV/JSONL.
7. **Bring-your-own-images (stretch):** upload a folder or zip of images + a small labeled CSV, extract features with a chosen model, and run the same probe + triage workflow. This is what makes it useful for a professor's own data.

Add a `docker-compose.yml` so the app runs with one command (CPU-only mode is fine for the app; feature extraction for new uploads can be slow on CPU, so show progress).

---

## 7. Engineering requirements

- **Python 3.11+**, managed with `uv` (or `poetry`). Pin dependencies, including `torch`, `timm`, `open_clip_torch`, `scikit-learn`.
- **Config-driven runs:** YAML configs in `configs/` (Hydra or `pydantic` + YAML). Key commands:
  - `labellite extract --config configs/extract/dinov2_pcam.yaml`
  - `labellite probe --config configs/main.yaml`
  - `labellite finetune --config configs/finetune/resnet50_pcam.yaml`
  - `labellite conformal --config configs/main.yaml`
  - `labellite activelearn --config configs/al.yaml`
  - `labellite analyze` (builds all tables and figures)
- **Feature cache** with integrity checks (model name + weights hash + dataset split hash); never silently reuse stale features.
- **Resumable** extraction and training.
- **Run manifest** per run: git commit, config, model identifiers and weight versions, split hashes, hardware, timings.
- **Tests (pytest):** split stratification, metrics on hand-checked examples, ECE on a synthetic case, conformal coverage on synthetic data (coverage ≥ target within tolerance), feature-cache invalidation, probe training on tiny fake features. GitHub Actions CI running tests + `ruff` lint, CPU-only (use tiny dummy models/data in tests).
- **No data, weights, or secrets committed.** `.gitignore` covers data, caches, weights, `.env`. Any Hugging Face token comes from an environment variable.

---

## 8. Repository structure

```
labellite/
├── PROJECT_2_SPEC.md
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── main.yaml
│   ├── extract/*.yaml
│   ├── finetune/*.yaml
│   ├── al.yaml
│   └── datasets/*.yaml
├── data/
│   ├── DATASETS.md          # sources, licenses, sizes, splits
│   └── splits/              # fixed split/subset IDs (committed); raw data git-ignored
├── src/labellite/
│   ├── cli.py
│   ├── data/                # dataset loaders, splits, label-budget subsets, transforms
│   ├── models/              # backbone loading (timm / open_clip / HF), zero-shot CLIP
│   ├── features/            # extraction + cache
│   ├── train/               # linear probe, kNN, fine-tuning
│   ├── calibration/         # temperature scaling, ECE, reliability diagrams
│   ├── conformal/           # LAC, APS, triage policy, class-conditional coverage
│   ├── active/              # query strategies, simulated loop
│   ├── explain/             # Grad-CAM, ViT attribution maps
│   ├── stats/               # bootstrap CIs, paired tests
│   └── analysis/            # tables, figures, error galleries
├── app/
│   ├── backend/             # FastAPI
│   └── frontend/            # React + Vite
├── results/                 # tables (CSV/Markdown), figures, manifests, galleries
├── report/                  # LaTeX source of the 4-page report
├── tests/
└── docker-compose.yml
```

---

## 9. Figures and tables to produce

1. **Main results table:** rows = model × method (linear probe, kNN, zero-shot, fine-tune), columns = per-dataset accuracy / macro-F1 / AUROC at 1%, 10%, 100% labels (with CIs or mean ± std over seeds).
2. **Label-efficiency curves:** one plot per dataset, all models.
3. **Calibration table + reliability diagrams:** before vs. after temperature scaling.
4. **Conformal triage table:** target coverage, empirical coverage, mean set size, % routed to expert, auto-label error rate, per method (LAC/APS).
5. **Class-conditional coverage plot.**
6. **Distribution-shift table and per-hospital plot** (Camelyon17-WILDS): ID vs. OOD accuracy, ECE, coverage.
7. **Active-learning curves:** performance vs. labels for each query strategy.
8. **Explanation gallery figure:** a few correct, wrong, and uncertain examples with heatmaps (clearly labeled as qualitative).

---

## 10. Optional variant: time-series instead of images

Out of scope unless time allows. Same structure (frozen pretrained model → lightweight head → calibration → conformal → shift test), but with a **pretrained time-series forecasting model** on public data such as traffic, weather/precipitation, power-grid load, or fMRI signals, compared against a classical statistical or simulation baseline. Conformal prediction then produces **prediction intervals** instead of label sets.

---

## 11. Phases and acceptance criteria

Each phase ends with a check of its acceptance criteria.

**Phase 0: Setup (≈1 day)**
- Repo scaffold, environment, CI, lint, `.env.example`.
- ✅ `pytest` and `ruff` pass in CI on the skeleton.

**Phase 1: Data (≈2–3 days)**
- Loaders for PCam, EuroSAT, Camelyon17-WILDS; verify sources/licenses; write `data/DATASETS.md`.
- Fixed splits, calibration split, label-budget subsets (3 seeds); save IDs.
- ✅ A script prints dataset sizes, image shapes, class distributions per split (and hospital counts for WILDS), matching the documented sources.

**Phase 2: Feature extraction (≈3–4 days)**
- Backbone loaders for ResNet-50, DINOv2, CLIP, and the pathology model (or documented substitute). Feature cache with integrity checks.
- Before running full extraction, **estimate the GPU time and disk usage** to make sure it fits the machine.
- ✅ Features extracted for every (model × dataset); re-running hits the cache; a sanity check (e.g., kNN accuracy on 10% labels) is clearly above chance.

**Phase 3: Probes, zero-shot, fine-tuning (≈4–5 days)**
- Linear probe and kNN at all label budgets and seeds; CLIP zero-shot; ResNet-50 fine-tuning.
- ✅ Main results table (Section 9, item 1) and label-efficiency curves generated with CIs.

**Phase 4: Calibration, conformal, shift (≈4–5 days)**
- Temperature scaling, ECE, reliability diagrams; LAC + APS conformal; triage policy; class-conditional coverage; Camelyon17-WILDS shift analysis.
- ✅ Tables/figures 3–6 generated; conformal coverage on in-distribution test is within expected tolerance of target; unit tests pass.

**Phase 5: Active learning + explanations (≈3–4 days)**
- Simulated AL loop with all strategies and 3 seeds; Grad-CAM / ViT maps; error galleries.
- ✅ Figures 7–8 generated.

**Phase 6: Web app (≈5–6 days)**
- FastAPI + React app per Section 6; docker-compose.
- ✅ `docker compose up` starts the app; I can filter the triage queue, view a heatmap overlay, correct a label, tag a failure type, run an active-learning round and see metrics update, and export labels.

**Phase 7: Write-up (≈3–4 days)**
- Error analysis (with a manual review of sampled errors), README, 4-page LaTeX report (intro, related work, method, results, failure modes, limitations, ethics), demo GIF/screenshots.
- ✅ README has the results table, key figures, one-command reproduction instructions, and limitations. The report compiles to PDF.

---

## 12. Limitations and ethics (must appear in README and report)

- This is a **research benchmark on public datasets**, not a diagnostic tool. Make **no clinical claims**; pathology results do not imply clinical usefulness.
- Patch-level labels simplify real pathology workflows (slide-level diagnosis, context).
- Conformal guarantees assume calibration and test data are exchangeable; the shift experiment shows what happens when that fails.
- Explanation maps are qualitative and can be misleading.
- Foundation-model weights and gated models have their own licenses; respect them, and do not redistribute weights or raw data.

---

## 13. Target outcome (as planned; final numbers are in the README)

> **LabelLite: Label-Efficient, Uncertainty-Aware Image Analysis** · PyTorch, timm, OpenCLIP, scikit-learn, FastAPI, React
> - Compared 4 vision backbones (ResNet-50, DINOv2, CLIP, a pathology foundation model) across pathology and satellite imagery; with 10% of labels, frozen foundation-model features reached X% of fully supervised performance.
> - Built split-conformal triage (LAC/APS) that auto-labels Z% of images at ≤ E% error; showed coverage degrades from A% to B% under cross-hospital shift (Camelyon17-WILDS).
> - Uncertainty-driven active learning matched random-sampling performance with N% fewer expert labels; shipped a full-stack expert review tool with heatmap explanations and one-click retraining.
