# Datasets

Each fact below is tagged with where it comes from:

- **[pub]**: stated by the publisher (page linked in the entry), checked 2026-10-27
- **[local]**: measured by this repo on the downloaded files (`labellite data stats`)

Raw data lives in `data/raw/<dataset>/` and is git-ignored; it is never redistributed. Only the
split ID files in `data/splits/<dataset>/` are committed.

Camelyon17-WILDS was in the original plan for the distribution-shift question (RQ4). It was
dropped on 2026-10-27, when its only official download host (CodaLab) was serving an expired TLS
certificate. RQ4 uses a synthetic stain/color shift on PCam instead.

| Dataset | Status |
|---|---|
| EuroSAT (RGB) | Downloaded, MD5 verified, splits built |
| PatchCamelyon (PCam) | Downloaded, MD5 verified, splits built |

---

## EuroSAT (RGB)

- **Source [pub]:** <https://github.com/phelber/EuroSAT>; files on Zenodo record
  [7711810](https://zenodo.org/records/7711810) (the DFKI mirror is marked deprecated by the
  authors).
- **File:** `EuroSAT_RGB.zip`, 94,658,721 bytes, MD5 `f46e308c4d50d4bf32fedad2d3d62f3b`
  **[pub]**; downloaded file matches both **[local]**.
- **License [pub]:** MIT (GitHub README and Zenodo record). The underlying imagery is Copernicus
  Sentinel-2 data, subject to the Copernicus Sentinel Data Terms and Conditions.
- **Content:** 27,000 labeled, geo-referenced Sentinel-2 images in 10 land-use/land-cover classes
  **[pub]**. The RGB version holds the red, green, and blue bands as JPEG **[pub]**.
- **Measured [local]:** 27,000 JPEGs, all 64 x 64 px, RGB. Per class: AnnualCrop 3,000,
  Forest 3,000, HerbaceousVegetation 3,000, Highway 2,500, Industrial 2,500, Pasture 2,000,
  PermanentCrop 2,500, Residential 3,000, River 2,500, SeaLake 3,000.
- **Official splits:** none **[pub]**.
- **Our splits:** stratified by class, seed 0: train 60% / val 10% / cal 10% / test 20%
  (16,200 / 2,700 / 2,700 / 5,400) **[local]**. Config: `configs/datasets/eurosat.yaml`.
- **Image IDs:** `<ClassName>/<file>.jpg`, relative to `EuroSAT_RGB/`.
- **Citation:** Helber, Bischke, Dengel, Borth. *EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019; and IGARSS 2018.

## PatchCamelyon (PCam)

- **Source [pub]:** <https://github.com/basveeling/pcam>; files on Zenodo record
  [2546921](https://zenodo.org/records/2546921) (also Google Drive and Baidu mirrors).
- **Files [pub]:** 9 files, 8.03 GB compressed in total. The size and MD5 of each are pinned in
  `src/labellite/data/sources.py` (from the Zenodo API) and checked on download; all 9 downloaded
  files match **[local]**. The `.h5.gz` files are decompressed to `.h5` next to them (about 9.1 GB).
- **License [pub]:** the GitHub README says the data is CC0, following the Camelyon16 license,
  with the code under MIT. The Zenodo record's license field says MIT. Both permit this use. We
  cite the README's CC0 statement and note the mismatch.
- **Content [pub]:** 327,680 color patches, 96 x 96 px, from H&E lymph-node sections (Camelyon16).
  A positive label means the center 32 x 32 px region holds at least one pixel of tumor tissue.
- **Official splits [pub]:** train 262,144 / valid 32,768 / test 32,768, each 50/50
  positive/negative, with no whole-slide-image overlap between splits.
- **Measured [local]:**
  - HDF5 layout: each `_x.h5` holds one dataset `x` of shape (N, 96, 96, 3), uint8; each `_y.h5`
    one dataset `y` of shape (N, 1, 1, 1), uint8 with values {0, 1}.
  - Split sizes match the publisher: 262,144 / 32,768 / 32,768.
  - Class balance: train is exactly 131,072 / 131,072 (normal / tumor). Valid (16,399 / 16,369)
    and test (16,391 / 16,377) are close to, but not exactly, 50/50.
  - Labels equal the `center_tumor_patch` column of the `_meta.csv` files for every patch.
  - Slides (`wsi` column): 216 train, 54 valid, 129 test, with no slide in more than one split.
- **Our splits:** official train and test kept as-is; the official valid set is divided 50/50,
  stratified by label, into val (model selection) and cal (temperature scaling and conformal
  only): train 262,144 / val 16,385 / cal 16,383 / test 32,768 **[local]**. Val and cal share
  slides with each other but not with test. Config: `configs/datasets/pcam.yaml`.
- **Label budgets [local]:** 1% = 2,622 images (1,311 per class), 10% = 26,214 (13,107 per
  class), for each of seeds 0-2.
- **Image IDs:** `<official split>/<row>`, e.g. `valid/123` is row 123 of the valid HDF5 files.
- **Citation:** Veeling et al., *Rotation Equivariant CNNs for Digital Pathology*,
  arXiv:1806.03962 (2018); Ehteshami Bejnordi et al., *JAMA* 318(22):2199-2210 (2017).

---

## Split files

`labellite data splits --config configs/datasets/<name>.yaml` writes, to `data/splits/<name>/`:

- `splits.csv.gz`: `id, label, split, eval_subset`
- `budgets.csv.gz`: `id, budget, seed`, the label-budget subsets of the train split (1% and 10%,
  seeds 0-2, stratified by class, nested so that 1% is contained in 10% for each seed; at least one
  image per class). The 100% budget is the full train split.
- `manifest.json`: config, split counts, SHA-256 of the whole split table and of each split, and
  a hash of every budget subset

The files are byte-reproducible: re-running with the same config gives identical files.
`labellite data stats --config ...` prints sizes, image shapes, and class counts per split.
