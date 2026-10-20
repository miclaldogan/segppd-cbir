# Dataset audit (SegPPD-101, Zenodo 10.5281/zenodo.15114160)

All numbers below were measured locally from the downloaded archive (`artifacts/audits/`).

| Item | Result |
|---|---|
| Archive | `SegPPD-101.rar`, 247,220,581 B, MD5 `2242cf24f0c84a80e15633e049a9c0df` = Zenodo record checksum (**integrity PASS**) |
| Extraction | 7z; no path traversal, no symlinks; 247 MB unpacked; `image/` 2,263 + `mask/` 2,263 PNG + `fullname.txt` |
| Images | 2,263; **all 256x256 RGB PNG** (archive is pre-resized); EXIF orientation=1 for all; 0 decode errors |
| Masks | 2,263, identical names; two pixel values each (0 and one non-zero value); 0 empty, 0 full-frame |
| Classes | 101 (`fullname.txt`, ids 001-101, `crop_condition`), 9-68 images/class (median 22). **27 of the 101 classes are "healthy"** (not all classes are pest/disease) |
| Crops | **27 distinct crop tokens** in the file (incl. the typo `cucmber`) - supports 27, not the 21 also quoted in the literature |
| Filename -> class | `CCCNN.png` = class id + index; consistent with `fullname.txt` and with image content on a 25-image visual check (`contact_sheet.png`); no ambiguity |

## Mask semantics (verified visually + numerically)
Foreground = non-zero pixels; its value equals the class id in 2,262/2,263 masks (`07016` is a palette-mode PNG with value 1; still binary).
The foreground is the **annotated target region**: small lesions for spot diseases (apple frog eye, rust, pear rust), the **whole affected leaf** for diffuse diseases (angular leaf spot, mosaic, mold) and the **leaf** for healthy classes.
Therefore "occupancy" = annotated target-region area / image area, *not* strictly lesion area, and it is confounded with disease type. Healthy classes are excluded from the occupancy analysis.
Occupancy median 0.249 (all), 0.194 (non-healthy); bins small/medium/large = 910 / 809 / 544 images (all images, incl. healthy).

## Duplicates and label noise
* Exact duplicates: 7 byte-identical pairs, **3 with conflicting class labels**.
* Perceptual duplicates: rule = identical SHA-256 **or** (pHash<=12 **and** dHash<=18); thresholds chosen by visual review of every candidate pair (true duplicates had dHash<=17; false positives >=20). 26 groups (25 pairs + 1 triple).
* **16 groups (32 images) show the same photo under different class labels** (e.g. lemon vs orange anthracnose, hops black rot vs downy mildew, eggplant early blight vs necrotic leaf spot). These are source-data label conflicts; all members are **excluded** (no ground truth exists). 11 same-class duplicates are excluded (lowest id kept). **43 excluded, 2,220 usable.** Nothing was deleted from disk.

## Class support (usable images)
75 classes have >=13 usable images (primary cohort); 26 classes have 2-12 (secondary). Per-class counts: `artifacts/audits/class_counts.csv`; mask checks: `artifacts/audits/mask_integrity.csv`; full manifest: `dataset_manifest.csv`.

## Rights
The Zenodo record's license field states **MIT** (the task text said no clear license was shown). The record does not document the rights of the underlying internet-sourced photographs, so MIT on the record is **not** assumed to cover them. Raw images are not redistributed; `artifacts/retrieval_examples/*_LOCAL_ONLY.png` contain dataset images and must not be published.
