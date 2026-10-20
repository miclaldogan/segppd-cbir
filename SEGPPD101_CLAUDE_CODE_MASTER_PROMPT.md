# MASTER IMPLEMENTATION PROMPT — SEGPPD-101 TRAINING-FREE CBIR RESEARCH SYSTEM

You are Claude Code acting as a senior computer vision researcher, ML systems engineer, data engineer, research software engineer, and skeptical academic reviewer. **Execute this project end to end in the current working environment**, not merely propose a plan or write placeholder code. Inspect the working directory first, preserve existing user work, implement the system, download and audit the dataset, extract features, run retrieval evaluations, write the reports, and provide verifiable results.

## 0. Mission and non-negotiable assignment constraints

**Working title:** *Evaluating Frozen ResNet50 for Content-Based Retrieval of In-the-Wild Plant Pest and Disease Images: A Lesion-Scale Analysis on SegPPD-101*.

**Real research question:** When pest/disease evidence occupies only a small part of a real outdoor photograph, how well can a training-free, whole-image ResNet50–Euclidean CBIR system retrieve previously labeled examples of the same category? How does accuracy vary with target-region occupancy and retrieval depth K, and what failure modes suggest reliance on crop species or background instead of symptoms?

**Instructor-mandated method:** multiclass labeled image dataset -> pretrained **ResNet50** frozen deep feature extraction -> **Euclidean (L2) distance** ranking -> increasing Top-K retrieval -> retrieval metrics. **Do not train, fine-tune, or update any neural-network weights. Do not implement feature selection, PCA, mRMR, NCA, metric learning, classifiers, learned reranking, or replace ResNet50.** No augmentation that creates artificial evaluation images. Do not call image classification accuracy “retrieval accuracy.”

The scientific contribution is the **experimental analysis**, not a novel neural architecture. Do not claim to be the first plant-disease CBIR paper. A literature search through October 2026 found no direct SegPPD-101 CBIR paper, but absence of indexed evidence is not proof no such work exists.

**Source-of-truth links:**
- Dataset/metadata: https://zenodo.org/records/15114160
- Candidate direct archive: https://zenodo.org/records/15114160/files/SegPPD-101.rar?download=1
- DOI: https://doi.org/10.5281/zenodo.15114160
- Original article: https://pmc.ncbi.nlm.nih.gov/articles/PMC13109344/
- Related code to inspect only if relevant: https://github.com/zihan303/SegPPD-FS

Previously reported (UNVERIFIED UNTIL LOCAL AUDIT): archive `SegPPD-101.rar` roughly 247.2 MB; Zenodo MD5 `2242cf24f0c84a80e15633e049a9c0df`; 2,263 images; 101 pest/disease classes; pixel-level LabelMe JSON / semantic masks; 27 versus 21 crop types inconsistently reported; ~43.5% of images have target occupancy <=20%, ~78.3% <=40%. Verify locally rather than assuming archive layout or machine-readable class counts. The Zenodo license field did not display a clear dataset license; do not infer dataset redistribution rights from the article's license. Do not publish or rehost raw images or masks without confirmed permission.

## 1. Autonomous execution agreement

1. First inspect `pwd`, `git status`, filesystem, disk space, Python/uv, OS, hardware, and CUDA/CPU capabilities. If inside an existing project, integrate non-destructively; otherwise make a clearly named `segppd-cbir-research/` project directory. Never overwrite or delete unrelated work. No `sudo`, no global installs, no pushes, no uploads of dataset, no publishing of copyrighted images, no destructive actions. Ask permission only for genuinely required privileged/external action.
2. Write a short `docs/IMPLEMENTATION_PLAN.md` and begin execution immediately. Do **not** stop after planning. Proceed autonomously through data audit, implementation, feature extraction, experiments, visualization, and reporting if resources allow. Track status as `DONE`, `PARTIAL`, `BLOCKED`, with actual evidence.
3. Use Python 3.12 if available, otherwise a supported installed Python; prefer project-local `uv` environment and `pyproject.toml` with reproducible locked dependencies. Core dependencies: PyTorch, torchvision, Pillow, NumPy, pandas, scikit-learn, scipy, matplotlib, tqdm, pydantic or dataclasses, pytest; FAISS CPU if compatible, otherwise exact vectorized NumPy/Torch Euclidean ranking with an explicit equivalence test. Optional: `streamlit`, `imagehash`. Be dependency-minimal and version-aware.
4. Log command invocations, package versions, runtime, hardware, random seeds, input hashes and model-weight identifier; never make up timings or scores. If internet/download/model-weight retrieval is blocked, retry through official channels with bounded retries, document the actual error, and stop the dependent stages clearly. Do not silently switch to a synthetic dataset or a randomly initialized model. Do not report mock/dummy results as empirical results.
5. Prioritize correctness and finished science before UI polish. Do not use secret API keys, third-party paid services, an LLM classifier, or an external diagnosis API.

## 2. Repository structure (adapt intelligently to any existing repository)

Aim for something close to:

```text
segppd-cbir-research/
  pyproject.toml
  uv.lock                          # if uv is available
  README.md
  .gitignore
  configs/
    baseline.yaml
  data/
    raw/                         # ignored by Git
    extracted/                   # ignored by Git
    processed/                   # ignored by Git
  artifacts/
    features/                    # ignored if large; manifest preserved
    indices/
    splits/
    audits/
    figures/
    metrics/
    retrieval_examples/
  docs/
    IMPLEMENTATION_PLAN.md
    DATASET_AUDIT.md
    DATA_RIGHTS.md
    RESEARCH_PROTOCOL.md
    ARCHITECTURE.md
    RESULTS.md
    LIMITATIONS.md
    RUN_LOG.md
    FINAL_HANDOFF.md
  src/segppd_cbir/
    __init__.py
    cli.py
    download.py
    extract.py
    dataset_audit.py
    labels.py
    masks.py
    duplicate_groups.py
    split.py
    feature_extractor.py
    index.py
    metrics.py
    evaluate.py
    lesion_analysis.py
    roi_experiment.py
    visualize.py
    app.py
    reproducibility.py
  tests/
    test_dataset_parsing.py
    test_mask_area.py
    test_duplicate_groups.py
    test_split_leakage.py
    test_embeddings.py
    test_l2_ranking.py
    test_metrics.py
    test_class_eligibility.py
```

Commands should be documented and executable via `uv run python -m segppd_cbir.cli ...` (or equivalent). Include a sequential `run-all` path that resumes completed steps. Do not require manual notebook cell execution to reproduce results.

## 3. Phase A — official dataset acquisition and verification (hard gate)

- Read the Zenodo REST record metadata (e.g., `https://zenodo.org/api/records/15114160`) and/or official page. Record the actual file names, sizes, checksums, DOI, published version, available rights/license fields, archive URLs. Do not blindly trust the previously quoted checksum; compare against current official metadata.
- Download the complete official archive to `data/raw/` using a resumable approach (`curl -L --retry ... -C -` or an equivalent robust Python downloader); handle HTTP failures, content-length and redirects, verify expected compressed size and MD5/SHA when official checksums exist. Maintain a manifest.
- Safely extract `.rar` using an already installed extractor (`7z`, `unar`, `bsdtar`, or a supported Python package); do not silently install privileged packages. Detect archive-path traversal, symlinks, archive bombs, unexpected extraction paths; preserve originals, report actual layout. The compressed file might contain nested directories; discover, do not guess.
- Inventory every image and annotation. Verify images can actually be decoded, orientation handled, dimensions, channels, color format, filename collisions, missing/extra masks, malformed LabelMe JSON, bad polygons, label inconsistency, and class-folder mappings. Preserve original labels and add normalized IDs without rewriting the raw data.
- Derive a reliable CSV/Parquet manifest with `image_id`, relative path, raw class label, normalized class label, available crop label, dimensions, hash, mask/JSON path, mask pixel count, total pixel count, occupancy, source metadata if available, and validation flags. **Never infer crop names, disease labels, or mask alignment from ambiguous filenames without confirming the actual schema.** If class labels are multi-label, record this and design honest handling; do not force arbitrary classes.
- Render at least 20 actual dataset examples as a contact sheet, covering different classes, mask sizes, annotation types, and images with problems. Confirm annotation overlays by aligning each mask to the actual original image dimensions. Do not use illustrative internet images as proof of archive contents.
- Output `docs/DATASET_AUDIT.md`, `artifacts/audits/dataset_manifest.csv`, `class_counts.csv`, `mask_integrity.csv`, `file_inventory.csv`, `contact_sheet.png`, plus human-readable warnings. Resolve or transparently document the published 21-vs-27-crop discrepancy from the files if possible.
- Dataset acceptance gate: a meaningful number of images must have reliable class labels; query/gallery evaluation must be feasible; annotation occupancy must be reliable for the lesion-scale analysis. If masks fail but labels pass, run the baseline and explicitly mark lesion analysis BLOCKED. If class labels fail, STOP before any claimed evaluation. If counts differ from 2,263/101, report actual values and investigate; do not massage them to match a paper.
- Examine redistribution terms. Classroom local evaluation can proceed when legitimately accessible, but avoid publishing raw imagery; clearly flag unclear rights for public artifacts.

## 4. Phase B — leakage/duplicate audit and evaluation split (hard gate)

Important: 2,263 images over 101 classes averages only ~22.4 images per class; some categories may be too small for reliable Top-20. Never force all classes into an unsupported protocol. The dataset aggregates internet/public sources and may include reused or near-identical pictures.

- Calculate exact-byte SHA-256 and a documented perceptual hash (pHash/dHash) to identify exact/near duplicates, optionally assisted by visual review. Similar-looking-but-distinct diseased leaves are not automatically duplicates. Treat uncertain near-duplicate decisions as reviewable. Flag duplicates with conflicting class labels. Keep audit evidence.
- Build conservative provenance/duplicate groups. If source/plant/scene IDs exist, preserve those as groups; otherwise use validated near-duplicate clusters. All members of a duplicate/source group must be assigned entirely to either query or gallery. Never claim guaranteed geographic or temporal independence if the data has no field/date metadata.
- Produce deterministic query/gallery splits with configuration and seed 42. **There is no training split** because the backbone is frozen and no learning is performed. Keep any reference/model weight cache wholly separate from query/gallery sample allocation.
- Before splits, choose fixed, published eligibility rules and record coverage. Proposed *primary Top-10 cohort*: a class is eligible only if it can retain >=3 independent query images and >=10 positive gallery images after leakage prevention; put exactly 3 query images per eligible class, with all eligible remaining nonduplicate images in the gallery. If duplicates/source grouping prevents this, adapt transparently at class level but do not silently alter the predeclared rules. Gallery also contains eligible negatives from other classes.
- *Secondary coverage cohort*: include otherwise ineligible classes only when at least one independent query and one relevant gallery image exist. Report this separately, with class support and clear limitations. Unscorable classes are listed, not erased. Never conflate different cohorts in one aggregate headline.
- Optionally repeat query-group sampling with 3 fixed seeds if sample support permits; never tune seeds based on best results. Persist split manifests and audit checks that there is no query/gallery path/image/group overlap.
- Freeze cohort definitions and metadata **before looking at retrieval scores**. Report number of original classes, eligible classes, eligible images, query images, gallery images, duplicates excluded, and class frequency distribution. Run class-balanced macro metrics to prevent dominant categories from masking failures.
- Compute and report random-gallery-ranking baseline, with multiple fixed permutations or exact expected P@K when possible. It must use the **identical gallery, queries and relevance definition**. Do not fabricate a pretrained model comparison outside instructor scope.

## 5. Phase C — canonical, frozen ResNet50 descriptor extraction

Baseline inputs are **whole original RGB images**; ground-truth disease masks are NOT fed into, multiplied with, or used to crop the baseline embeddings. Using the mask to stratify results afterwards is allowed and is the primary scientific experiment.

- Use an authentic torchvision ImageNet-pretrained ResNet50 (`ResNet50_Weights.IMAGENET1K_V2` if available; pin and record exact weight enum, package versions, official source and cache hash). If that weight file is unavailable, transparently use another genuine official ImageNet ResNet50 weight version and record the justified change; do not use random initialization.
- Remove/replace the final classification `fc` with `nn.Identity()` so the output is the **2048-dimensional vector after global average pooling**. Freeze all parameters; `eval()`; `torch.inference_mode()`; assert that embedding dimension is exactly 2048 and no optimizer/backpropagation or weight changes occur.
- Apply the exact weight-associated deterministic inference preprocessing (commonly resize shorter side to 232 for V2 and center-crop 224, ImageNet normalization; **do not hard-code Resize(256) if the chosen weight transform says otherwise**). Resolve EXIF orientation before applying preprocessing; RGB conversion and deterministic image handling. Record actual preprocessing transforms.
- Use GPU CUDA if available, else MPS if available, else CPU. For GPU memory, conservative initial batch of 16–32, automatic OOM batch backoff, no aggressive resource consumption, progress bars and corrupt-file handling; produce one deterministic embedding per valid image. Prefer fp32 features and exact fp32 L2 rankings; optional autocast only if explicitly documented and validated for ranking stability.
- Cache features by image-content hash + checkpoint fingerprint + transform config. Ensure query/gallery features are identical in extraction semantics and cached results cannot be mistaken for other model versions. Verify non-finite vectors, all-zero rows, duplicate cache keys, and dimensionality.
- **Primary variant:** raw, unnormalized 2048-D descriptors with Euclidean distance. Do not secretly normalize them. If and only if the instructor's interpretation allows, optionally run an explicitly labeled *L2-normalized embedding + Euclidean* sensitivity check; no feature selection or learned transform, and never replace the mandated raw baseline with it.
- Save fp32 features `.npy`/`.npz`, manifest/IDs, model metadata and `artifacts/features/feature_report.json`. Store checkpoint metadata; no training logs, epochs, losses or GPU training claims.

## 6. Phase D — exact Euclidean nearest-neighbor retrieval

- Use `faiss.IndexFlatL2` on gallery-only descriptors when supported; otherwise NumPy/Torch exact vectorized pairwise squared-Euclidean calculation. No approximate HNSW, IVF, cosine similarity, classification head or learned reranker. FAISS IndexFlatL2 returns squared L2 distances; the ordering is exactly the same as Euclidean distance. If showing actual Euclidean distance, take the square root; clearly distinguish squared values in saved outputs.
- Explicit deterministic tie handling (`distance`, then stable `image_id`), gallery membership assertions, no self-match and no query leakage. Return Top-K image ID/path/actual class/Euclidean distance/relevance flag; top-K must contain distinct images.
- Evaluate **K = [1, 3, 5, 10, 20, 50, 100]** whenever `K <= gallery size`; `K = [1, 5, 10]` are primary; `20, 50, 100` are exploratory due to positive-gallery ceilings and class sparsity. Do not declare high-K precision a failure without acknowledging the maximum attainable relevant count. Save an empirical/class-specific precision ceiling `min(R_q, K)/K` and relevant gallery count `R_q` for every query. If K exceeds total gallery size, report unsupported rather than silently inventing results.
- Relevance for the main study is **exact confirmed dataset category equality** (same crop/pest/disease class as defined by original labels). This is a proxy for case-label agreement, **not verified lesion visual similarity and not proof of correct clinical/agronomic diagnosis**. Clearly state this throughout.

## 7. Phase E — retrieval metrics: correct mathematics and denominators

Implement from scratch with unit tests, and cross-check against trusted reference implementations where appropriate.

For each query q, let R_q be the number of relevant images actually in the gallery; for K, let rel_i in {0,1} indicate the label-relevance of the i-th result.

- `Precision@K = (sum_{i=1..K} rel_i) / K` (for supported K and full K results).
- `Recall@K = (sum_{i=1..K} rel_i) / R_q` when R_q > 0.
- `Hit@K = 1 if any of Top-K is relevant, else 0` (useful for small-support categories; do not mislabel as classification accuracy).
- Define truncated `AP@K` explicitly: sum over ranks i<=K of `P@i * rel_i`, divided by `min(R_q, K)` for the declared common AP@K convention; additionally implement **full-list AP** with denominator `R_q` and report macro mean AP across eligible queries. Do not switch conventions unnoticed. Mark undefined for R_q=0 and exclude from scored aggregates while reporting counts.
- Macro average across classes (classwise means averaged equally) as the primary headline; micro/query-weighted metrics as supplementary. Per-class distributions, worst classes and count of participating classes. Show class-supported precision ceilings at K.
- Bootstrap confidence intervals grouped at the class level or hierarchical class->query, using fixed seed; use paired uncertainty estimates for differences between the same-query comparison groups. Descriptive association != causation.
- Report retrieval computation latency excluding image decoding/model extraction, plus separate preprocessing/feature-extraction timing and optional full query-to-result latency. State hardware, warm-up protocol, number of queries, p50/p95, and what was timed; do not report invented speedups.
- Output CSV and JSON tables, publication-readable figures (P@K, R@K, mAP@K curves, per-class, query-count coverage), and a mathematical definitions section in `docs/RESEARCH_PROTOCOL.md`.

## 8. Phase F — PRIMARY research experiment: lesion-area sensitivity

The dataset's annotated disease/pest mask is used **only to compute target occupancy and analyze the retrieval results after ranking**; the baseline does not receive a mask. Compute `occupancy = count(non-background target pixels) / (image_height * image_width)` from VALIDATED aligned masks. If masks are polygons, rasterize carefully using original image dimensions, correct overlapping regions, include all target polygons, and check equivalence against provided masks where both exist. Document excluded invalid masks.

Preregister bins:
- `small`: 0 < occupancy <= 0.20
- `medium`: 0.20 < occupancy <= 0.40
- `large`: occupancy > 0.40

Zero/empty or malformed masks are `unknown/invalid`, never silently placed in `small`. Compute cohort participation in each bin; metrics P@1, P@5, P@10, Recall@K, AP/mAP@K and Hit@K, plus confidence intervals. Plot results by K and occupancy with **n queries, n classes and relevant-gallery support** visible.

**Critical confounding rule:** occupancy and class/crop may be correlated. Therefore produce (a) pooled-by-query descriptive analysis and (b) class-balanced/within-class contrasts only for classes represented in multiple occupancy bins with enough queries; report which classes permit comparison. Do not interpret pooled bin differences as a causal effect of lesion area. Also calculate correlations/stratified summaries involving available crop metadata and image resolution; no unvalidated crop inferred from class naming.

Analyze how retrieval depth affects scarce positives: P@20/50/100 can be mathematically capped. Clearly separate actual ranking quality from the dataset's number of positives. Report if the small-bin hypothesis is **not** supported; never force a “small lesions perform worse” conclusion.

Produce at least 12 actual examples of successful/failing retrieval results, selected by transparent criteria (high P@10, low P@10, rare classes, small/large lesion); annotate true category, top results, match/mismatch and crop information if validated. Avoid claiming examples show model attention/causation: nearest neighbors alone do not reveal which pixels the network uses.

## 9. Phase G — SECONDARY diagnostic experiment (only after baseline is complete)

Question: Is full-image background dilution a plausible explanation for poor small-target retrieval?

If and only if mask/JSON localization is validated, create **oracle disease-region bounding-box crops** from provided annotations (union of all validated target pixels, with a small, documented contextual margin). Pass these crops through the **same fixed ResNet50 and weight-associated preprocessing**, then Euclidean retrieval with the **same gallery and query IDs and split**. No training or feature selection. Separately report `WHOLE_IMAGE` versus `ORACLE_ROI_CROP`, using paired metrics and uncertainty.

**Absolutely essential interpretation:** using ground-truth masks for query and gallery localization is **privileged information unavailable for arbitrary user-uploaded field photos**. This secondary condition is an *oracle diagnostic upper-information scenario*, not an operationally deployable baseline. It is not necessarily an upper bound on actual performance; cropping may hurt. If masks are imperfect or crop methods are incomparable, stop and report. Never combine whole-image and oracle ROI results into one headline or label the oracle performance a production-ready system.

If there is time, an optional realistic demo may allow the user to select a rectangle manually for a query and compare against gallery crops; mark it clearly separate from academic experiments. Do not train a detector/segmenter to automate ROI, since instructor prohibits training.

## 10. Phase H — light local CBIR demonstrator

After quantitative science is working, implement a modest, functional local Streamlit app or similarly simple interface:
- Select an actual dataset query image OR upload a local photo.
- Display its nearest neighbor images with Euclidean distance, source file/label and ranking; explain labels are known only for dataset entries and uploaded images are not automatically diagnosed.
- Adjustable K and optional class/crop metadata filtering **only if this is clearly distinguished from the default unrestricted experimental ranking**.
- Show reference crops/mask overlay where legally and technically permissible; never publish raw copyrighted images to an external service.
- If oracle ROI mode exists, visibly label it as using research ground-truth annotations and disable that option for unknown user-uploaded photos without annotations.
- Responsive UI is welcome but must not displace science/testing. Ensure the app uses exactly the previously saved feature/index pipeline rather than reimplementing a different similarity metric.

## 11. Mandatory quality checks and tests

Minimum tests must cover:
- Image/label/annotation pairing and explicit failure on ambiguous classes.
- Polygon rasterization or binary mask conversion yields expected occupancy for simple synthetic shapes, not dummy performance data.
- Stable ID and duplicate-group construction; no group appears on both sides of a split; exact duplicates never create self-matches.
- ResNet50 is frozen, no gradients/optimizer, deterministic inference and 2048-D outputs.
- Distances rank known 2-D toy points correctly, FAISS squared L2 vs NumPy true L2 ordering equivalence, stable tie handling.
- Hand-calculated P@K, R@K, Hit@K, full AP and truncated AP@K on small known relevance arrays, including sparse positives and R=0.
- Class eligibility, negative-gallery inclusion, high-K precision upper bound, macro vs micro aggregation, lesion-bin boundaries and missing masks.
- Re-running the same configuration reproduces splits, feature IDs and ranking order within declared numeric tolerance.

Run tests in the actual environment. Fix failures before claiming completion. A project that only contains source files without downloading the data and generating actual evaluation metrics is **not complete**.

## 12. Deliverables and final report: no fabricated outcomes

Write these with evidence and actual generated artifacts:

- `README.md`: installation, environment, official dataset source, complete minimal commands, reproducibility and expected outputs.
- `docs/ARCHITECTURE.md`: baseline dataflow and why **no training**, 2048-D ResNet50 descriptor, exact Euclidean retrieval, indexing complexity, mask exclusion from baseline.
- `docs/RESEARCH_PROTOCOL.md`: research hypothesis, relevance, full metrics formulas, split and eligibility rules, confidence intervals, class-macro, positive-count ceiling, preregistered lesion bins, oracle-ROI restrictions.
- `docs/DATASET_AUDIT.md`: verified structure/counts, labels, class support, crop discrepancy, annotations, duplicates, rights/license uncertainties, excluded data, contact sheet.
- `docs/RESULTS.md`: actual numbers, hardware and runtimes, P@K/Recall@K/mAP curves and tables, per-class outcomes, lesion occupancy groups, statistical uncertainty, baseline random comparison, independent oracle ROI results if available, qualitative retrieval examples. For each result, say exactly how it was measured.
- `docs/LIMITATIONS.md`: small classes, label vs visual similarity proxy, source/near-duplicate leakage limits, no field/date IDs, correlation confounding, oracle annotations unavailable for fresh photos, out-of-distribution performance unknown, no agronomic diagnosis validation, data rights.
- `docs/RUN_LOG.md`: exact commands, outputs, exit codes, package versions, observed errors, repairs, and resumed steps.
- `docs/FINAL_HANDOFF.md`: `DONE/PARTIAL/BLOCKED` status for all phases, number of usable images/classes/query/gallery samples, result table and artifact paths, runnable next command if interrupted, safe next steps, and a concise Turkish summary of the research findings for the student's professor.
- Actual `.csv/.json/.png` artifacts and cached features/index. Use relative paths and clear manifests. Do not commit large dataset or weight binaries. Provide sensible `.gitignore`.

## 13. Practical execution and stopping policy

Execute in order:

`environment check -> official download/checksum -> safe extraction -> schema discovery + masks -> duplicate audit -> leakage-safe split -> frozen ResNet50 features -> exact Euclidean retrieval -> full metric suite -> lesion-scale analysis -> (optional) oracle ROI -> tests + results -> optional local demo -> final handoff`.

You may parallelize independent CPU work safely but **do not** launch unnecessary GPU training; this is an **inference-only** pipeline. If runtime is limited, first finish the complete whole-image baseline and at least K=[1,5,10,20] across the valid primary cohort, then the lesion analysis. The oracle experiment and demo are lower priority. Resumability and output correctness take precedence over adding architectural complexity.

At the end, tell me truthfully, with artifacts and commands:
1. Did the full SegPPD-101 archive download and pass integrity checks?
2. What are the *actual* counts per category, mask/label integrity and verified crop types?
3. Which primary/secondary cohorts were possible and how many samples were excluded?
4. Did ImageNet-pretrained **frozen** ResNet50 extraction complete with 2048-D embeddings?
5. Does the exact **Euclidean** retrieval work with no training or feature selection?
6. What actual P@K, R@K, AP/mAP results were produced, with which K and which denominators?
7. Is there evidence that small target occupancy correlates with worse retrieval? Include uncertainty/confounding and contrary findings.
8. What did the oracle ROI experiment show, if performed, and what are its deployment limitations?
9. What tests passed/failed and where are the figures, results and reproducibility instructions?
10. What exact command should I run next to reproduce or continue the work?

**Start now. Inspect the environment, create the project plan, download the official archive, and perform the real dataset audit immediately. Do not stop at a design document, and do not fabricate any result if a resource is unavailable.**
