# Representations

**Mandated baseline (training-free).** torchvision ResNet50 `IMAGENET1K_V2` (weights file sha256 `11ad3fa6...`), frozen, eval, `inference_mode`; weight transform (resize 232, center-crop 224, ImageNet norm). One forward pass gives
A = 1000-D raw fc logits (pre-softmax; ImageNet class-evidence scores, not generic features) and B = 2048-D global-average-pooled features. fp32, raw (unnormalised), exact L2. Extraction 92 s for 2,263 images on CPU; weights verified unchanged (state-dict fingerprint). `artifacts/features/feature_report.json`.

**Extensions (not training-free; reported separately).**
* Multi-block/multi-scale ResNet50 features, hand-crafted descriptors (HSV/LBP/HOG/GLCM/Gabor), and hand-crafted *maps rendered as images* fed to ResNet50 (LBP, Gabor, Lab, HSV, gradient, vegetation indices, CLAHE): exploratory on a dev split; hand-crafted vectors and map branches (concat or attention) did **not** help (`hand_*_dev.csv`, `handimg_dev.csv`).
* DINOv2 (timm `vit_small/base_patch14_dinov2.lvd142m`, official LVD-142M weights): frozen CLS / mean-patch / GeM at 224 and 336 px; CLS@224 best.
* Trained embeddings (Protocol B, `train_B` only): last 4 transformer blocks + final norm unfrozen (layer-wise lr decay 0.8, base lr 3e-5, head lr 1e-3, AdamW wd 1e-4) + Linear(d->256) head, L2-normalised 256-D embedding; class-balanced P=16 x K=4 batches; RandomResizedCrop(0.4-1)/flip/vflip(0.2)/ColorJitter; bf16 autocast; early stopping on dev mAP (patience 10, best-epoch weights restored); embeddings = mean of image + horizontal flip. Losses: ArcFace (s=30,m=0.3), Proxy-Anchor (alpha=32, delta=0.1), SupCon (tau=0.1).
* Fusion: concat(sqrt(w)*Za, sqrt(1-w)*Zb) of L2-normalised embeddings => squared L2 = w*d_a^2+(1-w)*d_b^2 (still exact Euclidean); weights chosen on dev.
