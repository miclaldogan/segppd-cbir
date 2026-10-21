# Final handoff (status as of 2026-10-20)

| Phase | Status | Evidence |
|---|---|---|
| A. Download + integrity + safe extraction + audit | DONE | MD5 match; `docs/DATASET_AUDIT.md`, `artifacts/audits/` |
| B. Duplicate audit + Protocol A split | DONE | 43 excluded, 75 primary / 26 secondary classes |
| B2. Protocol A audit + leakage-controlled Protocol B | DONE | `protocol_audit.json`, `tests/test_protocols.py` |
| C-F. Frozen ResNet50 (1000-D, 2048-D), exact L2, metrics, FAISS check | DONE | `docs/EXPERIMENT_RESULTS.md` s1 |
| G. Occupancy analysis; oracle ROI | DONE (hypothesis not supported) | s2 |
| DINOv2 frozen / head / partial fine-tune; ArcFace vs Proxy-Anchor vs SupCon; seeds | DONE | s3 |
| Complementary errors + hybrid | DONE (no significant gain) | s3-s4 |
| Locked Protocol B evaluation (one shot) | DONE - **locked set now consumed** | `artifacts/metrics/protocolB_locked/` |
| Second locked set, Protocol C (142 queries from gallery_B) | DONE - one shot, now consumed | `artifacts/metrics/protocolC_locked/`, `artifacts/audits/protocolC_audit.json` |
| Precision-Recall curves (B and C) | DONE | `artifacts/figures/protocol{B,C}_PR_curve.png` |
| Hit@1 >= 0.80 | **PARTIAL**: B 0.807 [0.756, 0.856]; C 0.767 [0.689, 0.840]; pooled B+C (367 q) 0.794 [0.741, 0.844] | `artifacts/figures/protocolBC_hit1.png` |
| External validation (PlantSeg, CC BY-NC-ND), leakage-screened, one shot | DONE: headline Hit@1 0.608 on 11 mapped classes (1,485 queries); 867 PlantSeg images found to overlap SegPPD and removed | `artifacts/metrics/external_plantseg/`, `artifacts/figures/external_plantseg_hit1.png` |
| Streamlit demo | NOT DONE (lower priority) | |
| Tests | 35 passed (`uv run pytest -q`) | |

## Usable data
2,263 images / 101 classes; 2,220 usable after 43 exclusions. Protocol B: 75 classes; train 772, gallery 809, dev queries 150, locked test queries 225. Protocol C: 142 queries carved from gallery_B (gallery_C 667).

## Reproduce
```
uv sync                                          # CPU env; GPU env: see docs/RESUME.md (.venv-gpu)
uv run python -m segppd_cbir.cli audit && uv run python -m segppd_cbir.cli split
uv run python -m segppd_cbir.cli extract && uv run python -m segppd_cbir.cli evaluate
.venv-gpu/bin/python -m segppd_cbir.train_embed --backbone dinov2s --variant base --mode partial --lr_bb 3e-5 --loss supcon --seed 0 --tag D5_dinoB_p4_supcon_lr3e-5_s0
uv run python -m segppd_cbir.final_locked artifacts/metrics/locked_candidates.json   # rerunning only re-evaluates; do NOT use it to select models
```
## Safe next steps
Any further model selection needs a NEW locked set (B and C are consumed) or the external set; DINOv2-B with ArcFace/Proxy-Anchor for a fair loss comparison on B; more queries per class (merge Protocol B train/gallery roles in a new split); optional demo (must use the saved embeddings + exact L2).

## Ozet (Turkce, hocaya)
SegPPD-101 (2.263 gorsel, 101 sinif) uzerinde, egitimsiz dondurulmus ResNet50 + tam Oklid mesafesiyle yapilan CBIR'de (hocanin yontemi) Hit@1 yaklasik %32 (2048-D) ve %26 (1000-D logit) olcuyoruz; rastgele P@1 ~%1. Kaynak veride ayni fotografin farkli etiketle gectigi 16 grup (32 gorsel) bulundu ve cikarildi. Hedef bolge buyuklugu (kucuk/orta/buyuk) ile retrieval basarisi arasinda anlamli bir iliski bulamadik; maske kirpma (oracle ROI) da neredeyse hic fark yaratmadi. Egitimli uzantilarda (hocanin kuralinin disinda, ayri raporlandi) DINOv2-B'nin son 4 blogu SupCon kaybiyla ince ayarlandiginda, egitim/galeri/sorgu gorselleri birbirinden ayrik ve daha once hic kullanilmamis 225 test sorgusunda Hit@1 ortalama %80,7 (95% GA %75-86), 3 tohumun ayri degerleri %81,3/%79,6/%81,3. Ayni protokolde ince ayarli ResNet50 %51'de kaldi. Nokta tahmini %80'i geciyor ama guven araligi alti da iceriyor; ek olarak bu veri kumesi kilitli test setini tuketti. ResNet50 ile birlestirme istatistiksel olarak anlamli kazanc vermedi. Ikinci, bagimsiz kilitli sorgu setinde (142 sorgu, hic egitimde gorulmemis gorseller) ayni sistem %76,7 (GA %69-84) verdi; iki set birlestirildiginde (367 sorgu) %79,4 (GA %74-84). Yani %80 hedefi dogrulanmadi: durust ifade "Hit@1 yaklasik %79-80"; ResNet50 sistemlerine (%33 ve %51) uzaklik ise iki sette de ayni ve cok buyuk. Precision-Recall egrileri (B ve C) `artifacts/figures/`'ta. Harici dogrulama (PlantSeg, SegPPD ile ayni 11 sinif, sizinti taramasindan gecmis 1.485 sorgu): ayni sistem %60,8 (ResNet50 %14,9, donmus DINOv2-B %48,4). Siralama korunuyor ama mutlak basari alan degisiminde yaklasik 17 puan dusuyor; harici sette %80 yok. PlantSeg ile SegPPD'nin ortak kaynak kullandigi (867 gorsel kopya/kirpim) tespit edildi ve cikarildi.
