"""Checks numbers quoted in the report text against the saved artifacts (no model run). Prints PASS/FAIL per claim."""
import json
import numpy as np
import pandas as pd

R = []
def chk(name, claim, actual, tol=0.0006):
    ok = abs(claim - actual) <= tol
    R.append((ok, name, claim, round(float(actual), 4)))

L = "artifacts/metrics/protocolB_locked/"
ci = pd.read_csv(L + "macro_ci.csv"); cl = pd.read_csv(L + "classification_roc_metrics.csv").set_index("rep")
v = lambda rep, m, K=1, col="macro_mean": float(ci[(ci.rep == rep) & (ci.K == K) & (ci.metric == m)][col].iloc[0])
sd = lambda p: [f"{p}_s{i}" for i in range(3)]
mean = lambda reps, m, K=1: np.mean([v(r, m, K) for r in reps])
# Protocol B table / narrative
chk("B resnet logits hit1", .262, v("BASE_resnet50_frozen_logits1000_raw", "P")); chk("B resnet pooled hit1", .320, v("BASE_resnet50_frozen_pooled2048_raw", "P"))
chk("B R1 hit1", .511, v("R1_resnet50_ft_arcface_existing_system", "P")); chk("B dinoS frozen", .538, v("F_dinov2s_frozen_cls", "P")); chk("B dinoB frozen", .627, v("F_dinov2b_frozen_cls", "P"))
chk("B D2 arcface", .760, v("D2_dinoS_p4_arcface_s0", "P")); chk("B D3 PA mean", .748, mean(sd("D3_dinoS_p4_proxyanchor"), "P")); chk("B D4 SupCon mean", .717, mean(sd("D4_dinoS_p4_supcon"), "P"))
chk("B D4 P@10", .591, mean(sd("D4_dinoS_p4_supcon"), "P", 10)); chk("B D4 mAP", .658, mean(sd("D4_dinoS_p4_supcon"), "full_AP", 10))
h = sd("D5_dinoB_p4_supcon"); chk("B headline", .807, mean(h, "P")); [chk(f"B seed{i}", x, v(h[i], "P")) for i, x in enumerate((.813, .796, .813))]
chk("B headline P@10", .651, mean(h, "P", 10)); chk("B headline Hit@10", .967, mean(h, "Hit", 10)); chk("B headline mAP", .742, mean(h, "full_AP", 10))
e = "E_dinoB_supcon_3seed_ensemble"; chk("ensemble", .822, v(e, "P")); chk("ensemble lo", .769, v(e, "P", 1, "ci_lo")); chk("ensemble hi", .876, v(e, "P", 1, "ci_hi"))
chk("fuse resnet", .840, v("FUSE_dinoBens0.9_resnetft0.1_(dev-selected)", "P"))
chk("AUC resnet pooled", .765, cl.loc["BASE_resnet50_frozen_pooled2048_raw", "roc_auc_pairwise"]); chk("AUC R1", .914, cl.loc["R1_resnet50_ft_arcface_existing_system", "roc_auc_pairwise"])
chk("AUC dinoB frozen", .959, cl.loc["F_dinov2b_frozen_cls", "roc_auc_pairwise"]); chk("AUC headline", .989, np.mean([cl.loc[r, "roc_auc_pairwise"] for r in h])); chk("F1 headline", .794, np.mean([cl.loc[r, "macro_f1_1nn"] for r in h]))
pdf = pd.read_csv(L + "paired_differences.csv"); r = pdf[(pdf.A.str.startswith("FUSE_dinoBens0.9")) & (pdf.K == 1) & (pdf.metric == "P")].iloc[0]
chk("fusion diff", .018, r.mean_diff_A_minus_B, .001); chk("fusion lo", -.004, r.ci_lo, .001); chk("fusion hi", .044, r.ci_hi, .001)
# B / C / pooled / external
s = pd.read_csv("artifacts/metrics/protocolBC_hit1_summary.csv")
g = lambda st, col: float(s[(s.set.str.startswith(st)) & (s.rep == "HEADLINE")][col].iloc[0])
for st, vals in (("B (", (.807, .756, .856)), ("C (", (.767, .689, .840)), ("B+C", (.794, .741, .844))):
    for col, x in zip(("hit1", "ci_lo", "ci_hi"), vals): chk(f"{st} {col}", x, g(st, col))
ex = pd.read_csv("artifacts/metrics/external_plantseg/summary_ci.csv"); e1 = lambda m, c: float(ex[(ex.rep == "HEADLINE_D5ck_mean3seeds") & (ex.tier == "strict") & (ex.metric == m)][c].iloc[0])
chk("ext hit1", .608, e1("hit1", "macro_mean")); chk("ext lo", .466, e1("hit1", "ci_class_lo")); chk("ext hi", .760, e1("hit1", "ci_class_hi")); chk("ext p10", .536, e1("p10", "macro_mean")); chk("ext hit10", .869, e1("hit10", "macro_mean"))
pc = pd.read_csv("artifacts/metrics/external_plantseg/per_class_in_vs_external.csv"); chk("in-domain 11 classes", .778, pc.B_in_domain.mean())
for rep, x in (("BASE_resnet50_frozen_pooled2048_raw", .149), ("F_dinov2b_frozen_cls", .484)): chk("ext " + rep, x, e1("hit1", "macro_mean") * 0 + float(ex[(ex.rep == rep) & (ex.tier == "strict") & (ex.metric == "hit1")].macro_mean.iloc[0]))
# Protocol A
a = pd.read_csv("artifacts/metrics/macro_ci.csv"); aa = lambda rep, m, K, c="macro_mean": float(a[(a.rep == rep) & (a.K == K) & (a.metric == m)][c].iloc[0])
chk("A logits hit1", .293, aa("A_logits1000", "P", 1)); chk("A pooled hit1", .391, aa("B_pooled2048", "P", 1)); chk("A logits mAP", .118, aa("A_logits1000", "full_AP", 10), .001); chk("A pooled mAP", .145, aa("B_pooled2048", "full_AP", 10), .001)
chk("A logits P@10", .170, aa("A_logits1000", "P", 10)); chk("A pooled P@10", .212, aa("B_pooled2048", "P", 10))
sn = pd.read_csv("artifacts/metrics/sensitivity_l2norm.csv"); sv = lambda rep: float(sn[(sn.rep == rep) & (sn.K == 1) & (sn.metric == "P")].macro_mean.iloc[0])
chk("L2norm logits", .262, sv("A_logits1000_L2NORM")); chk("L2norm pooled", .373, sv("B_pooled2048_L2NORM"))
# handcrafted (dev)
hb = pd.read_csv("artifacts/metrics/hand_blocks_dev.csv").set_index("block"); chk("hsv mAP", .050, hb.loc["hsv", "l2n_mAP"], .001); chk("hog mAP", .025, hb.loc["hog", "l2n_mAP"], .001)
hi = pd.read_csv("artifacts/metrics/handimg_dev.csv"); cc = hi[hi.exp == "head_RGB+concat"]; at = hi[hi.exp == "head_RGB+attn"]
chk("concat min", .516, cc.mAP.min(), .001); chk("concat max", .557, cc.mAP.max(), .001); chk("attn min", .475, at.mAP.min(), .001); chk("attn max", .524, at.mAP.max(), .001); chk("RGB head mAP", .548, hi[hi.exp == "head_RGB_only"].mAP.iloc[0], .001)
chk("clahe alone mAP", .160, hi[(hi.exp == "variant_alone_raw") & (hi.variant == "clahe")].mAP.iloc[0], .001)
hf = pd.read_csv("artifacts/metrics/hand_fusion_dev.csv"); chk("hand fusion w1 P1", .447, hf[hf.w_hand == 1.0].raw_P1.iloc[0], .001)
ro = pd.read_csv("artifacts/metrics/oracle_roi_minus_whole_paired.csv"); rr = ro[(ro.rep.str.startswith("B_")) & (ro.K == 10) & (ro.metric == "P")]
if len(rr): chk("ROI P@10 diff (2048-D)", .016, rr.mean_diff_A_minus_B.iloc[0], .001)
# dev logs
import glob
def dev(pat):
    xs = [json.load(open(f))["best_dev"] for f in sorted(glob.glob(f"artifacts/extensions/B/{pat}"))]; return {k: np.mean([x[k] for x in xs]) for k in ("mAP", "P1", "P10")}
for nm, pat, vals in (("dev D1 head-only", "D1_*_log.json", (.514, .667, .464)), ("dev D2", "D2_*_log.json", (.597, .720, .546)), ("dev D3", "D3_*_log.json", (.636, .724, .572)), ("dev D4", "D4_*_log.json", (.663, .725, .596)),
                      ("dev D5", "D5_dinoB*_log.json", (.750, .815, .667)), ("dev R1", "R1_*_log.json", (.420, .527, .399))):
    d = dev(pat)
    for k, x in zip(("mAP", "P1", "P10"), vals): chk(f"{nm} {k}", x, d[k], .0012)
# counts
b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str}); rc = b.role_B.value_counts()
for k, x in (("train_B", 772), ("gallery_B", 809), ("dev_query_B", 150), ("test_query_B", 225)): chk("count " + k, x, rc[k], 0)
c = pd.read_csv("artifacts/splits/protocolC_seed777.csv", dtype={"image_id": str}).role_C.value_counts(); chk("count C query", 142, c["test_query_C"], 0); chk("count C gallery", 667, c["gallery_C"], 0)
pa = json.load(open("artifacts/audits/protocol_audit.json"))["protocolA"]; chk("A gallery", 1731, pa["final_train_gallery_n"], 0); chk("A dev gallery", 1581, pa["dev_train_gallery_n"], 0)
m = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}); u = m[m.dup_status != "excluded"].groupby("class_id").size()
chk("usable images", 2220, u.sum(), 0); chk("primary classes", 75, (u >= 13).sum(), 0); chk("primary images", 1956, u[u >= 13].sum(), 0); chk("secondary images", 264, u[u < 13].sum(), 0); chk("class min", 7, u.min(), 0); chk("class max", 67, u.max(), 0); chk("class median", 21, u.median(), 0)
chk("excluded images", 43, (m.dup_status == "excluded").sum(), 0)
ps = json.load(open("artifacts/external/plantseg_leak_screen.json")); chk("flagged", 867, ps["n_flagged"], 0); chk("hash flagged", 229, ps["n_hash"], 0); chk("emb flagged", 862, ps["n_emb"], 0)
idx = pd.read_csv("artifacts/external/plantseg_leak_screen.csv"); st = idx[idx.tier == "strict"]; chk("strict before", 1621, len(st), 0); chk("strict after", 1485, (~st.leak_flag).sum(), 0)
meta = pd.read_csv("data/external/plantseg/plantsegv2/Metadatav2.csv", encoding="utf-8-sig"); chk("plantseg images", 11458, len(meta), 0); chk("plantseg diseases", 114, meta.Disease.nunique(), 0); chk("plantseg plants", 35, meta.Plant.nunique(), 0)
cj = json.load(open("data/external/plantseg/plantsegv2/coco_annotations.json")); chk("coco categories", 117, len(cj["categories"]), 0); chk("coco images", 11480, len(cj["images"]), 0)
# per class (B)
pcl = pd.read_csv(L + "per_class_hit1.csv", index_col=0); fin = pcl[h].mean(axis=1); cnt = lambda x: (x * 3).round().astype(int).value_counts()
chk("final classes 0/3", 1, cnt(fin.round(2)).get(0, 0), 0); chk("final classes 3/3", 40, cnt(fin.round(2)).get(3, 0), 0)
chk("resnet frozen 0/3", 29, cnt(pcl["BASE_resnet50_frozen_pooled2048_raw"]).get(0, 0), 0); chk("resnet frozen 3/3", 4, cnt(pcl["BASE_resnet50_frozen_pooled2048_raw"]).get(3, 0), 0)
# cost
ct = pd.read_csv(L + "cost_table.csv").set_index("run"); chk("params M", 28.6, ct.loc["D5_dinoB_p4_supcon_lr3e-5_s0", "trainable_params_M"], .05); chk("vram", 2.98, ct.loc["D5_dinoB_p4_supcon_lr3e-5_s0", "peak_vram_gb"], .01)
lt = pd.read_csv(L + "retrieval_latency.csv"); chk("latency p50 ms", .26, lt[lt.rep == "D5_dinoB_p4_supcon_s1"].p50_ms.iloc[0], .01)
bad = [r for r in R if not r[0]]
print(f"{len(R) - len(bad)}/{len(R)} claims match saved artifacts")
for r in bad: print("MISMATCH:", r[1:])
open("report/verify_numbers_output.txt", "w").write(f"{len(R) - len(bad)}/{len(R)} claims match\n" + "\n".join(f"{'PASS' if r[0] else 'FAIL'}  {r[1]}: report={r[2]} artifact={r[3]}" for r in R) + "\n")
