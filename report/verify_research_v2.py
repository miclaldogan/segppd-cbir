"""Checks key numbers quoted in docs/*research_v2* documents against the saved artifacts."""
import json
import numpy as np
import pandas as pd

R = []
def chk(name, claim, actual, tol=0.0006): R.append((abs(claim - actual) <= tol, name, claim, round(float(actual), 4)))
A = "artifacts/research_v2/trackA/"; B = "artifacts/research_v2/trackB/"; Cc = "artifacts/research_v2/trackC/"
t = pd.read_csv(A + "final_summary.csv"); g = lambda m, s, c: float(t[(t.model.str.startswith(m)) & (t.set == s)][c].iloc[0])
for m, vals in (("arcface", (.7756, .7926, .7444)), ("proxyanchor", (.8200, .8252, .7822)), ("supcon (Hit", (.8200, .8059, .7711))):
    for s, v in zip(("dev", "testB_exploratory", "testC_exploratory"), vals): chk(f"{m} {s} hit1", v, g(m, s, "macro_hit1"), .0006)
for m, vals in (("arcface", (.7034, .7076, .6961)), ("proxyanchor", (.7572, .7526, .7514)), ("supcon (Hit", (.7474, .7403, .7309))):
    for s, v in zip(("dev", "testB_exploratory", "testC_exploratory"), vals): chk(f"{m} {s} mAP", v, g(m, s, "macro_mAP"), .0006)
p = pd.read_csv(A + "paired_comparisons.csv"); q = lambda a, b, s, c: float(p[(p.a == a) & (p.b == b) & (p.set == s)][c].iloc[0])
chk("PA-SC B hit1", .019, q("proxyanchor", "supcon", "testB_exploratory", "hit1_diff"), .001); chk("AF-SC dev mAP", -.044, q("arcface", "supcon", "dev", "mAP_diff"), .001); chk("AF-SC B mAP hi", -.0138, q("arcface", "supcon", "testB_exploratory", "mAP_hi"), .001)
sel = json.load(open(A + "loss_selection_dev_only.json")); chk("dev PA hit1", .82, sel["per_loss"]["proxyanchor"]["hit1"]); chk("dev SC hit1", .82, sel["per_loss"]["supcon"]["hit1"])
rb = pd.read_csv(B + "results_by_seed.csv"); gm = lambda m, c: rb[rb.model == m][c].mean()
for m, a in (("M1", .794), ("M2", .6883), ("M3", .8471), ("M4", .8522), ("M5", .8471)): chk(f"B {m} acc", a, gm(m, "accuracy"), .001)
chk("B M3 mAP", .834, gm("M3", "mAP"), .001); chk("B M3 p10", .836, gm("M3", "p10"), .001); chk("B M2 hit10", .933, gm("M2", "hit10"), .001)
pb = pd.read_csv(B + "paired_by_seed.csv"); chk("B M3-M4 p seed0", .2461, float(pb[(pb.a == "M3") & (pb.b == "M4") & (pb.seed == 0)].p.iloc[0]), .001); chk("B M3-M5 p seed2", .0433, float(pb[(pb.a == "M3") & (pb.b == "M5") & (pb.seed == 2)].p.iloc[0]), .001)
P = pd.read_csv(B + "protocol_P.csv"); chk("P train", 6744, (P.role == "train").sum(), 0); chk("P dev", 899, (P.role == "dev").sum(), 0); chk("P test", 1646, (P.role == "test").sum(), 0); chk("P classes", 98, P.Disease.nunique(), 0)
au = json.load(open(B + "audit_summary.json")); chk("dup pairs", 1773, au["dup_pairs"], 0); chk("groups span splits", 575, au["groups_spanning_splits"], 0); chk("groups span classes", 256, au["groups_spanning_classes"], 0); chk("test url share", 728, au["test_images_sharing_url_with_train"], 0)
ab = pd.read_csv(Cc + "aug_ablation_summary.csv").set_index("variant")
for v, a in (("base", .6175), ("color", .622), ("robust", .6192), ("res", .6001), ("geo", .5806)): chk(f"aug {v} ext", a, ab.loc[v, "ext_macro_hit1_mean"], .001)
ds = json.load(open(Cc + "diag_summary.json")); chk("ext macro", .6081, ds["external_hit1_class_macro"], .001); chk("ext micro", .5403, ds["external_hit1_seed_mean"], .001); chk("organ leaf", .675, ds["organ_hit1"]["L"], .001); chk("organ fruit", .246, ds["organ_hit1"]["F"], .001)
rp = pd.read_csv(Cc + "local_rerank_paired.csv"); r = lambda s, M, a, c: float(rp[(rp.set == s) & (rp.M == M) & (rp.alpha == a)][c].iloc[0])
chk("rerank B diff", .0252, r("testB_exploratory", 20, .5, "hit1_diff"), .001); chk("rerank B lo", .003, r("testB_exploratory", 20, .5, "lo"), .001); chk("rerank ext diff", .018, r("external_plantseg_exploratory", 20, .5, "hit1_diff"), .001); chk("rerank dev diff", -.0044, r("dev", 20, .5, "hit1_diff"), .001)
bad = [x for x in R if not x[0]]; print(f"{len(R) - len(bad)}/{len(R)} research_v2 claims match saved artifacts")
for x in bad: print("MISMATCH", x[1:])
open("report/verify_research_v2_output.txt", "w").write(f"{len(R) - len(bad)}/{len(R)}\n" + "\n".join(f"{'PASS' if x[0] else 'FAIL'} {x[1]}: doc={x[2]} artifact={x[3]}" for x in R) + "\n")
