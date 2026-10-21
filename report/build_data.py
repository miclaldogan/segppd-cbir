import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


def table(tag):
    L = f"artifacts/metrics/protocol{tag}_locked/"
    ci = pd.read_csv(L + "macro_ci.csv"); cl = pd.read_csv(L + "classification_roc_metrics.csv").set_index("rep")
    def v(rep, m, K): return float(ci[(ci.rep == rep) & (ci.K == K) & (ci.metric == m)].macro_mean.iloc[0])
    def lohi(rep): r = ci[(ci.rep == rep) & (ci.K == 1) & (ci.metric == "P")].iloc[0]; return float(r.ci_lo), float(r.ci_hi)
    def row(name, reps):
        reps = reps if isinstance(reps, list) else [reps]
        f = lambda g: float(np.mean([g(r) for r in reps]))
        lo, hi = (np.mean([lohi(r)[0] for r in reps]), np.mean([lohi(r)[1] for r in reps]))
        return dict(name=name, hit1=f(lambda r: v(r, "P", 1)), lo=float(lo), hi=float(hi), p10=f(lambda r: v(r, "P", 10)), hit10=f(lambda r: v(r, "Hit", 10)),
                    mAP=f(lambda r: v(r, "full_AP", 10)), f1=f(lambda r: cl.loc[r, "macro_f1_1nn"]), auc=f(lambda r: cl.loc[r, "roc_auc_pairwise"]))
    s = lambda p: [f"{p}_s{i}" for i in range(3)]
    return [row("ResNet50 donmuş, 1000-B logit", "BASE_resnet50_frozen_logits1000_raw"),
            row("ResNet50 donmuş, 2048-B", "BASE_resnet50_frozen_pooled2048_raw"),
            row("ResNet50 ince ayar + ArcFace", "R1_resnet50_ft_arcface_existing_system"),
            row("DINOv2-S donmuş (CLS)", "F_dinov2s_frozen_cls"),
            row("DINOv2-B donmuş (CLS)", "F_dinov2b_frozen_cls"),
            row("DINOv2-S son 4 blok + ArcFace (1 tohum)", "D2_dinoS_p4_arcface_s0"),
            row("DINOv2-S son 4 blok + Proxy-Anchor (3 tohum ort.)", s("D3_dinoS_p4_proxyanchor")),
            row("DINOv2-S son 4 blok + SupCon (3 tohum ort.)", s("D4_dinoS_p4_supcon")),
            row("DINOv2-B son 4 blok + SupCon (3 tohum ort.) - önerilen", s("D5_dinoB_p4_supcon")),
            row("DINOv2-B + SupCon, 3 tohum topluluğu", "E_dinoB_supcon_3seed_ensemble"),
            row("topluluk + ResNet50 (w = 0,1) füzyonu", "FUSE_dinoBens0.9_resnetft0.1_(dev-selected)")]


B, C = table("B"), table("C")
ex = pd.read_csv("artifacts/metrics/external_plantseg/summary_ci.csv")
e = lambda m: ex[(ex.rep == "HEADLINE_D5ck_mean3seeds") & (ex.tier == "strict") & (ex.metric == m)].iloc[0]
pooled = pd.read_csv("artifacts/metrics/protocolBC_hit1_summary.csv"); pooled = pooled[(pooled.set.str.startswith("B+C")) & (pooled.rep == "HEADLINE")].iloc[0]
les = pd.read_csv("artifacts/metrics/protocolB_locked/lesion_by_bin.csv"); les = les[les.K == 1]
lesion = {}
for b in ("small", "medium", "large"):
    sd = les[(les.bin == b) & les.rep.str.startswith("D5_")]; en = les[(les.bin == b) & (les.rep == "E_dinoB_supcon_3seed_ensemble")].iloc[0]
    rs = les[(les.bin == b) & (les.rep == "BASE_resnet50_frozen_pooled2048_raw")].iloc[0]
    lesion[b] = dict(n=int(en.n_queries), cls=int(en.n_classes), seeds=float(sd.P_macro.mean()), ens=float(en.P_macro), ens_lo=float(en.P_lo), ens_hi=float(en.P_hi), res=float(rs.P_macro))
figs = {}
for p in Path("report/figs").glob("*.png"):
    w, h = Image.open(p).size; figs[p.stem] = dict(w=w, h=h)
json.dump(dict(B=B, C=C, ext=dict(hit1=float(e("hit1").macro_mean), lo=float(e("hit1").ci_class_lo), hi=float(e("hit1").ci_class_hi), p10=float(e("p10").macro_mean), hit10=float(e("hit10").macro_mean)),
               pooled=dict(hit1=float(pooled.hit1), lo=float(pooled.ci_lo), hi=float(pooled.ci_hi)), lesion=lesion, figs=figs), open("report/data.json", "w"), indent=1)
print(pd.DataFrame(B).round(3).to_string()); print(pd.DataFrame(C).round(3).iloc[[8]].to_string()); print(lesion)
