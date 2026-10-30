"""Stage Z (registered in artifacts/research_v4/registered.json): zero-parameter post-processing on existing embeddings. No training of the network.
Families: DBA (gallery-side augmentation), aQE (query expansion), gallery-indexed k-reciprocal re-ranking (each query alone against the gallery; gallery
reciprocal sets precomputed without the query; no query-query information), NCA and shrinkage-LDA projections fitted on train_B only, and a greedy
dev-selected stack. Selection: dev class-macro Hit@1, tie band 1/150, then dev mAP. B/C tests consumed -> exploratory.
python -m segppd_cbir.rv4.zero_param [PA_single] [PA_single_RESNET]"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from ..rv2 import common as C
from ..rv2.metric_study import NSEEDS, load_model, pq_from_D

OUT = Path("artifacts/research_v4/zero"); OUT.mkdir(parents=True, exist_ok=True); S = C.segppd_sets()
SETS = (("dev", "dev_q", "gallery_B"), ("testB_exploratory", "testB_q", "gallery_B"), ("testC_exploratory", "testC_q", "gallery_C"))
L2n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------- families
def dba(G, k, p=1.0):
    sim = G @ G.T; nn = np.argsort(-sim, 1)[:, :k + 1]; w = np.take_along_axis(sim, nn, 1).clip(min=0) ** p
    return L2n((w[:, :, None] * G[nn]).sum(1))


def aqe(Q, G, k, alpha):
    sim = Q @ G.T; nn = np.argsort(-sim, 1)[:, :k]; w = np.take_along_axis(sim, nn, 1).clip(min=0) ** alpha
    return L2n(Q + (w[:, :, None] * G[nn]).sum(1))


class KRecip:
    """Gallery-indexed k-reciprocal (Zhong et al. 2017). V vectors of gallery items are computed on the gallery alone; a query's V uses its own
    k1 nearest gallery items and the reciprocity test 'd(q,j) < d(j, its k1-th gallery neighbour)'."""
    def __init__(self, G, k1, k2):
        self.G, self.k1, self.k2 = G, k1, k2; D = cdist(G, G, "sqeuclidean"); n = len(G); self.D = D
        order = np.argsort(D, 1); self.N = {k: order[:, :k] for k in (k1, k1 // 2, k2)}; self.kth = np.take_along_axis(D, order[:, [k1 - 1]], 1)[:, 0]  # self at rank 0
        R = [self._recip(i, D[i], self.N[k1][i], k1) for i in range(n)]; Rh = [self._recip(i, D[i], self.N[k1 // 2][i], k1 // 2) for i in range(n)]
        self.Rh = Rh; Rs = [self._expand(R[i], Rh) for i in range(n)]
        V = np.zeros((n, n), dtype=np.float64)
        for i in range(n): V[i, Rs[i]] = np.exp(-D[i, Rs[i]])
        V /= V.sum(1, keepdims=True); self.V = V[self.N[k2]].mean(1)  # local query expansion over the gallery (N(i,k2) includes i)

    def _recip(self, i, d, cand, k):
        # j in cand is reciprocal if i is among j's k nearest (gallery-only sets)
        return np.array([j for j in cand if i in self.N[k][j]], dtype=int)

    def _expand(self, R, Rh):
        Rs = set(R.tolist())
        for j in R:
            if len(np.intersect1d(Rh[j], R)) >= 2 / 3 * len(Rh[j]): Rs |= set(Rh[j].tolist())
        return np.array(sorted(Rs), dtype=int)

    def dist(self, Q, lam):
        Dq = cdist(Q, self.G, "sqeuclidean"); out = np.empty_like(Dq); n = len(self.G)
        for a in range(len(Q)):
            d = Dq[a]; order = np.argsort(d); cand = order[:self.k1]; R = cand[d[cand] < self.kth[cand]]  # reciprocity against the gallery's own k1-th distance
            Rs = set(R.tolist())
            for j in R:
                if len(np.intersect1d(self.Rh[j], R)) >= 2 / 3 * len(self.Rh[j]): Rs |= set(self.Rh[j].tolist())
            Rs = np.array(sorted(Rs), dtype=int) if Rs else cand[:1]; v = np.zeros(n); v[Rs] = np.exp(-d[Rs]); v /= v.sum()
            v = (v + self.V[order[:self.k2 - 1]].sum(0)) / self.k2  # local expansion with the query's k2-1 nearest gallery items
            jac = 1 - np.minimum(v, self.V).sum(1) / np.maximum(v, self.V).sum(1); out[a] = (1 - lam) * jac + lam * d
        return out


def fit_proj(kind, Ztr, ytr, d, seed=0):
    if kind == "nca":
        m = NeighborhoodComponentsAnalysis(n_components=min(d, Ztr.shape[1]), init="pca", max_iter=50, random_state=seed).fit(Ztr, ytr)
    else:
        m = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=min(d, len(np.unique(ytr)) - 1)).fit(Ztr, ytr)
    return lambda x: L2n(m.transform(x))


# ---------------------------------------------------------------- evaluation helpers
def evaluate(Q, G, q, g, D=None):
    D = cdist(Q, G, "sqeuclidean") if D is None else D
    return pq_from_D(D, q.class_id.to_numpy(), g.class_id.to_numpy(), g.image_id.to_numpy(dtype=str), q.image_id.to_numpy(dtype=str))


def pick(dev_rows):
    """dev rows: DataFrame with columns setting, hit1, mAP (seed means). Returns the setting chosen by the registered rule."""
    d = dev_rows.set_index("setting"); mx = d.hit1.max(); band = d[d.hit1 >= mx - C.TIE_EPS + 1e-12]; return band.mAP.idxmax()


def main(model):
    t0 = time.time(); rows, pqs = [], []; seeds = range(NSEEDS(model)); emb = {}
    for seed in seeds:
        ids, Z = load_model(model, seed); pos = {i: k for k, i in enumerate(ids)}; emb[seed] = (lambda df, Z=Z, pos=pos: L2n(Z[[pos[i] for i in df.image_id]].astype(np.float64)))  # bind per seed
    classes = sorted(S["train"].class_id.unique()); cm = {c: i for i, c in enumerate(classes)}; ytr = np.array([cm[c] for c in S["train"].class_id])

    def record(family, setting, seed, name, pq):
        s = C.summarize(pq); rows.append(dict(family=family, setting=setting, seed=seed, set=name, hit1=s["macro_hit1"], mAP=s["macro_mAP"], P10=s["macro_p10"])); pqs.append(pq.assign(family=family, setting=setting, seed=seed, set=name))

    # single families
    for seed in seeds:
        E = emb[seed]; Ztr = E(S["train"]); proj = {}
        for kind, d in [("nca", 256), ("nca", 128), ("nca", 64), ("lda", 74), ("lda", 64), ("lda", 32)]:
            proj[f"{kind}_{d}"] = fit_proj(kind, Ztr, ytr, d, seed)
        for name, qs, gs in SETS:
            q, g = S[qs], S[gs]; Q, G = E(q), E(g)
            record("baseline", "l2", seed, name, evaluate(Q, G, q, g))
            for k in (1, 2, 3, 5): record("dba", f"k{k}", seed, name, evaluate(Q, dba(G, k), q, g))
            for k, a in itertools.product((1, 2, 3, 5), (1.0, 3.0)): record("aqe", f"k{k}_a{a:g}", seed, name, evaluate(aqe(Q, G, k, a), G, q, g))
            for k1, k2 in itertools.product((10, 20), (3, 6)):
                kr = KRecip(G, k1, k2)
                for lam in (0.3, 0.5): record("krecip", f"k1{k1}_k2{k2}_l{lam:g}", seed, name, evaluate(Q, G, q, g, kr.dist(Q, lam)))
            for key, P in proj.items(): record("proj", key, seed, name, evaluate(P(Q), P(G), q, g))
        print(f"[{model}] seed {seed} single families done {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); M = R.groupby(["family", "setting", "set"])[["hit1", "mAP", "P10"]].mean().reset_index()
    # greedy dev-selected stack: proj -> dba -> aqe -> krecip; a step is adopted only if it beats the current stack by the rule
    dev = M[M.set == "dev"]; base = dev[dev.family == "baseline"].iloc[0]; stack, cur = [], dict(hit1=base.hit1, mAP=base.mAP)
    for fam in ("proj", "dba", "aqe", "krecip"):
        cand = dev[dev.family == fam][["setting", "hit1", "mAP"]]
        if fam == "proj" or not stack:
            best = pick(cand); b = cand.set_index("setting").loc[best]
        else:  # re-evaluate the family on top of the current stack (dev only), then pick
            srows = []
            for seed in seeds:
                E = emb[seed]; q, g = S["dev_q"], S["gallery_B"]; Q, G = apply_stack(stack, E, q, g, ytr, seed)
                for setting in cand.setting:
                    pq = apply_family(fam, setting, Q, G, q, g); s = C.summarize(pq); srows.append(dict(setting=setting, hit1=s["macro_hit1"], mAP=s["macro_mAP"]))
            cand = pd.DataFrame(srows).groupby("setting").mean().reset_index(); best = pick(cand); b = cand.set_index("setting").loc[best]
        adopt = b.hit1 > cur["hit1"] + C.TIE_EPS or (b.hit1 >= cur["hit1"] - C.TIE_EPS and b.mAP > cur["mAP"] + 1e-4 and b.hit1 >= cur["hit1"])
        if adopt: stack.append((fam, best)); cur = dict(hit1=float(b.hit1), mAP=float(b.mAP))
    # evaluate the stack on all sets
    for seed in seeds:
        E = emb[seed]
        for name, qs, gs in SETS:
            q, g = S[qs], S[gs]; record("stack", "+".join(f"{f}:{s}" for f, s in stack) or "none", seed, name, evaluate_stack(stack, E, q, g, ytr, seed))
    R = pd.DataFrame(rows); M = R.groupby(["family", "setting", "set"])[["hit1", "mAP", "P10"]].mean().reset_index()
    R.to_csv(OUT / f"zero_by_seed_{model}.csv", index=False); M.to_csv(OUT / f"zero_mean_{model}.csv", index=False); pd.concat(pqs).to_csv(OUT / f"per_query_{model}.csv.gz", index=False)
    sel = {fam: pick(dev[dev.family == fam][["setting", "hit1", "mAP"]]) for fam in ("dba", "aqe", "krecip", "proj")}
    json.dump(dict(model=model, rule="dev Hit@1 (seed mean), tie band 1/150, then dev mAP", selected_per_family=sel, stack=stack, stack_dev=cur), open(OUT / f"selection_dev_only_{model}.json", "w"), indent=1)
    pd.set_option("display.width", 250); show = M[M.setting.isin(list(sel.values()) + ["l2"]) | (M.family == "stack")]
    print(f"[{model}] dev-selected per family: {sel}; stack: {stack}"); print(show.pivot(index=["family", "setting"], columns="set", values="hit1").round(3).to_string())
    print(f"[{model}] done {time.time() - t0:.0f}s", flush=True)


def apply_family(fam, setting, Q, G, q, g):
    if fam == "dba": return evaluate(Q, dba(G, int(setting[1:])), q, g)
    if fam == "aqe": k, a = setting.split("_"); return evaluate(aqe(Q, G, int(k[1:]), float(a[1:])), G, q, g)
    if fam == "krecip":
        k1, k2, l = setting.split("_"); kr = KRecip(G, int(k1[2:]), int(k2[2:])); return evaluate(Q, G, q, g, kr.dist(Q, float(l[1:])))
    raise ValueError(fam)


def apply_stack(stack, E, q, g, ytr, seed):
    """Apply the adopted steps that transform Q/G (proj, dba, aqe); krecip changes distances only and is handled in evaluate_stack."""
    Q, G = E(q), E(g)
    for fam, setting in stack:
        if fam == "proj": kind, d = setting.split("_"); P = fit_proj(kind, E(S["train"]), ytr, int(d), seed); Q, G = P(Q), P(G)
        elif fam == "dba": G = dba(G, int(setting[1:]))
        elif fam == "aqe": k, a = setting.split("_"); Q = aqe(Q, G, int(k[1:]), float(a[1:]))
    return Q, G


def evaluate_stack(stack, E, q, g, ytr, seed):
    Q, G = apply_stack(stack, E, q, g, ytr, seed); kr = [st for f, st in stack if f == "krecip"]
    if kr: return apply_family("krecip", kr[0], Q, G, q, g)
    return evaluate(Q, G, q, g)


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["PA_single", "PA_single_RESNET"]): main(m)
