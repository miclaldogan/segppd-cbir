"""Protocol A (train-as-gallery, existing) audit + Protocol B (train / gallery / query mutually disjoint) construction.

Protocol B per primary class c (usable pool U_c, |U_c| >= 13):
  test_query_B : 3 images from U_c minus (A-test-queries, A-dev-queries)  -> never touched by any evaluation or selection
  dev_query_B  : the SAME 2 images/class used as dev queries in protocol A (never trained on in dev runs)
  rest         : U_c minus the above; split ~50/50 into train_B and gallery_B (seeded)
Retrieval for B: queries vs gallery_B only. Training uses train_B only. Dev selection uses dev_query_B vs gallery_B.
"""
from __future__ import annotations

import json

import imagehash
import numpy as np
import pandas as pd

from . import ext_dev as D

B_SEED = 4242


def _hash_mat(df, col):
    return np.array([imagehash.hex_to_hash(h).hash.flatten() for h in df[col]])


def nearest_dups(q: pd.DataFrame, ref: pd.DataFrame, p_max=14, d_max=20):
    """Near-duplicate screen of q vs ref using the SAME hash family as the duplicate-group step (looser thresholds)."""
    Pq, Pr = _hash_mat(q, "phash"), _hash_mat(ref, "phash")
    Dq, Dr = _hash_mat(q, "dhash"), _hash_mat(ref, "dhash")
    P = (Pq[:, None, :] != Pr[None, :, :]).sum(2)
    Dm = (Dq[:, None, :] != Dr[None, :, :]).sum(2)
    ii, jj = np.nonzero((P <= p_max) & (Dm <= d_max))
    return pd.DataFrame({"query": q["image_id"].to_numpy()[ii], "ref": ref["image_id"].to_numpy()[jj],
                         "q_class": q["class_id"].to_numpy()[ii], "ref_class": ref["class_id"].to_numpy()[jj],
                         "phash_d": P[ii, jj], "dhash_d": Dm[ii, jj]}), int(P.min())


def audit_protocol_a(sp: pd.DataFrame) -> dict:
    dev_q, dev_g = D.make_dev(sp)
    test_q = sp[sp.role == "query_primary"]
    gal = sp[sp.role == "gallery_primary"]
    ids = lambda d: set(d["image_id"])  # noqa: E731
    checks = {
        "dev_queries_n": len(dev_q), "dev_train_gallery_n": len(dev_g), "test_queries_n": len(test_q), "final_train_gallery_n": len(gal),
        "dev_queries_disjoint_from_dev_run_training": ids(dev_q).isdisjoint(ids(dev_g)),
        "dev_queries_disjoint_from_test_queries": ids(dev_q).isdisjoint(ids(test_q)),
        "test_queries_disjoint_from_dev_run_training": ids(test_q).isdisjoint(ids(dev_g)),
        "test_queries_disjoint_from_final_training": ids(test_q).isdisjoint(ids(gal)),
        "dev_queries_inside_final_training(by design: final run retrains on full gallery after selection)": ids(dev_q) <= ids(gal),
        "test_query_groups_disjoint_from_final_training": set(test_q.group_id).isdisjoint(set(gal.group_id)),
        "excluded_dup_or_conflict_images_in_any_training": bool(
            (ids(sp[sp.role == "excluded_duplicate_or_conflict"]) & (ids(gal) | ids(dev_g))) != set()),
    }
    nd, pmin = nearest_dups(test_q, gal)
    nd_dev, _ = nearest_dups(test_q, dev_q)
    checks["test_vs_final_training_near_dup_candidates(pHash<=14&dHash<=20)"] = len(nd)
    checks["test_vs_dev_queries_near_dup_candidates"] = len(nd_dev)
    checks["min_phash_distance_test_vs_training"] = pmin
    return checks, nd


def build_protocol_b(sp: pd.DataFrame, seed: int = B_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dev_q, _ = D.make_dev(sp)
    a_test = set(sp[sp.role == "query_primary"]["image_id"])
    a_dev = set(dev_q["image_id"])
    prim = sp[sp.cohort_class == "primary"]
    pool = prim[prim.dup_status.isin(["unique", "kept_representative"])]
    role = pd.Series("not_in_protocol_B", index=sp.index, dtype=object)
    for cid, sub in pool.groupby("class_id"):
        sub = sub.sort_values("image_id")
        eligible = sub[~sub.image_id.isin(a_test | a_dev)]
        q = eligible.index[rng.choice(len(eligible), 3, replace=False)]
        role[q] = "test_query_B"
        role[sub.index[sub.image_id.isin(a_dev)]] = "dev_query_B"
        rest = sub.index.difference(q).difference(sub.index[sub.image_id.isin(a_dev)])
        rest = list(sub.loc[rest].sort_values("image_id").index)
        perm = rng.permutation(len(rest))
        n_tr = len(rest) // 2
        for k, p in enumerate(perm):
            role[rest[p]] = "train_B" if k < n_tr else "gallery_B"
    out = sp.copy()
    out["role_B"] = role
    return out


def audit_protocol_b(b: pd.DataFrame) -> dict:
    g = lambda r: b[b.role_B == r]  # noqa: E731
    tr, ga, dq, tq = g("train_B"), g("gallery_B"), g("dev_query_B"), g("test_query_B")
    ids = lambda d: set(d["image_id"])  # noqa: E731
    gro = lambda d: set(d["group_id"])  # noqa: E731
    res = {
        "n_train": len(tr), "n_gallery": len(ga), "n_dev_query": len(dq), "n_test_query": len(tq),
        "n_classes": int(tq.class_id.nunique()),
        "all_sets_mutually_disjoint": all(ids(x).isdisjoint(ids(y)) for x, y in
                                          [(tr, ga), (tr, dq), (tr, tq), (ga, dq), (ga, tq), (dq, tq)]),
        "group_ids_mutually_disjoint": all(gro(x).isdisjoint(gro(y)) for x, y in
                                           [(tr, ga), (tr, dq), (tr, tq), (ga, dq), (ga, tq), (dq, tq)]),
        "test_queries_not_in_protocolA_test_or_dev": bool(ids(tq).isdisjoint(set(b[b.role == "query_primary"].image_id) | ids(dq))),
        "min_gallery_positives_per_class": int(ga.groupby("class_id").size().min()),
        "min_train_images_per_class": int(tr.groupby("class_id").size().min()),
        "every_class_has_3_test_2_dev_train_gallery": bool(
            all(x.groupby("class_id").size().reindex(sorted(tq.class_id.unique())).notna().all() for x in (tr, ga, dq, tq))),
    }
    for name, other in (("test_vs_train", tr), ("test_vs_gallery", ga), ("dev_vs_train", tr), ("dev_vs_gallery", ga)):
        nd, pmin = nearest_dups(tq if name.startswith("test") else dq, other)
        res[f"near_dup_candidates_{name}"] = len(nd)
        res[f"min_phash_{name}"] = pmin
    return res
