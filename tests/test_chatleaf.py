from pathlib import Path

import pandas as pd


def test_chatleaf_queries_are_screened_and_unique():
    p = Path("artifacts/metrics/chatleaf_matched/trackS_per_query.csv")
    g = Path("artifacts/chatleaf/audit_inscope_groups.csv")
    if not (p.exists() and g.exists()):
        return
    q = pd.read_csv(p); gr = pd.read_csv(g).set_index("Name")
    names = set(q.name)
    assert not gr.loc[list(names)].leak_flag.any()                       # no SegPPD copies among queries
    sub = gr.loc[list(names)]
    assert sub.group.is_unique                                            # one image per duplicate group
    conflict = pd.read_csv(g).groupby("group").Disease.nunique()
    assert not set(sub.group) & set(conflict[conflict > 1].index)         # no cross-category label-conflict groups
