"""Stage S follow-ups and stage D (distillation) configs. python -m segppd_cbir.rv4.make_configs2 followup <small1> <small2> | kd <student> [...]"""
import sys

from .make_configs import BASE, write

FOLLOW = "artifacts/research_v4/followup"; KD = "artifacts/research_v4/kd"
TEACH = {"ens": "artifacts/research_v4/teachers/pa_ens.npz", "ensres": "artifacts/research_v4/teachers/pa_ens_resnet.npz"}
LAMS = (30.0, 100.0)  # revised for small students (300 worst on DINOv2-B); the DINOv2-B runs used (30,100,300)


def followup(smalls):
    cfgs = {}
    for bb in smalls:
        cfgs[f"F_{bb}_lr1e-4"] = dict(backbone=bb, lr_bb=1e-4)
        cfgs[f"F_{bb}_full"] = dict(backbone=bb, full=True)
        cfgs[f"F_{bb}_full_lr1e-4"] = dict(backbone=bb, full=True, lr_bb=1e-4)
        cfgs[f"F_{bb}_336"] = dict(backbone=bb, img_size=336)
    return write(cfgs, FOLLOW)


def kd(students):
    cfgs = {}
    for st in students:  # st = "backbone[:opt=val,...]"
        bb, *opts = st.split(":"); extra = dict((k, (float(v) if "." in v or "e" in v else (v == "True" if v in ("True", "False") else int(v)))) for k, v in (o.split("=") for o in opts))
        for t, path in TEACH.items():
            for lam in LAMS: cfgs[f"D_{st.replace(':', '_').replace('=', '')}_{t}_l{lam:g}"] = dict(backbone=bb, kd=dict(teacher=path, lam=lam), **extra)
    return write(cfgs, KD)


if __name__ == "__main__":
    (followup if sys.argv[1] == "followup" else kd)(sys.argv[2:])
