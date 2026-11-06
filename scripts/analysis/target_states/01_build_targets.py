#!/usr/bin/env python
"""Build the target sets (anatomical masks, Yeo networks, oriented gradients) and list their parcels.

Outputs (<run>/target_states/):
  parcel_network_assignment.csv   per-parcel Yeo7/Yeo17 majority label, purity, anatomical-mask membership
  target_mask_parcels.csv         exact parcel list for every target mask per scale
  gradient_orientation.csv        vertex-level orientation check of Margulies G1/G2
  targets_scale-{s}.npz           unit-norm target vectors (all families)
  target_catalog.csv              target name, family, n support nodes

Run: python scripts/analysis/target_states/01_build_targets.py --run <name> [--overwrite]  (reads/writes runs/<name>/)
"""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np
import pandas as pd

from hoc.analysis import target_states as C

ctx = C.start_stage(__file__, ["primary/00_prepare_data"])

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- gradient orientation (vertex level, mask independent)
yeo_v = {h: nib.freesurfer.read_annot(str(C.YEO_DIR / f"{'lh' if h == 'L' else 'rh'}.Yeo2011_7Networks_N1000.annot"))[0] for h in ("L", "R")}
orient_rows = []
for g in (1, 2):
    vg = C.margulies_vertex(g)
    vals = np.concatenate([vg["L"], vg["R"]])
    yeo = np.concatenate([yeo_v["L"], yeo_v["R"]])
    row = {"gradient": f"G{g}"}
    for nid, name in C.YEO7_NAMES.items():
        row[f"mean_{name}"] = float(vals[yeo == nid].mean())
    orient_rows.append(row)
orient = pd.DataFrame(orient_rows)
g1 = orient.iloc[0]
# Rule: published Margulies orientation is transmodal-positive (DMN at the top of G1). Keep the published
# sign iff the vertex-wise mean in Yeo Default exceeds the mean in Yeo Visual and SomMot; flip otherwise.
G1_SIGN = 1.0 if (g1.mean_Default > g1.mean_Visual and g1.mean_Default > g1.mean_SomMot) else -1.0
g2 = orient.iloc[1]
# G2 (visual vs somatomotor): orient so that Visual is positive (documented convention; used only for spin tests).
G2_SIGN = 1.0 if g2.mean_Visual > g2.mean_SomMot else -1.0
orient["applied_sign"] = [G1_SIGN, G2_SIGN]
orient.to_csv(C.OUT / "gradient_orientation.csv", index=False)
print(orient.to_string())

assign_all, mask_rows, catalog = [], [], []
for scale in C.SCALES:
    lab = C.labels(scale)
    n = len(lab)
    base = lab.base_region.to_numpy()
    y7 = C.yeo_assignment(scale, 7)
    y17 = C.yeo_assignment(scale, 17)

    assoc = np.isin(base, C.ASSOCIATION_BASE)
    occ = np.isin(base, C.OCCIPITAL_BASE)

    a = y7[["node_index", "region_label", "yeo7_id", "yeo7_name", "yeo7_purity", "yeo7_medialwall_frac", "n_vertices"]].copy()
    a.insert(0, "scale", scale)
    a["base_region"] = base
    a["yeo17_id"] = y17.yeo17_id
    a["yeo17_purity"] = y17.yeo17_purity
    a["association"] = assoc
    a["occipital"] = occ
    a["G1_parcel"] = G1_SIGN * C.margulies_parcel(scale, 1)
    a["G2_parcel"] = G2_SIGN * C.margulies_parcel(scale, 2)
    assign_all.append(a)

    T: dict[str, np.ndarray] = {}
    fam: dict[str, str] = {}

    def add(name, vec, family):
        T[name] = C.unit(vec)
        fam[name] = family

    for name, regions in C.LANDMARKS.items():
        add(name, np.isin(base, regions).astype(float), "landmark")
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1
        add(f"single_cortical_node_{i:03d}", v, "single_cortical_parcel")
    # anatomical masks (exact base-region names)
    add("kw_association_module", assoc.astype(float), "keyword_module")
    add("kw_occipital_module", occ.astype(float), "keyword_module")
    add("kw_association_vs_occipital", assoc.astype(float) - occ.astype(float), "keyword_module")
    # Yeo 7 networks
    ynet = y7.yeo7_id.to_numpy()
    for nid, name in C.YEO7_NAMES.items():
        mask = ynet == nid
        if mask.any():
            add(f"net_{name}", mask.astype(float), "yeo7_network")
    uni = np.isin(ynet, [1, 2])
    trans = np.isin(ynet, [6, 7])
    add("net_unimodal_VisSomMot", uni.astype(float), "yeo7_network")
    add("net_transmodal_DefFP", trans.astype(float), "yeo7_network")
    add("net_transmodal_vs_unimodal", trans.astype(float) - uni.astype(float), "yeo7_network")
    # Yeo 17 (supplementary)
    y17id = y17.yeo17_id.to_numpy()
    for nid in range(1, 18):
        mask = y17id == nid
        if mask.sum() >= 2:
            add(f"net17_{nid:02d}", mask.astype(float), "yeo17_network")
    # distributed
    G1 = a.G1_parcel.to_numpy()
    G2 = a.G2_parcel.to_numpy()
    add("principal_gradient", G1, "distributed")
    add("bipolar_gradient", np.sign(G1), "distributed")
    add("uniform_cortical", np.ones(n), "distributed")
    add("second_gradient", G2, "distributed_supp")
    add("bipolar_second_gradient", np.sign(G2), "distributed_supp")
    np.savez(C.OUT / f"targets_scale-{scale}.npz", **T)
    for name, vec in T.items():
        if fam[name] == "single_cortical_parcel":
            continue
        pos = np.flatnonzero(vec > 1e-12)
        neg = np.flatnonzero(vec < -1e-12)
        mask_rows.append(
            dict(scale=scale, target_name=name, family=fam[name], n_positive=len(pos), n_negative=len(neg),
                 positive_parcels=";".join(lab.region_label[pos]), negative_parcels=";".join(lab.region_label[neg]))
        )
        catalog.append(dict(scale=scale, target_name=name, family=fam[name], n_support=len(pos) + len(neg)))
    for i in range(n):
        catalog.append(dict(scale=scale, target_name=f"single_cortical_node_{i:03d}", family="single_cortical_parcel", n_support=1))

pd.concat(assign_all).to_csv(C.OUT / "parcel_network_assignment.csv", index=False)
masks = pd.DataFrame(mask_rows)
# gradient targets: list sign split rather than "support" (all parcels)
masks.to_csv(C.OUT / "target_mask_parcels.csv", index=False)
pd.DataFrame(catalog).to_csv(C.OUT / "target_catalog.csv", index=False)

A = pd.concat(assign_all)
print(A.groupby(["scale", "yeo7_name"]).size().unstack(0))
print("purity median/min by scale:", A.groupby("scale").yeo7_purity.agg(["median", "min"]).to_dict())
print(masks[["scale", "target_name", "n_positive", "n_negative"]].to_string())
ctx.finish()
