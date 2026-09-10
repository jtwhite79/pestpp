"""build a conditional-independence graph for the synth2d pilot points.

the graph is the sparsity pattern of the prior PRECISION, not the covariance.
for a matern-like field on a lattice the covariance is dense but the precision is
essentially nearest-neighbour, which is what makes the graph worth supplying:
you can write it down from the grid geometry without ever forming a p x p
covariance.

verify_against_prior() checks that claim on any problem where the prior
covariance happens to be available, by inverting it and comparing the partial
correlations to the proposed edges.
"""
import os
import numpy as np
import pandas as pd
import pyemu


def lattice_graph(pst, kind="rook", extra_within=None):
    """adjacency over the adjustable parameters from their (i, j) grid indices.

    kind      "rook" (4 neighbours) or "queen" (8, i.e. including diagonals)
    extra_within  optionally also connect any pair within this many lattice
                  steps, for a deliberately denser graph
    """
    names = list(pst.adj_par_names)
    pdata = pst.parameter_data.loc[names]
    if "i" not in pdata.columns or "j" not in pdata.columns:
        raise ValueError("parameter_data has no i/j columns to build a lattice from")
    ii = pdata["i"].astype(int).values
    jj = pdata["j"].astype(int).values

    # lattice step = the spacing between distinct pilot point indices
    ui = np.unique(ii)
    step = float(np.min(np.diff(ui))) if len(ui) > 1 else 1.0
    di = (ii[:, None] - ii[None, :]) / step
    dj = (jj[:, None] - jj[None, :]) / step

    if kind == "rook":
        adj = ((np.abs(di) + np.abs(dj)) == 1)
    elif kind == "queen":
        adj = (np.maximum(np.abs(di), np.abs(dj)) == 1)
    else:
        raise ValueError(f"unknown graph kind '{kind}'")
    if extra_within:
        adj |= (np.sqrt(di ** 2 + dj ** 2) <= extra_within)
    np.fill_diagonal(adj, True)

    n = len(names)
    print(f"{kind} graph: {n} nodes, {int((adj.sum() - n) // 2)} edges, "
          f"mean degree {(adj.sum() - n) / n:.1f}, density {100 * adj.mean():.1f}%")
    return pyemu.Matrix(x=adj.astype(float), row_names=names, col_names=names)


def verify_against_prior(pst, cov_file, thresh=0.05):
    """is the proposed sparsity actually what the prior precision looks like?

    only possible where a prior covariance exists - which is exactly the case
    where the graph is not needed.  the point is to establish that the geometric
    rule reproduces the real structure, so it can be trusted at scales where the
    covariance cannot be formed.
    """
    names = list(pst.adj_par_names)
    C = pyemu.Cov.from_binary(cov_file).get(names).as_2d
    L = np.linalg.inv(C)
    d = np.sqrt(np.diag(L))
    pcorr = -L / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    keep = (np.abs(pcorr) > thresh) & ~np.eye(len(names), dtype=bool)
    return keep, pcorr


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "synth2d_hkpp/template"
    pst = pyemu.Pst(os.path.join(d, "synth2d.pst"))
    for kind in ("rook", "queen"):
        m = lattice_graph(pst, kind=kind)
        # binary coo keeps long names and stays sparse on disk; pest++ autodetects
        m.to_coo(os.path.join(d, f"graph_{kind}.jcb"))
        print(f"   wrote {d}/graph_{kind}.jcb")
    cov = os.path.join(d, "prior_cov.jcb")
    if os.path.exists(cov):
        keep, _ = verify_against_prior(pst, cov)
        rook = lattice_graph(pst, kind="rook").x.astype(bool)
        np.fill_diagonal(rook, False)
        agree = (keep == rook).mean()
        print(f"\nrook graph vs thresholded prior precision: {100 * agree:.1f}% of "
              f"entries agree ({int(keep.sum() // 2)} edges implied by the precision)")
