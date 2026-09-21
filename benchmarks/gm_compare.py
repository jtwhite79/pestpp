"""
compare the reference EnIF implementation (equinor/graphite-maps) against the
pest++ EnIF update (EnsembleSolver::solve_enif in EnsembleMethodUtils.cpp) on
the synth2d modflow6 test problem.

this is pure post-processing on ensembles already on disk - nothing is run.

three experiments:
  0. sanity check on small random data (guards against harness bugs)
  1. update algebra with identical inputs (same Prec_u, same H, same noise)
  2. the residual-inflation term
  3. the H estimator: reference LASSO vs pest++ ridge at several ridge_frac

everything is done in log10 space, since that is where pest++ solves and where
prior_cov.jcb lives.
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sps

sys.path.insert(0, "/tmp/claude-501/graphite-maps")

import pyemu  # noqa: E402
import graphite_maps.enif as gm_enif  # noqa: E402
import graphite_maps.utils as gm_utils  # noqa: E402
from graphite_maps.enif import EnIF  # noqa: E402


# ----------------------------------------------------------------------------
# scikit-sparse shim.  graphite-maps pins scikit-sparse < 0.5.0 and calls the
# old Factor api (cholesky(A, ordering_method=...) -> .solve_A / .L()).  the
# installed build is 0.5.0, which renamed everything.  shim the old api onto
# the new CholeskyFactor so the reference code runs unmodified - still real
# cholmod, same factorization, just a different entry point.
# ----------------------------------------------------------------------------
import sksparse.cholmod as _chol  # noqa: E402


class _OldFactor:
    def __init__(self, A, order="metis"):
        A = sps.csc_array(A)
        self._f = _chol.CholeskyFactor(A, order=order)
        self._f.factorize(A)

    def solve_A(self, b):
        return self._f.solve(b)

    def L(self):
        return self._f.L()

    def logdet(self):
        return self._f.logdet()


def _cholesky_shim(A, ordering_method="default", **kw):
    order = ordering_method if ordering_method != "default" else "default"
    return _OldFactor(A, order=order)


gm_enif.cholesky = _cholesky_shim
gm_utils.cholesky = _cholesky_shim

# quiet the progress bars - they wreck the report
import graphite_maps.linear_regression as gm_lr  # noqa: E402

gm_lr.tqdm = lambda it, **kw: it
gm_enif.tqdm = lambda it, **kw: it

D = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(D, "synth2d")
PST_FILE = os.path.join(CASE, "template", "synth2d.pst")
COV_FILE = os.path.join(CASE, "template", "prior_cov.jcb")
M_DIR = os.path.join(CASE, "nsweep_enif_50")


# ----------------------------------------------------------------------------
# pest++ side: a literal transcription of solve_enif()
# ----------------------------------------------------------------------------
def pestpp_enif(X, Y, dpert, C, w, ridge_frac=1.0e-6, lam=0.0, inflate=True,
                X0=None):
    """the pest++ enif upgrade.

    X     : (p,N) current par ensemble, log10 space
    Y     : (n,N) simulated obs ensemble
    dpert : (n,N) obs+noise realizations
    C     : (p,p) prior covariance, log10 space
    w     : (n,)  obs weights
    X0    : (p,N) base (prior) par ensemble; defaults to X so that e = 0
    """
    p, N = X.shape
    n = Y.shape[0]
    if X0 is None:
        X0 = X
    e = X - X0
    r = Y - dpert

    scale = 1.0 / np.sqrt(N - 1.0)
    A = (X - X.mean(axis=1, keepdims=True)) * scale
    B = (Y - Y.mean(axis=1, keepdims=True)) * scale

    gram_raw = A.T @ A
    gamma = ridge_frac * np.trace(gram_raw) / float(N)
    if gamma <= 0.0:
        gamma = 1.0e-12
    gram = gram_raw + gamma * np.eye(N)

    def gsolve(rhs):
        return np.linalg.solve(gram, rhs)

    Ht = A @ gsolve(B.T)                       # (p,n)
    s = 1.0 / (1.0 + lam)
    CHt = s * (C @ Ht)                         # (p,n)
    Gm = B @ gsolve(A.T @ CHt)                 # (n,n)

    unexplained = np.zeros(n)
    if inflate:
        resid = (B - B @ gsolve(gram_raw)) * np.sqrt(N - 1.0)
        unexplained = (resid ** 2).sum(axis=1) / float(N)
    Gm = Gm + np.diag(1.0 / (w ** 2) + unexplained)

    He = B @ gsolve(A.T @ e)
    upgrade = -((s * (e - CHt @ np.linalg.solve(Gm, He)))
                + CHt @ np.linalg.solve(Gm, r))
    return {
        "X_post": X + upgrade,
        "H": Ht.T,                 # (n,p)
        "unexplained": unexplained,
        "gamma": gamma,
    }


def pestpp_enif_with_H(X, Y, dpert, C, w, H, lam=0.0, unexplained=None, X0=None):
    """the same update but with an externally supplied (explicit) H.

    used so the reference's own LASSO H can be pushed through the pest++
    formula, and so the two sides can be given bit-identical H.
    """
    p, N = X.shape
    n = Y.shape[0]
    if X0 is None:
        X0 = X
    e = X - X0
    r = Y - dpert
    if unexplained is None:
        unexplained = np.zeros(n)

    s = 1.0 / (1.0 + lam)
    CHt = s * (C @ H.T)
    Gm = H @ CHt + np.diag(1.0 / (w ** 2) + unexplained)
    He = H @ e
    upgrade = -((s * (e - CHt @ np.linalg.solve(Gm, He)))
                + CHt @ np.linalg.solve(Gm, r))
    return X + upgrade


# ----------------------------------------------------------------------------
# reference side
# ----------------------------------------------------------------------------
class _NoInflateEnIF(EnIF):
    """EnIF with the residual-variance inflation switched off."""

    def response_residual(self, U, Y):
        res = super().response_residual(U, Y)
        self.unexplained_variance = np.zeros_like(self.unexplained_variance)
        return res


def reference_enif_run(X, Y, dpert, d, Cinv, w, H, inflate=True):
    cls = EnIF if inflate else _NoInflateEnIF
    enif = cls(
        Prec_u=sps.csc_array(Cinv),
        Prec_eps=sps.diags_array(w ** 2, offsets=0, format="csc"),
        H=sps.csc_array(H),
    )
    # inject pest++'s perturbations: eps = d - dpert  (shape (N, n))
    eps = (d[:, None] - dpert).T.copy()

    def _fixed_noise(n_reals, seed=None):
        assert n_reals == eps.shape[0]
        return eps

    enif.generate_observation_noise = _fixed_noise
    U_post = enif.transport(U=X.T.copy(), Y=Y.T.copy(), d=d)
    return {
        "X_post": U_post.T,
        "unexplained": np.asarray(enif.unexplained_variance).ravel(),
    }


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def cmp(a, b, label, out=sys.stdout):
    d = np.abs(a - b)
    scl = max(np.abs(a).max(), np.abs(b).max(), 1.0e-300)
    print("  {:<34s} max abs {:12.4e}   mean abs {:12.4e}   max rel {:12.4e}"
          .format(label, d.max(), d.mean(), d.max() / scl), file=out)
    return d.max() / scl


def phi_of(Ysim, obsval, w):
    """sum_i w_i^2 (y_i - obs_i)^2 per realization."""
    resid = Ysim - obsval[:, None]
    return ((w[:, None] ** 2) * resid ** 2).sum(axis=0)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
def experiment0():
    """small random self-test: does the harness itself line the two up?"""
    hr("EXPERIMENT 0 - sanity check on small random data")
    rng = np.random.default_rng(0)
    p, n, N = 12, 5, 8
    L = rng.normal(size=(p, p))
    C = L @ L.T + p * np.eye(p)
    Cinv = np.linalg.inv(C)
    w = rng.uniform(0.5, 2.0, n)
    X = rng.normal(size=(p, N))
    Y = rng.normal(size=(n, N))
    d = rng.normal(size=n)
    dpert = d[:, None] + rng.normal(size=(n, N)) / w[:, None]
    H = rng.normal(size=(n, p))

    for infl, unexp_getter in (("off", None), ("on", "ref")):
        if infl == "off":
            unexp = np.zeros(n)
        else:
            resid = Y.T - X.T @ H.T
            unexp = resid.var(axis=0, ddof=0)
        pp = pestpp_enif_with_H(X, Y, dpert, C, w, H, unexplained=unexp)
        ref = reference_enif_run(X, Y, dpert, d, Cinv, w, H,
                                 inflate=(infl == "on"))
        cmp(pp, ref["X_post"], "posterior, inflation " + infl)


def main():
    t0 = time.time()
    print("loading data ...")
    pst = pyemu.Pst(PST_FILE)
    pnames = pst.adj_par_names
    onames = pst.nnz_obs_names
    w = pst.observation_data.loc[onames, "weight"].values.astype(float)
    obsval = pst.observation_data.loc[onames, "obsval"].values.astype(float)

    pe = pyemu.ParameterEnsemble.from_binary(
        pst=pst, filename=os.path.join(M_DIR, "synth2d.0.par.jcb"))
    oe = pyemu.ObservationEnsemble.from_binary(
        pst=pst, filename=os.path.join(M_DIR, "synth2d.0.obs.jcb"))
    noe = pyemu.ObservationEnsemble.from_binary(
        pst=pst, filename=os.path.join(M_DIR, "synth2d.obs+noise.jcb"))
    for df in (pe, oe, noe):
        df.index = [str(i) for i in df.index]
    reals = [r for r in pe.index if r in oe.index and r in noe.index]
    print("  aligned realizations: {} of pe {} / oe {} / noe {}"
          .format(len(reals), pe.shape[0], oe.shape[0], noe.shape[0]))

    # native -> log10 for log-transformed pars (all of them here)
    log_par = pst.parameter_data.loc[pnames, "partrans"] == "log"
    Xnat = pe.loc[reals, pnames].values.T.astype(float)     # (p,N)
    X = Xnat.copy()
    X[log_par.values, :] = np.log10(X[log_par.values, :])
    Y = oe.loc[reals, onames].values.T.astype(float)        # (n,N)
    dpert = noe.loc[reals, onames].values.T.astype(float)   # (n,N)

    cov = pyemu.Cov.from_binary(os.path.join(CASE, "template", "prior_cov.jcb"))
    C = cov.get(pnames).as_2d.astype(float)
    p, N = X.shape
    n = Y.shape[0]
    print("  p={} n={} N={}  all-log={}".format(p, n, N, bool(log_par.all())))
    print("  C symmetric: {:.3e}   cond(C): {:.4e}"
          .format(np.abs(C - C.T).max(), np.linalg.cond(C)))
    Cinv = np.linalg.inv(C)
    print("  ||C Cinv - I||_max = {:.4e}".format(
        np.abs(C @ Cinv - np.eye(p)).max()))
    print("  prior phi mean = {:.4f}".format(phi_of(Y, obsval, w).mean()))
    print("  load time {:.1f}s".format(time.time() - t0))

    experiment0()

    # ------------------------------------------------------------------
    hr("EXPERIMENT 1 - update algebra, identical inputs (no inflation)")
    print("  both sides get: Prec_u = inv(C), the same explicit ridge H")
    print("  (ridge_frac=1e-6), the same perturbations (eps = d - dpert),")
    print("  and inflation disabled on both sides.")
    base = pestpp_enif(X, Y, dpert, C, w, ridge_frac=1.0e-6, inflate=False)
    H = base["H"]
    print("  ridge gamma = {:.6e}, H shape {}, ||H||_max = {:.4e}"
          .format(base["gamma"], H.shape, np.abs(H).max()))
    t = time.time()
    ref = reference_enif_run(X, Y, dpert, obsval, Cinv, w, H, inflate=False)
    print("  reference transport: {:.1f}s".format(time.time() - t))
    print()
    r1 = cmp(base["X_post"], ref["X_post"], "posterior ensemble (log10)")
    cmp(base["X_post"] - X, ref["X_post"] - X, "upgrade (X_post - X)")
    upmax = np.abs(base["X_post"] - X).max()
    print("  max abs upgrade (either side)        {:12.4e}".format(upmax))
    print("  posterior mean diff (per-par max)    {:12.4e}".format(
        np.abs(base["X_post"].mean(1) - ref["X_post"].mean(1)).max()))
    # cross-check: pest++ formula fed the explicit H must equal the
    # never-formed-H version
    ppH = pestpp_enif_with_H(X, Y, dpert, C, w, H)
    cmp(base["X_post"], ppH, "pest++ implicit-H vs explicit-H")

    # ------------------------------------------------------------------
    hr("EXPERIMENT 1b - python transcription vs the compiled pest++ C++")
    print("  the pest++ side above is a python transcription of solve_enif().")
    print("  check it against the real thing: synth2d.1.par.jcb is what")
    print("  pest++ actually wrote at iteration 1.  the rec file says the")
    print("  accepted step was lambda 1000, scale factor 1.")
    try:
        pe1 = pyemu.ParameterEnsemble.from_binary(
            pst=pst, filename=os.path.join(M_DIR, "synth2d.1.par.jcb"))
        pe1.index = [str(i) for i in pe1.index]
        X1 = pe1.loc[reals, pnames].values.T.astype(float)
        X1[log_par.values, :] = np.log10(X1[log_par.values, :])
        lb = pst.parameter_data.loc[pnames, "parlbnd"].values.astype(float)
        ub = pst.parameter_data.loc[pnames, "parubnd"].values.astype(float)
        lb = np.log10(lb)
        ub = np.log10(ub)
        for lam in (1000.0, 750.0, 100.0, 0.0):
            res = pestpp_enif(X, Y, dpert, C, w, ridge_frac=1.0e-6, lam=lam,
                              inflate=True)
            Xc = np.clip(res["X_post"], lb[:, None], ub[:, None])
            d = np.abs(Xc - X1)
            print("    lambda {:>7.1f}  max abs {:12.4e}  mean abs {:12.4e}"
                  .format(lam, d.max(), d.mean()))
        print("    (bounds enforced by clipping in log10 space, as pest++ does)")
    except Exception as ex:
        print("    could not run: {}".format(ex))

    # ------------------------------------------------------------------
    hr("EXPERIMENT 2 - residual inflation term")
    infl = pestpp_enif(X, Y, dpert, C, w, ridge_frac=1.0e-6, inflate=True)
    ref_i = reference_enif_run(X, Y, dpert, obsval, Cinv, w, H, inflate=True)
    print("  per-obs unexplained variance:")
    ue_pp = infl["unexplained"]
    ue_rf = ref_i["unexplained"]
    print("    pest++    min {:.6e}  mean {:.6e}  max {:.6e}"
          .format(ue_pp.min(), ue_pp.mean(), ue_pp.max()))
    print("    reference min {:.6e}  mean {:.6e}  max {:.6e}"
          .format(ue_rf.min(), ue_rf.mean(), ue_rf.max()))
    cmp(ue_pp, ue_rf, "unexplained variance vector")
    print("    obs error var 1/w^2: mean {:.6e}  max {:.6e}"
          .format((1.0 / w ** 2).mean(), (1.0 / w ** 2).max()))
    print("    inflation / obs var ratio: mean {:.3f}  max {:.3f}"
          .format((ue_pp * w ** 2).mean(), (ue_pp * w ** 2).max()))
    print()
    cmp(infl["X_post"], ref_i["X_post"], "posterior, inflation ON  both")
    cmp(base["X_post"], ref["X_post"], "posterior, inflation OFF both")
    print()
    print("  effect of the inflation term itself (same implementation):")
    cmp(infl["X_post"], base["X_post"], "pest++ on vs off")
    cmp(ref_i["X_post"], ref["X_post"], "reference on vs off")

    # ------------------------------------------------------------------
    hr("EXPERIMENT 3 - H estimator: reference LASSO vs pest++ ridge")
    print("  fitting reference H with LASSO (LassoCV, cv=10) ...")
    t = time.time()
    fitter = EnIF(Prec_u=sps.csc_array(Cinv),
                  Prec_eps=sps.diags_array(w ** 2, offsets=0, format="csc"))
    fitter.fit_H(U=X.T.copy(), Y=Y.T.copy(), learning_algorithm="LASSO")
    H_lasso = fitter.H.toarray()
    ue_lasso = np.asarray(fitter.unexplained_variance).ravel()
    print("  LASSO fit: {:.1f}s".format(time.time() - t))

    # a common linearization operator for phi so the comparison is not
    # self-serving: the plain (essentially unregularized) ensemble regression
    Hcommon = pestpp_enif(X, Y, dpert, C, w, ridge_frac=1.0e-6,
                          inflate=False)["H"]

    # projector onto the ensemble anomaly subspace (where any ensemble-fit
    # H is actually identified).  a step that leaves this subspace cannot be
    # scored honestly by any ensemble-derived linearization.
    Aa = X - X.mean(axis=1, keepdims=True)
    Qa, _ = np.linalg.qr(Aa)

    rows = []

    def add_row(name, Hm, ue):
        Xp = pestpp_enif_with_H(X, Y, dpert, C, w, Hm, unexplained=ue)
        dX = Xp - X
        Yp_self = Y + Hm @ dX
        Yp_com = Y + Hcommon @ dX
        insub = (np.linalg.norm(Qa @ (Qa.T @ dX)) ** 2
                 / max(np.linalg.norm(dX) ** 2, 1.0e-300))
        rows.append(dict(
            name=name,
            nnz=int((np.abs(Hm) > 0).sum()),
            dens=100.0 * (np.abs(Hm) > 0).sum() / Hm.size,
            ue=ue.mean(),
            hmax=np.abs(Hm).max(),
            step=np.abs(dX).max(),
            insub=100.0 * insub,
            phi_self=phi_of(Yp_self, obsval, w).mean(),
            phi_com=phi_of(Yp_com, obsval, w).mean(),
        ))

    add_row("reference LASSO", H_lasso, ue_lasso)
    for rf in (1.0e-6, 1.0e-2, 1.0e-1, 1.0):
        res = pestpp_enif(X, Y, dpert, C, w, ridge_frac=rf, inflate=True)
        add_row("pest++ ridge {:g}".format(rf), res["H"], res["unexplained"])

    print()
    print("  {:<22s} {:>9s} {:>7s} {:>12s} {:>10s} {:>8s} {:>11s} {:>11s}"
          .format("H estimator", "nnz(H)", "dens%", "mean unexpl", "max|dX|",
                  "dX in A%", "phi(self-H)", "phi(common)"))
    print("  " + "-" * 98)
    for r in rows:
        print("  {:<22s} {:>9d} {:>7.2f} {:>12.4e} {:>10.4e} {:>8.2f} "
              "{:>11.3f} {:>11.3f}"
              .format(r["name"], r["nnz"], r["dens"], r["ue"], r["step"],
                      r["insub"], r["phi_self"], r["phi_com"]))
    print("  " + "-" * 98)
    print("  prior phi mean = {:.3f}   (dense H would be {} entries)"
          .format(phi_of(Y, obsval, w).mean(), n * p))
    print()
    print("  NOTE: phi here is LINEARIZED (Y_post = Y + H dX); the model was")
    print("        not run, so it is indicative ONLY.  'self-H' scores each")
    print("        step with its own H; 'common' scores every step with the")
    print("        ridge-1e-6 ensemble regression - which is rigged in favour")
    print("        of that row and is unidentified off the ensemble subspace,")
    print("        so read the 'dX in A%' column (fraction of step energy")
    print("        inside the ensemble anomaly span) alongside it.  neither")
    print("        column is a substitute for running the model.")

    # ------------------------------------------------------------------
    hr("EXPERIMENT 3c - how much of the step is verifiable by the ensemble")
    print("  split each step dX into the part inside the ensemble anomaly")
    print("  span (dIn, the only part any ensemble-fit H can speak to) and")
    print("  the part outside it (dOut, which the prior covariance produces).")
    print("  note H_ridge @ dOut == 0 by construction, so the 'common' phi")
    print("  above scores dIn only.")
    print()
    print("  {:<18s} {:>9s} {:>13s} {:>13s} {:>13s}".format(
        "H estimator", "|dOut|/|dX|", "phi self(dX)", "phi self(dIn)",
        "phi common(dIn)"))
    for name, Hm, ue in (("reference LASSO", H_lasso, ue_lasso),
                         ("pest++ ridge 1e-6", Hcommon,
                          pestpp_enif(X, Y, dpert, C, w, ridge_frac=1.0e-6,
                                      inflate=True)["unexplained"])):
        Xp = pestpp_enif_with_H(X, Y, dpert, C, w, Hm, unexplained=ue)
        dX = Xp - X
        dIn = Qa @ (Qa.T @ dX)
        dOut = dX - dIn
        print("  {:<18s} {:>11.3f} {:>13.1f} {:>13.1f} {:>13.1f}".format(
            name, np.linalg.norm(dOut) / np.linalg.norm(dX),
            phi_of(Y + Hm @ dX, obsval, w).mean(),
            phi_of(Y + Hm @ dIn, obsval, w).mean(),
            phi_of(Y + Hcommon @ dIn, obsval, w).mean()))
    print("  max |H_ridge @ dOut| = {:.3e}  (zero to round-off, as expected)"
          .format(np.abs(Hcommon @ (dX - Qa @ (Qa.T @ dX))).max()))

    # cross-implementation check with the LASSO H
    hr("EXPERIMENT 3b - reference LASSO H pushed through BOTH updates")
    ref_l = reference_enif_run(X, Y, dpert, obsval, Cinv, w, H_lasso,
                               inflate=True)
    pp_l = pestpp_enif_with_H(X, Y, dpert, C, w, H_lasso,
                              unexplained=ue_lasso)
    cmp(pp_l, ref_l["X_post"], "posterior with LASSO H")
    cmp(ue_lasso, np.asarray(ref_l["unexplained"]).ravel(),
        "unexplained variance")

    print("\ntotal wall time {:.1f}s".format(time.time() - t0))
    return r1


if __name__ == "__main__":
    main()
