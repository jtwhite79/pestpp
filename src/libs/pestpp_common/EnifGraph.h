#ifndef ENIFGRAPH_H_
#define ENIFGRAPH_H_

#include <string>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "covariance.h"

using namespace std;

/* A conditional-independence graph over the adjustable parameters, and the
sparse prior precision estimated on it.

This is the input mode the ensemble information filter is actually built around:
rather than supplying a p x p prior covariance - which does not scale - the user
supplies the SPARSITY PATTERN of the prior precision and the precision itself is
estimated from the prior ensemble.  A dense covariance generally has a nearly
sparse inverse, which is the whole premise.

The graph is read with Mat::from_file(), so it comes for free in every matrix
format pest++ already understands:

    .jcb / .jco   binary; the extended/coo variant is genuine (i,j,value) triplet
                  storage with long names, so it stays sparse on disk
    .mat / .cov   ascii PEST matrix format
    .csv          comma separated

Any structural non-zero is an edge; the value is ignored.  The matrix is
symmetrised (G union G^T) and the diagonal forced on, so a thresholded
correlation matrix, a localiser product, or a hand-built adjacency all mean the
same thing.

Note on how the precision is USED: H^T R^-1 H is dense unless H is also sparse,
so forming the posterior precision explicitly would throw the sparsity away.
Instead the solve keeps the woodbury form and obtains C*H^T by sparse-solving
Lam * X = H^T for the n observation right-hand sides.  No p x p dense matrix is
ever formed and no sparsity is required of H.
*/
class EnifGraph
{
public:
	EnifGraph() : initialized(false), nnz_offdiag(0) {}

	/* read the graph and align it to par_names.  throws if the file does not
	cover every adjustable parameter - a silently mis-aligned graph would be
	worse than no graph at all. */
	void from_file(const string& filename, const vector<string>& par_names,
		ofstream& frec);

	/* estimate the prior precision on the graph from ensemble anomalies
	(p x N, already scaled by 1/sqrt(N-1)) by neighbourhood regression:
	for each node i, regress u_i on its graph neighbours, which gives one row
	of the cholesky-like factor.  shrink is a stein-type pull toward the
	diagonal, needed when a node's neighbourhood approaches the ensemble size. */
	void estimate_precision(const Eigen::MatrixXd& anomalies, double shrink,
		ofstream& frec);

	/* apply the implied prior covariance: returns C * M, computed as
	solve(Lam, M) so the covariance is never formed */
	Eigen::MatrixXd apply_cov(const Eigen::MatrixXd& M) const;

	/* damped gauss-newton step in the information form, for use when H is sparse:

	    Lam_post = (1+lam) Lam_prior + H^T Rinv H
	    g        = Lam_prior (x - x0) + H^T Rinv (y - d)
	    delta    = -solve(Lam_post, g)

	violating no sparsity: with a sparse H the posterior precision stays sparse and
	this is one sparse cholesky, instead of the n right-hand-side solves woodbury
	needs.  returns the upgrade (p x N). */
	Eigen::MatrixXd information_step(const Eigen::SparseMatrix<double>& H,
		const Eigen::VectorXd& rinv, const Eigen::MatrixXd& e,
		const Eigen::MatrixXd& resid, double lam, ofstream& frec) const;

	bool is_initialized() const { return initialized; }
	bool has_precision() const { return prec_ready; }
	int num_nodes() const { return (int)names.size(); }
	int num_edges() const { return nnz_offdiag / 2; }
	const Eigen::SparseMatrix<double>& precision() const { return prec; }
	void report(ofstream& frec) const;

private:
	bool initialized = false;
	bool prec_ready = false;
	int nnz_offdiag;
	vector<string> names;
	Eigen::SparseMatrix<double> adj;
	Eigen::SparseMatrix<double> prec;
	mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
};


/* Estimate a SPARSE observation operator H (n x p) by lasso regression of the
observation anomalies on the parameter anomalies, one independent problem per
observation row.

Sparsity here is load-bearing, not cosmetic.  A structural zero in H is the
statement "this observation carries no information about this parameter", which
is what makes H an inspectable influence map.  It is also what keeps
H^T Rinv H - and therefore the posterior precision - sparse, so the graph is
worth having at all.

lasso_frac scales the penalty relative to the smallest value that would zero a
whole row, so it is dimensionless and lives in (0,1).  unexplained returns the
per-observation residual variance, which is the quantity the observation error is
inflated by.  Rows are independent, so this parallelises over num_threads. */
Eigen::SparseMatrix<double> estimate_sparse_H(const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& B, double lasso_frac, int num_threads,
	Eigen::VectorXd& unexplained, ofstream& frec);

#endif // ENIFGRAPH_H_
