#include <set>
#include <thread>
#include <atomic>
#include <map>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "EnifGraph.h"
#include "utilities.h"

using namespace std;


void EnifGraph::from_file(const string& filename, const vector<string>& par_names,
	ofstream& frec)
{
	stringstream ss;
	Mat m;
	//Mat::from_file dispatches on the extension: jcb/jco binary (the extended
	//coo variant is genuine sparse triplet storage), mat/cov ascii, csv
	m.from_file(filename);

	vector<string> mrow = m.get_row_names();
	set<string> have(mrow.begin(), mrow.end());
	vector<string> missing;
	for (const auto& n : par_names)
		if (have.find(n) == have.end())
			missing.push_back(n);
	if (!missing.empty())
	{
		ss << "EnifGraph::from_file(): graph file '" << filename << "' is missing "
			<< missing.size() << " adjustable parameters, e.g. '" << missing[0] << "'";
		frec << ss.str() << endl;
		frec << "   a partially covered graph would silently drop parameters from the "
			<< "prior precision, so this is treated as an error" << endl;
		throw runtime_error(ss.str());
	}

	//reorder to the solve ordering
	Mat sub = m.get(par_names, par_names);
	Eigen::SparseMatrix<double> a = sub.get_matrix();

	//structural pattern only: any non-zero is an edge, the value is ignored.
	//symmetrise (G union G^T) and force the diagonal on, so a thresholded
	//correlation matrix, a localiser product or a hand-built adjacency all mean
	//the same thing.
	int p = (int)par_names.size();
	vector<Eigen::Triplet<double>> trips;
	trips.reserve(a.nonZeros() * 2 + p);
	for (int k = 0; k < a.outerSize(); k++)
		for (Eigen::SparseMatrix<double>::InnerIterator it(a, k); it; ++it)
		{
			if (it.value() == 0.0)
				continue;
			trips.push_back(Eigen::Triplet<double>((int)it.row(), (int)it.col(), 1.0));
			trips.push_back(Eigen::Triplet<double>((int)it.col(), (int)it.row(), 1.0));
		}
	for (int i = 0; i < p; i++)
		trips.push_back(Eigen::Triplet<double>(i, i, 1.0));

	adj.resize(p, p);
	//duplicate entries are summed by setFromTriplets, so clamp back to a pattern
	adj.setFromTriplets(trips.begin(), trips.end());
	for (int k = 0; k < adj.outerSize(); k++)
		for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it)
			it.valueRef() = 1.0;
	adj.makeCompressed();

	names = par_names;
	nnz_offdiag = (int)adj.nonZeros() - p;
	initialized = true;
	report(frec);
}


void EnifGraph::report(ofstream& frec) const
{
	if (!initialized)
		return;
	int p = (int)names.size();
	//degree statistics tell the user whether the graph they supplied is sane
	int mind = p, maxd = 0;
	double sumd = 0.0;
	for (int i = 0; i < p; i++)
	{
		int d = (int)(adj.col(i).nonZeros()) - 1;
		mind = min(mind, d);
		maxd = max(maxd, d);
		sumd += d;
	}
	frec << endl << "  ---  enif conditional-independence graph  ---  " << endl;
	frec << "...nodes (adjustable parameters): " << p << endl;
	frec << "...edges: " << num_edges() << endl;
	frec << "...neighbours per node: min " << mind << ", mean "
		<< setprecision(3) << (sumd / (double)p) << ", max " << maxd << endl;
	frec << "...density: " << setprecision(4)
		<< (100.0 * (double)adj.nonZeros() / ((double)p * (double)p)) << " percent" << endl;
	if (maxd == 0)
		frec << "...WARNING: the graph has no edges - the prior precision will be "
		<< "diagonal, which discards all parameter correlation" << endl;
}


void EnifGraph::estimate_precision(const Eigen::MatrixXd& anomalies, double shrink,
	ofstream& frec)
{
	stringstream ss;
	if (!initialized)
		throw runtime_error("EnifGraph::estimate_precision(): no graph");
	int p = (int)names.size();
	if (anomalies.rows() != p)
		throw runtime_error("EnifGraph::estimate_precision(): anomaly rows != graph nodes");
	int nreal = (int)anomalies.cols();

	//Estimate the precision through its cholesky-like factor, which guarantees a
	//symmetric positive definite result.  For node i, regress its anomalies on the
	//anomalies of the graph neighbours that PRECEDE it in the solve ordering:
	//
	//    a_i = sum_{j in pred(i)} b_j a_j + e_i,   var(e_i) = d_i
	//
	//then the row of the upper-triangular factor is L(i,i) = 1/sqrt(d_i) and
	//L(i,j) = -b_j/sqrt(d_i), and Lam = L^T L.  Restricting to predecessors is
	//what makes the factor triangular; the resulting precision carries the full
	//(symmetric) graph sparsity.
	vector<Eigen::Triplet<double>> ltrips;
	ltrips.reserve(adj.nonZeros());
	int n_shrunk = 0, max_pred = 0;

	for (int i = 0; i < p; i++)
	{
		vector<int> pred;
		for (Eigen::SparseMatrix<double>::InnerIterator it(adj, i); it; ++it)
		{
			int j = (int)it.row();
			if (j < i)
				pred.push_back(j);
		}
		//never ask for more predecessors than the ensemble can support
		if ((int)pred.size() > nreal - 2)
		{
			pred.resize(max(0, nreal - 2));
			n_shrunk++;
		}
		max_pred = max(max_pred, (int)pred.size());

		double var_i = anomalies.row(i).squaredNorm();
		double d_i = var_i;
		Eigen::VectorXd b;

		if (!pred.empty())
		{
			Eigen::MatrixXd Ap(pred.size(), nreal);
			for (size_t k = 0; k < pred.size(); k++)
				Ap.row(k) = anomalies.row(pred[k]);
			Eigen::MatrixXd G = Ap * Ap.transpose();
			//stein-type pull toward the diagonal: needed when a neighbourhood
			//approaches the ensemble size, where the local gram is ill-conditioned
			double tr = G.trace() / (double)pred.size();
			G.diagonal().array() += max(shrink * tr, 1.0e-12 * max(tr, 1.0));
			Eigen::VectorXd rhs = Ap * anomalies.row(i).transpose();
			b = G.ldlt().solve(rhs);
			d_i = (anomalies.row(i).transpose() - Ap.transpose() * b).squaredNorm();
		}

		//a degenerate residual means this node is fully explained by its
		//neighbours in-sample; floor it rather than divide by ~zero
		double floor_d = 1.0e-10 * max(var_i, 1.0);
		if (!(d_i > floor_d))
			d_i = floor_d;

		double s = 1.0 / sqrt(d_i);
		ltrips.push_back(Eigen::Triplet<double>(i, i, s));
		for (size_t k = 0; k < pred.size(); k++)
			if (b[k] != 0.0)
				ltrips.push_back(Eigen::Triplet<double>(i, pred[k], -b[k] * s));
	}

	Eigen::SparseMatrix<double> L(p, p);
	L.setFromTriplets(ltrips.begin(), ltrips.end());
	prec = (Eigen::SparseMatrix<double>)(L.transpose() * L);
	prec.makeCompressed();

	solver.compute(prec);
	if (solver.info() != Eigen::Success)
		throw runtime_error("EnifGraph::estimate_precision(): failed to factor the "
			"estimated prior precision - try a sparser graph or more shrinkage");
	prec_ready = true;

	frec << "...estimated prior precision on the graph: " << prec.nonZeros()
		<< " non-zeros, density " << setprecision(4)
		<< (100.0 * (double)prec.nonZeros() / ((double)p * (double)p)) << " percent" << endl;
	frec << "...largest neighbourhood used: " << max_pred << " of " << nreal
		<< " realizations" << endl;
	if (n_shrunk > 0)
		frec << "...WARNING: " << n_shrunk << " nodes had more neighbours than the "
		<< "ensemble can support and were truncated - consider a sparser graph "
		<< "or more realizations" << endl;
}


Eigen::MatrixXd EnifGraph::apply_cov(const Eigen::MatrixXd& M) const
{
	if (!prec_ready)
		throw runtime_error("EnifGraph::apply_cov(): precision not estimated");
	//C * M == solve(Lam, M): the covariance implied by the graph is never formed
	Eigen::MatrixXd out = solver.solve(M);
	if (solver.info() != Eigen::Success)
		throw runtime_error("EnifGraph::apply_cov(): sparse solve failed");
	return out;
}


Eigen::MatrixXd EnifGraph::information_step(const Eigen::SparseMatrix<double>& H,
	const Eigen::VectorXd& rinv, const Eigen::MatrixXd& e,
	const Eigen::MatrixXd& resid, double lam, ofstream& frec) const
{
	if (!prec_ready)
		throw runtime_error("EnifGraph::information_step(): precision not estimated");
	int p = (int)names.size();

	//H^T Rinv H stays sparse because H is sparse - that is the whole reason the
	//graph is worth having.  with a dense H this product would be a full p x p
	//matrix and the sparse cholesky below would be pointless.
	Eigen::SparseMatrix<double> Rinv((int)H.rows(), (int)H.rows());
	{
		vector<Eigen::Triplet<double>> t;
		t.reserve(H.rows());
		for (int i = 0; i < (int)H.rows(); i++)
			t.push_back(Eigen::Triplet<double>(i, i, rinv[i]));
		Rinv.setFromTriplets(t.begin(), t.end());
	}
	Eigen::SparseMatrix<double> HtRH = (H.transpose() * Rinv * H).eval();

	Eigen::SparseMatrix<double> lam_post = ((1.0 + lam) * prec + HtRH).eval();
	lam_post.makeCompressed();

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> post_solver;
	post_solver.compute(lam_post);
	if (post_solver.info() != Eigen::Success)
		throw runtime_error("EnifGraph::information_step(): failed to factor the "
			"posterior precision");

	//gradient of the rml objective at the current iterate
	Eigen::MatrixXd wresid = rinv.asDiagonal() * resid;
	Eigen::MatrixXd g = prec * e + H.transpose() * wresid;
	Eigen::MatrixXd delta = -post_solver.solve(g);
	if (post_solver.info() != Eigen::Success)
		throw runtime_error("EnifGraph::information_step(): posterior solve failed");

	frec << "...posterior precision: " << lam_post.nonZeros() << " non-zeros, density "
		<< setprecision(4)
		<< (100.0 * (double)lam_post.nonZeros() / ((double)p * (double)p))
		<< " percent (H contributed " << HtRH.nonZeros() << ")" << endl;
	return delta;
}


Eigen::SparseMatrix<double> estimate_sparse_H(const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& B, double lasso_frac, int num_threads,
	Eigen::VectorXd& unexplained, ofstream& frec)
{
	int p = (int)A.rows();
	int nreal = (int)A.cols();
	int nobs = (int)B.rows();
	unexplained.setZero(nobs);

	//squared norms of each parameter's anomaly vector - the coordinate descent
	//denominators, computed once
	Eigen::VectorXd anorm(p);
	for (int j = 0; j < p; j++)
		anorm[j] = A.row(j).squaredNorm();

	vector<vector<Eigen::Triplet<double>>> row_trips(nobs);

	//each observation row is an independent lasso problem
	auto solve_row = [&](int i)
	{
		Eigen::VectorXd b = B.row(i).transpose();
		//penalty scaled by the smallest value that would zero the whole row, so
		//lasso_frac is dimensionless
		double amax = 0.0;
		for (int j = 0; j < p; j++)
			amax = max(amax, abs(A.row(j).dot(b)));
		double alpha = lasso_frac * amax;

		Eigen::VectorXd x = Eigen::VectorXd::Zero(p);
		Eigen::VectorXd r = b;
		for (int sweep = 0; sweep < 100; sweep++)
		{
			double max_chg = 0.0;
			for (int j = 0; j < p; j++)
			{
				if (anorm[j] <= 0.0)
					continue;
				double xj = x[j];
				double rho = A.row(j).dot(r) + anorm[j] * xj;
				double nx = 0.0;
				if (rho > alpha)
					nx = (rho - alpha) / anorm[j];
				else if (rho < -alpha)
					nx = (rho + alpha) / anorm[j];
				if (nx != xj)
				{
					r -= A.row(j).transpose() * (nx - xj);
					x[j] = nx;
					max_chg = max(max_chg, abs(nx - xj));
				}
			}
			if (max_chg < 1.0e-10)
				break;
		}
		//residual variance is what the observation error gets inflated by
		unexplained[i] = r.squaredNorm() * (double)(nreal - 1) / (double)nreal;
		for (int j = 0; j < p; j++)
			if (x[j] != 0.0)
				row_trips[i].push_back(Eigen::Triplet<double>(i, j, x[j]));
	};

	if (num_threads < 2)
	{
		for (int i = 0; i < nobs; i++)
			solve_row(i);
	}
	else
	{
		vector<thread> threads;
		std::atomic<int> next(0);
		int nt = min(num_threads, nobs);
		for (int t = 0; t < nt; t++)
			threads.push_back(thread([&]() {
				int i;
				while ((i = next.fetch_add(1)) < nobs)
					solve_row(i);
				}));
		for (auto& th : threads)
			th.join();
	}

	vector<Eigen::Triplet<double>> all;
	for (auto& rt : row_trips)
		all.insert(all.end(), rt.begin(), rt.end());
	Eigen::SparseMatrix<double> H(nobs, p);
	H.setFromTriplets(all.begin(), all.end());
	H.makeCompressed();

	frec << "...sparse H: " << H.nonZeros() << " of " << ((long)nobs * (long)p)
		<< " entries (" << setprecision(3)
		<< (100.0 * (double)H.nonZeros() / ((double)nobs * (double)p))
		<< " percent), lasso fraction " << lasso_frac << endl;
	frec << "...mean unexplained variance per observation: "
		<< unexplained.mean() << endl;
	return H;
}
