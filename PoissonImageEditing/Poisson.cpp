// Copyright (C) 2019 Amazing Zhen <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "Poisson.h"

#include <chrono>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif

#include <opencv2/core/eigen.hpp>

using namespace Eigen;

/*
	This anonymous namespace actual contains the kernel of the algorithm.
*/
namespace {
	// Set off size for neighbours
	const int noff = 4;
	int offx[] = { 0, -1, 1, 0 };
	int offy[] = { -1, 0, 0, 1 };
	//const int noff = 8;
	//int offx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	//int offy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

	// Kernal function.
	void construct_linear_system(
		const vector<MatrixXd> &src,
		const vector<MatrixXd> &dst,
		const MatrixXi &mask,
		const bool mixing_gradients,
		SparseMatrix<double> &Lmat,
		vector<VectorXd> &Lbvec);

	/*
		Lmat * x = Lbvec
	*/
	void solve_linear_system(
		const SparseMatrix<double> &Lmat,
		const vector<VectorXd> &Lbvec,
		vector<VectorXd> &x);

	void cv_to_eigen(const Mat &img, MatrixXd &R, MatrixXd &G, MatrixXd &B);

	void eigen_to_cv(const MatrixXd &R, const MatrixXd &G, const MatrixXd &B, Mat &img);

	void create_index_from_mask(const MatrixXi &mask, MatrixXi &MI, MatrixXi &IM);

	void create_boundary_from_mask(const MatrixXi &mask, MatrixXi &MB);
}

Mat poisson_image_editing(Mat src_, Mat dst_, Mat mask_, const bool mixed)
{
	/*
		Preprocessing image data.
	*/
	if (src_.channels() != dst_.channels()) {
		throw std::invalid_argument("channels of src and dst is not equal!");
	}

	if (src_.channels() != 3 && src_.channels() != 1) {
		throw std::invalid_argument("channels of src and dst only surpport 3 or 1!");
	}

	const int channels = src_.channels();

	vector<MatrixXd> src(channels), dst(channels);
	if (channels == 3) {  // RGB
		cv_to_eigen(src_, src[0], src[1], src[2]);
		cv_to_eigen(dst_, dst[0], dst[1], dst[2]);
	} else {
		cv2eigen(src_, src[0]);
		cv2eigen(dst_, dst[0]);
	}

	MatrixXi mask;
	cv2eigen(mask_, mask);

	MatrixXi MI, IM;
	create_index_from_mask(mask, MI, IM);

	/*
		Solve Poisson equation with Dirichlet boundary conditions.
	*/

	SparseMatrix<double> Lmat;
	vector<VectorXd> Lbvec;
	construct_linear_system(src, dst, mask, mixed, Lmat, Lbvec);

	vector<VectorXd> res;
	solve_linear_system(Lmat, Lbvec, res);

	/*
		Postprocessing image data and return.
	*/
	const int n = IM.rows();
	const int rows = MI.rows();
	const int cols = MI.cols();

	for (int ch = 0; ch < channels; ch++) {
		for (int i = 0; i < n; i++) {
			int r = IM(i, 0), c = IM(i, 1);

			dst[ch](r, c) = res[ch](i);
			//dst[ch](r, c) = src[ch](r, c);
		}
	}

	Mat seamless_cloning_res;
	eigen_to_cv(dst[0], dst[1], dst[2], seamless_cloning_res);

	return seamless_cloning_res;
}

namespace {
	void construct_linear_system(
		const vector<MatrixXd> &src,
		const vector<MatrixXd> &dst,
		const MatrixXi &mask,
		const bool mixing_gradients,
		SparseMatrix<double> &Lmat,
		vector<VectorXd> &Lbvec)
	{
		if (src.size() != dst.size()) {
			throw std::invalid_argument("channels of src and dst is not equal!");
		}

		MatrixXi MI, IM;
		create_index_from_mask(mask, MI, IM);

		/*
			Construct Lmat from Poisson equations
		*/

		typedef Eigen::Triplet<double> tri;
		vector<tri> tri_list;

		const int n = IM.rows();
		const int rows = MI.rows();
		const int cols = MI.cols();

		for (int i = 0; i < n; i++) {
			int r = IM(i, 0), c = IM(i, 1);
			int count = 0;

			for (int k = 0; k < noff; ++k) {
				int kr = r + offy[k];
				int kc = c + offx[k];

				if (kr >= 0 && kr < rows && kc >= 0 && kc < cols) {
					count++;

					int ki = MI(kr, kc);
					if (ki != -1) {
						tri_list.push_back(tri(i, ki, -1));
					}
				}
			}

			tri_list.push_back(tri(i, i, count));
		}

		Lmat = SparseMatrix<double>(n, n);
		Lmat.setFromTriplets(tri_list.begin(), tri_list.end());

		/*
			Construct Lbvec from Dirichlet boundary conditions.
		*/

		const int channels = src.size();
		Lbvec = vector<VectorXd>(channels);

		for (int ch = 0; ch < channels; ch++) {
			VectorXd bvec = VectorXd::Zero(n);

			for (int i = 0; i < n; i++) {
				int r = IM(i, 0), c = IM(i, 1);
				
				for (int k = 0; k < noff; ++k) {
					int kr = r + offy[k];
					int kc = c + offx[k];

					if (kr >= 0 && kr < rows && kc >= 0 && kc < cols) {
						if (mixing_gradients) {
							double src_grad = src[ch](r, c) - src[ch](kr, kc);
							double dst_grad = dst[ch](r, c) - dst[ch](kr, kc);
							bvec(i) += (std::abs(src_grad) > std::abs(dst_grad) ? src_grad : dst_grad);
						}
						else {
							bvec(i) += (src[ch](r, c) - src[ch](kr, kc));
						}

						int ki = MI(kr, kc);
						if (ki == -1) { // This is the actual boundary condition
							bvec(i) += dst[ch](kr, kc);
						}
					}
				}
			}

			Lbvec[ch] = bvec;
		}
	}

	void solve_linear_system(
		const SparseMatrix<double> &Lmat,
		const vector<VectorXd> &Lbvec,
		vector<VectorXd> &x)
	{
		SparseMatrix<double> Amat = Lmat.transpose() * Lmat;

		// Sparse solving
		auto t1 = std::chrono::high_resolution_clock::now();

#ifdef EIGEN_USE_MKL_ALL
		PardisoLDLT<SparseMatrix<double>> solver;
#else
		SimplicialLDLT<SparseMatrix<double>> solver;
#endif
		solver.compute(Amat);
		if (solver.info() != Success)
			throw std::runtime_error("decomposition failed");

		const int channels = Lbvec.size();
		x = vector<VectorXd>(channels);

		for (int ch = 0; ch < channels; ch++) {
			VectorXd bvec = Lmat.transpose() * Lbvec[ch];

			VectorXd res = solver.solve(bvec);
			if (solver.info() != Success)
				throw std::runtime_error("sovling failed");

			x[ch] = res;
		}

		auto t2 = std::chrono::high_resolution_clock::now();
		auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout << "Sparse solving time: " << timespan.count() << " seconds.\n";
	}


	void cv_to_eigen(const Mat &img, MatrixXd &R, MatrixXd &G, MatrixXd &B)
	{
		vector<Mat> bgr;
		cv::split(img, bgr);
		cv::cv2eigen(bgr[0], B);
		cv::cv2eigen(bgr[1], G);
		cv::cv2eigen(bgr[2], R);
	}

	void eigen_to_cv(const MatrixXd &R, const MatrixXd &G, const MatrixXd &B, Mat &img)
	{
		vector<Mat> bgr(3);
		cv::eigen2cv(B, bgr[0]);
		cv::eigen2cv(G, bgr[1]);
		cv::eigen2cv(R, bgr[2]);
		cv::merge(bgr, img);
	}

	void create_index_from_mask(const MatrixXi &mask, MatrixXi &MI, MatrixXi &IM) {
		const int rows = mask.rows(), cols = mask.cols();

		MI.setConstant(rows, cols, -1);
		IM.setConstant(rows * cols, 2, -1);

		int count = 0;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (mask(r, c) == 255) {
					MI(r, c) = count;  // mat to index
					IM(count, 0) = r;  // index to mat
					IM(count, 1) = c;
					count++;
				}
			}
		}

		IM.conservativeResize(count, 2);
	}

	void create_boundary_from_mask(const MatrixXi &mask, MatrixXi &MB) {
		const int rows = mask.rows(), cols = mask.cols();

		MB.setConstant(rows, cols, 0);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (mask(r, c) != 255) {
					for (int k = 0; k < noff; ++k) {
						int kr = r + offy[k];
						int kc = c + offx[k];

						if (kr >= 0 && kr < rows && kc >= 0 && kc < cols) {
							if (mask(kr, kc) == 255) {
								MB(r, c) = 255;
								break;
							}
						}
					}
				}
			}
		}

		Mat boundary;
		eigen2cv(MB, boundary);
		imwrite("boundary.png", boundary);
	}
}
