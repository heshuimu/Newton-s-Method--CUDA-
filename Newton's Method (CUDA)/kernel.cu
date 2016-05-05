#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixUtility.h"
#include "LDLUtility.h"


int dimension = 10;

#define ___PROBLEM_SIZE___ dimension
#define ___TOLERANCE___ 0.0001

void GetABForCalculation(double* X, double* A, double* B, int dim, int blocks, int threads);

void LUDecomposition(double* A, double* A_res, int dim);

double ComputeResidual_dev(double* v, int dim, int blocks, int threads);

__device__ double F_xn(const double* v, const int n, const int dim)
{
	int res = 0;

	if (n < dim - 1)
		res = 400 * (v[n] * v[n] - v[n + 1]) * v[n] + 2 * (v[n] - 1);

	if (n > 0)
		res += -200 * (v[n - 1] * v[n - 1] - v[n]);

	return res;
}

__device__ double F_xn_xn(const double* v, const int n, const int dim) 
{
	int res = 0;

	if (n < dim - 1)
		res = 1200 * v[n] * v[n] - 400 * v[n + 1] + 2;

	if (n > 0)
		res += 200;

	return res;
}

__device__ double F_xn_xn_minus_1(const double* v, const int n) 
{
	return -400 * v[n - 1];
}

__device__ double F_xn_xn_plus_1(const double* v, const int n) 
{
	return -400 * v[n];
}

__global__ void F_gradient_dev(double* v, double* res, const int dim, const int offset) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x + offset;

	if (i < dim)
		res[i] = -F_xn(v, i, dim);
}

__global__  void F_hessian_dev(double* v, double* res, const int dim, const int offset)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x + offset;

	if (i < dim)
	{
		res[i*dim + i] = F_xn_xn(v, i, dim);

		if (i < dim - 1)
		{
			res[i*dim + i + 1] = F_xn_xn_plus_1(v, i);
		}

		if (i > 0)
		{
			res[i*dim + i - 1] = F_xn_xn_minus_1(v, i);
		}
	}
}

__global__ void LU_scale(double *a, int dim, int offset)
{
	int start = (offset*dim + offset);
	int end = (offset*dim + dim);

	for (int i = start + 1; i < end; i++)
	{
		a[i] = (a[i] / a[start]);
	}
}

__global__ void LU_reduce(double *a, int dim, int offset)
{
	int start = ((offset + blockIdx.x + 1)*dim + offset);
	int end = ((offset + blockIdx.x + 1)*dim + dim);

	for (int i = start + 1; i < end; i++)
	{
		a[i] = a[i] - (a[start] * a[(offset*dim) + (offset + (i - start))]);
	}
}

__global__ void Residual_add(double* v, double* sum, const int dim, const int offset)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x + offset;

	if (i < dim)
		sum[0] += v[i] * v[i];
}

int main(int argc, char** argv)
{
	int blocks = ___PROBLEM_SIZE___, threads = 1;
	double initial_guess = 0;
	
	if(argc > 1)
		___PROBLEM_SIZE___ = atoi(argv[1]);
	if(argc > 2)
		blocks = atoi(argv[2]);
	if(argc > 3)
		threads = atoi(argv[3]);
	if(argc > 4)
		initial_guess = atof(argv[4]);

	double* A = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___ * ___PROBLEM_SIZE___);
	double* A_res = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___ * ___PROBLEM_SIZE___);
	double* A_t = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___ * ___PROBLEM_SIZE___);
	double* B = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___);
	double* X = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___);
	double* X_res = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___);
	double* Z = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___);
	double* Y = MatrixUtility::InitializeEmptyVector(___PROBLEM_SIZE___);

	for (int i = 0; i < ___PROBLEM_SIZE___; i++) 
	{
		X[i] = initial_guess;
	}
	
	timeval start, end;

	double residual = 0;
	
	gettimeofday(&start, NULL);

	do
	{
		GetABForCalculation(X, A, B, ___PROBLEM_SIZE___, blocks, threads);

		LUDecomposition(A, A_res, ___PROBLEM_SIZE___);

		MatrixUtility::TransposeMatrixAsVector(A_res, A_t, ___PROBLEM_SIZE___, ___PROBLEM_SIZE___);

		double** A_m = MatrixUtility::GetMatrixFromFlattenVector(A_t, ___PROBLEM_SIZE___, ___PROBLEM_SIZE___);
		double* A_d = MatrixUtility::ExtractDiagnoal(A_m, ___PROBLEM_SIZE___);

		LDLUtility::SolveXWithLD(A_m, ___PROBLEM_SIZE___, A_d, B, Z, Y, X_res);

		MatrixUtility::DeleteMatrix(A_m, ___PROBLEM_SIZE___, ___PROBLEM_SIZE___);
		delete[] A_d;

		if (___PROBLEM_SIZE___ <= 30)
		{
			printf("\nX:\n");
			for (int i = 0; i < ___PROBLEM_SIZE___; i++) {
				std::cout << std::setw(12) << X[i];
			}
			printf("\nA:\n");
			MatrixUtility::PrintVectorAsMatrix(A, ___PROBLEM_SIZE___, ___PROBLEM_SIZE___);
			printf("\nB:\n");
			for (int i = 0; i < ___PROBLEM_SIZE___; i++) {
				std::cout << std::setw(12) << B[i];
			}
			printf("\n:L & D in one matrix: \n");
			MatrixUtility::PrintVectorAsMatrix(A_t, ___PROBLEM_SIZE___, ___PROBLEM_SIZE___);
			printf("\n:Step of X for next iteration: \n");
			MatrixUtility::PrintVector(X_res, ___PROBLEM_SIZE___);
		}

		for (int i = 0; i < ___PROBLEM_SIZE___; i++)
		{
			X[i] += X_res[i];
		}

		//device residual method has WOR hazard, fallback to serial
		residual = MatrixUtility::ComputeResidual(X_res, ___PROBLEM_SIZE___);

		printf("\nResidual: %f\n", residual);

	} while (residual > ___TOLERANCE___);
	
	gettimeofday(&end, NULL);
	
	printf("\nRuntime is %f for %d blocks and %d threads for a problem size of %d. Initial guess is that x_i = %f\n", (end.tv_sec*1000 + end.tv_usec/1000) - (start.tv_sec*1000 + start.tv_usec/1000), blocks, threads, ___PROBLEM_SIZE___, initial_guess);
	printf("\nFinal X:\n");
	for (int i = 0; i < ___PROBLEM_SIZE___; i++) {
		std::cout << std::setw(12) << X[i];
	}
	
	cudaDeviceReset();
	
	return 0;
}

void GetABForCalculation(double* X, double* A, double* B, int dim, int blocks, int threads)
{
	double* A_dev = NULL, *B_dev = NULL, *X_dev = NULL;

	cudaSetDevice(0);

	cudaMalloc(&A_dev, sizeof(double) * dim * dim);
	cudaMalloc(&B_dev, sizeof(double) * dim);
	cudaMalloc(&X_dev, sizeof(double) * dim);

	cudaMemcpy(X_dev, X, sizeof(double) * dim, cudaMemcpyHostToDevice);

	int offset = 0;
	while (offset < dim)
	{
		//Any number of blocks or threads would work because there is no dependency involved 
		F_gradient_dev<<<blocks, threads>>>(X_dev, B_dev, dim, offset);
		F_hessian_dev<<<blocks, threads>>>(X_dev, A_dev, dim, offset);
		cudaDeviceSynchronize();
		offset += blocks*threads;
	}
	
	cudaMemcpy(B, B_dev, sizeof(double) * dim, cudaMemcpyDeviceToHost);
	cudaMemcpy(A, A_dev, sizeof(double) * dim * dim, cudaMemcpyDeviceToHost);

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(X_dev);

}

void LUDecomposition(double* A, double* A_res, int dim)
{
	double* A_dev = NULL;
	cudaMalloc(&A_dev, sizeof(double) * dim * dim);
	cudaMemcpy(A_dev, A, sizeof(double) * dim * dim, cudaMemcpyHostToDevice);

	for (int i = 0; i < dim; i++)
	{
		//Scale and reduce one row by one row because of the dependency
		LU_scale<<<1, 1>>>(A_dev, dim, i);
		LU_reduce<<<(dim - i - 1), 1>>>(A_dev, dim, i);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(A_res, A_dev, sizeof(double) * dim * dim, cudaMemcpyDeviceToHost);

	cudaFree(A_dev);
}

double ComputeResidual_dev(double* v, int dim, int blocks, int threads)
{
	double* sum = NULL;
	cudaMalloc(&sum, sizeof(double));

	double* v_dev = NULL;
	cudaMalloc(&v_dev, sizeof(double) * dim);
	cudaMemcpy(v_dev, v, sizeof(double) * dim, cudaMemcpyHostToDevice);

	int offset = 0;
	while (offset < dim)
	{
		Residual_add << <blocks, threads >> >(v_dev, sum, dim, offset);
		cudaDeviceSynchronize();
		offset += blocks*threads;
	}

	double sum_local = 0;

	cudaMemcpy(&sum_local, sum, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(sum);
	cudaFree(v_dev);

	return sum_local;;
}