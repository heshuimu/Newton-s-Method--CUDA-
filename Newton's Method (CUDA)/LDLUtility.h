//
//  LDLUtility.h
//  LDLt Solver
//
//  Created by 雪竜 on 16/2/24.
//  Copyright © 2016年 雪竜. All rights reserved.
//

#pragma once

#include "MatrixUtility.h"

namespace LDLUtility {
	
	void GetLDFactor(double** A, int dim, double** L, double* D)
	{
		for (int i = 0; i < dim; i++)
		{
			for(int j = 0; j <= i; j++)
			{
				//Compute diagnal matrix
				double t = 0;
				double s = 0;
				for(int k = 0; k < j; k++)
				{
					t += L[j][k] * L[j][k] * D[k];
					s += L[i][k] * D[k] * L[j][k];
				}
				D[j] = A[j][j]  - t;
				L[i][j] = (A[i][j] - s) / D[j];
			}
		}
	}
	
	void SolveXWithLD(
					  double** L,
					  int dim,
					  double* D,
					  double* B,
					  double* Z,
					  double* Y,
					  double* X)
	{
		for (int i = 0; i < dim; i++)
		{
			double t = 0;
			for(int k = 0; k < i; k++)
			{
				t += L[i][k] * Z[k];
			}
			Z[i] = B[i] - t;
		}
		
		for (int i = 0; i < dim; i++)
		{
			Y[i] = Z[i] / D[i];
		}
		
		for (int i = dim - 1; i >=0; i--)
		{
			double t = 0;
			for(int k = i + 1; k < dim; k++)
			{
				t +=  L[k][i] * X[k];
			}
			X[i] = Y[i] - t;
		}
	}
}