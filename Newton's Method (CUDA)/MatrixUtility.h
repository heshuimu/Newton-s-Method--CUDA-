//
//  MaxtrixUtility.h
//  LDLt Solver
//
//  Created by 雪竜 on 16/2/24.
//  Copyright © 2016年 雪竜. All rights reserved.
//
#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>

namespace MatrixUtility {
	
	double** InitializeEmptyMatrix(int x, int y)
	{
		double** m = new double*[x];
		for(int i = 0; i <  x; i++)
		{
			m[i] = new double[y];
			for(int j = 0; j <  y; j++)
			{
				m[i][j] = 0;
			}
		}
		
		return m;
	}
	
	double** GetNewSymmetricMatrix(int d)
	{
		double** m = InitializeEmptyMatrix(d, d);
		int token = 1;
		
		for(int i = 0; i <  d; i++)
		{
			for(int j = 0; j <=  i; j++)
			{
				m[i][j] = token;
				m[j][i] = token++;
			}
		}
		
		return m;
	}
	
	double** GetNewSquareMatrix(int d)
	{
		double** m = InitializeEmptyMatrix(d, d);		
		int token = 1;
		
		for(int i = 0; i <  d; i++)
		{
			for(int j = 0; j <=  d; j++)
			{
				m[i][j] = token++;
			}
		}
		
		return m;
	}
	
	void PrintMatrix(double** m, int x, int y)
	{
		for(int i = 0; i <  x; i++)
		{
			for(int j = 0; j <  y; j++)
			{
				std::cout  << std::setw(12) << m[i][j];
			}
			std::cout << std::endl;
		}
	}
	
	double** GetMatrixTranspose(double** m, int x, int y)
	{
		double** t = new double*[y];
		for(int i = 0; i <  y; i++)
		{
			t[i] = new double[x];
		}
		
		for(int i = 0; i <  x; i++)
		{
			for(int j = 0; j <  y; j++)
			{
				t[j][i] = m[i][j];
			}
		}
		
		return t;
	}
	
	void DeleteMatrix(double** m, int x, int y)
	{
		for(int i = 0; i <  x; i++)
		{
			delete[] m[i];
		}
		delete[] m;
	}
	
	double* InitializeEmptyVector(int n)
	{
		double* v = new double[n];
		for(int i = 0; i <  n; i++)
		{
			v[i] = 0;
		}
		return v;
	}
	
	void PrintVector(double* v, int n)
	{
		for(int i = 0; i < n; i++)
		{
			std::cout << v[i] << std::endl;;
		}
		std::cout << std::endl;
	}
	
	double* ExtractDiagnoal(double** m, int n)
	{
		double* v = new double[n];
		for(int i = 0; i <  n; i++)
		{
			v[i] = m[i][i];
		}
		return v;
	}
	
	double ComputeResidual(double* v1, int n)
	{
		double val = 0;

		for (int i = 0; i < n; i++)
		{
			double diff = v1[i];
			val += diff * diff;
		}

		return sqrt(val);
	}

	double GetMatrixUnitFromVector(double* v, int w, int x, int y)
	{
		return v[w * x + y];
	}

	void TransposeMatrixAsVector(double* v, double* t, int x, int y)
	{
		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				t[y*j + i] = GetMatrixUnitFromVector(v, y, i, j);
			}
		}
	}

	void PrintVectorAsMatrix(double* vector, int width, int height)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				std::cout << std::setw(12) << vector[width * i + j];
			}
			std::cout << std::endl;
		}
	}

	double** GetMatrixFromFlattenVector(double* v, int x, int y)
	{
		double** m = InitializeEmptyMatrix(x, y);

		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				m[i][j] = GetMatrixUnitFromVector(v, y, i, j);
			}
		}

		return m;
	}
	
}
