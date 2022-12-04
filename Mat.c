#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Mat.h"
#include <time.h>
Matrix* InitMatrix(int row,int line)				//初始化一个矩阵
{
	if (row>0 && line>0)
	{
		Matrix *matrix;
		matrix = (Matrix*)malloc(sizeof(Matrix));
		matrix->row = row;
		matrix->line = line;
		matrix->data = (float*)malloc(sizeof(float)*row*line);
		memset(matrix->data,0,sizeof(float)*row*line);
		return matrix;
	}
	else 
		return NULL;
} 

void ValueMatrix(Matrix *matrix,float *array) 	
{
	if (matrix->data != NULL)
	{
		memcpy(matrix->data, array, matrix->row*matrix->line*sizeof(float));
	}
}

int SizeMatrix(Matrix *matrix)
{
	return matrix->row*matrix->line;
}

void FreeMatrix(Matrix *matrix)
{
	free(matrix->data);		
	matrix->data = NULL;
	free(matrix);
	//printf("free\n");
}

void CopyMatrix(Matrix *matrix_A, Matrix *matrix_B)
{
	matrix_B->row = matrix_A->row;
	matrix_B->line = matrix_A->line;
	memcpy(matrix_B->data, matrix_A->data, SizeMatrix(matrix_A)*sizeof(float));
}

void PrintMatrix(Matrix *matrix)
{
	for (int i=0;i<SizeMatrix(matrix);i++)
	{
		printf("%f\t", matrix->data[i]);
		if ((i+1)%matrix->line == 0)
			printf("\n");
	}
			
}
//加法
Matrix* AddMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	if (matrix_A->row == matrix_B->row && matrix_A->line == matrix_B->line)
	{
		Matrix *matrix_C = InitMatrix(matrix_A->row,matrix_A->line);
		for (int i=0;i<matrix_A->line;i++)
		{
			for (int j=0;j<matrix_A->row;j++)
			{
				matrix_C->data[i*matrix_C->row + j] = \
				matrix_A->data[i*matrix_A->row + j] + matrix_B->data[i*matrix_A->row + j];
			}
		}
		return matrix_C;
	}
	else 
	{
		printf("不可相加\n");
		return NULL;
	}
}

Matrix* DecMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	if (matrix_A->row == matrix_B->row && matrix_A->line == matrix_B->line)
	{
		Matrix *matrix_C = InitMatrix(matrix_A->row,matrix_A->line);
		for (int i=0;i<matrix_A->line;i++)
		{
			for (int j=0;j<matrix_A->row;j++)
			{
				matrix_C->data[i*matrix_C->row + j] = \
				matrix_A->data[i*matrix_A->row + j] - matrix_B->data[i*matrix_A->row + j];
			}
		}
		return matrix_C;
	}
	else 
	{
		printf("不可相减\n");
		return NULL;
	}
}
Matrix* HadamaMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	if (matrix_A->row == matrix_B->row && matrix_A->line == matrix_B->line)
	{
		Matrix *matrix_C = InitMatrix(matrix_A->row,matrix_A->line);
		for (int i=0;i<matrix_A->line;i++)
		{
			for (int j=0;j<matrix_A->row;j++)
			{
				matrix_C->data[i*matrix_C->row + j] = \
				matrix_A->data[i*matrix_A->row + j] * matrix_B->data[i*matrix_A->row + j];
			}
		}
		return matrix_C;
	}
	else 
	{
		printf("fail to hadama\n");
		return NULL;
	}
}

Matrix* DotmultMatrix(Matrix *matrix_A,float step)
{

		Matrix *matrix_C = InitMatrix(matrix_A->row,matrix_A->line);
		for (int i=0;i<matrix_A->line;i++)
		{
			for (int j=0;j<matrix_A->row;j++)
			{
				matrix_C->data[i*matrix_C->row + j] = matrix_A->data[i*matrix_A->row + j]*step ;
			}
		}
		return matrix_C;
}
Matrix* FXMatrix(Matrix *matrix_A,float (*p)(float,Matrix*))
{

		Matrix *matrix_C = InitMatrix(matrix_A->row,matrix_A->line);
		for (int i=0;i<matrix_A->line;i++)
		{
			for (int j=0;j<matrix_A->row;j++)
			{
				matrix_C->data[i*matrix_C->row + j] = p(matrix_A->data[i*matrix_A->row + j],matrix_A);
			}
		}
		return matrix_C;
}
void RandomMatrix(Matrix *matrix_A,float upnum)
{

		for (int i=0;i<matrix_A->line;i++)
		{
			for (int j=0;j<matrix_A->row;j++)
			{
			matrix_A->data[i*matrix_A->row + j]=upnum*(rand()/32767.0-0.5);
			}
		}
}
//乘法
Matrix* MulMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	if (matrix_A->line == matrix_B->row)		
	{
		Matrix *matrix_C = InitMatrix(matrix_A->row,matrix_B->line);
		// matrix_C->line = matrix_A->line;	
		// matrix_C->row = matrix_B->row;		
		for (int i=0;i<matrix_A->row;i++)
		{
			for (int j=0;j<matrix_B->line;j++)
			{
				matrix_C->data[i*matrix_C->line + j]=0;
				for (int k=0;k<matrix_A->line;k++)
				{
					matrix_C->data[i*matrix_C->line + j] += \
					matrix_A->data[i*matrix_A->line + k] * matrix_B->data[k*matrix_B->line + j];
				//printf("%f*%f\n",matrix_A->data[i*matrix_A->line + k],matrix_B->data[k*matrix_B->row + j]);
				}
				
			}
		}
		return matrix_C;
	}
	else
	{
		printf("fail to x\n");
		return NULL;
	}
}

//矩阵转置
Matrix * TransMatrix(Matrix *matrix)			
{
	Matrix *matrixTemp = InitMatrix(matrix->row,matrix->line);       
		matrixTemp->line=matrix->row;
		matrixTemp->row=matrix->line;
		for (int i=0;i<matrixTemp->row;i++)
		{
			for (int j=0;j<matrixTemp->line;j++)
			{
				matrixTemp->data[i*matrixTemp->line + j] = matrix->data[j*matrix->line + i];
			}
		}
	return matrixTemp;
}