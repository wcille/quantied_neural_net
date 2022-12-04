typedef struct 
{
	int row,line;	                           	//line为行,row为列
	float *data;
}Matrix;
void ValueMatrix(Matrix *matrix,float *array);				//给一个矩阵赋值
int SizeMatrix(Matrix *matrix);								//获得一个矩阵的大小
void FreeMatrix(Matrix *matrix);							//释放一个矩阵
void CopyMatrix(Matrix *matrix_A, Matrix *matrix_B);		//复制一个矩阵的值
void PrintMatrix(Matrix *matrix);	//打印一个矩阵
void RandomMatrix(Matrix *matrix_A,float upnum);			//随机化矩阵			
//矩阵的基本运算
Matrix* InitMatrix(int row,int line);		          //初始化矩阵
Matrix* AddMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的加法
Matrix* DecMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的jian法
Matrix* MulMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的乘法
Matrix* TransMatrix(Matrix *matrix);			
Matrix* DotmultMatrix(Matrix *matrix_A,float step);
Matrix* FXMatrix(Matrix *matrix_A,float (*p)(float,Matrix*));//F(M)
Matrix* HadamaMatrix(Matrix *matrix_A,Matrix *matrix_B);