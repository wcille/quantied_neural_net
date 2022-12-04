#include<stdio.h>
#include<math.h>
//#include<memory.h>
#include"Mat.h"
//#include<malloc.h>
#include <stdlib.h>
struct layer{
	int num;//神经元数
	Matrix *nerus;//神经元矩阵
	Matrix *nerusA;//激活后神经元矩阵
	struct weightnet *nwnet;//下层突触矩阵
	int lnum;//层数
};
struct weightnet{
	int num;//突触数
	int lnum;//行数
	int rnum;//列数
	Matrix *tc;//权重矩阵
	Matrix *qtc;//量化权重矩阵
};
struct net{
	FILE *ftdp,*ftlp,*ftestdp,*ftestlp; 
	struct layer *Flay;
	float studya;
	int laynum;
	int *lays;
	int trainnum;
	int testnum;
	int epochs;
	float cr;
};
float atan1(float x,Matrix *M);
float atanm(float x,Matrix *M);
float quantify(float da,Matrix *M);
float softmax(float z,Matrix *M);
void qweightnet(struct layer *flayer);//更新权重网络
void creatwnet(struct layer *flayer,int layernum);//生成突触矩阵
void printweight(struct layer *flayer,Matrix **dweight);//更新权重网络
struct layer * newbpnet(int layernum,int *layers);//新建网络
void updatenet(struct layer *flayer);//更新网络
void  printnet(struct layer *flayer);//打印网络
void getgrad(struct layer *flayer,Matrix *y0,Matrix *yloss,Matrix **theta,Matrix **dweight);//计算梯度
void upweightnet(struct layer *flayer,Matrix **dweight,float step);//更新权重网络
void addweight(struct layer *flayer,Matrix **dweight,Matrix **dweight1,float step);//更新权重网络
void Inputdata_net(struct layer *Flay,FILE *data,FILE *lab,Matrix *Out);//导入输入输出矩阵
void trainnet(struct layer *flayer,Matrix *y0,float step,int datasize,int ecoh,FILE *ftdp,FILE *ftlp);//训练网络--step步长，ecoh训练总轮数，y0正确输出
int Fmax(float *data,int num);//找最大值
float predict(struct layer *Flay,FILE *ftestdp,FILE *ftestlp,Matrix *y0,int num);//预测函数
void Train_Pre_net(struct net *net);