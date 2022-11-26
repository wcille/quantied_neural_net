#include<stdio.h>
#include<math.h>
#include<memory.h>
#include"Mat.h"
#include<malloc.h>
struct layer{
	int num;//神经元数
	Matrix *nerus;//神经元矩阵
	Matrix *nerusA;//神经元矩阵
	struct weightnet *nwnet;//下层突触矩阵
	int lnum;//层数
};
struct weightnet{
	int num;//突触数
	int lnum;//行数
	int rnum;//列数
	Matrix *tc;//权重矩阵
};
double atan1(double x)
{
	return 1/(1+x*x);
}
double quantify(double da,int level,double kedu)
{
	if(da<0)
	da=da-0.000001;
	else
	da=da+0.000001;
	switch(int(da/kedu)*2/(level-1))
	{
		case 0: return int(da/kedu)*kedu;break;
		default:if(da<0) return kedu*(1-level)/2; else return kedu*(level-1)/2;break;
	}
	
}
void creatwnet(struct layer *flayer,int layernum)//生成突触矩阵
{
	int ii=0;
	flayer[0].nwnet=(struct weightnet *)malloc((layernum-1)*sizeof(struct weightnet));
	for(ii=0;ii<layernum-1;ii++)
	{
		flayer[0].nwnet[ii].num=flayer[ii].num*flayer[ii+1].num;
		flayer[0].nwnet[ii].lnum=flayer[ii].num;
		flayer[0].nwnet[ii].rnum=flayer[ii+1].num;
		flayer[0].nwnet[ii].tc=InitMatrix(flayer[ii].num,flayer[ii+1].num);
		RandomMatrix(flayer[0].nwnet[ii].tc,1);
		//printf("%dx%d\n",flayer[0].nwnet[ii].tc->row,flayer[0].nwnet[ii].tc->line);
	}
}
struct layer * newbpnet(int layernum,int *layers)//新建网络
{
	int i=0,j=0,k=0;
	struct layer *flayer;
	flayer=(struct layer*)malloc(layernum*sizeof(struct layer));
	flayer[0].lnum=layernum;
	for(i=0;i<layernum;i++)
	{
		flayer[i].num=layers[i];
		flayer[i].nerus=InitMatrix(1,flayer[i].num);
		flayer[i].nerusA=InitMatrix(1,flayer[i].num);
	}
	creatwnet(flayer,layernum);
	return flayer;
};

void updatenet(struct layer *flayer)//更新网络
{
	int i=0,j=0,laynum=0;
	Matrix *M;
	laynum=(*flayer).lnum;
	for(i=0;i<laynum-1;i++)
		{
			M=MulMatrix(flayer[i].nerusA,flayer[0].nwnet[i].tc);
			CopyMatrix(M, flayer[i+1].nerus);		
			FreeMatrix(M);
			M=NULL;
			
			M=FXMatrix(flayer[i+1].nerus,atan);//激活函数aan
			CopyMatrix(M, flayer[i+1].nerusA);		
			FreeMatrix(M);
			
			M=NULL;			
		}
}

void  printnet(struct layer *flayer)//打印网络
{
	int i=0,j=0,k=0,laynum=0;
	laynum=(*flayer).lnum;
	for(i=0;i<laynum-1;i++)	
	{
		PrintMatrix(flayer[i].nerusA);							//打印一个矩阵
		PrintMatrix(flayer[0].nwnet[i].tc);	
	}
	PrintMatrix(flayer[laynum-1].nerus);	
	printf("\n");
}

void getgrad(struct layer *flayer,Matrix *y0,Matrix *yloss,Matrix **theta,Matrix **dweight)//计算梯度
{
	Matrix *M,*M1;
	int i=0,j=0;
	yloss=DecMatrix(y0,flayer[flayer->lnum-1].nerusA);//y=Y'
	
	M1=FXMatrix(flayer[flayer->lnum-1].nerus,atan1);//loss*激活函数导数
	M=HadamaMatrix(yloss,M1);
	CopyMatrix(M, theta[0]);
	FreeMatrix(M1);	
	M1=NULL;
	FreeMatrix(M);
	M=NULL;	
	FreeMatrix(yloss);
	for(i=1;i<flayer->lnum-1;i++)
	{
			M1=TransMatrix(flayer->nwnet[flayer->lnum-1-i].tc);	
			M=MulMatrix(theta[i-1],M1);
			FreeMatrix(M1);	
			M1=NULL;
			CopyMatrix(M, theta[i]);
			FreeMatrix(M);
			M=NULL;	
					
			M1=FXMatrix(flayer[flayer->lnum-i-1].nerus,atan1);//
			M=HadamaMatrix(theta[i],M1);
			CopyMatrix(M, theta[i]);
			FreeMatrix(M1);	
			M1=NULL;
			FreeMatrix(M);
			M=NULL;					
			
	}
			
		for(i=0;i<flayer->lnum-1;i++)
	{
			M1=TransMatrix(flayer[flayer->lnum-2-i].nerusA);
			M=MulMatrix(M1,theta[i]);
			CopyMatrix(M, dweight[flayer->lnum-2-i]);	
			FreeMatrix(M1);
			FreeMatrix(M);
			M1=NULL;
			M=NULL;		
	}
}
void upweightnet(struct layer *flayer,Matrix **dweight,double step)//更新权重网络
{
	Matrix *M,*M1;
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{
			M=DotmultMatrix(dweight[i],step);
			M1=AddMatrix(flayer->nwnet[i].tc,M);
			FreeMatrix(M);
			CopyMatrix(M1, flayer->nwnet[i].tc);	
			FreeMatrix(M1);	
			M=NULL;	
			M1=NULL;
		//PrintMatrix(flayer->nwnet[i].tc);
	}
}
void Inputdata_net(struct layer *Flay,FILE *data,FILE *lab,Matrix *Out)//导入输入输出矩阵
{
	int size_data,size_lab,temp;
	size_data=Flay[0].num;
	size_lab=Flay[Flay[0].lnum-1].num;
	for(int i=0;i<size_data;i++)
   {
		fscanf(data,"%d",&temp);
		Flay[0].nerusA->data[i]=0.001*(double)temp;
   }
   	fscanf(lab,"%d",&temp);
   	for(int i=0;i<size_lab;i++)
   {
		Out->data[i]=0;
   }
   Out->data[temp]=0.8;
}
void trainnet(struct layer *flayer,Matrix *y0,double step,int ecoh,FILE *ftdp,FILE *ftlp)//训练网络--step步长，ecoh训练总轮数，y0正确输出
{
	Matrix *yloss,**theta,**dweight,*M;//损失函数，theta，权重梯度，临时矩阵
	int i=0,j=0,k=0;
	long int posd=0,posl=0;
	//初始化矩阵参数
	theta=(Matrix**)malloc((flayer->lnum-1)*sizeof(Matrix*));
	dweight=(Matrix**)malloc((flayer->lnum-1)*sizeof(Matrix*));
	yloss=InitMatrix(1,flayer[flayer->lnum-1].num);
	for(i=0;i<flayer->lnum-1;i++)
	{
		theta[i]=InitMatrix(1,flayer[flayer->lnum-1-i].num);
	//	printf(" 1x%d",flayer[flayer->lnum-2-i].num);
		dweight[flayer->lnum-2-i]=InitMatrix(flayer->nwnet[flayer->lnum-2-i].tc->row,flayer->nwnet[flayer->lnum-2-i].tc->line);
	}
	//训练
	for(i=0;i<ecoh;i=i+15000)
	{
		posd=ftell(ftdp);
		posl=ftell(ftlp);
		for(j=0;j<5;j++)
	{
		fseek(ftdp,posd,SEEK_SET);
		fseek(ftlp,posl,SEEK_SET);
		for(k=0;k<15000;k++)
		{
			Inputdata_net(flayer,ftdp,ftlp,y0);
			updatenet(flayer);//更新网络
		
			getgrad(flayer,y0,yloss,theta,dweight);//得到权重
		
			upweightnet(flayer,dweight,step);//更新权重
		}

	}	
			printf("%lf%c\n",i*100.0/ecoh,'%');	
	}

}
int Fmax(double *data,int num)//找最大值
{
	int i,temp=0;
	for(i=0;i<num;i++)
	{
		if(data[i]>data[temp])
		temp=i;
	}
	return temp;
}
void predict(struct layer *Flay,FILE *ftestdp,FILE *ftestlp,Matrix *y0,int num)//预测函数
{
	int count=0,tem=0,tem2=0;
	double cr;
	for(int i=0;i<num;i++)
	{
		Inputdata_net(Flay,ftestdp,ftestlp,y0);
		updatenet(Flay);
		tem=Fmax(Flay[Flay->lnum-1].nerus->data,Flay[Flay->lnum-1].num);
		
		//
		tem2=Fmax(y0->data,Flay[Flay->lnum-1].num);
		
		if(tem==tem2)
			count++;
		else
		{
			
		/*	printf("PRE_value: %d ",tem);
			printf("Value: %d\n",tem2);
			PrintMatrix(y0);
			PrintMatrix(Flay[Flay->lnum-1].nerus);*/
		}
		
		//		
	}
	cr=count;
	printf("correct rate: %lf %c\n",100*cr/num,'%');
	
}
int main()
{
    struct layer *Flay;//定义网络
    Matrix *Outmat;
    FILE *ftdp,*ftlp,*ftestdp,*ftestlp; 
	             
    int laynum=4,lays[4]={784,20,10,10};//网络结构
    
    ftdp=fopen("traindata.txt","rb");//打开训练数据文件
    ftlp=fopen("trainlab.txt","rb");//打开训练标签文件
    ftestdp=fopen("testdata.txt","rb");//打开训练数据文件
    ftestlp=fopen("testlab.txt","rb");//打开训练标签文件    
    
    Flay=newbpnet(laynum,lays);//生成神经网络
   	Outmat=InitMatrix(1,Flay[Flay->lnum-1].num);//生成标签矩阵
   	trainnet(Flay,Outmat,0.13,60000,ftdp,ftlp);//训练网络
	predict(Flay,ftestdp,ftestlp,Outmat,10000);//预测
	//while(1);
   	printnet(Flay);//打印网络
}                   
   
   
       struct layer *Flay[10];//定义网络
    Matrix *Outmat;
    FILE *ftdp,*ftlp,*ftestdp,*ftestlp; 
	             
    int i=0,laynum=4,lays[4]={784,20,10,10};//网络结构
    
    ftdp=fopen("traindata.txt","rb");//打开训练数据文件
    ftlp=fopen("trainlab.txt","rb");//打开训练标签文件
    ftestdp=fopen("testdata.txt","rb");//打开训练数据文件
    ftestlp=fopen("testlab.txt","rb");//打开训练标签文件    
    
	for(i=0;i<10;i++)
    {
    Flay[i]=newbpnet(laynum,lays);//生成神经网络
	Outmat=InitMatrix(1,Flay[i][Flay[i]->lnum-1].num);//生成标签矩阵
   	trainnet(Flay[i],Outmat,0.001+0.0001*i,60000,ftdp,ftlp);//训练网络
   	printf("%d\n",i);
	}
	for(i=0;i<10;i++)
	{
		fseek(ftestdp,0,SEEK_SET);
		fseek(ftestlp,0,SEEK_SET);
		printf("α: %lf   ",0.001+0.0001*i);
		predict(Flay[i],ftestdp,ftestlp,Outmat,10000);//预测	
		
	}

	
	//while(1);
	
	#include<stdio.h>
#include<math.h>
#include<memory.h>
#include<Mat.h>
#include<malloc.h>
struct layer{
	int num;//神经元数
	Matrix *nerus;//神经元矩阵
	Matrix *nerusA;//神经元矩阵
	struct weightnet *nwnet;//下层突触矩阵
	int lnum;//层数
};
struct weightnet{
	int num;//突触数
	int lnum;//行数
	int rnum;//列数
	Matrix *tc;//权重矩阵
};
double atan1(double x)
{
	return 1/(1+x*x);
}
double wclear(double w)
{
	if((w<(0.1/0.02))&&(w>(-0.1/0.02)))
	return w;
	else
	return 0.0;
}
double quantify(double da)
{
	int level=15;
	double kedu=0.1;
	if(da<0)
	da=da-0.000001;
	else
	da=da+0.000001;
	switch((int)(da/kedu)*2/(level-1))
	{
		case 0: return (int)(da/kedu)*kedu;break;
		default:if(da<0) return kedu*(1-level)/2; else return kedu*(level-1)/2;break;
	}
	
}
void creatwnet(struct layer *flayer,int layernum)//生成突触矩阵
{
	int ii=0;
	Matrix *M;
	flayer[0].nwnet=(struct weightnet *)malloc((layernum-1)*sizeof(struct weightnet));
	for(ii=0;ii<layernum-1;ii++)
	{
		flayer[0].nwnet[ii].num=flayer[ii].num*flayer[ii+1].num;
		flayer[0].nwnet[ii].lnum=flayer[ii].num;
		flayer[0].nwnet[ii].rnum=flayer[ii+1].num;
		flayer[0].nwnet[ii].tc=InitMatrix(flayer[ii].num,flayer[ii+1].num);
		RandomMatrix(flayer[0].nwnet[ii].tc,1.6);
	/*	M=FXMatrix(flayer[0].nwnet[ii].tc,quantify);
		CopyMatrix(M, flayer[0].nwnet[ii].tc);		
		FreeMatrix(M);
		M=NULL;		*/
		//printf("%dx%d\n",flayer[0].nwnet[ii].tc->row,flayer[0].nwnet[ii].tc->line);
	}
}
struct layer * newbpnet(int layernum,int *layers)//新建网络
{
	int i=0,j=0,k=0;
	struct layer *flayer;
	flayer=(struct layer*)malloc(layernum*sizeof(struct layer));
	flayer[0].lnum=layernum;
	for(i=0;i<layernum;i++)
	{
		flayer[i].num=layers[i];
		flayer[i].nerus=InitMatrix(1,flayer[i].num);
		flayer[i].nerusA=InitMatrix(1,flayer[i].num);
	}
	creatwnet(flayer,layernum);
	return flayer;
};

void updatenet(struct layer *flayer)//更新网络
{
	int i=0,j=0,laynum=0;
	Matrix *M;
	laynum=(*flayer).lnum;
	for(i=0;i<laynum-1;i++)
		{
			M=MulMatrix(flayer[i].nerusA,flayer[0].nwnet[i].tc);
			CopyMatrix(M, flayer[i+1].nerus);		
			FreeMatrix(M);
			M=NULL;
			
			M=FXMatrix(flayer[i+1].nerus,atan);//激活函数aan
			CopyMatrix(M, flayer[i+1].nerusA);		
			FreeMatrix(M);
			
			M=NULL;			
		}
}

void  printnet(struct layer *flayer)//打印网络
{
	int i=0,j=0,k=0,laynum=0;
	laynum=(*flayer).lnum;
	for(i=0;i<laynum-1;i++)	
	{
		PrintMatrix(flayer[i].nerusA);	
		printf("*******************\n");						//打印一个矩阵
		PrintMatrix(flayer[0].nwnet[i].tc);	
		printf("*******************\n");	
	}
	PrintMatrix(flayer[laynum-1].nerusA);	
	printf("\n");
}

void getgrad(struct layer *flayer,Matrix *y0,Matrix *yloss,Matrix **theta,Matrix **dweight)//计算梯度
{
	Matrix *M,*M1;
	int i=0,j=0;
	yloss=DecMatrix(y0,flayer[flayer->lnum-1].nerusA);//y=Y'
	
	M1=FXMatrix(flayer[flayer->lnum-1].nerus,atan1);//loss*激活函数导数
	M=HadamaMatrix(yloss,M1);
	CopyMatrix(M, theta[0]);
	FreeMatrix(M1);	
	M1=NULL;
	FreeMatrix(M);
	M=NULL;	
	FreeMatrix(yloss);
	for(i=1;i<flayer->lnum-1;i++)
	{
			M1=TransMatrix(flayer->nwnet[flayer->lnum-1-i].tc);	
			M=MulMatrix(theta[i-1],M1);
			FreeMatrix(M1);	
			M1=NULL;
			CopyMatrix(M, theta[i]);
			FreeMatrix(M);
			M=NULL;	
					
			M1=FXMatrix(flayer[flayer->lnum-i-1].nerus,atan1);//
			M=HadamaMatrix(theta[i],M1);
			CopyMatrix(M, theta[i]);
			FreeMatrix(M1);	
			M1=NULL;
			FreeMatrix(M);
			M=NULL;					
			
	}
			
		for(i=0;i<flayer->lnum-1;i++)
	{
			M1=TransMatrix(flayer[flayer->lnum-2-i].nerusA);
			M=MulMatrix(M1,theta[i]);
			CopyMatrix(M, dweight[flayer->lnum-2-i]);	
			FreeMatrix(M1);
			FreeMatrix(M);
			M1=NULL;
			M=NULL;		
	}
}
void printweight(struct layer *flayer,Matrix **dweight)//更新权重网络
{
	Matrix *M,*M1;
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{	
			PrintMatrix(dweight[i]);
			printf("*******************\n");
			getchar();
	}
}
void upweightnet(struct layer *flayer,Matrix **dweight,double step)//更新权重网络
{
	Matrix *M,*M1;
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{
			M=DotmultMatrix(dweight[i],step);
			M1=AddMatrix(flayer->nwnet[i].tc,M);
			FreeMatrix(M);
			CopyMatrix(M1, flayer->nwnet[i].tc);	
			FreeMatrix(M1);	
			M=NULL;	
			M1=NULL;
			
/*			M=FXMatrix(flayer->nwnet[i].tc,quantify);
			CopyMatrix(M, flayer->nwnet[i].tc);		
			FreeMatrix(M);
			M=NULL;	*/
			M=DotmultMatrix(dweight[i],0.0);
			CopyMatrix(M, dweight[i]);		
			FreeMatrix(M);
			M=NULL;	
			/*			
			//printweight(flayer,dweight);
			M=FXMatrix(dweight[i],wclear);
			CopyMatrix(M, dweight[i]);		
			FreeMatrix(M);
			M=NULL;	*/
		//PrintMatrix(flayer->nwnet[i].tc);
	}
}
void addweight(struct layer *flayer,Matrix **dweight,Matrix **dweight1,double step)//更新权重网络
{
	Matrix *M,*M1;
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{
			M=DotmultMatrix(dweight1[i],step);
			M1=AddMatrix(dweight[i],M);
			FreeMatrix(M);
			CopyMatrix(M1, dweight[i]);	
			FreeMatrix(M1);	
			M=NULL;	
			M1=NULL;
			
			/*PrintMatrix(dweight[i]);
			printf("*******************\n");
			getchar();*/
			
	}
}

void Inputdata_net(struct layer *Flay,FILE *data,FILE *lab,Matrix *Out)//导入输入输出矩阵
{
	int size_data,size_lab,temp;
	size_data=Flay[0].num;
	size_lab=Flay[Flay[0].lnum-1].num;
	for(int i=0;i<size_data;i++)
   {
		fscanf(data,"%d",&temp);
		Flay[0].nerusA->data[i]=0.0028*(double)temp;
   }
   	fscanf(lab,"%d",&temp);
   	for(int i=0;i<size_lab;i++)
   {
		Out->data[i]=0;
   }
   Out->data[temp]=1.6;
}
void trainnet(struct layer *flayer,Matrix *y0,double step,int ecoh,FILE *ftdp,FILE *ftlp)//训练网络--step步长，ecoh训练总轮数，y0正确输出
{
	Matrix *yloss,**theta,**dweight1,**dweight,*M;//损失函数，theta，权重梯度，临时矩阵
	int i=0,j=0,k=0;
	long int posd=0,posl=0;
	//初始化矩阵参数
	theta=(Matrix**)malloc((flayer->lnum-1)*sizeof(Matrix*));
	dweight=(Matrix**)malloc((flayer->lnum-1)*sizeof(Matrix*));
	dweight1=(Matrix**)malloc((flayer->lnum-1)*sizeof(Matrix*));
	yloss=InitMatrix(1,flayer[flayer->lnum-1].num);
	for(i=0;i<flayer->lnum-1;i++)
	{
		theta[i]=InitMatrix(1,flayer[flayer->lnum-1-i].num);
	//	printf(" 1x%d",flayer[flayer->lnum-2-i].num);
		dweight[flayer->lnum-2-i]=InitMatrix(flayer->nwnet[flayer->lnum-2-i].tc->row,flayer->nwnet[flayer->lnum-2-i].tc->line);
		dweight1[flayer->lnum-2-i]=InitMatrix(flayer->nwnet[flayer->lnum-2-i].tc->row,flayer->nwnet[flayer->lnum-2-i].tc->line);
	}
	//训练
	for(i=0;i<ecoh;i=i+1000)
	{
		posd=ftell(ftdp);
		posl=ftell(ftlp);
		for(j=0;j<3;j++)
	{
		fseek(ftdp,posd,SEEK_SET);
		fseek(ftlp,posl,SEEK_SET);
		
		for(k=0;k<1000;k++)
		{
			Inputdata_net(flayer,ftdp,ftlp,y0);
			updatenet(flayer);//更新网络
			getgrad(flayer,y0,yloss,theta,dweight1);//得到权重
			addweight(flayer,dweight,dweight1,1.0);
		}
		//printweight(flayer,dweight);
		upweightnet(flayer,dweight,step);//更新权重

	}	
			printf("%lf%c\n",i*100.0/ecoh,'%');	
	}
	

}
int Fmax(double *data,int num)//找最大值
{
	int i,temp=0;
	for(i=0;i<num;i++)
	{
		if(data[i]>data[temp])
		temp=i;
	}
	return temp;
}
void predict(struct layer *Flay,FILE *ftestdp,FILE *ftestlp,Matrix *y0,int num)//预测函数
{
	int count=0,tem=0,tem2=0;
	double cr;
	for(int i=0;i<num;i++)
	{
		Inputdata_net(Flay,ftestdp,ftestlp,y0);
		updatenet(Flay);
		tem=Fmax(Flay[Flay->lnum-1].nerus->data,Flay[Flay->lnum-1].num);
		
		//
		tem2=Fmax(y0->data,Flay[Flay->lnum-1].num);
		
		if(tem==tem2)
			count++;
		else
		{
			
		/*	printf("PRE_value: %d ",tem);
			printf("Value: %d\n",tem2);
			PrintMatrix(y0);
			PrintMatrix(Flay[Flay->lnum-1].nerus);*/
		}
		
		//		
	}
	cr=count;
	printf("correct rate: %lf %c\n",100*cr/num,'%');
	
}
int main()
{
    struct layer *Flay;//定义网络
    Matrix *Outmat;
    FILE *ftdp,*ftlp,*ftestdp,*ftestlp; 
	             
    int i=0,laynum=3,lays[3]={784,20,10};//网络结构
    
    ftdp=fopen("traindata.txt","rb");//打开训练数据文件
    ftlp=fopen("trainlab.txt","rb");//打开训练标签文件
    ftestdp=fopen("testdata.txt","rb");//打开训练数据文件
    ftestlp=fopen("testlab.txt","rb");//打开训练标签文件    
    
    Flay=newbpnet(laynum,lays);//生成神经网络
	Outmat=InitMatrix(1,Flay[Flay->lnum-1].num);//生成标签矩阵
   	trainnet(Flay,Outmat,0.0014,60000,ftdp,ftlp);//训练网络
	predict(Flay,ftestdp,ftestlp,Outmat,10000);//预测	

	
	//while(1);
   	printnet(Flay);//打印网络
}                   
   