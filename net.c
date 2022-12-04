#include"net.h"
float atan1(float x,Matrix *M)
{
	return 2/3.1415926/(1+x*x);
}
float atanm(float x,Matrix *M)
{
	
	return atan((double)x)*2/3.1415926;
}
float quantify(float da,Matrix *M)
{
	int level=15;
	float kedu=0.2;
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
float softmax(float z,Matrix *M)
{
	int i=0;
	float sum=0;
	for(i=0;i<(M->line)*(M->row);i++)
		sum+=exp(M->data[i]);
	return exp(z)/sum;
}

void qweightnet(struct layer *flayer)//更新权重网络
{
	Matrix *M;
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{
			M=FXMatrix(flayer->nwnet[i].tc,quantify);
			CopyMatrix(M, flayer->nwnet[i].qtc);			
			FreeMatrix(M);
			M=NULL;	
		//PrintMatrix(flayer->nwnet[i].tc);
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
		flayer[0].nwnet[ii].qtc=InitMatrix(flayer[ii].num,flayer[ii+1].num);
		RandomMatrix(flayer[0].nwnet[ii].tc,1);
		//printf("%dx%d\n",flayer[0].nwnet[ii].tc->row,flayer[0].nwnet[ii].tc->line);
	}
		qweightnet(flayer);//更新量化权重网络
}
void printweight(struct layer *flayer,Matrix **dweight)//更新权重网络
{
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{	
			PrintMatrix(dweight[i]);
			printf("*******************\n");
			getchar();
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
	for(i=0;i<laynum-2;i++)
		{
			M=MulMatrix(flayer[i].nerusA,flayer[0].nwnet[i].qtc);
			CopyMatrix(M, flayer[i+1].nerus);		
			FreeMatrix(M);
			M=NULL;
			
			M=FXMatrix(flayer[i+1].nerus,atanm);//激活函数aan
			CopyMatrix(M, flayer[i+1].nerusA);		
			FreeMatrix(M);
			
			M=NULL;			
		}
			M=MulMatrix(flayer[i].nerusA,flayer[0].nwnet[i].qtc);
			CopyMatrix(M, flayer[i+1].nerus);		
			FreeMatrix(M);
			M=NULL;
			
			M=FXMatrix(flayer[i+1].nerus,softmax);//激活函数aan
			CopyMatrix(M, flayer[i+1].nerusA);		
			FreeMatrix(M);			
			M=NULL;	
					
}
void  printnet(struct layer *flayer)//打印网络
{
	int i=0,j=0,k=0,laynum=0;
	laynum=(*flayer).lnum;
	for(i=0;i<laynum-1;i++)	
	{
		PrintMatrix(flayer[i].nerusA);		
		printf("**************\n");					//打印一个矩阵
		PrintMatrix(flayer[0].nwnet[i].qtc);	
		printf("**************\n");
	}
	PrintMatrix(flayer[laynum-1].nerusA);	
	printf("**************\n");
}

void getgrad(struct layer *flayer,Matrix *y0,Matrix *yloss,Matrix **theta,Matrix **dweight)//计算梯度
{
	Matrix *M,*M1;
	int i=0,j=0;
	yloss=DecMatrix(flayer[flayer->lnum-1].nerusA,y0);//y=Y'
	CopyMatrix(yloss, theta[0]);
	FreeMatrix(yloss);
	yloss=NULL;
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
void upweightnet(struct layer *flayer,Matrix **dweight,float step)//更新权重网络
{
	Matrix *M,*M1;
	int i=0;
	for(i=0;i<flayer->lnum-1;i++)
	{
			M=DotmultMatrix(dweight[i],step);
			M1=DecMatrix(flayer->nwnet[i].tc,M);
			FreeMatrix(M);
			CopyMatrix(M1, flayer->nwnet[i].tc);	
			FreeMatrix(M1);	
			M=NULL;	
			M1=NULL;
			//printweight(flayer,dweight);
			M=DotmultMatrix(dweight[i],0.0);
			CopyMatrix(M, dweight[i]);		
			FreeMatrix(M);
			M=NULL;	
		//PrintMatrix(flayer->nwnet[i].tc);
	}
		qweightnet(flayer);//更新量化权重网络
}
void addweight(struct layer *flayer,Matrix **dweight,Matrix **dweight1,float step)//更新权重网络
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
		Flay[0].nerusA->data[i]=0.001*(float)temp;
   }
   	fscanf(lab,"%d",&temp);
   	for(int i=0;i<size_lab;i++)
   {
		Out->data[i]=0;
   }
   Out->data[temp]=1;
}
void trainnet(struct layer *flayer,Matrix *y0,float step,int datasize,int ecoh,FILE *ftdp,FILE *ftlp)//训练网络--step步长，ecoh训练总轮数，y0正确输出
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
		posd=ftell(ftdp);
		posl=ftell(ftlp);	
	for(j=0;j<ecoh;j++)
	{	
		fseek(ftdp,posd,SEEK_SET);
		fseek(ftlp,posl,SEEK_SET);		
		for(i=0;i<datasize;i=i+100)
	{
		for(k=0;k<100;k++)
		{
			Inputdata_net(flayer,ftdp,ftlp,y0);
			updatenet(flayer);//更新网络
			getgrad(flayer,y0,yloss,theta,dweight1);//得到权重
			addweight(flayer,dweight,dweight1,1.0);	
		}
		upweightnet(flayer,dweight,step);//更新权重
		//printnet(flayer);
		//getchar();
		printf("%d,%d\n",j,i);	
	}	
	}
}
int Fmax(float *data,int num)//找最大值
{
	int i,temp=0;
	for(i=0;i<num;i++)
	{
		if(data[i]>data[temp])
		temp=i;
	}
	return temp;
}
float predict(struct layer *Flay,FILE *ftestdp,FILE *ftestlp,Matrix *y0,int num)//预测函数
{
	int count=0,tem=0,tem2=0;
	float cr;
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
		}
		//		
	}
	cr=count;
    cr=cr/num;
	//printf("correct rate: %f %c\n",100*cr,'%');
    return cr;
}
void Train_Pre_net(struct net *net)
{
    Matrix *Outmat;//输出矩阵
    net->Flay=newbpnet(net->laynum,net->lays);//生成神经网络
   	Outmat=InitMatrix(1,net->Flay[net->Flay->lnum-1].num);//生成标签矩阵
   	trainnet(net->Flay,Outmat,net->studya,net->trainnum,net->epochs,net->ftdp,net->ftlp);//训练网络15-95.72,20-95.91,22-95.94,25-96.02
	net->cr=predict(net->Flay,net->ftestdp,net->ftestlp,Outmat,net->testnum);//预测
   // printnet(net->Flay);//打印网络
}     