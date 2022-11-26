#include<stdio.h>
#include<math.h>
#include<malloc.h>
#include"net.h"
int main()
{
	struct net *net0;
	net0=(struct net*)malloc(sizeof(struct net));
	int laynum=4,lays[4]={784,20,20,10};//网络结构

    net0->ftdp=fopen("D:/design/mnist_dataset/mnist_dataset/traindata.txt","rb");//打开训练数据文件
    net0->ftlp=fopen("D:/design/mnist_dataset/mnist_dataset/trainlab.txt","rb");//打开训练标签文件
    net0->ftestdp=fopen("D:/design/mnist_dataset/mnist_dataset/testdata.txt","rb");//打开训练数据文件
    net0->ftestlp=fopen("D:/design/mnist_dataset/mnist_dataset/testlab.txt","rb");//打开训练标签文件    	
	net0->laynum=laynum;
	net0->lays=lays;

	net0->studya=0.006;
	net0->trainnum=60000;
	net0->testnum=10000;
	net0->epochs=5;
	Train_Pre_net(net0);
	
	printf("%lf\n",net0->cr);
	printnet(net0->Flay);//打印网络
}