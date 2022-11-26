#include<stdio.h>
#include<math.h>
#include<memory.h>
#include<malloc.h>
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
int main()
{
	int i,j,*p;
	double t;
	printf("%lf",atan(999.9));
	while(1)
	{
		
	//	p=(int*)malloc(sizeof(int));
		//free(p);
	}
	
	
}