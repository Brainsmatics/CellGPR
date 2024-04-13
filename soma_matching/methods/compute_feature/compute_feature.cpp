#include <cmath>
#include <iostream>

extern "C"{
    void get_graph(int N1, int *p){
        // int* p;
        int count = 0;
        for(int i=0; i<N1; i++){
            for(int j=i+1; j<N1; j++){
                for(int k=j+1; k<N1; k++){
                    p[count] = i;
                    p[count +1] = j;
                    p[count +2] = k;
                    count+=3;
                }
        }
        }
        // return p;
    }


    void computeFeatureSimple( double* pP1, int i , int j, int k , double* pF)
    { 
    // pP1是点集，ijk分别是三个点的索引，pF是提取的特征
    const int nFeature=3;
    double vecX[nFeature];
    double vecY[nFeature];
    // 索引
    int ind[nFeature];
    ind[0]=i;ind[1]=j;ind[2]=k;
    double n;
    int f;
    // 只要有相等的就返回最小值
    if((ind[0]==ind[1])||(ind[0]==ind[2])||(ind[1]==ind[2]))
    {
        pF[0]=pF[1]=pF[2]=-10;
        return;
    }
    for(f=0;f<nFeature;f++)
    {
        // 为什么要乘以2（只取xy的坐标，这里pP1拉伸为一维的向量
        // 这里理解为x1-x2, y1-y2
        vecX[f]=pP1[ind[((f+1)%3)]*2]-pP1[ind[f]*2];
        vecY[f]=pP1[ind[((f+1)%3)]*2+1]-pP1[ind[f]*2+1];
        // std::cout << "data2 " << pP1[ind[((f+1)%3)]*2] << " " << pP1[ind[((f+1)%3)]*2+1]<< std::endl;
        // std::cout << "data1 " << pP1[ind[f]*2] << " " << pP1[ind[f]*2+1] << std::endl;
        double norm=std::sqrt(vecX[f]*vecX[f]+vecY[f]*vecY[f]);
        // 归一化
        // std::cout << "norm = " << norm << std::endl;
        // std::cout << "vecX = " << vecX[f] << std::endl;
        if(norm!=0)
        {
        vecX[f]/=norm;
        vecY[f]/=norm;
        }else{
        vecX[f]=0;
        vecY[f]=0;
        }
    }
    for(f=0;f<nFeature;f++)
    {
        // 因为做了归一化，等价于计算角度
        pF[f] = vecX[((f+1)%3)]*vecY[f]-vecY[((f+1)%3)]*vecX[f];
        // std::cout << pF[f] << std::endl;
    }
    }


    void computeFeature( double* pP1 , int nP1 , double* pP2 , int nP2 ,
                        int* pT1 , int nT1 , double* pF1 , double* pF2)
    { 
    const int nFeature=3;
    for(int t=0;t<nT1;t++)
    {
        // 计算t1构成的三角形的特征，写入pF1中
        computeFeatureSimple(pP1,pT1[t*3],pT1[t*3+1],pT1[t*3+2],pF1+t*nFeature);
    }
    
    // 计算所有的graph2的三角形特征，写入pF2中
    for(int i=0;i<nP2;i++){
        for(int j=0;j<nP2;j++){
        for(int k=0;k<nP2;k++){
            // std::cout << pP2[(i*nP2+j)*nP2+k]
            computeFeatureSimple(pP2,i,j,k,pF2+((i*nP2+j)*nP2+k)*nFeature);
        }
        }
    }
    
    }
}