#include <iostream>
#include <cmath>

extern "C"{
    void tensormatching(double* pX, int N1, int N2, 
                        int* pIndH3, double* pValH3, int Nt3,
                        double* pXout, double* pScoreOut){
        int NN=N1*N2;
        double* pXtemp = new double[NN];
        // double* pXtemp = (double*)malloc(sizeof(double)*NN);
        for(int n=0;n<NN;n++)
            pXout[n]=pX[n];
        double score;
        int maxIter = 100, maxIter2 = 10;
        for(int iter=0;iter<maxIter;iter++){
            // std::cout << iter << std::endl;
            *pScoreOut=0;
            for(int n=0;n<NN;n++)
                pXtemp[n]=1*pX[n];

            for(int t=0;t<Nt3;t++)
                {
                std::cout << "ind: " << pIndH3[t] << " "<< pIndH3[t+Nt3] << " " << pIndH3[t+2*Nt3];
                std::cout << "  val: " << pValH3[t] << std::endl;
                score=pXout[pIndH3[t]]*pXout[pIndH3[t+Nt3]]*pXout[pIndH3[t+2*Nt3]];
                pXtemp[pIndH3[t]] += score*
                    pValH3[t]*pXout[pIndH3[t+Nt3]]*pXout[pIndH3[t+2*Nt3]];
                pXtemp[pIndH3[t+Nt3]] += score*
                    pValH3[t]*pXout[pIndH3[t+2*Nt3]]*pXout[pIndH3[t]];
                pXtemp[pIndH3[t+2*Nt3]] += score*
                    pValH3[t]*pXout[pIndH3[t]]*pXout[pIndH3[t+Nt3]];
                if(iter==(maxIter-1))
                {
                    score= pXout[pIndH3[t]]*pXout[pIndH3[t+Nt3]]*pXout[pIndH3[t+2*Nt3]];
                    *pScoreOut=*pScoreOut+3*score*score;
                }
                }

            for(int iter2=0;iter2<maxIter2;iter2++)
                {
                    for(int n1=0;n1<N1;n1++)
                    {
                    double pXnorm=0;
                    for(int n2=0;n2<N2;n2++)
                        pXnorm+=pXtemp[n1*N2+n2]*pXtemp[n1*N2+n2];
                    pXnorm=std::sqrt(pXnorm);
                    if(pXnorm!=0)
                        for(int n2=0;n2<N2;n2++)
                        pXout[n1*N2+n2]=pXtemp[n1*N2+n2]/pXnorm;
                    }
                }
            for(int n2=0;n2<N2;n2++)
            {
                double pXnorm=0;
                for(int n1=0;n1<N1;n1++)
                pXnorm+=pXtemp[n1*N2+n2]*pXtemp[n1*N2+n2];
                pXnorm=std::sqrt(pXnorm);
                if(pXnorm!=0)
                for(int n1=0;n1<N1;n1++)
                    pXout[n1*N2+n2]=pXtemp[n1*N2+n2]/pXnorm;
            }

        }
        delete[] pXtemp;
    }
}