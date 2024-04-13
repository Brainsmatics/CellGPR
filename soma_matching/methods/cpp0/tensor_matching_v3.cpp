#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>



extern "C"{
    void tensormatching(double* pX, int N1, int N2, 
                        int* pIndH3, double* pValH3, int Nt3,
                        double* pXout, double* pScoreOut){
        std::queue<int> q;
        int count = 0;
        int a = pIndH3[0], b = pIndH3[Nt3]/N2, c = pIndH3[2*Nt3]/N2;
        for(int i = 0; i < Nt3; i++){
            if (pIndH3[i] == a && pIndH3[i+Nt3]/N2 == b && pIndH3[i+2*Nt3]/N2 == c){
                count++;
            }
            else{
                q.push(count);
                count = 1;
                a = pIndH3[i], b = pIndH3[i + Nt3]/N2, c = pIndH3[i + 2*Nt3]/N2;
            }
        }
        q.push(count);


        int NN=N1*N2;
        double* pXtemp = new double[NN];
        // double* pXtemp = (double*)malloc(sizeof(double)*NN);
        for(int n=0;n<NN;n++)
            pXout[n]=pX[n];
        double score;
        int maxIter = 100, maxIter2 = 10;
        // std::cout << "dfdcds" << std::endl;

        // std::vector<int> s = counter(pIndH3, Nt3, N2);
        // print_vector(s);
        

        for(int iter=0;iter<maxIter;iter++){
            // std::cout << iter << std::endl;
            // vector<vector<double>> 
            std::queue<int> q0 = q;

            int num = q0.front();
            // std::cout << num << std::endl;
            q0.pop();
            *pScoreOut=0;
            for(int n=0;n<NN;n++)
                pXtemp[n]=1*pX[n];

            double score0 = 0;

            for(int t=0;t<Nt3;t++)
                {
                score=pXout[pIndH3[t]]*pXout[pIndH3[t+Nt3]]*pXout[pIndH3[t+2*Nt3]];
                score = score*
                    pValH3[t]*pXout[pIndH3[t+Nt3]]*pXout[pIndH3[t+2*Nt3]];
                num --;
                if (num > 0){
                    score0 = std::max(score, score0);
                }
                else{
                    pXtemp[pIndH3[t]] += score0;
                    score0 = 0;
                    num = q0.front();
                    q0.pop();                    
                }
                // pXtemp[pIndH3[t]] += score*
                //     pValH3[t]*pXout[pIndH3[t+Nt3]]*pXout[pIndH3[t+2*Nt3]];
                // pXtemp[pIndH3[t+Nt3]] += score*
                //     pValH3[t]*pXout[pIndH3[t+2*Nt3]]*pXout[pIndH3[t]];
                // pXtemp[pIndH3[t+2*Nt3]] += score*
                //     pValH3[t]*pXout[pIndH3[t]]*pXout[pIndH3[t+Nt3]];
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