#include<bits/stdc++.h>
using namespace std;
inline void naive_func()   
{
    double head,tail,freq,head1,tail1,timess=0;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&head);
    for(int i=1; i<=n; i++) a[i]=i; //a[i]为给定向量
    for(int i=1 ;i<=n; i++)
    for(int j=1; j<=n ;j++) b[i][j]=i+j;//数组b存储给定矩阵
    for (int i=1; i<=n; i++)
        for (int j=1; j<=n; j++)
            sum[i] += (b[j][i]*a[j]); //sum[i]存储第i列与给定向量内积 
    QueryPerformanceCounter ((LARGE_INTEGER *)& tail) ;
    cout << "\nordCol:" <<(tail-head)*1000.0 / freq<< "ms" << endl;
}
inline void opti_func()   
{
    double head,tail,freq,head1,tail1,timess=0;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&head);
    for(int i=1; i<=n; i++) a[i]=i; //a[i]为给定向量
    for(int i=1 ;i<=n; i++)
    for(int j=1; j<=n ;j++) b[i][j]=i+j;//数组b存储给定矩阵
    for(int j=1; j<=n; j++){
    for(int i=1; i<=n ;i++)
        sum[i] += b[j][i] ∗ a[j];
    }
    QueryPerformanceCounter ((LARGE_INTEGER *)& tail) ;
    cout << "\nordCol:" <<(tail-head)*1000.0 / freq<< "ms" << endl;
  }
int main(){
    int n=1000;
    int k=100;
    for(int i=1;i<=k;i++) naive_func(n);
}
