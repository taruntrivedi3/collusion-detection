#include <algorithm>
#include <cmath>
#include <vector>
extern "C" void batch_mi(int b,int v,int r,int n,int k,const int* x,const int* states,double* output) {
    int e=n*(n-1)/2,smax=*std::max_element(states,states+r)+1;
    std::vector<int> ns(smax),own(smax*n*k),joint(smax*e*k*k),li,ri;
    std::vector<double> f(r+1),ent(smax*n);
    for(int t=0;t<r;++t)++ns[states[t]];
    for(int t=1;t<=r;++t)f[t]=t*std::log2(double(t));
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){li.push_back(i);ri.push_back(j);}
    for(int d=0;d<b*v;++d) {
        std::fill(own.begin(),own.end(),0);std::fill(joint.begin(),joint.end(),0);
        for(int t=0;t<r;++t) {
            int s=states[t];if(ns[s]<2)continue;
            const int* row=x+(d*r+t)*n;
            for(int i=0;i<n;++i)++own[(s*n+i)*k+row[i]];
            for(int q=0;q<e;++q)++joint[(s*e+q)*k*k+row[li[q]]*k+row[ri[q]]];
        }
        double* out=output+d*e;std::fill(out,out+e,0);
        for(int s=0;s<smax;++s) {
            if(ns[s]<2)continue;
            for(int i=0;i<n;++i) {
                double h=0;for(int z=0;z<k;++z)h+=f[own[(s*n+i)*k+z]];
                ent[s*n+i]=h;
            }
            for(int q=0;q<e;++q) {
                double h=f[ns[s]]-ent[s*n+li[q]]-ent[s*n+ri[q]];
                for(int z=0;z<k*k;++z)h+=f[joint[(s*e+q)*k*k+z]];
                out[q]+=std::max(0.0,h)/r;
            }
        }
    }
}
