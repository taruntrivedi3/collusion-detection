#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

// Independent games use separate, recorded seeds. No shared global RNG/state.
namespace nullstudy {
struct Engine {
    int n, r, regime, kind, k, e;
    double reaction, hold;
    std::mt19937_64 rng;
    std::vector<int> a, s, ns, own, joint, left, right;
    std::vector<double> f, ent, p0, p01;
    Engine(int n_, int r_, int regime_, int kind_, double p, double h, uint64_t seed)
        : n(n_), r(r_), regime(regime_), kind(kind_), k((n+1)*(n+1)+1),
          e(n*(n-1)/2), reaction(p), hold(h), rng(seed), a(r*n), s(r), ns(k),
          own(k*n*3), joint(k*e*9), f(r+1), ent(k*n), p0(k,1.0/3), p01(k,2.0/3) {
        for (int i=1;i<=r;++i) f[i]=i*std::log2(double(i));
        for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {left.push_back(i);right.push_back(j);}
        for (int c0=0;c0<=n;++c0) for(int c1=0;c1<=n-c0;++c1) {
            int c[3]={c0,c1,n-c0-c1}, sorted[3]={c[0],c[1],c[2]};
            std::sort(sorted,sorted+3);
            int count=0;
            for(int m=0;m<3;++m) count+=(c[m]==sorted[2]);
            double p=regime==0 ? 0.0 : reaction;
            if(regime==3 && sorted[2]-sorted[1]<3) p=0;
            double probs[3]={(1-p)/3,(1-p)/3,(1-p)/3};
            for(int m=0;m<3;++m) if(c[m]==sorted[2]) probs[(m+1)%3]+=p/count;
            int code=c0*(n+1)+c1+1;
            p0[code]=probs[0];p01[code]=probs[0]+probs[1];
        }
    }
    double unif() {return std::generate_canonical<double,53>(rng);}
    int integer(int hi) {return std::uniform_int_distribution<int>(0,hi-1)(rng);}
    void simulate(double cp) {
        int public_counts[3]={0,0,0};
        for(int t=0;t<r;++t) {
            int code=0;
            if(t>0 && regime!=0) {
                if(regime==1) {
                    std::fill(public_counts,public_counts+3,0);
                    for(int i=0;i<n;++i) ++public_counts[integer(3)];
                } else if(regime!=4 || t==1 || unif()>=hold) {
                    std::fill(public_counts,public_counts+3,0);
                    for(int i=0;i<n;++i) ++public_counts[a[(t-1)*n+i]];
                }
                code=public_counts[0]*(n+1)+public_counts[1]+1;
            }
            if(regime==0 || kind==0 || t==0) s[t]=0;
            else if(kind==1) s[t]=code;
            else {
                int top=*std::max_element(public_counts,public_counts+3), mask=0;
                for(int m=0;m<3;++m) if(public_counts[m]==top) mask|=1<<m;
                s[t]=mask;
            }
            bool fire=cp>0 && unif()<cp;
            double shared=fire ? unif() : 0;
            for(int i=0;i<n;++i) {
                double u=(fire && i<3) ? shared : unif();
                double t0=i<n/2 ? p0[code] : 1.0/3;
                double t1=i<n/2 ? p01[code] : 2.0/3;
                a[t*n+i]=(u<t0) ? 0 : (u<t1 ? 1 : 2);
            }
        }
    }
    void gamma(double* out) {
        std::fill(ns.begin(),ns.end(),0);
        std::fill(own.begin(),own.end(),0);
        std::fill(joint.begin(),joint.end(),0);
        for(int t=0;t<r;++t) {
            int state=s[t]; ++ns[state];
            const int* row=&a[t*n];
            int* o=&own[state*n*3];
            int* jnt=&joint[state*e*9];
            for(int i=0;i<n;++i) ++o[i*3+row[i]];
            for(int q=0;q<e;++q) ++jnt[q*9+row[left[q]]*3+row[right[q]]];
        }
        std::fill(out,out+e,0);
        for(int state=0;state<k;++state) {
            if(ns[state]<2) continue;
            for(int i=0;i<n;++i) {
                const int* o=&own[(state*n+i)*3];
                ent[state*n+i]=f[o[0]]+f[o[1]]+f[o[2]];
            }
            for(int q=0;q<e;++q) {
                double v=f[ns[state]]-ent[state*n+left[q]]-ent[state*n+right[q]];
                const int* c=&joint[(state*e+q)*9];
                for(int j=0;j<9;++j) v+=f[c[j]];
                out[q]+=v/r;
            }
        }
    }
};
}

extern "C" {
void simulate_games(int n,int r,int regime,int kind,double p,double hold,double cp,
                    int b,uint64_t seed,int* actions,int* states,double* gammas) {
    nullstudy::Engine x(n,r,regime,kind,p,hold,seed);
    for(int d=0;d<b;++d) {
        x.simulate(cp);
        if(actions) std::copy(x.a.begin(),x.a.end(),actions+d*r*n);
        if(states) std::copy(x.s.begin(),x.s.end(),states+d*r);
        if(gammas) x.gamma(gammas+d*x.e);
    }
}
void surrogate_games(int n,int r,int regime,int kind,double p,double hold,int method,
                     int b,uint64_t seed,const int* observed_a,const int* observed_s,
                     int* actions,int* states,double* gammas) {
    nullstudy::Engine x(n,r,regime,kind,p,hold,seed);
    std::copy(observed_s,observed_s+r,x.s.begin());
    std::vector<std::vector<int>> groups(x.k);
    for(int t=0;t<r;++t) groups[observed_s[t]].push_back(t);
    for(int d=0;d<b;++d) {
        if(method==2) x.simulate(0);
        else if(method==1) {
            for(int i=0;i<n;++i) {
                int offset=x.integer(r);
                for(int t=0;t<r;++t) x.a[t*n+i]=observed_a[((t+offset)%r)*n+i];
            }
        } else {
            std::copy(observed_a,observed_a+r*n,x.a.begin());
            for(const auto& rows:groups) for(int i=0;i<n;++i) {
                // Uniform Fisher-Yates within each state and independently for each agent.
                for(int j=int(rows.size())-1;j>0;--j) {
                    int q=x.integer(j+1);
                    std::swap(x.a[rows[j]*n+i],x.a[rows[q]*n+i]);
                }
            }
        }
        if(actions) std::copy(x.a.begin(),x.a.end(),actions+d*r*n);
        if(states) std::copy(x.s.begin(),x.s.end(),states+d*r);
        if(gammas) x.gamma(gammas+d*x.e);
    }
}
void gamma_games(int n,int r,int b,const int* actions,const int* states,double* gammas) {
    nullstudy::Engine x(n,r,0,0,0,0,0);
    for(int d=0;d<b;++d) {
        std::copy(actions+d*r*n,actions+(d+1)*r*n,x.a.begin());
        std::copy(states+d*r,states+(d+1)*r,x.s.begin());
        x.gamma(gammas+d*x.e);
    }
}
void score_games(int n,int b,const double* gammas,const double* emean,const double* esd,
                 const double* tmean,const double* tsd,double* maxima,int* winners) {
    int edges=n*(n-1)/2, count=1<<n;
    std::vector<double> z(n*n), sums(count);
    for(int d=0;d<b;++d) {
        int q=0;
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j,++q)
            z[i*n+j]=z[j*n+i]=(gammas[d*edges+q]-emean[q])/esd[q];
        double best=-INFINITY;int winner=0;
        std::fill(sums.begin(),sums.end(),0);
        for(int mask=1;mask<count;++mask) {
            int i=__builtin_ctz(unsigned(mask)), rest=mask&(mask-1);
            double value=sums[rest];
            for(int bits=rest;bits;bits&=bits-1) value+=z[i*n+__builtin_ctz(unsigned(bits))];
            sums[mask]=value;
            if(!rest) continue;
            double score=(value-tmean[mask])/tsd[mask];
            // Increasing bitmask order gives deterministic tie-breaking.
            if(score>best) {best=score;winner=mask;}
        }
        maxima[d]=best;winners[d]=winner;
    }
}
}
