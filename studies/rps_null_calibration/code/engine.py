"""Small checked ctypes interface; all simulation and scoring are float64/int32."""
from pathlib import Path
import ctypes as C
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = C.CDLL(str(ROOT / 'code/libnullstudy.so'))
I, D, U, P = C.c_int, C.c_double, C.c_uint64, C.c_void_p
LIB.simulate_games.argtypes = [I,I,I,I,D,D,D,I,U,P,P,P]
LIB.surrogate_games.argtypes = [I,I,I,I,D,D,I,I,U,P,P,P,P,P]
LIB.gamma_games.argtypes = [I,I,I,P,P,P]
LIB.score_games.argtypes = [I,I,P,P,P,P,P,P,P]
for name in ['simulate_games','surrogate_games','gamma_games','score_games']:
    getattr(LIB,name).restype = None

def ptr(a):
    return None if a is None else a.ctypes.data_as(P)

def arrays(c,b,keep):
    n,r=c['n'],c['rounds']
    return (np.empty((b,r,n),np.int32) if keep else None,
            np.empty((b,r),np.int32) if keep else None,
            np.empty((b,n*(n-1)//2),np.float64))

def simulate(c,b,seed,cp=0,keep=False):
    a,s,g=arrays(c,b,keep)
    LIB.simulate_games(c['n'],c['rounds'],c['regime'],c['state_kind'],c['reaction'],c['hold'],
                       cp,b,seed,ptr(a),ptr(s),ptr(g))
    return a,s,g

def surrogate(c,a,s,b,seed,method,keep=False):
    a=np.ascontiguousarray(a,dtype=np.int32);s=np.ascontiguousarray(s,dtype=np.int32)
    assert a.shape==(c['rounds'],c['n']) and s.shape==(c['rounds'],)
    out_a,out_s,g=arrays(c,b,keep)
    LIB.surrogate_games(c['n'],c['rounds'],c['regime'],c['state_kind'],c['reaction'],c['hold'],
                        method,b,seed,ptr(a),ptr(s),ptr(out_a),ptr(out_s),ptr(g))
    return out_a,out_s,g

def gamma(a,s):
    a=np.ascontiguousarray(a,dtype=np.int32);s=np.ascontiguousarray(s,dtype=np.int32)
    if a.ndim==2:a=a[None];s=s[None]
    b,r,n=a.shape
    assert s.shape==(b,r) and s.min()>=0 and s.max()<(n+1)**2+1
    g=np.empty((b,n*(n-1)//2),np.float64)
    LIB.gamma_games(n,r,b,ptr(a),ptr(s),ptr(g))
    return g

def fit(g,n):
    emean=g.mean(0);esd=np.maximum(g.std(0),1e-12)
    z=(g-emean)/esd
    masks=np.arange(1<<n,dtype=np.int32)
    i,j=np.triu_indices(n,1)
    membership=(((masks[:,None]>>i)&1)*((masks[:,None]>>j)&1)).astype(float)
    t=z@membership.T
    tmean=t.mean(0);tsd=np.maximum(t.std(0),1e-9)
    return dict(emean=emean,esd=esd,tmean=tmean,tsd=tsd)

def score(g,n,bank):
    g=np.ascontiguousarray(g,dtype=np.float64)
    b=len(g);m=np.empty(b);w=np.empty(b,np.int32)
    LIB.score_games(n,b,ptr(g),*[ptr(bank[k]) for k in ['emean','esd','tmean','tsd']],ptr(m),ptr(w))
    return m,w

def rank_p(observed,reference):
    # The relative tolerance treats floating-point versions of mathematical ties conservatively.
    return (1+np.count_nonzero(reference>=observed-1e-10*max(1,abs(observed))))/(len(reference)+1)

def seed_for(master,*parts):
    return int(np.random.SeedSequence([master,*parts]).generate_state(1,dtype=np.uint64)[0])
