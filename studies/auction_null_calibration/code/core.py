import ctypes as C
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIB=C.CDLL(str(ROOT/'code/mi.so'))
LIB.batch_mi.argtypes=[C.c_int]*5+[C.c_void_p]*3
LIB.batch_mi.restype=None
N=5;V=2;R=29;K=6
PAIRS=np.array([(i,j) for i in range(N) for j in range(i+1,N)])
MASKS=np.array([m for m in range(1<<N) if m.bit_count()>=2])
MEMBERSHIP=np.array([[(m>>i&1)*(m>>j&1) for i,j in PAIRS] for m in MASKS],float)
METHODS=['within_state','circular_shift','trajectory_reference']
STATES=['none','thirds','blocks5','exact_round']

def seed_for(*parts):return int(np.random.SeedSequence([2026102307,*parts]).generate_state(1,np.uint64)[0])

def time_state(name):
    rounds=np.arange(2,31)
    return {'none':np.zeros(R,int),'thirds':(rounds-1)//10,'blocks5':(rounds-1)//5,
            'exact_round':np.arange(R)}[name].astype(np.int32)

def gamma(x,states):
    x=np.ascontiguousarray(x,dtype=np.int32)
    if x.ndim==3:x=x[None]
    b,v,r,n=x.shape
    assert (v,r,n)==(V,R,N) and x.min()>=0 and x.max()<K
    states=np.ascontiguousarray(np.unique(states,return_inverse=True)[1],dtype=np.int32)
    out=np.empty((b,v,len(PAIRS)))
    LIB.batch_mi(b,v,r,n,K,x.ctypes.data,states.ctypes.data,out.ctypes.data)
    return out

def draw(pool,b,rng,clone=False):
    ids=rng.integers(len(pool),size=(b,N))
    if clone:ids[:,:3]=ids[:,:1]
    return np.ascontiguousarray(pool[ids].transpose(0,2,3,1)),ids

def surrogate(x,states,b,rng,method):
    index=np.broadcast_to(np.arange(R)[None,:,None],(b,R,N)).copy()
    if method==0:
        for state in np.unique(states):
            rows=np.flatnonzero(states==state)
            order=np.argsort(rng.random((b,len(rows),N)),axis=1)
            index[:,rows,:]=rows[order]
    elif method==1:index=(index+rng.integers(R,size=(b,1,N)))%R
    else:raise ValueError(method)
    return np.stack([view[index,np.arange(N)] for view in x],axis=1)

def fit(g):
    mean=g.mean(0);sd=g.std(0,ddof=1);ev=sd>1e-10
    inv=np.divide(1,sd,out=np.zeros_like(sd),where=ev)
    z=(g-mean)*inv
    t=z@MEMBERSHIP.T
    tm=t.mean(0);ts=t.std(0,ddof=1);valid=ts>1e-10
    ti=np.divide(1,ts,out=np.zeros_like(ts),where=valid)
    return dict(mean=mean,inv=inv,tmean=tm,tinv=ti,edge_valid=ev,candidate_valid=valid)

def score(g,bank):
    z=(g-bank['mean'])*bank['inv']
    t=z@MEMBERSHIP.T
    s=((t-bank['tmean'])*bank['tinv']).reshape(len(g),-1)
    winner=s.argmax(1)
    return s.max(1),MASKS[winner%len(MASKS)],winner//len(MASKS)

def pvalue(real,reference):
    return (1+np.count_nonzero(reference>=real-1e-10*max(1,abs(real))))/(len(reference)+1)

def state_diagnostics(s):
    _,ix,count=np.unique(s,return_inverse=True,return_counts=True)
    return dict(states=len(count),singleton_fraction=float(np.mean(count[ix]==1)),
                small_state_fraction=float(np.mean(count[ix]<5)),
                movable_fraction=float(np.mean(count[ix]>1)),median_occupancy=float(np.median(count)))
