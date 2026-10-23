"""Reference checks against the frozen supplied implementation and process invariants."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from engine import ROOT, simulate, surrogate, gamma, fit, score, rank_p

sys.path.insert(0,str(ROOT/'source_snapshot'))
from collusion.estimate import gamma as reference_gamma, encode_states
from collusion.detect import subset_scores
from collusion.null import NullContext
from collusion.game import _react, LeaderReactive, MarginReactive

def run():
    checks=[];rng=np.random.default_rng(822)
    def checked(name):checks.append(name);print('PASS',name,flush=True)
    n,r=6,137;i,j=np.triu_indices(n,1)
    for state_count in [1,7,28,49]:
        a=rng.integers(3,size=(r,n),dtype=np.int32)
        s=rng.integers(state_count,size=r,dtype=np.int32)
        np.testing.assert_allclose(gamma(a,s)[0],reference_gamma(a,s)[i,j],atol=3e-14,rtol=0)
    checked('C++ conditional MI matches supplied NumPy implementation, including sparse states')
    c=dict(n=6,rounds=300,regime=2,state_kind=1,reaction=.5,hold=0)
    a,s,g=simulate(c,401,11,keep=True)
    for row in [0,15,200]:
        ref=encode_states(a[row],'full')
        np.testing.assert_array_equal(np.unique(s[row],return_inverse=True)[1],ref)
        np.testing.assert_allclose(g[row],reference_gamma(a[row],ref)[i,j],atol=3e-14)
    checked('Endogenous state updates match original full-history encoder')
    dense=np.zeros((len(g),n,n));dense[:,i,j]=g;dense[:,j,i]=g
    bank=fit(g[:200],n);m,w=score(g[200:],n,bank)
    ctx=NullContext(dense[200],dense[:200],dense[201:])
    masks,sreal,mb=subset_scores(ctx)
    np.testing.assert_allclose(m[0],sreal.max(),atol=3e-6,rtol=3e-6)
    np.testing.assert_allclose(m[1:],mb,atol=3e-6,rtol=3e-6)
    assert w[0]==masks[sreal.argmax()]
    # Independent brute-force float64 scores for every subset.
    z=(g[200:]-bank['emean'])/bank['esd']
    brute=[]
    for mask in range(1<<n):
        if mask.bit_count()<2:continue
        edges=[q for q,(l,h) in enumerate(zip(i,j)) if (mask>>l)&(mask>>h)&1]
        brute.append((z[:,edges].sum(1)-bank['tmean'][mask])/bank['tsd'][mask])
    np.testing.assert_allclose(m,np.max(brute,axis=0),atol=3e-14)
    checked('Recursive subset maximum agrees with source and float64 brute force')
    aa,ss,gg=surrogate(c,a[0],s[0],25,12,0,True)
    for b in range(len(aa)):
        np.testing.assert_array_equal(ss[b],s[0])
        for state in np.unique(s[0]):
            rows=s[0]==state
            for agent in range(n):
                np.testing.assert_array_equal(np.bincount(aa[b,rows,agent],minlength=3),
                                              np.bincount(a[0,rows,agent],minlength=3))
        np.testing.assert_allclose(gg[b],reference_gamma(aa[b],ss[b])[i,j],atol=3e-14)
    checked('Within-state shuffling preserves all individual conditional action counts')
    short=dict(c,rounds=7)
    a0=rng.integers(3,size=(7,n),dtype=np.int32);s0=np.zeros(7,np.int32)
    aa,ss,_=surrogate(short,a0,s0,100,99,1,True)
    zero_seen=False
    for b in range(100):
        for agent in range(n):
            assert any(np.array_equal(aa[b,:,agent],np.roll(a0[:,agent],offset)) for offset in range(7))
            zero_seen|=np.array_equal(aa[b,:,agent],a0[:,agent])
    assert zero_seen
    checked('Circular shifts preserve cyclic order and permit zero offset')
    aa,ss,_=surrogate(c,a[0],s[0],20,124,2,True)
    assert len({x.tobytes() for x in ss})==20
    for b in range(20):
        np.testing.assert_array_equal(np.unique(ss[b],return_inverse=True)[1],encode_states(aa[b],'full'))
    checked('Fresh-game null recomputes dynamically consistent histories')
    sticky=dict(c,regime=4,hold=.9)
    aa,ss,_=simulate(sticky,10,93,keep=True)
    for b in range(10):
        for t in range(2,300):
            cnt=np.bincount(aa[b,t-1],minlength=3)
            code=cnt[0]*7+cnt[1]+1
            assert ss[b,t] in [ss[b,t-1],code]
    checked('Sticky public state either persists or refreshes from actual prior actions')
    for cls in [LeaderReactive(.5),MarginReactive(.5,3)]:
        for counts in [(6,0,0),(3,3,0),(2,2,2),(4,2,0),(4,1,1)]:
            counts=np.array(counts);top=counts.max();leaders=np.flatnonzero(counts==top)
            p=.5 if not isinstance(cls,MarginReactive) or top-np.sort(counts)[-2]>=3 else 0
            probs=np.full(3,(1-p)/3);probs[(leaders+1)%3]+=p/len(leaders)
            draws=np.array([_react(cls,counts,rng) for _ in range(10000)])
            empirical=np.bincount(draws,minlength=3)/len(draws)
            assert np.max(abs(empirical-probs))<.023
    checked('Categorical probability construction agrees with original independent tie/reaction policy')
    for cp in [0,.5,1]:
        long=dict(c,regime=1,rounds=30000)
        aa,ss,_=simulate(long,1,1234,cp,True);aa=aa[0];ss=ss[0]
        for state in np.unique(ss):
            rows=ss==state;count=rows.sum()
            if state==0 or count<100:continue
            code=int(state)-1;counts=np.array([code//7,code%7,6-code//7-code%7])
            leaders=np.flatnonzero(counts==counts.max())
            expected=np.full(3,1/6);expected[(leaders+1)%3]+=.5/len(leaders)
            for agent in range(n):
                probs=expected if agent<3 else np.full(3,1/3)
                actual=np.bincount(aa[rows,agent],minlength=3)/count
                assert np.all(abs(actual-probs)<6*np.sqrt(probs*(1-probs)/count))
    checked('Shared-uniform coordination retains each conditional marginal policy')
    assert rank_p(3,np.array([1,2,3,4]))==.6
    assert rank_p(3,np.array([1,2]))==1/3
    assert rank_p(1,np.ones(999))==1
    checked('Rank p-values include observed game and conservative ties')
    iid=dict(c,regime=0,state_kind=0)
    _,_,base=simulate(iid,2000,332)
    bk=fit(base,n)
    _,_,alt=simulate(iid,100,334,.7)
    sm,sw=score(alt,n,bk)
    assert np.mean(sw==7)>.9
    checked('Strong planted triangle wins the subset scan')
    timings={}
    a,s,_=simulate(c,1,129,keep=True)
    for method in range(3):
        start=time.perf_counter()
        for trial in range(5):
            _,_,gg=surrogate(c,a[0],s[0],999,1000+trial,method)
            score(gg,n,bank)
        timings[str(method)]=(time.perf_counter()-start)/5
    result=dict(checks=checks,seconds_per_999_surrogates=timings)
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
