import json
import sys
import time
import numpy as np
from core import ROOT,N,R,PAIRS,MASKS,STATES,draw,surrogate,gamma,fit,score,pvalue,time_state
sys.path.insert(0,str(ROOT/'source_snapshot'))
from information import information_scores,subset_scores

def main():
    checks=[];rng=np.random.default_rng(145)
    def ok(name):checks.append(name);print('PASS',name,flush=True)
    x=rng.integers(0,6,size=(50,2,29,5));s=rng.integers(0,8,size=29)
    g=gamma(x,s)
    for v in range(2):
        ref=information_scores(x[:,v],s,include_ipf=False)[0]
        np.testing.assert_allclose(g[:,v],ref,atol=2e-14,rtol=0)
    ok('Conditional MI matches the supplied auction implementation for two views and six action categories')
    bank=fit(g[:25]);sm,mask,view=score(g[25:],bank)
    groups=[[i for i in range(5) if m>>i&1] for m in MASKS]
    ref=np.concatenate([subset_scores(g[:,v],PAIRS,groups,slice(0,25))[0][25:] for v in range(2)],axis=1)
    np.testing.assert_allclose(sm,ref.max(1),rtol=0,atol=1e-13)
    np.testing.assert_array_equal(mask,MASKS[ref.argmax(1)%26])
    ok('All 52 subset/view scores agree with the source A/B normalization')
    obs=x[0]
    for method in [0,1]:
        fake=surrogate(obs,s,20,rng,method)
        for row in fake:
            for j in range(5):
                if method==0:
                    for label in np.unique(s):
                        a=sorted(map(tuple,obs[:,s==label,j].T));b=sorted(map(tuple,row[:,s==label,j].T));assert a==b
                else:assert any(np.array_equal(row[:,:,j],np.roll(obs[:,:,j],shift,axis=1)) for shift in range(29))
    ok('Both surrogates keep the direction/level bundle together; shuffles preserve conditional counts and shifts cyclic order')
    pool=rng.integers(6,size=(50,2,29))
    data,ids=draw(pool,1000,rng)
    np.testing.assert_array_equal(data[:,0,:,0],pool[ids[:,0],0])
    assert np.corrcoef(ids.T)[0,1]<.1
    copied,cids=draw(pool,1000,rng,clone=True)
    np.testing.assert_array_equal(copied[:,:,:,0],copied[:,:,:,1]);np.testing.assert_array_equal(copied[:,:,:,1],copied[:,:,:,2])
    assert np.all(cids[:,:3]==cids[:,:1])
    ok('Independent donor draws and planted exact three-agent copying preserve complete paths')
    exact=gamma(x,time_state('exact_round'));assert np.all(exact==0)
    z=score(exact,fit(exact));assert np.all(z[0]==0) and pvalue(0,z[0])==1
    flat=np.zeros((1,2,29,5),int);flat[:,0]=1;flat[:,1]=2
    assert np.all(gamma(flat,time_state('none'))==0)
    ok('Singleton states and a constant $99 agreement have zero information, with no artificial infinite scores')
    assert pvalue(3,np.array([1,2,3,4]))==.6 and pvalue(1,np.ones(999))==1
    ok('Rank p-values count ties conservatively and include the observed statistic')
    null=rng.integers(3,size=(2000,2,29,5));nb=fit(gamma(null,np.zeros(29)))
    alt=rng.integers(3,size=(200,2,29,5));alt[:,:,:,1]=alt[:,:,:,0];alt[:,:,:,2]=alt[:,:,:,0]
    av,aw,_=score(gamma(alt,np.zeros(29)),nb)
    assert np.mean(aw==7)>.9
    ok('High-entropy copied triples are recovered in the simple strong-signal check')
    start=time.perf_counter()
    for method in [0,1]:
        fake=surrogate(obs,time_state('thirds'),1998,rng,method)
        gg=gamma(fake,time_state('thirds'));bb=fit(gg[:999]);score(gg[999:],bb)
    result=dict(checks=checks,seconds_for_two_A_B_tests=time.perf_counter()-start)
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
