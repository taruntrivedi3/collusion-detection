"""Post-result control: equal donor counts for training and held-out empirical references."""
import concurrent.futures as cf
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
from core import ROOT,STATES,R,N,seed_for,time_state,draw,gamma,fit,score,pvalue
from run import load_pool

def work(job):
    sid,fold,train_ids,test_ids=job
    pool=load_pool();s=time_state(STATES[sid]);train=np.array(train_ids);test=np.array(test_ids)
    rng=np.random.default_rng(seed_for(70,sid,fold))
    x,ids=draw(pool[train],2000,rng)
    ix=np.broadcast_to(np.arange(R)[None,:,None],(len(x),R,N)).copy()
    for st in np.unique(s):
        rows=np.flatnonzero(s==st);ix[:,rows,:]=rows[np.argsort(rng.random((len(x),len(rows),N)),axis=1)]
    gx=gamma(np.stack([np.take_along_axis(x[:,v],ix,axis=1) for v in range(2)],axis=1),s)
    bank=fit(gx);p=[];donors=[];seeds=[];maximum=[]
    for rep in range(500):
        seed=seed_for(71,sid,fold,rep);rng=np.random.default_rng(seed)
        obs,oi=draw(pool[test],1,rng);ref,_=draw(pool[train],999,rng)
        m,_,_=score(gamma(obs,s),bank);mb,_,_=score(gamma(ref,s),bank)
        p.append(pvalue(m[0],mb));maximum.append(m[0]);donors.append(test[oi[0]]);seeds.append(seed)
    np.savez_compressed(ROOT/f'results/matched_pool/{STATES[sid]}_fold{fold}.npz',p=p,donors=donors,
                        seeds=np.array(seeds,np.uint64),maximum=maximum,training_ids=train,test_ids=test,**bank)
    return STATES[sid],fold,float(np.mean(np.array(p)<=.05))

def main():
    config=json.loads((ROOT/'config.json').read_text());prov=json.loads((ROOT/'results/pool_provenance.json').read_text())
    jobs=[]
    for sid in range(4):
        for f in range(5):
            test=config['folds'][f];train=config['folds'][(f+1)%5]
            ti=[i for i,p in enumerate(prov) if p['run'] in train];vi=[i for i,p in enumerate(prov) if p['run'] in test]
            assert len(ti)==len(vi)==10 and not set(ti)&set(vi)
            jobs.append((sid,f+1,ti,vi))
    path=ROOT/'supplement_config.json'
    document=dict(created_utc=datetime.now(timezone.utc).isoformat(),reason='Control the training 40 versus held-out 10 donor support-size difference seen in the primary transfer experiment',
                  timing='Added after primary transfer results; not part of the original frozen protocol',
                  design='Two source markets train versus two disjoint source markets test, 10 donor paths each; next fixed fold supplies training markets',
                  jobs=jobs,draws=999,normalization_games=2000,evaluation_games=500,
                  code_sha256=hashlib.sha256(__file__ and open(__file__,'rb').read()).hexdigest())
    if path.exists():assert json.loads(path.read_text())['jobs']==json.loads(json.dumps(jobs))
    else:path.write_text(json.dumps(document,indent=2)+'\n')
    (ROOT/'results/matched_pool').mkdir(exist_ok=True)
    todo=[j for j in jobs if not (ROOT/f'results/matched_pool/{STATES[j[0]]}_fold{j[1]}.npz').exists()]
    with cf.ProcessPoolExecutor(max_workers=12) as pool:
        for result in pool.map(work,todo):print(result,flush=True)

if __name__=='__main__':main()
