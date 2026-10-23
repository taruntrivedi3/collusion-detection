"""Exploratory transfer of the repaired, source-style raw maximum calibration."""
import concurrent.futures as cf
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from core import ROOT,STATES,seed_for,time_state,draw,surrogate,gamma,fit,score
from run import load_pool,partition,load_config

def work(job):
    sid,fold,size=job;config=load_config();pool=load_pool();train,test=partition(config,fold)
    if size==10:
        provenance=json.loads((ROOT/'results/pool_provenance.json').read_text())
        chosen=config['folds'][fold%5]
        train=np.array([i for i,p in enumerate(provenance) if p['run'] in chosen])
    assert len(train)==size and not set(train)&set(test)
    s=time_state(STATES[sid]);maximum=np.zeros((1000,2));donors=np.zeros((1000,5),int);seeds=np.zeros((1000,3),np.uint64)
    for rep in range(1000):
        seeds[rep]=[seed_for(80,sid,fold,size,rep,k) for k in range(3)]
        x,ids=draw(pool[train],1,np.random.default_rng(int(seeds[rep,0])))
        donors[rep]=train[ids[0]];g=gamma(x,s)
        for m in range(2):
            ga=gamma(surrogate(x[0],s,999,np.random.default_rng(int(seeds[rep,1+m])),m),s)
            maximum[rep,m]=score(g,fit(ga))[0][0]
    path=ROOT/f'results/score_transfer_cal/{STATES[sid]}_f{fold}_n{size}.npz'
    np.savez_compressed(path,maximum=maximum,donors=donors,seeds=seeds,train_ids=train,held_ids=test)
    return STATES[sid],fold,size

def main():
    jobs=[(s,f,size) for s in range(4) for f in range(1,6) for size in [40,10]]
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),timing='Added after raw-score calibration repaired the same-pool benchmark',
                question='Does source-style raw-maximum calibration transfer better than the fixed-normalization empirical reference?',
                source_style_A=999,training_calibration_groups=1000,evaluation='The existing 500 held-out groups per fold, with no refitting to their outcomes',
                pools=[40,10],jobs=jobs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'score_transfer_config.json'
    if not path.exists():path.write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'results/score_transfer_cal').mkdir(exist_ok=True)
    todo=[j for j in jobs if not (ROOT/f'results/score_transfer_cal/{STATES[j[0]]}_f{j[1]}_n{j[2]}.npz').exists()]
    with cf.ProcessPoolExecutor(max_workers=12) as pool:
        for result in pool.map(work,todo):print(result,flush=True)

if __name__=='__main__':main()
