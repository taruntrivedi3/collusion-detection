"""Frozen study, deterministic chunk seeds, resumable independent process batches."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from engine import ROOT, simulate, surrogate, fit, score, rank_p, seed_for

METHODS=['within_state','circular_shift','fresh_simulation']

def make_config():
    settings=[]
    def add(name,regime,r,kind=1,p=.5,hold=0,n=6,role='primary'):
        settings.append(dict(id=len(settings),name=name,n=n,rounds=r,regime=regime,state_kind=kind,
                             reaction=p,hold=hold,role=role))
    for name,regime in [('iid',0),('exogenous',1),('leader',2),('margin',3),('sticky',4)]:
        for r in [100,300,1000]:add(f'{name}_r{r}',regime,r,kind=0 if regime==0 else 1,hold=.9 if regime==4 else 0)
    for r in [100,300,1000]:add(f'sticky_strong_r{r}',4,r,p=.95,hold=.99,role='stress')
    for name,regime in [('leader',2),('margin',3)]:
        for r in [300,1000]:add(f'{name}_top_r{r}',regime,r,kind=2,role='state_diagnostic')
        add(f'{name}_none_r300',regime,300,kind=0,role='state_diagnostic')
    add('leader_n10_r300',2,300,n=10,role='size_diagnostic')
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),master_seed=2026102207,
                normalization_games=2000,null_draws=999,calibration_games=1000,evaluation_games=1000,
                power_games=500,coordination_probabilities=[.15,.3,.5],alpha=.05,chunk_size=25,
                settings=settings,
                power_settings=[s['name'] for s in settings if s['role']=='primary' and s['regime'] in [0,2,3,4] and s['rounds'] in [300,1000]],
                native_settings=['iid_r300','leader_r300','leader_r1000','margin_r300','margin_r1000',
                                 'sticky_r300','sticky_r1000','sticky_strong_r300','sticky_strong_r1000',
                                 'leader_none_r300','margin_top_r300'],
                native_games=1000,native_A=999,
                inference='First scan only; rank p with ties counted in the upper tail; no peeling/grow.',
                calibration='Independent empirical rank calibration of method p-values; no tuning on evaluation or power games.')

def freeze():
    path=ROOT/'config.json'
    if path.exists():raise RuntimeError('Configuration is already frozen')
    c=make_config();path.write_text(json.dumps(c,indent=2)+'\n')
    manifest=dict(created_utc=c['created_utc'],python=platform.python_version(),numpy=np.__version__,
                  machine=platform.machine(),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files={})
    for p in sorted((ROOT/'code').glob('*')):
        if p.is_file():manifest['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(c,indent=2))

def normalization(config):
    (ROOT/'results/banks').mkdir(exist_ok=True)
    for c in config['settings']:
        dest=ROOT/f"results/banks/{c['name']}.npz"
        if dest.exists():continue
        seed=seed_for(config['master_seed'],c['id'],10)
        a,s,g=simulate(c,config['normalization_games'],seed,keep=True)
        bank=fit(g,c['n'])
        np.savez_compressed(dest,**bank,gamma=g,actions=a.astype(np.uint8),states=s.astype(np.uint16),seed=np.uint64(seed))
        print('BANK',c['name'],flush=True)

def jobs(config,phase):
    for c in config['settings']:
        if phase=='honest':cohorts=[(0,config['calibration_games'],0),(1,config['evaluation_games'],0)]
        elif phase=='power':
            if c['name'] not in config['power_settings']:continue
            cohorts=[(2+k,config['power_games'],cp) for k,cp in enumerate(config['coordination_probabilities'])]
        elif phase=='native':
            if c['name'] not in config['native_settings']:continue
            cohorts=[(8,config['native_games'],0)]
        else:raise ValueError(phase)
        for cohort,total,cp in cohorts:
            for start in range(0,total,config['chunk_size']):
                yield c,phase,cohort,start,min(total,start+config['chunk_size']),cp

def destination(job):
    c,phase,cohort,start,stop,cp=job
    return ROOT/f"results/chunks/{phase}_{c['name']}_c{cohort}_{start:04d}.npz"

def work(job):
    started=time.monotonic()
    config=json.loads((ROOT/'config.json').read_text())
    c,phase,cohort,start,stop,cp=job
    bk=np.load(ROOT/f"results/banks/{c['name']}.npz")
    bank={k:bk[k] for k in ['emean','esd','tmean','tsd']}
    methods=range(2) if phase=='native' else range(3)
    count=stop-start;seeds=np.zeros((count,7),np.uint64)
    pvalues=np.ones((count,3));maxima=np.full((count,3),np.nan)
    winners=np.zeros((count,3),np.int32);quantiles=np.full((count,3,5),np.nan)
    actions=np.empty((count,c['rounds'],c['n']),np.uint8)
    states=np.empty((count,c['rounds']),np.uint16)
    gammas=np.empty((count,c['n']*(c['n']-1)//2))
    diagnostics=np.zeros((count,5))
    for local,rep in enumerate(range(start,stop)):
        seeds[local]=[seed_for(config['master_seed'],c['id'],cohort,rep,stream) for stream in range(7)]
        a,s,g=simulate(c,1,int(seeds[local,0]),cp,True);a=a[0];s=s[0]
        actions[local]=a;states[local]=s;gammas[local]=g[0]
        occ=np.bincount(s);positive=occ[occ>0]
        diagnostics[local]=[len(positive),positive.min(),np.median(positive),
                            np.mean(occ[s]<5),np.mean(s[1:]!=s[:-1])]
        for method in methods:
            scoring_bank=bank
            if phase=='native':
                _,_,ga=surrogate(c,a,s,config['native_A'],int(seeds[local,4+method]),method)
                scoring_bank=fit(ga,c['n'])
            observed,winner=score(g,c['n'],scoring_bank)
            _,_,gn=surrogate(c,a,s,config['null_draws'],int(seeds[local,1+method]),method)
            reference,_=score(gn,c['n'],scoring_bank)
            pvalues[local,method]=rank_p(observed[0],reference)
            maxima[local,method]=observed[0];winners[local,method]=winner[0]
            quantiles[local,method]=np.quantile(reference,[0,.5,.9,.95,1])
    np.savez_compressed(destination(job),pvalues=pvalues,maxima=maxima,winners=winners,quantiles=quantiles,
                        actions=actions,states=states,gamma=gammas,seeds=seeds,diagnostics=diagnostics,
                        rep=np.arange(start,stop),setting_id=c['id'],cohort=cohort,cp=cp,phase=phase,
                        elapsed_seconds=time.monotonic()-started)
    return c['name'],phase,cohort,start,time.monotonic()-started

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','banks','honest','power','native','all'])
    ap.add_argument('--workers',type=int,default=16);args=ap.parse_args()
    if args.phase=='freeze':freeze();return
    config=json.loads((ROOT/'config.json').read_text())
    if args.phase in ['banks','all']:normalization(config)
    if args.phase=='banks':return
    (ROOT/'results/chunks').mkdir(exist_ok=True)
    phases=['honest','power','native'] if args.phase=='all' else [args.phase]
    for phase in phases:
        todo=[j for j in jobs(config,phase) if not destination(j).exists()]
        print('START',phase,'chunks',len(todo),'workers',args.workers,flush=True)
        start=time.monotonic()
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(work,j) for j in todo]
            for done,fut in enumerate(cf.as_completed(futures),1):
                result=fut.result()
                if done%20==0 or done==len(todo):
                    print('PROGRESS',phase,done,'/',len(todo),'wall_seconds',round(time.monotonic()-start,1),'last',result,flush=True)
        print('FINISH',phase,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
