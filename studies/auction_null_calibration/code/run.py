import argparse
import concurrent.futures as cf
import json
import time
import numpy as np
from core import ROOT,N,R,STATES,METHODS,seed_for,time_state,draw,surrogate,gamma,fit,score,pvalue

def load_config():return json.loads((ROOT/'config.json').read_text())
def load_pool():return np.load(ROOT/'results/pool.npz')['paths']
def partition(config,fold):
    prov=json.loads((ROOT/'results/pool_provenance.json').read_text())
    if fold==0:return np.arange(len(prov)),np.arange(len(prov))
    held=config['folds'][fold-1]
    return np.array([i for i,p in enumerate(prov) if p['run'] not in held]),np.array([i for i,p in enumerate(prov) if p['run'] in held])

def bank_path(state,fold):return ROOT/f'results/banks/{state}_fold{fold}.npz'
def load_bank(state,fold):
    f=np.load(bank_path(state,fold))
    return {k:f[k] for k in ['mean','inv','tmean','tinv','edge_valid','candidate_valid']}

def make_banks(config):
    (ROOT/'results/banks').mkdir(exist_ok=True)
    pool=load_pool()
    for sid,state in enumerate(STATES):
        s=time_state(state)
        for fold in range(6):
            path=bank_path(state,fold)
            if path.exists():continue
            train,_=partition(config,fold)
            rng=np.random.default_rng(seed_for(90,sid,fold))
            x,ids=draw(pool[train],config['normalization_games'],rng)
            index=np.broadcast_to(np.arange(R)[None,:,None],(len(x),R,N)).copy()
            for label in np.unique(s):
                rows=np.flatnonzero(s==label)
                index[:,rows,:]=rows[np.argsort(rng.random((len(x),len(rows),N)),axis=1)]
            shuffled=np.stack([np.take_along_axis(x[:,v],index,axis=1) for v in range(2)],axis=1)
            g=gamma(shuffled,s);bank=fit(g)
            np.savez_compressed(path,**bank,gamma=g,donors=train[ids],seed=np.uint64(seed_for(90,sid,fold)))
            print('BANK',state,fold,flush=True)

def jobs(config,phase):
    if phase=='real':
        cases=json.loads((ROOT/'results/cases.json').read_text())
        for rid,c in enumerate(cases):
            for sid,state in enumerate(config['real_states']):yield dict(phase=phase,rid=rid,state=state,sid=sid)
    else:
        for sid,state in enumerate(STATES):
            setups=[]
            if phase=='benchmark':setups=[(0,0,config['calibration_games']),(1,0,config['evaluation_games']),(2,0,config['clone_games'])]
            if phase=='transfer':setups=[(3,f,config['transfer_games_per_fold']) for f in range(1,6)]
            for cohort,fold,total in setups:
                for start in range(0,total,config['chunk_size']):
                    yield dict(phase=phase,sid=sid,state=state,cohort=cohort,fold=fold,start=start,stop=min(total,start+config['chunk_size']))

def dest(j):
    if j['phase']=='real':return ROOT/f"results/real/r{j['rid']:03d}_{j['state']}.json"
    return ROOT/f"results/chunks/{j['phase']}_{j['state']}_c{j['cohort']}_f{j['fold']}_{j['start']:04d}.npz"

def real_work(j):
    config=load_config();case=json.loads((ROOT/'results/cases.json').read_text())[j['rid']]
    x=np.array(case['x'],np.int32);s=np.array(case['states'][j['state']],np.int32);g=gamma(x,s)
    rows=[]
    for m in range(2):
        sa=seed_for(10,j['rid'],j['sid'],m,0);sb=seed_for(10,j['rid'],j['sid'],m,1)
        ga=gamma(surrogate(x,s,config['null_draws'],np.random.default_rng(sa),m),s)
        gb=gamma(surrogate(x,s,config['null_draws'],np.random.default_rng(sb),m),s)
        bank=fit(ga);sm,w,v=score(g,bank);ref,_,_=score(gb,bank)
        rows.append(dict(method=METHODS[m],p=pvalue(sm[0],ref),maximum=float(sm[0]),winner=int(w[0]),view=int(v[0]),
                         valid_candidates=int(bank['candidate_valid'].sum()),valid_edges=int(bank['edge_valid'].sum()),
                         seeds=[sa,sb],reference_quantiles=np.quantile(ref,[0,.5,.95,1]).tolist()))
    result=dict(**j,run=case['run'],condition=case['condition'],rows=rows,gamma=g[0].tolist())
    dest(j).write_text(json.dumps(result,indent=2)+'\n')
    return j['phase'],j['rid'],j['state']

def benchmark_work(j):
    config=load_config();allpool=load_pool();train,test=partition(config,j['fold'])
    bank=load_bank(j['state'],j['fold']);s=time_state(j['state']);b=config['null_draws']
    size=j['stop']-j['start'];p=np.ones((size,3));pn=np.ones((size,2));maximum=np.zeros((size,3));mn=np.zeros((size,2))
    winner=np.zeros((size,3),int);wn=np.zeros((size,2),int);view=np.zeros((size,3),int)
    seeds=np.zeros((size,6),np.uint64);donors=np.zeros((size,5),int);gs=np.zeros((size,2,10));valid=np.zeros((size,2),int)
    quant=np.zeros((size,3,4));qnative=np.zeros((size,2,4))
    for pos,rep in enumerate(range(j['start'],j['stop'])):
        seeds[pos]=[seed_for(20,j['sid'],j['cohort'],j['fold'],rep,k) for k in range(6)]
        x,ids=draw(allpool[test],1,np.random.default_rng(int(seeds[pos,0])),clone=j['cohort']==2)
        x=x[0];donors[pos]=test[ids[0]];g=gamma(x,s);gs[pos]=g[0]
        sm,w,v=score(g,bank)
        for m in range(3):
            rng=np.random.default_rng(int(seeds[pos,1+m]))
            if m<2:xs=surrogate(x,s,b,rng,m)
            else:xs,_=draw(allpool[train],b,rng)
            gb=gamma(xs,s);ref,_,_=score(gb,bank)
            p[pos,m]=pvalue(sm[0],ref);maximum[pos,m]=sm[0];winner[pos,m]=w[0];view[pos,m]=v[0]
            quant[pos,m]=np.quantile(ref,[0,.5,.95,1])
            if m<2:
                ga=gamma(surrogate(x,s,b,np.random.default_rng(int(seeds[pos,4+m])),m),s)
                nativebank=fit(ga);sn,nn,_=score(g,nativebank);rn,_,_=score(gb,nativebank)
                pn[pos,m]=pvalue(sn[0],rn);mn[pos,m]=sn[0];wn[pos,m]=nn[0]
                valid[pos,m]=nativebank['candidate_valid'].sum();qnative[pos,m]=np.quantile(rn,[0,.5,.95,1])
    np.savez_compressed(dest(j),p=p,native_p=pn,maximum=maximum,native_maximum=mn,winner=winner,native_winner=wn,
                        view=view,seeds=seeds,donors=donors,gamma=gs,native_valid=valid,reference_quantiles=quant,
                        native_quantiles=qnative,rep=np.arange(j['start'],j['stop']),cohort=j['cohort'],fold=j['fold'])
    return j['phase'],j['state'],j['cohort'],j['fold'],j['start']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['banks','real','benchmark','transfer','all']);ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
    config=load_config()
    for sub in ['real','chunks']:(ROOT/f'results/{sub}').mkdir(exist_ok=True)
    if args.phase in ['banks','all']:make_banks(config)
    if args.phase=='banks':return
    for phase in (['real','benchmark','transfer'] if args.phase=='all' else [args.phase]):
        todo=[j for j in jobs(config,phase) if not dest(j).exists()]
        start=time.monotonic();print('START',phase,len(todo),flush=True)
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs=[pool.submit(real_work if phase=='real' else benchmark_work,j) for j in todo]
            for i,f in enumerate(cf.as_completed(futs),1):
                result=f.result()
                if i%20==0 or i==len(todo):print('PROGRESS',phase,i,'/',len(todo),'seconds',round(time.monotonic()-start,1),'last',result,flush=True)
        print('FINISH',phase,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
