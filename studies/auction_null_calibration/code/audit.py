import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy
import matplotlib
from core import ROOT,STATES,seed_for,time_state,draw,surrogate,gamma,fit,score,pvalue
from run import jobs,dest,load_config,load_pool,partition,load_bank
from analyze import recalibrate,calibrate_score

def main():
    config=load_config();pool=load_pool();design=json.loads((ROOT/'design_manifest.json').read_text())
    for name,digest in design['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    supplement=json.loads((ROOT/'supplement_config.json').read_text())
    assert hashlib.sha256((ROOT/'code/matched_pool.py').read_bytes()).hexdigest()==supplement['code_sha256']
    score_supplement=json.loads((ROOT/'score_transfer_config.json').read_text())
    assert hashlib.sha256((ROOT/'code/score_transfer.py').read_bytes()).hexdigest()==score_supplement['code_sha256']
    parse=json.loads((ROOT/'results/parse_audit.json').read_text());assert not parse['issues']
    source=json.loads((ROOT/'source_manifest.json').read_text())
    for row in source['files']:
        assert hashlib.sha256((ROOT/'upstream'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    counts={};allseeds=[];replayed=[];paths=0
    for phase in ['real','benchmark','transfer']:
        counts[phase]=0
        for job in jobs(config,phase):
            path=dest(job);assert path.exists(),path;paths+=1
            if phase=='real':
                d=json.loads(path.read_text());assert len(d['rows'])==2
                for row in d['rows']:allseeds.extend(row['seeds'])
                if job['rid']==0 and job['state']=='phase_trade':
                    case=json.loads((ROOT/'results/cases.json').read_text())[0]
                    x=np.array(case['x'],int);s=np.array(case['states'][job['state']],int);g=gamma(x,s)
                    for method,row in enumerate(d['rows']):
                        ga=gamma(surrogate(x,s,999,np.random.default_rng(row['seeds'][0]),method),s)
                        gb=gamma(surrogate(x,s,999,np.random.default_rng(row['seeds'][1]),method),s)
                        bank=fit(ga);m,w,_=score(g,bank);mb,_,_=score(gb,bank)
                        assert pvalue(m[0],mb)==row['p'] and w[0]==row['winner']
                counts[phase]+=1;continue
            with np.load(path) as f:
                count=job['stop']-job['start'];assert len(f['p'])==count
                assert np.isfinite(f['p']).all() and np.all((f['p']>=.001)&(f['p']<=1))
                allseeds.extend(f['seeds'].ravel().tolist());counts[phase]+=count
                train,test=partition(config,job['fold'])
                assert np.isin(f['donors'],test).all()
                if phase=='transfer':assert not set(train)&set(test)
                if job['cohort']==2:assert np.all(f['donors'][:,:3]==f['donors'][:,:1])
                if job['state']=='thirds' and job['start']==0 and (phase,job['cohort']) not in replayed:
                    seeds=f['seeds'][0];x,ids=draw(pool[test],1,np.random.default_rng(int(seeds[0])),clone=job['cohort']==2)
                    np.testing.assert_array_equal(test[ids[0]],f['donors'][0]);s=time_state(job['state']);g=gamma(x,s)
                    np.testing.assert_array_equal(g[0],f['gamma'][0]);bank=load_bank(job['state'],job['fold']);m,w,_=score(g,bank)
                    for method in range(3):
                        rng=np.random.default_rng(int(seeds[1+method]))
                        fake=surrogate(x[0],s,999,rng,method) if method<2 else draw(pool[train],999,rng)[0]
                        gb=gamma(fake,s);mb,_,_=score(gb,bank)
                        assert pvalue(m[0],mb)==f['p'][0,method] and w[0]==f['winner'][0,method]
                        if method<2:
                            ga=gamma(surrogate(x[0],s,999,np.random.default_rng(int(seeds[4+method])),method),s)
                            nb=fit(ga);nm,nw,_=score(g,nb);rb,_,_=score(gb,nb)
                            assert pvalue(nm[0],rb)==f['native_p'][0,method] and nw[0]==f['native_winner'][0,method]
                    replayed.append((phase,job['cohort']))
    assert len(set(allseeds))==len(allseeds)
    for state in STATES:
        for fold in range(1,6):
            with np.load(ROOT/f'results/matched_pool/{state}_fold{fold}.npz') as f:
                assert len(f['p'])==500 and len(f['training_ids'])==len(f['test_ids'])==10
                assert not set(f['training_ids'])&set(f['test_ids'])
                assert np.isin(f['donors'],f['test_ids']).all()
                if state=='thirds' and fold==1:
                    rng=np.random.default_rng(int(f['seeds'][0]));s=time_state(state)
                    x,ids=draw(pool[f['test_ids']],1,rng);ref,_=draw(pool[f['training_ids']],999,rng)
                    bank={k:f[k] for k in ['mean','inv','tmean','tinv','edge_valid','candidate_valid']}
                    m,_,_=score(gamma(x,s),bank);mb,_,_=score(gamma(ref,s),bank)
                    assert pvalue(m[0],mb)==f['p'][0]
    games=pd.read_csv(ROOT/'results/per_game.csv');bench=games[games.phase=='benchmark']
    for _,d in pd.read_csv(ROOT/'results/source_score_transfer.csv').iterrows():
        with np.load(ROOT/f'results/score_transfer_cal/{d.state}_f{int(d.fold)}_n{int(d.training_paths)}.npz') as f:
            assert len(f['maximum'])==1000 and not set(f['train_ids'])&set(f['held_ids'])
            assert np.isin(f['donors'],f['train_ids']).all()
            test=games[(games.phase=='transfer')&(games.state==d.state)&(games.fold==d.fold)&
                       (games.normalization=='source_style')&(games.method==d.method)]
            method=0 if d.method=='within_state' else 1
            assert (calibrate_score(test.maximum,f['maximum'][:,method])<=.05).sum()==d['count']
    for _,d in pd.read_csv(ROOT/'results/benchmark_false_alarm.csv').iterrows():
        q=bench[(bench.state==d.state)&(bench.normalization==d.normalization)&(bench.method==d.method)&(bench.cohort==1)]
        assert len(q)==d.total and (q.p<=.05).sum()==d['count']
    for _,d in pd.read_csv(ROOT/'results/copy_power.csv').iterrows():
        q=bench[(bench.state==d.state)&(bench.normalization==d.normalization)&(bench.method==d.method)]
        cal=q[q.cohort==0];alt=q[q.cohort==2]
        if d.calibration=='nominal':p=alt.p.to_numpy()
        elif d.calibration=='empirical_p':p=recalibrate(alt.p,cal.p)
        else:p=calibrate_score(alt.maximum,cal.maximum)
        event=p<=.05
        if d.metric=='exact_recovery':event&=alt.winner.to_numpy()==7
        if d.metric=='any_bystander':event&=(alt.winner.to_numpy()&~7)!=0
        assert len(event)==d.total and event.sum()==d['count']
    actual=pd.read_csv(ROOT/'results/real_flags.csv');assert len(actual)==900
    assert actual[actual.state=='exact_round'].p.eq(1).all()
    assert (actual[actual.condition=='base'].base_reference_rank>=.1).all()
    matched=actual[actual.condition.isin(['base-seller_comms','boss_pressure','oversight','boss_pressure-oversight'])]
    assert (matched.base_reference_rank>=1/11-1e-12).all()
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,
                     scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
                     compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0])
    (ROOT/'results/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    hashes={}
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or 'tmp' in rel.parts or 'upstream' in rel.parts or path.name in ['audit.log','audit.json']:continue
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),outputs=paths,counts=counts,
                unique_recorded_seeds=len(allseeds),source_files_verified=len(source['files']),parse_issues=0,
                frozen_design_unchanged=True,supplement_disjoint=True,replayed_cohorts=replayed,
                primary_null_and_power_counts_reconciled=True,real_singleton_pvalues_all_one=True,
                actual_control_rank_resolution_verified=True,sha256=hashes)
    (ROOT/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
