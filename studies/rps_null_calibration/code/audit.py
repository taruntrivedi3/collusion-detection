"""Final integrity/replay audit without changing experiment outputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from engine import ROOT, simulate, surrogate, score, fit, rank_p
from run_study import jobs, destination

def main():
    config=json.loads((ROOT/'config.json').read_text())
    original=json.loads((ROOT/'source_manifest.json').read_text())
    for record in original['files']:
        path=ROOT/'source_snapshot'/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],path
    frozen=json.loads((ROOT/'experiment_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()==frozen['config_sha256']
    for name,digest in frozen['files'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==digest,name
    allseeds=[];counts={};replayed=[]
    for phase in ['honest','power','native']:
        counts[phase]=0
        for job in jobs(config,phase):
            c,_,cohort,start,stop,cp=job
            with np.load(destination(job)) as f:
                assert len(f['pvalues'])==stop-start
                np.testing.assert_array_equal(f['rep'],np.arange(start,stop))
                assert np.isfinite(f['pvalues']).all()
                assert np.all((f['pvalues']>=.001)&(f['pvalues']<=1))
                assert np.all(f['actions']<=2)
                allseeds.append(f['seeds'].ravel())
                counts[phase]+=stop-start
                # Replay the first complete reference test per phase at a feedback setting.
                if c['name']=='leader_r300' and start==0 and phase not in replayed and cohort!=0:
                    seeds=f['seeds'][0];a,s,g=simulate(c,1,int(seeds[0]),cp,True)
                    np.testing.assert_array_equal(a[0],f['actions'][0]);np.testing.assert_array_equal(s[0],f['states'][0])
                    np.testing.assert_allclose(g[0],f['gamma'][0],rtol=0,atol=0)
                    loaded=np.load(ROOT/f"results/banks/{c['name']}.npz")
                    bank={k:loaded[k] for k in ['emean','esd','tmean','tsd']}
                    for method in range(2 if phase=='native' else 3):
                        localbank=bank
                        if phase=='native':
                            _,_,ga=surrogate(c,a[0],s[0],config['native_A'],int(seeds[4+method]),method)
                            localbank=fit(ga,c['n'])
                        m,w=score(g,c['n'],localbank)
                        _,_,gn=surrogate(c,a[0],s[0],config['null_draws'],int(seeds[1+method]),method)
                        mn,_=score(gn,c['n'],localbank)
                        assert rank_p(m[0],mn)==f['pvalues'][0,method]
                        assert m[0]==f['maxima'][0,method] and w[0]==f['winners'][0,method]
                    replayed.append(phase)
    seeds=np.concatenate(allseeds)
    assert len(np.unique(seeds))==len(seeds)
    nominal=pd.read_csv(ROOT/'results/nominal_false_alarm.csv')
    empirical=pd.read_csv(ROOT/'results/empirical_false_alarm.csv')
    pergame=pd.read_csv(ROOT/'results/per_game.csv')
    for _,d in nominal.iterrows():
        q=pergame[(pergame.phase=='honest_evaluation')&(pergame.setting==d.setting)&(pergame.method==d.method)]
        assert len(q)==d.total and (q.p<=.05).sum()==d['count']
        e=empirical[(empirical.setting==d.setting)&(empirical.method==d.method)].iloc[0]
        assert (q.adjusted_p<=.05).sum()==e['count']
    power=pd.read_csv(ROOT/'results/power.csv')
    for _,d in power.iterrows():
        q=pergame[(pergame.phase=='power')&(pergame.setting==d.setting)&(pergame.method==d.method)&(pergame.cp==d.cp)]
        p=q.p if d.calibration=='nominal' else q.adjusted_p
        event=p<=.05
        if d.metric=='exact_recovery':event&=q.winner==7
        if d.metric=='any_innocent':event&=(q.winner.astype(int)&~7)!=0
        assert len(q)==d.total and event.sum()==d['count']
    hashes={}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or 'tmp' in path.relative_to(ROOT).parts or path.name in ['audit.json','audit.log']:continue
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    audit=dict(completed_utc=datetime.now(timezone.utc).isoformat(),observed_games=counts,
               unique_recorded_seeds=len(seeds),replayed_phases=replayed,source_hashes_unchanged=True,
               frozen_design_and_engine_unchanged=True,all_summary_counts_reconciled=True,sha256=hashes)
    (ROOT/'results/audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
