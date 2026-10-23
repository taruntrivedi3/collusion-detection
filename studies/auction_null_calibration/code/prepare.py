import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from core import ROOT,STATES,seed_for,state_diagnostics,time_state

def compact(rows):return np.unique(np.array([json.dumps(x) for x in rows]),return_inverse=True)[1].astype(np.int32)

def main():
    cases=[];pool=[];pool_provenance=[];audit=[]
    episodes=sorted((ROOT/'results/parsed').glob('*.json'))
    for path in episodes:
        d=json.loads(path.read_text());n=d['summary'];agents=d['sellers']
        actions={(a['round'],a['agent']):a for a in d['actions']}
        x=np.array([[[actions[t,a][view] for a in agents] for t in range(2,31)] for view in ['direction','level']],np.int32)
        state={name:time_state(name) for name in STATES}
        state['phase_trade']=compact([[int((t-1)//10),int(d['rounds'][t-2]['trades']>0)] for t in range(2,31)])
        # This is a richness/occupancy diagnostic, not a claim of sufficient or causal conditioning.
        def band(value):return -1 if value is None else (0 if value<=90 else 1 if value<=95 else 2)
        state['rich_observed']=compact([
            [int((t-1)//10),int(d['rounds'][t-2]['trades']>0),
             [actions[t-1,a]['level'] for a in agents],
             [[band(actions[t,a]['visible_best_bid']),band(actions[t,a]['visible_best_ask'])] for a in agents]]
            for t in range(2,31)])
        states_order=['none','thirds','blocks5','phase_trade','rich_observed','exact_round']
        cases.append(dict(run=d['run'],condition=n['condition'],x=x.tolist(),
                          states={s:state[s].tolist() for s in states_order},sellers=agents,
                          summary=n,parsed_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for name,s in state.items():
            within_constant=sum(all(len(np.unique(x[v,s==st,j]))==1 for st in np.unique(s)) for v in range(2) for j in range(5))
            audit.append(dict(run=d['run'],condition=n['condition'],state=name,**state_diagnostics(s),
                              constant_agent_views=within_constant,agent_views=10))
        if n['condition']=='base':
            assert not n['seller_comms'] and n['sellers']==5 and n['buyers']==5
            assert set(d['config']['seller_valuations'])=={80} and set(d['config']['buyer_valuations'])=={100}
            for j,a in enumerate(agents):
                pool.append(x[:,:,j]);pool_provenance.append(dict(run=d['run'],agent=a))
    assert len(cases)==75 and len(pool)==50
    (ROOT/'results/cases.json').write_text(json.dumps(cases)+'\n')
    np.savez_compressed(ROOT/'results/pool.npz',paths=np.array(pool,np.int32))
    (ROOT/'results/pool_provenance.json').write_text(json.dumps(pool_provenance,indent=2)+'\n')
    (ROOT/'results/state_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    base=sorted({p['run'] for p in pool_provenance})
    folds=[base[i:i+2] for i in range(0,10,2)]
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),states=STATES,real_states=list(cases[0]['states']),
                null_draws=999,normalization_games=2000,calibration_games=1000,evaluation_games=1000,
                clone_games=1000,transfer_games_per_fold=500,folds=folds,chunk_size=20,
                views=['direction','level'],candidates=52,alpha=.05,seed_scheme='SeedSequence([2026102307, ...])',
                normalizer='Independent empirical paths, shuffled within state before fitting bank A',
                benchmark='Independent whole seller paths drawn with replacement from the frozen base pool',
                clone='Agents 0,1,2 copy one full random donor path; all marginal path distributions preserved',
                source_style='Additional independent A bank per observed trajectory and surrogate method',
                live_llm_runs=0)
    path=ROOT/'config.json'
    if path.exists():raise RuntimeError('Refusing to overwrite frozen configuration')
    path.write_text(json.dumps(config,indent=2)+'\n')
    manifest=dict(created_utc=config['created_utc'],files={})
    for path in [ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'results/cases.json',ROOT/'results/pool.npz',ROOT/'results/pool_provenance.json']+list((ROOT/'code').glob('*'))+list((ROOT/'source_snapshot').glob('*')):
        if path.is_file():manifest['files'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'design_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(cases),donor_paths=len(pool),folds=folds),indent=2))

if __name__=='__main__':main()
