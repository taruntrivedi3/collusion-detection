"""Aggregate frozen study; independent empirical calibration, CIs and standalone plots."""
import base64
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
from engine import ROOT
from run_study import jobs, destination, METHODS

LABELS=dict(within_state='Within-state shuffle',circular_shift='Circular shift',fresh_simulation='Fresh honest simulation')
COLORS=['#1976a3','#d68117','#258456']

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return min(p,max(0,mid-half)),max(p,min(1,mid+half))

def rate(events):
    k=int(np.count_nonzero(events));n=len(events);lo,hi=wilson(k,n)
    return dict(count=k,total=n,rate=k/n,ci_low=lo,ci_high=hi)

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def recalibrate(p,cal):
    return (1+np.searchsorted(np.sort(cal),p,side='right'))/(len(cal)+1)

def load(config,phase,c,cohort):
    parts=[]
    for job in jobs(config,phase):
        if job[0]['name']!=c['name'] or job[2]!=cohort:continue
        with np.load(destination(job)) as f:
            parts.append({k:f[k] for k in ['pvalues','maxima','winners','diagnostics','rep']})
    return {k:np.concatenate([part[k] for part in parts]) for k in parts[0]}

def table(frame,columns):
    rows=['| '+' | '.join(label for key,label in columns)+' |',
          '| '+' | '.join('---' for _ in columns)+' |']
    for _,row in frame.iterrows():rows.append('| '+' | '.join(str(row[key]) for key,_ in columns)+' |')
    return '\n'.join(rows)

def compact_table(df,settings):
    rows=[]
    for name in settings:
        row={'setting':name}
        for method in METHODS:
            d=df[(df.setting==name)&(df.method==method)].iloc[0]
            row[method]=f"{100*d['rate']:.1f}% ({int(d['count'])}/{int(d['total'])}; {100*d.ci_low:.1f}–{100*d.ci_high:.1f}%)"
        rows.append(row)
    return table(pd.DataFrame(rows),[('setting','Setting')]+[(m,LABELS[m]) for m in METHODS])

def main():
    config=json.loads((ROOT/'config.json').read_text())
    expected=[destination(j) for phase in ['honest','power','native'] for j in jobs(config,phase)]
    missing=[str(p) for p in expected if not p.exists()]
    if missing:raise RuntimeError(f'{len(missing)} chunks missing, e.g. {missing[:3]}')
    nominal=[];calibrated=[];native=[];power=[];occupancy=[];individual=[];cuts=[]
    for c in config['settings']:
        cal=load(config,'honest',c,0);obs=load(config,'honest',c,1)
        for method_id,method in enumerate(METHODS):
            p=obs['pvalues'][:,method_id];calp=cal['pvalues'][:,method_id]
            adjusted=recalibrate(p,calp)
            identity=dict(setting=c['name'],method=method,rounds=c['rounds'],n=c['n'],role=c['role'])
            d=dict(**identity,**rate(p<=.05));d['inflation_p']=binomtest(d['count'],d['total'],.05,alternative='greater').pvalue
            nominal.append(d);calibrated.append(dict(**identity,**rate(adjusted<=.05)))
            grid=np.arange(1,1001)/1000
            allowed=grid[recalibrate(grid,calp)<=.05]
            cuts.append(dict(**identity,largest_accepted_raw_p=allowed.max() if len(allowed) else 0,
                             calibration_nominal_rate=np.mean(calp<=.05)))
            for rep in range(len(p)):
                individual.append(dict(**identity,phase='honest_evaluation',rep=rep,cp=0,
                                       p=p[rep],adjusted_p=adjusted[rep],winner=int(obs['winners'][rep,method_id]),
                                       maximum=obs['maxima'][rep,method_id]))
        occupancy.append(dict(setting=c['name'],visited_states_mean=obs['diagnostics'][:,0].mean(),
                              median_state_occupancy_mean=obs['diagnostics'][:,2].mean(),
                              fraction_rounds_in_states_with_less_than_5=obs['diagnostics'][:,3].mean(),
                              state_change_rate=obs['diagnostics'][:,4].mean()))
        if c['name'] in config['native_settings']:
            data=load(config,'native',c,8)
            for method_id,method in enumerate(METHODS[:2]):
                d=dict(setting=c['name'],method=method,**rate(data['pvalues'][:,method_id]<=.05))
                d['inflation_p']=binomtest(d['count'],d['total'],.05,alternative='greater').pvalue
                native.append(d)
        if c['name'] in config['power_settings']:
            for cp_id,cp in enumerate(config['coordination_probabilities']):
                data=load(config,'power',c,2+cp_id)
                for method_id,method in enumerate(METHODS):
                    raw=data['pvalues'][:,method_id];adj=recalibrate(raw,cal['pvalues'][:,method_id])
                    w=data['winners'][:,method_id]
                    for calibration,p in [('nominal',raw),('empirical',adj)]:
                        reject=p<=.05
                        outcomes=dict(detection=reject,exact_recovery=reject&(w==7),
                                      any_innocent=reject&((w&~7)!=0))
                        for metric,events in outcomes.items():
                            power.append(dict(setting=c['name'],rounds=c['rounds'],cp=cp,method=method,
                                              calibration=calibration,metric=metric,**rate(events)))
                    for rep in range(len(raw)):
                        individual.append(dict(setting=c['name'],method=method,rounds=c['rounds'],n=c['n'],role=c['role'],
                                               phase='power',rep=rep,cp=cp,p=raw[rep],adjusted_p=adj[rep],
                                               winner=int(w[rep]),maximum=data['maxima'][rep,method_id]))
    frames=dict(nominal_false_alarm=pd.DataFrame(nominal),empirical_false_alarm=pd.DataFrame(calibrated),
                source_style_false_alarm=pd.DataFrame(native),power=pd.DataFrame(power),
                occupancy=pd.DataFrame(occupancy),thresholds=pd.DataFrame(cuts),per_game=pd.DataFrame(individual))
    for key in ['nominal_false_alarm','source_style_false_alarm']:
        frames[key]['inflation_holm_p']=holm(frames[key].inflation_p)
    for name,df in frames.items():df.to_csv(ROOT/f'results/{name}.csv',index=False)
    nominal=frames['nominal_false_alarm'];matched=frames['empirical_false_alarm'];power=frames['power'];native=frames['source_style_false_alarm']

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(2,3,figsize=(13,8.5))
    fig.subplots_adjust(top=.85,bottom=.075,hspace=.38,wspace=.32)
    for ax,regime,title in zip(axs.flat,['iid','exogenous','leader','margin','sticky','sticky_strong'],
                              ['IID uniform','External public signal','Leader feedback','Margin feedback','Sticky state (hold 0.9)','Stress: hold 0.99, reaction 0.95']):
        names=[f'{regime}_r{r}' for r in [100,300,1000]]
        for method,color in zip(METHODS,COLORS):
            d=nominal[(nominal.setting.isin(names))&(nominal.method==method)].sort_values('rounds')
            y=d.rate.to_numpy()*100
            ax.errorbar(d['rounds'],y,yerr=np.array([d.rate-d.ci_low,d.ci_high-d.rate])*100,
                        label=LABELS[method],color=color,marker='o',capsize=3)
        ax.axhline(5,color='black',linestyle=':',linewidth=1)
        ax.set(title=title,xlabel='Rounds',ylabel='False alarms (%)',xscale='log')
        ax.set_xticks([100,300,1000],['100','300','1,000']);ax.set_ylim(bottom=0)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.suptitle('Nominal 5% tests: 1,000 independent honest evaluation games per point',y=.99)
    fig.savefig(ROOT/'results/false_alarm.png',bbox_inches='tight');fig.savefig(ROOT/'results/false_alarm.pdf',bbox_inches='tight');plt.close(fig)

    fig,axs=plt.subplots(2,4,figsize=(14,7.8))
    fig.subplots_adjust(top=.83,bottom=.08,hspace=.42,wspace=.35)
    for row,r in enumerate([300,1000]):
        for col,regime in enumerate(['iid','leader','margin','sticky']):
            ax=axs[row,col]
            for method,color in zip(METHODS,COLORS):
                d=power[(power.setting==f'{regime}_r{r}')&(power.method==method)&(power.calibration=='empirical')&(power.metric=='detection')].sort_values('cp')
                ax.errorbar(d.cp,d.rate*100,yerr=np.array([d.rate-d.ci_low,d.ci_high-d.rate])*100,
                            marker='o',color=color,label=LABELS[method],capsize=3)
            ax.set(title=f'{regime.capitalize()}, {r:,} rounds',xlabel='Coordination probability',ylabel='Detected (%)',ylim=(-2,102))
            ax.set_xticks([.15,.3,.5]);ax.grid(alpha=.15)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.suptitle('Power after independent honest calibration: 500 games per point',y=.99)
    fig.savefig(ROOT/'results/power.png',bbox_inches='tight');fig.savefig(ROOT/'results/power.pdf',bbox_inches='tight');plt.close(fig)

    selected=[f'{name}_r300' for name in ['iid','exogenous','leader','margin','sticky','sticky_strong']]
    selected1000=[f'{name}_r1000' for name in ['iid','exogenous','leader','margin','sticky','sticky_strong']]
    highlights=nominal[nominal.inflation_holm_p<.05]
    nativehigh=native[native.inflation_holm_p<.05]
    nt=native.copy();nt['result']=nt.apply(lambda d:f"{100*d['rate']:.1f}% ({int(d['count'])}/{int(d['total'])}; {100*d.ci_low:.1f}–{100*d.ci_high:.1f}%)",axis=1)
    psel=power[(power.cp==.3)&(power.rounds==300)&(power.calibration=='empirical')].copy()
    psel['result']=psel.apply(lambda d:f"{100*d['rate']:.1f}% ({int(d['count'])}/{int(d['total'])})",axis=1)
    ptab=psel.pivot(index=['setting','method'],columns='metric',values='result').reset_index()
    result=f'''# Null-calibration experiment results

Generated {datetime.now(timezone.utc).isoformat()}. All experiments and new files are under `{ROOT}`. The supplied repository is unchanged.

## Scope and denominators

{len(config['settings'])} settings, each with 2,000 independent games for normalization, 1,000 independent honest calibration games, and 1,000 separate honest evaluation games. Each method generates 999 new reference games per trajectory. The power study has {len(config['power_settings'])*len(config['coordination_probabilities'])} cells, each with 500 coordinated games. The source-style sensitivity uses {len(config['native_settings'])} settings with 1,000 additional honest games each. There are no LLM runs.

The primary statistic scans all 57 subsets of size at least two among six agents. The ten-agent diagnostic scans 1,013 subsets. Counts below refer to the first detection in an entirely honest population. They do not establish strong family-wise error control under partial alternatives or correct membership after iterative removal.

The main comparison holds a single, independently fitted honest-simulation normalization fixed across null methods. The source-style sensitivity below restores per-trajectory, method-specific normalization. All use rank p-values and consistent float64 arithmetic, with conservative tie handling. Full definitions and frozen decisions are in [PROTOCOL.md](../PROTOCOL.md) and [config.json](../config.json).

## Nominal false alarms at 300 rounds

Each cell gives percent, count/1,000, and a Wilson 95% interval. The requested nominal rate is 5%.

{compact_table(nominal,selected)}

## Nominal false alarms at 1,000 rounds

{compact_table(nominal,selected1000)}

![All primary and stress false-alarm rates](false_alarm.png)

## Diagnostics for missing state and population size

{compact_table(nominal,[c['name'] for c in config['settings'] if c['role'] in ['state_diagnostic','size_diagnostic']])}

Full public counts are sufficient for the primary policies. Top-only state is sufficient for the leader policy, but omits the activation margin used by the margin policy. No-history states omit the common public driver. Fresh simulation remains calibrated against its complete, correctly specified honest process even when the statistic uses an insufficient state; this is a model-based benchmark and not a generic conditional-independence permutation guarantee.

The fixed normalization also uses known honest games for the two surrogate methods. It can absorb the mean and scale of honest unconditioned dependence and materially change the omitted-state results. Those rows are therefore not direct replications of the original unconditioned detector. Use the source-style sensitivity below to assess that detector's behavior. This distinction is one reason both normalizations were prespecified.

## Source-style A/B normalization sensitivity

Each trajectory receives its own independent A and B banks (999 games each), using that method's surrogate generator. These are new honest evaluation games. Only the two surrogate methods are shown; the fixed-score oracle argument does not automatically apply to a score fitted from the observed trajectory.

{table(nt,[('setting','Setting'),('method','Method'),('result','False alarms, 95% interval')])}

## Independent empirical recalibration

For each setting and method, compare a new raw p-value with the 1,000 independent honest calibration p-values. `adjusted_p = (1 + count(calibration_p <= new_p)) / 1001`. Count ties conservatively and reject at 0.05. This calibrates the complete procedure under the same known honest process; it is not available automatically for unknown LLM agents. The evaluation games were never used to select these cutoffs. Because p-values are discrete, actual rates can be below 5%.

Achieved false-alarm rates at 300 rounds:

{compact_table(matched,selected)}

## Planted coordination power

Agents 0,1,2 share a uniform draw with probability 0.15, 0.30, or 0.50 per round. Each applies its honest conditional action CDF, preserving conditional marginals. The full action/history process evolves from the resulting actions. This is planted dependence, not evidence of economic collusion or intent.

![Power after independent honest calibration](power.png)

At 300 rounds and coordination probability 0.30, with empirical calibration:

{table(ptab,[('setting','Setting'),('method','Method'),('detection','Detected'),('exact_recovery','Exact group'),('any_innocent','Any innocent member')])}

All power/membership denominators are 500 games, including games with no rejection. A rejection's argmax can contain honest bystanders; testing the all-honest null does not guarantee exact group recovery. Full nominal and recalibrated power, membership and intervals are in [power.csv](power.csv).

## Statistical interpretation and limitations

One-sided binomial tests of inflation above 5%, Holm-adjusted over all {len(nominal)} main setting/method rows, flag {len(highlights)} rows. These are descriptive multiple-comparison diagnostics, not a replacement for the intervals or model assumptions. The source-style family has {len(native)} rows and {len(nativehigh)} adjusted flags. See the CSV `inflation_holm_p` columns. All per-rate Wilson intervals are pointwise and conditional on each fixed independent normalization/calibration bank. They do not assert exact size equality; failure to reject inflation does not prove universal validity. Power comparisons must be read alongside each method's actual honest evaluation size.

The fresh-simulation rank test is valid when the observed game and fresh reference games follow the same specified honest process, because they are exchangeable conditional on the independent normalization bank. This does not require shuffled histories to be physically valid. Conversely, physically impossible shuffled histories alone do not prove test-size inflation. These experiments test particular honest processes with known policies and fully observed compact public state. They do not test unobserved private memory, fitted/misspecified LLM policies, nonstationary policies absent from state, auction data, or arbitrary strategic adaptation.

## Verification and reproduction

The custom engine passed {len(json.loads((ROOT/'results/validation.json').read_text())['checks'])} reference/invariant checks. The frozen supplied core tests passed (15 tests). The custom checks include exact conditional MI against the supplied implementation; subset scores against source and float64 brute force; state transitions; conditional margins under shuffling; valid circular offsets; conditional policy probabilities; preserved coordinated marginal policies; rank/tie handling; and strong planted-group recovery.

Run `bash ../code/reproduce.sh` from this directory (or invoke that file by absolute path). It rebuilds the engine, checks validation, resumes missing deterministic chunks and regenerates analysis. It does not overwrite the frozen configuration. Dependencies: Python 3.11, NumPy, SciPy, pandas, matplotlib, markdown-it-py, a C++17 compiler. The wrapper uses the existing Andromeda environment. The final [integrity audit](audit.json) verifies all expected chunks, reconciles all reported counts, checks frozen hashes and seed uniqueness, and exactly replays a saved test from each study phase.

Artifacts: [full nominal false alarms](nominal_false_alarm.csv), [recalibrated false alarms](empirical_false_alarm.csv), [source-style sensitivity](source_style_false_alarm.csv), [all per-game results](per_game.csv), [state occupancies](occupancy.csv), [empirical thresholds](thresholds.csv). Compact trajectories, p-values, winners, seeds and surrogate maximum quantiles are in `chunks/`; normalization trajectories and statistics are in `banks/`. Reference draws can be reconstructed from saved seeds. The original source and SHA-256 manifest are one level above.
'''
    interpretation=ROOT/'results/INTERPRETATION.md'
    if interpretation.exists():
        result=result.replace('# Null-calibration experiment results\n',
                              '# Null-calibration experiment results\n\n'+interpretation.read_text()+'\n',1)
    (ROOT/'results/REPORT.md').write_text(result)
    # Standalone HTML embeds the plots and all tables without a CDN or web service.
    body=MarkdownIt('commonmark').enable('table').render(result)
    for name in ['false_alarm','power']:
        body=body.replace('src="'+name+'.png"','src="data:image/png;base64,'+
                          base64.b64encode((ROOT/f'results/{name}.png').read_bytes()).decode()+'"')
    tables=[]
    for title,df in [('Nominal false alarms',nominal),('Recalibrated false alarms',matched),('Source-style sensitivity',native),('Power',power)]:
        tables.append(f'<details><summary>{html.escape(title)}: full data</summary>'+df.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)+'</details>')
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Null calibration study</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18242b}img{width:100%}code{font-size:14px;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}h2{margin-top:40px}summary{cursor:pointer;font-weight:600;margin:20px 0}a{color:#176b92}</style>'+body+''.join(tables)
    (ROOT/'results/report.html').write_text(page)
    overview=dict(settings=len(config['settings']),normalization_games=len(config['settings'])*config['normalization_games'],
                  honest_calibration_games=len(config['settings'])*config['calibration_games'],
                  honest_evaluation_games=len(config['settings'])*config['evaluation_games'],
                  power_games=len(config['power_settings'])*len(config['coordination_probabilities'])*config['power_games'],
                  source_style_games=len(config['native_settings'])*config['native_games'],
                  complete_chunks=len(expected),main_inflation_flags=highlights.to_dict('records'),
                  source_style_inflation_flags=nativehigh.to_dict('records'))
    (ROOT/'results/overview.json').write_text(json.dumps(overview,indent=2)+'\n')
    print(json.dumps(overview,indent=2))
    print(compact_table(nominal,selected1000))

if __name__=='__main__':main()
