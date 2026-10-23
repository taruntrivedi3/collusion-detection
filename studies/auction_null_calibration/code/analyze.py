import base64
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
from core import ROOT,STATES,METHODS,MASKS
from run import jobs,dest,load_config

LABELS=['No state','Time thirds','Five-round blocks','Exact round']
MLABELS=['Within-state shuffle','Circular shift','Empirical path reference']
COLORS=['#146e96','#d57814','#258454']

def rate(x):
    k=int(np.count_nonzero(x));n=len(x);p=k/n;z=1.959963984540054;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(count=k,total=n,rate=p,ci_low=min(p,max(0,mid-delta)),ci_high=max(p,min(1,mid+delta)))

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def recalibrate(p,reference):return (1+np.searchsorted(np.sort(reference),p,side='right'))/(len(reference)+1)

def calibrate_score(score,reference):
    score=np.asarray(score);reference=np.asarray(reference)
    cut=score-1e-10*np.maximum(1,np.abs(score))
    return (1+len(reference)-np.searchsorted(np.sort(reference),cut,side='left'))/(len(reference)+1)

def markdown_table(df,cols):
    out=['| '+' | '.join(label for key,label in cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
    for _,row in df.iterrows():out.append('| '+' | '.join(str(row[key]) for key,label in cols)+' |')
    return '\n'.join(out)

def result_cell(d):return f"{100*d['rate']:.1f}% ({int(d['count'])}/{int(d['total'])}; {100*d.ci_low:.1f}–{100*d.ci_high:.1f}%)"

def benchmark_table(frame,norm):
    out=[]
    for state,label in zip(STATES,LABELS):
        row={'state':label}
        for method in METHODS[:3 if norm=='common' else 2]:
            d=frame[(frame.state==state)&(frame.normalization==norm)&(frame.method==method)].iloc[0]
            row[method]=result_cell(d)
        out.append(row)
    return markdown_table(pd.DataFrame(out),[('state','State')]+list(zip(METHODS[:3 if norm=='common' else 2],MLABELS)))

def main():
    config=load_config();records=[];actual=[]
    for phase in ['benchmark','transfer']:
        for job in jobs(config,phase):
            with np.load(dest(job)) as f:
                donors=f['donors'];duplicates=np.array([5-len(set(row)) for row in donors])
                for norm,pkey,mkey,wkey,methods in [('common','p','maximum','winner',METHODS),('source_style','native_p','native_maximum','native_winner',METHODS[:2])]:
                    p=f[pkey];m=f[mkey];w=f[wkey]
                    for i,rep in enumerate(f['rep']):
                        for k,method in enumerate(methods):
                            records.append(dict(phase=phase,state=job['state'],cohort=job['cohort'],fold=job['fold'],normalization=norm,
                                                method=method,rep=int(rep),p=p[i,k],maximum=m[i,k],winner=int(w[i,k]),duplicate_donors=int(duplicates[i])))
    games=pd.DataFrame(records);null=[];power=[];calibrated=[];score_calibrated=[];transfer=[];floor=[];nodup=[]
    for (state,norm,method),g in games[games.phase=='benchmark'].groupby(['state','normalization','method']):
        cal=g[g.cohort==0];test=g[g.cohort==1];alt=g[g.cohort==2]
        ident=dict(state=state,normalization=norm,method=method)
        d=dict(**ident,**rate(test.p<=.05));d['inflation_p']=binomtest(d['count'],d['total'],.05,alternative='greater').pvalue;null.append(d)
        calibrated.append(dict(**ident,**rate(recalibrate(test.p,cal.p)<=.05)))
        score_calibrated.append(dict(**ident,**rate(calibrate_score(test.maximum,cal.maximum)<=.05)))
        nd=test[test.duplicate_donors==0];nodup.append(dict(**ident,**rate(nd.p<=.05)))
        floor.append(dict(**ident,calibration_at_p_floor=int((cal.p==.001).sum()),
                          calibration_size=len(cal),test_distinct_donor_groups=len(nd)))
        for label,p in [('nominal',alt.p.to_numpy()),('empirical_p',recalibrate(alt.p,cal.p)),
                        ('empirical_score',calibrate_score(alt.maximum,cal.maximum))]:
            reject=p<=.05
            for metric,event in [('detection',reject),('exact_recovery',reject&(alt.winner.to_numpy()==7)),
                                 ('any_bystander',reject&((alt.winner.to_numpy()&~7)!=0))]:
                power.append(dict(**ident,calibration=label,metric=metric,**rate(event)))
    null=pd.DataFrame(null);null['inflation_holm_p']=holm(null.inflation_p)
    for (state,norm,method,fold),g in games[games.phase=='transfer'].groupby(['state','normalization','method','fold']):
        transfer.append(dict(state=state,normalization=norm,method=method,fold=fold,**rate(g.p<=.05)))
    transfer=pd.DataFrame(transfer);matched=[]
    for state in STATES:
        for fold in range(1,6):
            f=np.load(ROOT/f'results/matched_pool/{state}_fold{fold}.npz')
            matched.append(dict(state=state,fold=fold,**rate(f['p']<=.05)))
    matched=pd.DataFrame(matched);score_transfer=[]
    for state in STATES:
        for fold in range(1,6):
            for size in [40,10]:
                calibration=np.load(ROOT/f'results/score_transfer_cal/{state}_f{fold}_n{size}.npz')['maximum']
                for m,method in enumerate(METHODS[:2]):
                    test=games[(games.phase=='transfer')&(games.state==state)&(games.fold==fold)&
                               (games.normalization=='source_style')&(games.method==method)]
                    score_transfer.append(dict(state=state,fold=fold,training_paths=size,method=method,
                                               **rate(calibrate_score(test.maximum,calibration[:,m])<=.05)))
    score_transfer=pd.DataFrame(score_transfer)
    for job in jobs(config,'real'):
        obj=json.loads(dest(job).read_text())
        for d in obj['rows']:actual.append(dict(run=obj['run'],condition=obj['condition'],state=job['state'],**d))
    actual=pd.DataFrame(actual);actual['base_reference_rank']=np.nan;actual['reference_markets']=0
    matched_conditions=['base','base-seller_comms','boss_pressure','oversight','boss_pressure-oversight']
    for (state,method),g in actual.groupby(['state','method']):
        reference=g[g.condition=='base']
        for index,row in g[g.condition.isin(matched_conditions)].iterrows():
            other=reference[reference.run!=row.run]
            actual.at[index,'base_reference_rank']=(1+np.count_nonzero(other.maximum>=row.maximum-1e-10))/(len(other)+1)
            actual.at[index,'reference_markets']=len(other)
    actual_summary=[]
    for (state,method,condition),g in actual.groupby(['state','method','condition']):
        actual_summary.append(dict(state=state,method=method,condition=condition,**rate(g.p<=.05),
                                   median_maximum=g.maximum.median(),median_valid_candidates=g.valid_candidates.median(),
                                   median_base_rank=g.base_reference_rank.median()))
    actual_summary=pd.DataFrame(actual_summary)
    state_audit=pd.DataFrame(json.loads((ROOT/'results/state_audit.json').read_text()))
    occupancy=state_audit.groupby('state').agg(markets=('run','size'),median_states=('states','median'),
                    mean_singleton_fraction=('singleton_fraction','mean'),mean_small_state_fraction=('small_state_fraction','mean'),
                    mean_constant_agent_view_fraction=('constant_agent_views',lambda x:float(x.mean()/10))).reset_index()
    frames=dict(benchmark_false_alarm=null,benchmark_calibrated_false_alarm=pd.DataFrame(calibrated),
                benchmark_score_calibrated_false_alarm=pd.DataFrame(score_calibrated),copy_power=pd.DataFrame(power),
                transfer_false_alarm=transfer,matched_pool_transfer=matched,source_score_transfer=score_transfer,
                real_flags=actual,real_condition_summary=actual_summary,
                state_occupancy=occupancy,p_saturation=pd.DataFrame(floor),distinct_donor_diagnostic=pd.DataFrame(nodup),per_game=games)
    for key,frame in frames.items():frame.to_csv(ROOT/f'results/{key}.csv',index=False)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(1,2,figsize=(12,5.5));fig.subplots_adjust(top=.77,bottom=.2,wspace=.25)
    for ax,norm,title in zip(axs,['common','source_style'],['Common independent normalization','Original A/B normalization']):
        for m,method in enumerate(METHODS[:3 if norm=='common' else 2]):
            d=null[(null.normalization==norm)&(null.method==method)].set_index('state').loc[STATES]
            y=d.rate.to_numpy()*100;xx=np.arange(4)+(m-1)*.08
            ax.errorbar(xx,y,yerr=np.array([d.rate-d.ci_low,d.ci_high-d.rate])*100,marker='o',color=COLORS[m],label=MLABELS[m],capsize=3)
        ax.axhline(5,color='black',ls=':',lw=1);ax.set_xticks(range(4),LABELS,rotation=15);ax.set(title=title,ylabel='Flagged no-interaction groups (%)',ylim=(-3,103))
    fig.suptitle('Independent recorded LLM paths: 1,000 evaluation groups per point',y=.98)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False)
    fig.savefig(ROOT/'results/benchmark.png',bbox_inches='tight');fig.savefig(ROOT/'results/benchmark.pdf',bbox_inches='tight');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(top=.8,bottom=.2,wspace=.25)
    p=frames['copy_power']
    for ax,norm,title in zip(axs,['common','source_style'],['Common normalization: raw flags','Original A/B: raw flags']):
        for m,method in enumerate(METHODS[:3 if norm=='common' else 2]):
            d=p[(p.normalization==norm)&(p.method==method)&(p.metric=='detection')&(p.calibration=='nominal')].set_index('state').loc[STATES]
            ax.plot(range(4),d.rate*100,marker='o',color=COLORS[m],label=MLABELS[m])
        ax.set_xticks(range(4),LABELS,rotation=15);ax.set(title=title,ylabel='Copied triples flagged (%)',ylim=(-3,103))
    fig.suptitle('Strong copied-path test — read alongside the false-alarm rates',y=.98)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.savefig(ROOT/'results/copy_power.png',bbox_inches='tight');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.7));fig.subplots_adjust(top=.84,bottom=.18,wspace=.3)
    for ax,frame,title in zip(axs,[transfer[(transfer.normalization=='common')&(transfer.method=='trajectory_reference')],matched],
                              ['40 training paths → 10 held-out paths','10 training paths → 10 held-out paths']):
        for i,state in enumerate(STATES[:3]):
            d=frame[frame.state==state].sort_values('fold');ax.plot(d.fold,d.rate*100,marker='o',label=LABELS[i])
        ax.axhline(5,color='black',ls=':',lw=1);ax.set(title=title,xlabel='Held-out fold',ylabel='Flagged independent groups (%)',ylim=(-2,102),xticks=range(1,6));ax.legend(frameon=False)
    fig.suptitle('Empirical path reference: transfer to different source markets',y=.98)
    fig.savefig(ROOT/'results/transfer.png',bbox_inches='tight');fig.savefig(ROOT/'results/transfer.pdf',bbox_inches='tight');plt.close(fig)

    real_rows=[]
    for state in config['real_states']:
        row={'state':state}
        for cond,key in [('base','base'),('base-seller_comms','comms')]:
            for method,label in zip(METHODS[:2],['shuffle','shift']):
                d=actual_summary[(actual_summary.state==state)&(actual_summary.condition==cond)&(actual_summary.method==method)].iloc[0]
                row[f'{key}_{label}']=f"{int(d['count'])}/{int(d['total'])}"
        real_rows.append(row)
    transfer_rows=[]
    score_transfer_rows=[]
    for state,label in zip(STATES,LABELS):
        a=transfer[(transfer.state==state)&(transfer.normalization=='common')&(transfer.method=='trajectory_reference')]
        b=matched[matched.state==state]
        transfer_rows.append(dict(state=label,primary=f'{100*a.rate.min():.1f}–{100*a.rate.max():.1f}%',matched=f'{100*b.rate.min():.1f}–{100*b.rate.max():.1f}%'))
        row=dict(state=label)
        for size in [40,10]:
            for method in METHODS[:2]:
                d=score_transfer[(score_transfer.state==state)&(score_transfer.training_paths==size)&(score_transfer.method==method)]
                row[f'{method}_{size}']=f'{100*d.rate.min():.1f}–{100*d.rate.max():.1f}%'
        score_transfer_rows.append(row)
    text=f'''# LLM double-auction calibration study

Generated {datetime.now(timezone.utc).isoformat()}. All files live in `{ROOT}`. No LLM API calls or new auction rollouts were made.

## Data and scope

The pinned published double-auction repository contains 75 actual LLM markets, each with five sellers, five buyers and 30 rounds. We verified all 1,119 upstream files against their Git blobs, parsed 22,500 prompt-response pairs, reconciled 22,393 numeric quotes, checked 4,110 current-order trade prices plus one carried-order case, and reconciled all 300 economic totals with zero issues. The source includes GPT-4.1 and Claude compositions. Only ten markets are matched GPT-only no-message controls. The no-message label is not certified noncollusion.

The statistic is the existing subset scan on two bundled action views: price band and quote direction. One maximum covers 52 candidates (26 subsets per view). Round one is excluded, leaving 29 observations. Every test uses 999 B draws and, for source-style scans, an independent 999-draw A bank. No group growth, peeling, IPF or new feature tuning was used. [Frozen protocol](../PROTOCOL.md), [configuration](../config.json), [source provenance](../source_integrity.json).

## What the independently resampled paths show

For this benchmark, independently select a complete recorded seller path for each of five artificial agents from the 50 paths in the ten base markets. Every direction/level pair and its entire temporal order remain intact. Conditional on this empirical pool, complete paths are independent across agents. The artificial groups never interacted, but the source agents reacted to their own original markets. This preserves trends, persistence and quantized actions; it does not reconstruct valid joint auction histories.

The target here is detecting interaction/dependence between those generated agents. Pooling times can create apparent dependence from common nonstationary schedules even when complete random paths are independent. Thus inflation here demonstrates that the surrogate is not a calibrated no-interaction comparison for this process. It is not a counterexample to a permutation theorem with satisfied exchangeability assumptions, nor a measured live-market collusion false-positive rate.

Each rate below uses 1,000 independent evaluation draws conditional on the frozen pool. Intervals are pointwise Wilson 95% intervals for this resampling process, not uncertainty over a population of real LLM markets. A separate 1,000 generated groups per state were reserved for optional empirical calibration.

### Original per-trajectory A/B normalization

{benchmark_table(null,'source_style')}

### Common independent normalization and the empirical reference

The common A bank is fit on 2,000 independent path groups shuffled within the chosen state. This keeps the scoring transformation fixed across null generators without normalizing away the temporal pattern under investigation. Fresh draws from the same empirical path distribution are an exact reference for this benchmark, conditional on its independent normalization. They are not a known honest simulator for real LLM markets.

{benchmark_table(null,'common')}

![No-interaction benchmark](benchmark.png)

All 20 state/normalization/method rows belong to one Holm-adjusted inflation-diagnostic family; {int((null.inflation_holm_p<.05).sum())} exceed 5% significantly after that correction. The exact-round rows are degenerate by design. [Full benchmark table](benchmark_false_alarm.csv). [Distinct-donor diagnostic](distinct_donor_diagnostic.csv) reports the subset with no repeated donor identity, so accidental copying can be inspected separately; it is a descriptive conditional diagnostic.

## Actual logged LLM markets

These are surrogate flags from the source-style scan, not known false alarms or verified collusion. All cells have ten markets. The richer state is an observational diagnostic, not sufficient full history.

{markdown_table(pd.DataFrame(real_rows),[('state','State'),('base_shuffle','No message: shuffle'),('base_shift','No message: shift'),('comms_shuffle','Messages: shuffle'),('comms_shift','Messages: shift')])}

The full set of 75 markets and all ten experimental conditions is in [real_condition_summary.csv](real_condition_summary.csv). The existing LLM evaluator and profit/price summaries were not used to choose features, state bins, or thresholds. Comparisons across model compositions also involve observation timing: some agents see different books within the same round. Flags do not establish economic collusion, agreements, or intent.

Ranking a treatment market against the ten complete matched base markets gives a minimum nonrandomized reference rank of **1/11 = 9.09%**. Leave-one-out base ranks have minimum 1/10. This dataset therefore cannot supply a conventional 5% empirical market-level cutoff. Thousands of synthetic path draws do not create additional independent real markets.

## State richness and information loss

{markdown_table(occupancy.round(3),[('state','State'),('median_states','Median states / 29 rounds'),('mean_singleton_fraction','Mean fraction in singleton states'),('mean_constant_agent_view_fraction','Mean fraction of agent/views constant within state')])}

Conditioning on exact round produces 29 singleton states. Every within-state empirical MI is zero, including for perfectly copied agents. The finite-sample estimator is degenerate; this does not mean that underlying conditional dependence is absent. Richer representations cannot be assessed by low flag rates alone. The scalar state also cannot capture an LLM's complete private/public context without stronger modeling assumptions.

## Strong copied-path dependence test

Agents 0,1,2 receive the same randomly selected complete donor path; the other two agents receive independent paths. Each individual's full-path marginal distribution is exactly the same as in the no-interaction benchmark. This is deliberately strong planted statistical dependence, not a demonstrated profitable auction agreement. The denominator is 1,000 groups per state; detection, exact triple recovery and inclusion of independent bystanders are saved separately.

![Copied-path flags](copy_power.png)

High raw detection rates from a method with high no-interaction flag rates are not evidence of useful power. [copy_power.csv](copy_power.csv) contains nominal results and results after ranking p-values against the independent calibration groups; [benchmark_calibrated_false_alarm.csv](benchmark_calibrated_false_alarm.csv) gives the achieved evaluation size. When many calibration p-values saturate at the minimum 0.001, conservative p-value recalibration cannot distinguish more extreme alternatives. See [p_saturation.csv](p_saturation.csv). This quantization is distinct from information loss in the action features.

As a post-result diagnostic for that saturation, we also ranked the unrounded maximum statistic directly against the same independent 1,000 calibration groups, for every state and normalization. Its upper-tail rank uses conservative ties. These `empirical_score` rows in the power CSV retain information lost at the Monte Carlo p-value floor; achieved sizes are in [benchmark_score_calibrated_false_alarm.csv](benchmark_score_calibrated_false_alarm.csv). No new test groups or fitted bins were selected. This is recorded in [analysis_additions.json](../analysis_additions.json). It is still calibration to the frozen empirical path distribution, not to live LLM markets. The three common-normalization methods have the identical raw statistic, so their score-calibrated results are identical by construction.

The same-pool empirical reference has a meaningful calibrated comparison but still can miss copied paths with little additional information beyond the common schedule. Exact-round power is zero. A stipulated agreement to quote a constant $99 produces zero MI in these action encodings; dependence scans cannot generally rule out fixed-price policy agreements.

## Transfer of the empirical reference

Five prespecified folds hold out two complete base markets (ten paths) and fit normalization/reference draws on the remaining eight markets (40 paths). Each fold has 500 independent generated evaluation groups. Their donor paths come entirely from the held-out markets; agents within each group are still independently sampled. A new empirical path distribution need not match the training distribution, so the same-pool reference guarantee does not apply.

The unequal donor-pool sizes also change accidental exact-copy probabilities. A post-result control therefore uses two training markets and two disjoint held-out markets, ten paths on each side, keeping those probabilities equal at the donor-identity level. The next fixed fold supplies training markets. This control was added after inspecting transfer failures and is recorded separately in [supplement_config.json](../supplement_config.json); it does not replace the primary experiment.

The ranges below cover the five fold-specific flag rates, with 500 groups per fold. They are not confidence intervals over thousands of actual market runs.

{markdown_table(pd.DataFrame(transfer_rows),[('state','State'),('primary','40 → 10 paths: range'),('matched','10 → 10 paths: range')])}

![Reference transfer](transfer.png)

Equalizing donor counts controls one resampling artifact. Two versus two real markets remains a small and heterogeneous comparison, and even matching counts does not make the empirical path distributions equal. These failures limit this proposed learned reference; they do not establish that every model of honest LLM play must fail.

### Transfer of the repaired source-style raw score

Because source-style raw-score calibration retained signal in the same-pool benchmark, a second exploratory check tests that specific repair rather than assuming it shares the fixed-score transfer results. For every state/fold, generate 1,000 independent calibration groups from the training donor pool, fit the source-style A bank separately for each generated group, and retain its maximum. Rank each of the existing 500 held-out maxima against that training reference. No held-out data are used to choose a cutoff. This adds 40,000 synthetic training groups across two pool sizes and two surrogate-score constructions; see [score_transfer_config.json](../score_transfer_config.json).

Ranges across the five folds:

{markdown_table(pd.DataFrame(score_transfer_rows),[('state','State'),('within_state_40','Shuffle: 40 paths'),('circular_shift_40','Shift: 40 paths'),('within_state_10','Shuffle: 10 paths'),('circular_shift_10','Shift: 10 paths')])}

Full per-fold counts and intervals are in [source_score_transfer.csv](source_score_transfer.csv). This post-result repair is reported alongside the original tests. Calibration can work for a fixed empirical distribution and still fail to transfer; it should not be promoted to a real-market guarantee from these results.

## Reproduction and audit

The accelerated MI and all 52 subset/view scores agree with the supplied Python implementation. Seven focused checks cover scoring, surrogate invariants, donor/clone construction, singleton and constant-action degeneracy, rank/tie handling, and a strong generic copied-triple example. The final [audit](audit.json) verifies frozen hashes, expected outputs, donor disjointness, independent generation/replay, and reported counts. The exact source dataset, donor provenance, raw parsed logs, seeds, per-game scores and reference quantiles remain here.

Use `bash ../code/reproduce.sh` (or that file's absolute path) to resume missing results and rebuild the reports. Dependencies are recorded in [environment.json](environment.json). No upstream auction runner or LLM API client is executed.
'''
    interp=ROOT/'results/INTERPRETATION.md'
    if interp.exists():text=text.replace('# LLM double-auction calibration study\n','# LLM double-auction calibration study\n\n'+interp.read_text()+'\n',1)
    (ROOT/'results/REPORT.md').write_text(text)
    body=MarkdownIt('commonmark').enable('table').render(text)
    for name in ['benchmark','copy_power','transfer']:
        body=body.replace(f'src="{name}.png"','src="data:image/png;base64,'+base64.b64encode((ROOT/f'results/{name}.png').read_bytes()).decode()+'"')
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM auction null calibration</title><style>body{font:16px/1.65 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#172b36}img{width:100%}table{display:block;overflow:auto;border-collapse:collapse;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:right}h2{margin-top:40px}code{overflow-wrap:anywhere;font-size:14px}a{color:#136e97}</style>'+body
    (ROOT/'results/report.html').write_text(page)
    overview=dict(actual_markets=75,actual_markets_with_no_message=10,benchmark_calibration_groups=4000,benchmark_evaluation_groups=4000,
                  cloned_groups=4000,transfer_groups=10000,equal_pool_transfer_groups=10000,
                  repaired_score_training_groups=40000,live_llm_calls=0,
                  benchmark_inflation_flags=null[null.inflation_holm_p<.05].to_dict('records'))
    (ROOT/'results/overview.json').write_text(json.dumps(overview,indent=2)+'\n')
    print(benchmark_table(null,'source_style'));print(benchmark_table(null,'common'))
    print(pd.DataFrame(transfer_rows).to_string(index=False))

if __name__=='__main__':main()
