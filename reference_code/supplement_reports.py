"""Post-process existing formal logs; never launches or changes solver runs.

Numerical conclusions require the same complete groups/comparisons as the main
statistics. Missing measurements remain missing. The input summary hash binds
this supplement to a particular statistics snapshot.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_results import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED, finite, mean
from experiment_io import atomic_json, sha256, timestamp

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'reports'/'supplement'
FRACTIONS=(.1,.25,.5,.75,1.)
LEARNING={'AC','DAC-Partial','DAC-Pre','DAC-Credit','Adapt'}


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,json.JSONDecodeError):return default


def csv_rows(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in row.items()} for row in rows)
    os.replace(temp,path)


def moment():return {'n':0.,'sum':0.,'sum_squares':0.,'min':None,'max':None,'nonfinite':0,'gt1':0}


def add_scalar(acc,value,weight=1):
    value=finite(value)
    if value is None:
        acc['nonfinite']+=1
        return
    acc['n']+=weight;acc['sum']+=weight*value;acc['sum_squares']+=weight*value*value
    acc['min']=value if acc['min'] is None else min(acc['min'],value)
    acc['max']=value if acc['max'] is None else max(acc['max'],value)
    acc['gt1']+=int(value>1)*weight


def add_distribution(acc,distribution,size):
    avg=finite(distribution.get('mean'));std=finite(distribution.get('std'))
    if avg is None or std is None:
        acc['nonfinite']+=1;return
    acc['n']+=size;acc['sum']+=size*avg;acc['sum_squares']+=size*(std*std+avg*avg)
    low=finite(distribution.get('min'));high=finite(distribution.get('max'))
    if low is not None:acc['min']=low if acc['min'] is None else min(acc['min'],low)
    if high is not None:acc['max']=high if acc['max'] is None else max(acc['max'],high)


def merged_moments(values):
    acc=moment()
    for item in values:
        for key in ('n','sum','sum_squares','nonfinite','gt1'):acc[key]+=item[key]
        if item['min'] is not None:acc['min']=item['min'] if acc['min'] is None else min(acc['min'],item['min'])
        if item['max'] is not None:acc['max']=item['max'] if acc['max'] is None else max(acc['max'],item['max'])
    n=acc['n'];avg=acc['sum']/n if n else None
    return {'count':n,'mean':avg,'std':math.sqrt(max(0,acc['sum_squares']/n-avg*avg)) if n else None,
            'rms':math.sqrt(max(0,acc['sum_squares']/n)) if n else None,'min':acc['min'],'max':acc['max'],
            'nonfinite_recorded':acc['nonfinite'],'greater_than_one_fraction':acc['gt1']/n if n else None}


def anytime_points(result):
    budget=finite(result.get('budget_value',result.get('config',{}).get('budget_value')))
    points=result.get('anytime') or []
    out=[]
    for fraction in FRACTIONS:
        exact=next((p for p in points if finite(p.get('fraction'))==fraction),None)
        if exact is not None:
            objective=finite(exact.get('objective'));cutoff=finite(exact.get('budget'))
        else:
            cutoff=fraction*budget if budget is not None else None
            eligible=[p for p in points if finite(p.get('elapsed_s',p.get('time'))) is not None
                      and cutoff is not None and float(p.get('elapsed_s',p.get('time')))<=cutoff]
            point=max(eligible,key=lambda p:float(p.get('elapsed_s',p.get('time')))) if eligible else None
            objective=finite(point.get('best_obj',point.get('objective'))) if point else None
        out.append({'fraction':fraction,'cutoff_seconds':cutoff,'objective_recorded':objective})
    return out


def summarize_run(row,cache):
    token=row['source_sha256']
    cached=cache.get(row['run_id'])
    if cached and cached.get('source_sha256')==token:return cached
    path=Path(row['source_file'])
    if not path.exists() or sha256(path)!=token:
        return {'run_id':row['run_id'],'input_valid':False,'error':'source file differs from main snapshot'}
    result=read(path,{})
    group={k:row[k] for k in ('experiment','problem','group','method','instance','run_id')}
    out={**group,'seed':int(row['seed']),'source_sha256':token,'input_valid':True,
         'diagnostic_log_sha256':result.get('diagnostic_log_sha256'),
         'anytime':anytime_points(result),'expected_updates':int(result.get('updates',0) or 0),
         'iterations':int(result.get('iterations',0) or 0),
         'untrained_rollout_records':result.get('untrained_rollout_records'),
         'action_counts':{k:result.get(k) for k in ('destroy_counts','repair_counts','pair_counts','mean_action_entropy')},
         'compatibility':{k:result.get(k) for k in ('upstream_commit','compatibility_changes','operator_library',
                          'restart_policy','distance_metric','distance_scale','checkpoint_sha256','checkpoint_training_customers')},
         'diagnostics_status':'not_recorded','recorded_updates':0,'batch_sizes':{},'singleton_batches':0,
         'tail_batches':0,'terminal_transitions':0,'scalar_moments':{},'distribution_moments':{},'boundary_counts':{}}
    logpath=result.get('diagnostic_log')
    if logpath:
        logpath=BASE/logpath
        if logpath.exists() and (not result.get('diagnostic_log_sha256') or sha256(logpath)==result['diagnostic_log_sha256']):
            log=read(logpath,{})
            updates=log.get('update_diagnostics')
            if isinstance(updates,list):
                out['recorded_updates']=len(updates)
                out['diagnostics_status']='complete' if len(updates)==out['expected_updates'] else 'count_mismatch'
                sizes=Counter()
                for update in updates:
                    size=int(update.get('batch_size',0));sizes[size]+=1
                    out['singleton_batches']+=int(size==1)
                    out['tail_batches']+=int(update.get('tail_batch',False))
                    out['terminal_transitions']+=int(update.get('terminal_count',0))
                    for key,value in update.items():
                        if isinstance(value,dict) and {'mean','std'}.issubset(value):
                            acc=out['distribution_moments'].setdefault(key,moment());add_distribution(acc,value,size)
                            boundary=finite(value.get('clip_fraction'))
                            if boundary is not None:
                                counts=out['boundary_counts'].setdefault(key,{'samples':0,'at_or_above_abs_one':0.,'eligible_multisample_samples':0,'multisample_at_or_above_abs_one':0.})
                                counts['samples']+=size;counts['at_or_above_abs_one']+=size*boundary
                                if size>1:
                                    counts['eligible_multisample_samples']+=size;counts['multisample_at_or_above_abs_one']+=size*boundary
                        elif key not in {'iteration','time','batch_size','terminal_count','tail_batch'} and value is not None:
                            acc=out['scalar_moments'].setdefault(key,moment());add_scalar(acc,value)
                out['batch_sizes']={str(k):v for k,v in sorted(sizes.items())}
        else:out['diagnostics_status']='missing_or_changed_log'
    cache[row['run_id']]=out
    return out


def diagnostic_groups(main,runs):
    output=[]
    for group in main.get('groups',[]):
        if group['experiment'] not in {'E1','E2','E3','E5'} or group['method'] not in LEARNING:continue
        chosen=[r for r in runs if all(r[k]==group[k] for k in ('experiment','problem','group','method'))]
        available=[r for r in chosen if r['diagnostics_status']=='complete']
        complete=group['status']=='complete' and len(available)==group['expected_runs']
        row={k:group[k] for k in ('experiment','problem','group','method')}
        row.update(status='complete' if complete else 'partial',expected_runs=group['expected_runs'],
                   valid_runs=len(chosen),runs_with_complete_diagnostics=len(available),
                   recorded_updates=sum(r['recorded_updates'] for r in available),metrics=None)
        if complete:
            keys=sorted({k for r in available for k in r['scalar_moments']})
            distributions=sorted({k for r in available for k in r['distribution_moments']})
            batch=Counter()
            for r in available:batch.update(r['batch_sizes'])
            boundaries={}
            for key in sorted({k for r in available for k in r['boundary_counts']}):
                counts={name:sum(r['boundary_counts'].get(key,{}).get(name,0) for r in available)
                        for name in ('samples','at_or_above_abs_one','eligible_multisample_samples','multisample_at_or_above_abs_one')}
                counts['multisample_boundary_fraction']=counts['multisample_at_or_above_abs_one']/counts['eligible_multisample_samples'] if counts['eligible_multisample_samples'] else None
                boundaries[key]=counts
            actions={}
            for key in ('destroy_counts','repair_counts','pair_counts'):
                values=[np.asarray(r['action_counts'][key]) for r in available if r['action_counts'].get(key) is not None]
                if len(values)==len(available):
                    total=np.sum(values,axis=0);denominator=total.sum()
                    actions[key]=total.tolist();actions[key.replace('counts','frequency')]=(total/denominator).tolist() if denominator else None
            row['metrics']={'batch_size_counts':dict(batch),'singleton_batches':sum(r['singleton_batches'] for r in available),
                'tail_batches':sum(r['tail_batches'] for r in available),
                'untrained_rollout_records':sum(r['untrained_rollout_records'] or 0 for r in available),
                'scalars':{k:merged_moments([r['scalar_moments'][k] for r in available if k in r['scalar_moments']]) for k in keys},
                'distributions':{k:merged_moments([r['distribution_moments'][k] for r in available if k in r['distribution_moments']]) for k in distributions},
                'advantage_boundaries':boundaries,'empirical_actions':actions,
                'raw_advantage_meaning':'destroy=V(partial)-V(full), repair=target-V(partial)' if group['method']=='DAC-Credit' else 'destroy=target-V(full); repair=target-V(partial), when present'}
        output.append(row)
    return output


def anytime_groups(main,runs):
    output=[]
    for group in main.get('groups',[]):
        if group['experiment'] not in {'E1','E3'}:continue
        chosen=[r for r in runs if all(r[k]==group[k] for k in ('experiment','problem','group','method'))]
        complete=group['status']=='complete' and len(chosen)==group['expected_runs']
        for fraction in FRACTIONS:
            values=[];by_instance=defaultdict(list);cutoffs=[]
            for run in chosen:
                point=next(p for p in run['anytime'] if p['fraction']==fraction)
                value=point['objective_recorded']
                if value is not None:values.append(value);by_instance[run['instance']].append(value)
                if point['cutoff_seconds'] is not None:cutoffs.append(point['cutoff_seconds'])
            row={k:group[k] for k in ('experiment','problem','group','method')}
            row.update(fraction=fraction,status='complete' if complete and len(values)==group['expected_runs'] else 'partial',
                       expected_runs=group['expected_runs'],available_points=len(values),
                       expected_instances=group['expected_instances'],objective_mean=None,
                       cutoff_seconds_min=min(cutoffs,default=None),cutoff_seconds_max=max(cutoffs,default=None))
            if row['status']=='complete':row['objective_mean']=mean([mean(v) for v in by_instance.values()])
            output.append(row)
    return output


def plot_anytime(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    for experiment,problem,group in sorted({(r['experiment'],r['problem'],r['group']) for r in rows}):
        expected={'Frozen','Adapt'} if experiment=='E3' else {'ALNS','AC','DAC-Pre','DAC-Partial'}|({'PPO-adapted'} if group=='CVRP-S' else set())
        selected=[r for r in rows if (r['experiment'],r['problem'],r['group'])==(experiment,problem,group)]
        if not all(any(r['method']==m and r['fraction']==1. and r['status']=='complete' for r in selected) for m in expected):continue
        fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
        for method in sorted(expected):
            series=sorted([r for r in selected if r['method']==method],key=lambda r:r['fraction'])
            ax.plot([r['fraction']*100 for r in series],[r['objective_mean'] if r['status']=='complete' else np.nan for r in series],marker='o',markersize=3,label=method)
        ax.set(xlabel='Wall-clock budget used (%)',ylabel='Mean incumbent distance (solver record)',title=f'{experiment} | {problem} | {group}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        stem=f'anytime_{experiment}_{problem}_{group}'
        for extension in ('png','pdf'):fig.savefig(out/(stem+'.'+extension),dpi=200)
        plt.close(fig)
        figures.append({'experiment':experiment,'problem':problem,'group':group,'png':str(out/(stem+'.png')),'pdf':str(out/(stem+'.pdf'))})
    return figures


def training_costs():
    rows=[]
    source_instances=read(BASE/'configs'/'source_training_instances.json',[])
    for directory in sorted((BASE/'models').iterdir()):
        if not directory.is_dir() or 'pilot' in directory.name.lower() or directory.name=='archive':continue
        protocol=read(directory/'protocol.json',{})
        config=read(directory/'training_config.json',{})
        if not protocol and not config:continue
        completed=read(directory/'completed.json',{})
        progress=read(directory/'progress.json',{})
        actual=completed or progress
        checkpoint=completed.get('checkpoint_path')
        verified=bool(checkpoint and Path(checkpoint).exists() and sha256(checkpoint)==completed.get('checkpoint_sha256'))
        status='complete' if completed.get('status') in {'complete','completed'} and verified else 'running' if progress else 'not_started'
        job=protocol.get('job',{})
        planned_transitions=config.get('timesteps')
        if job.get('mode')=='light':planned_transitions=protocol.get('epochs',2)*protocol.get('instances',0)*1000
        elif job.get('mode')=='source':planned_transitions=protocol.get('epochs',2)*sum(r['original_steps_per_epoch'] for r in source_instances if r['problem']==job.get('problem'))
        rows.append({'job_id':directory.name,'status':status,'mode':job.get('mode','official_external'),
                     'problem':job.get('problem','VRPTW' if config else None),'method':job.get('method','PPO-ALNS-original' if config else None),
                     'training_seed':job.get('training_seed',config.get('seed')),
                     'source_instances':protocol.get('instances',completed.get('training_instance_count')),
                     'epochs':protocol.get('epochs'),'completed_episodes':actual.get('completed_episodes'),
                     'planned_episodes':actual.get('total_episodes'),'completed_transitions':actual.get('total_steps',actual.get('timesteps')),
                     'planned_transitions':planned_transitions,
                     'elapsed_training_seconds':finite(actual.get('elapsed_training_seconds',actual.get('elapsed_s'))),
                     'measured_cpu_seconds':None,'checkpoint_sha256':completed.get('checkpoint_sha256') if verified else None,
                     'checkpoint_selection':protocol.get('checkpoint_selection',config.get('selection_rule')),
                     'source_protocol_path':str(directory/('protocol.json' if protocol else 'training_config.json')),
                     'status_path':str(directory/('completed.json' if completed else 'progress.json')),
                     'cost_scope':'Recorded source-training elapsed wall time; running rows are cumulative-to-date, not final; process CPU time was not measured.'})
    rows.append({'job_id':'DR_ALNS_published_weights','status':'published_model_training_cost_unknown','mode':'official_external',
                 'problem':'CVRP','method':'DR-ALNS-original','elapsed_training_seconds':None,'measured_cpu_seconds':None,
                 'cost_scope':'Author-provided checkpoints; historical author training compute is unavailable in the local archive and is not zero.'})
    return rows


def external_descriptions(instance_rows,main,runs):
    output=[]
    metadata=read(BASE/'configs'/'instances.json',[])
    lookup={(r['experiment'],r['problem'],r['instance'],r['method']):r for r in instance_rows}
    for problem,baseline in [('CVRP','DR-ALNS-original'),('VRPTW','PPO-ALNS-original')]:
        ids=sorted(r['id'] for r in metadata if r['problem']==problem and r['panel'])
        a=[];b=[]
        for inst in ids:
            left=lookup.get(('E4',problem,inst,baseline));right=lookup.get(('E4',problem,inst,'DAC-Partial'))
            if left and right and left['status']=='complete' and right['status']=='complete' and left['planned_seeds']==right['planned_seeds']:
                a.append(float(left['objective_mean']));b.append(float(right['objective_mean']))
        row={'experiment':'E4','problem':problem,'baseline':baseline,'target':'DAC-Partial','expected_pairs':len(ids),
             'complete_pairs':len(a),'status':'complete' if ids and len(a)==len(ids) else 'partial','statistics':None}
        if row['status']=='complete':
            aa=np.asarray(a);bb=np.asarray(b);dif=aa-bb;rel=100*dif/aa
            rng=np.random.default_rng(BOOTSTRAP_SEED)
            draws=np.mean(rel[rng.integers(0,len(ids),(BOOTSTRAP_REPEATS,len(ids)))],axis=1)
            low,high=np.percentile(draws,[2.5,97.5])
            row['statistics']={'baseline_mean':float(aa.mean()),'target_mean':float(bb.mean()),
                'mean_relative_improvement_pct':float(rel.mean()),'ci95_low':float(low),'ci95_high':float(high),
                'wins':int((dif>0).sum()),'ties':int((dif==0).sum()),'losses':int((dif<0).sum()),
                'inference_scope':'descriptive paired interval only; no added significance test or unregistered p value'}
        observed=[r['compatibility'] for r in runs if r['experiment']=='E4' and r['problem']==problem and r['method']==baseline]
        row['observed_compatibility']={key:sorted({json.dumps(r[key],ensure_ascii=False,sort_keys=True) for r in observed if r.get(key) is not None})
                                      for key in ('upstream_commit','compatibility_changes','operator_library','restart_policy','checkpoint_sha256','checkpoint_training_customers')}
        output.append(row)
    return output


def parameter_provenance():
    rows=[]
    for filename,problem,method in [('DAC_CVRP_online.py','CVRP','DAC'),('DAC_VRPTW_online.py','VRPTW','DAC'),
                                    ('AC_CVRP.py','CVRP','AC'),('AC_VRPTW.py','VRPTW','AC')]:
        path=BASE/'original_source'/filename;source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'DACAgent','SingleACAgent'})
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        for node in ast.walk(init):
            if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Attribute):continue
            name=node.targets[0].attr
            if name=='min_action_prob':
                rows.append({'problem':problem,'method':method,'parameter':name,'value':ast.literal_eval(node.value),
                             'source':str(path),'line':node.lineno,'source_sha256':sha256(path),'selection_origin':'inherited archived setting; historical grid provenance unavailable'})
            elif name.startswith('opt_') and isinstance(node.value,ast.Call):
                for keyword in node.value.keywords:
                    if keyword.arg in {'lr','weight_decay'}:
                        rows.append({'problem':problem,'method':method,'parameter':name+'.'+keyword.arg,'value':ast.literal_eval(keyword.value),
                                     'source':str(path),'line':node.lineno,'source_sha256':sha256(path),'selection_origin':'inherited archived setting; historical grid provenance unavailable'})
    engine=BASE/'src'/'controlled_engine.py'
    for parameter,value in [('encoder_and_hidden_dim',64),('gamma_repair',.99),('critic_coef',1.),('k_nn',5),
                            ('ALNS_reaction_factor',.1),('temperature_start',50.),('temperature_end_ratio',.001),('gradient_norm_clip',1.)]:
        rows.append({'problem':'CVRP/VRPTW','method':'core controlled methods where applicable','parameter':parameter,'value':value,
                     'source':str(engine),'line':None,'source_sha256':sha256(engine),'selection_origin':'explicit frozen controlled protocol; no new performance tuning on benchmark'})
    frozen=read(BASE/'configs'/'frozen_protocol.json',{})
    return rows,{'historical_grid_recovered':False,'new_hyperparameter_search_performed':False,
                 'independent_validation_generated_but_not_used_for_selection':True,
                 'frozen_protocol_path':str(BASE/'configs'/'frozen_protocol.json'),
                 'frozen_protocol_sha256':sha256(BASE/'configs'/'frozen_protocol.json') if frozen else None,
                 'frozen_at':frozen.get('created_at'),'hyperparameters_statement':frozen.get('hyperparameters'),
                 'historical_caveat':frozen.get('historical_grid_caveat'),
                 'claim_limit':'New runs evaluate a pre-fixed historical-parameter protocol. This does not establish that historical benchmark-informed parameter selection never occurred.'}


def generate(main_path,out):
    out.mkdir(parents=True,exist_ok=True)
    original_bytes=main_path.read_bytes();input_hash=hashlib.sha256(original_bytes).hexdigest();main=json.loads(original_bytes)
    per_run=csv_rows(main_path.parent/'per_run.csv')
    selected=[r for r in per_run if r['analysis_eligible']=='True' and r['planned']=='True']
    consistent=len(selected)==main['valid_runs']
    cache_document=read(out/'run_cache.json',{})
    code_hash=sha256(Path(__file__))
    cache=cache_document.get('runs',{}) if cache_document.get('aggregation_source_sha256')==code_hash else {}
    summaries=[summarize_run(row,cache) for row in selected]
    invalid=[r for r in summaries if not r['input_valid']]
    runs=[r for r in summaries if r['input_valid']]
    consistent=consistent and not invalid and sha256(main_path)==input_hash
    diagnostics=diagnostic_groups(main,runs)
    anytime=anytime_groups(main,runs)
    figures=plot_anytime(anytime,out) if consistent else []
    training=training_costs()
    external=external_descriptions(csv_rows(main_path.parent/'per_instance.csv'),main,runs)
    parameters,provenance=parameter_provenance()
    notes={'diagnostics_unit':'Pooled recorded updates or transitions, descriptive only; not independent samples for significance tests.',
           'raw_advantage_rms':'Reconstructed from logged within-batch population mean/std and batch size. E1 Partial/Pre raw advantages equal the corresponding shared-target TD residual; E5 destroy raw advantage is V(partial)-V(full).',
           'clipping_measurement':'clip_fraction records abs(value)>=1. For multi-sample normalized advantages this is boundary occupancy, not the exact count of values changed by clipping. Singleton batches follow the uncentered path and are reported separately.',
           'grad_norm_measurement':'clip_grad_norm_ return value is the pre-clipping norm. Fraction >1 is a threshold-exceedance measure.',
           'not_recorded':['per-transition minimum policy probability','individual TD samples/absolute TD errors','exact pre-vs-post advantage clipping change count','process CPU time','historical grid-search instance list'],
           'anytime_measurement':'Solver-recorded incumbent distances at fixed budget fractions. Historical incumbent routes are not saved for independent float64 revalidation; final endpoint main statistics instead use independent validation.distance. Do not call the full curve independently revalidated.',
           'missing_anytime':'No incumbent available by a checkpoint remains missing; no survivor-only mean is computed.',
           'training_measurement':'Recorded actual elapsed wall time; running rows are cumulative costs to date, pilots excluded, author checkpoint training cost unknown.',
           'e4_measurement':'Preselected 20 CVRP / 15 VRPTW instances; paired descriptive bootstrap intervals, no extra significance tests.'}
    summary={'schema_version':1,'updated_at':timestamp(),'input_summary_sha256':input_hash,
             'input_consistent':consistent,'input_main_status':main['status'],'input_valid_runs':main['valid_runs'],
             'status':'complete' if consistent and main['status']=='complete' else 'partial',
             'invalid_sources':invalid,'parameter_provenance':provenance,'parameter_table':parameters,
             'diagnostics':diagnostics,'anytime':anytime,'anytime_figures':figures,'training_costs':training,
             'external_comparisons':external,'external_source_audit':read(BASE/'reports'/'external_audit.json',{}).get('official_repositories',[]),
             'measurement_notes':notes}
    if not consistent:
        summary['consistency_note']='Main summary/CSV snapshot changed or source hash mismatch; updater must not use numerical supplement results.'
    write_csv(out/'parameters_and_provenance.csv',parameters)
    write_csv(out/'diagnostic_groups.csv',diagnostics)
    write_csv(out/'anytime_groups.csv',anytime)
    write_csv(out/'training_costs.csv',training)
    write_csv(out/'external_paired_descriptive.csv',external)
    summary['aggregation_source_sha256']=code_hash
    summary['derived_payload_sha256']=hashlib.sha256(json.dumps(summary,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest()
    atomic_json(out/'run_cache.json',{'aggregation_source_sha256':code_hash,'runs':cache})
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'summary_manifest.json',{'input_summary_sha256':input_hash,'summary_sha256':sha256(out/'summary.json'),
                'derived_payload_sha256':summary['derived_payload_sha256'],'aggregation_source_sha256':code_hash,'updated_at':summary['updated_at']})
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-summary',type=Path,default=BASE/'reports'/'statistics'/'summary.json')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/'supplement.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        import psutil
        previous=read(lock,{})
        pid=previous.get('pid')
        if pid and not psutil.pid_exists(pid):
            lock.unlink(missing_ok=True)
            try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError:
                print(json.dumps({'status':'busy','lock':str(lock)}));return
        else:
            print(json.dumps({'status':'busy','lock':str(lock),'pid':pid}));return
    try:
        os.write(fd,json.dumps({'pid':os.getpid(),'started_at':timestamp()}).encode('utf-8'))
        os.close(fd);summary=generate(args.main_summary,args.output)
        print(json.dumps({k:summary[k] for k in ('status','input_consistent','input_valid_runs','input_summary_sha256')},ensure_ascii=False))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
