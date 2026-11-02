"""Manifest-based summaries of independently validated revision experiments.

No inferential test is calculated until every planned run in its comparison is
present, status=ok and independently valid. Statistical units are instance means
over the planned seeds, never individual runs. Partial tables are descriptive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
FORMAL = {'E1','E2','E3','E4','E5'}
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_REPEATS = 2000


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    numbers = [v for v in values if v is not None]
    return float(np.mean(numbers)) if numbers else None


def sd(values):
    numbers = [v for v in values if v is not None]
    return float(np.std(numbers, ddof=1)) if len(numbers)>1 else None


def config_of(record):
    cfg = dict(record.get('config') or {})
    for key in ('experiment','problem','method','seed','instance','instance_id','instance_path','group','run_id',
                'budget_mode','budget_value','checkpoint_path','checkpoint_sha256',
                'initialization','data_sha256','distance_convention','fleet_objective',
                'protocol_version','device','rollout_size','log_stride','record_parameter_hash'):
        if key not in cfg and key in record:
            cfg[key] = record[key]
    cfg['instance'] = cfg.get('instance') or cfg.get('instance_id') or Path(cfg.get('instance_path','')).stem
    return cfg


def key_of(cfg):
    return (cfg.get('experiment',''), cfg.get('problem',''), cfg.get('instance',''),
            cfg.get('method',''), int(cfg.get('seed',0)))


def read_plan(path, instance_metadata):
    if not path.exists():
        return {}, ['planned_runs manifest missing; zero completed runs does not mean complete']
    content = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(content, dict):
        content = content.get('runs',content.get('tasks',content.get('planned_runs',[])))
    planned, issues = {}, []
    for entry in content:
        cfg = config_of(entry)
        if cfg.get('experiment') not in FORMAL:
            continue
        key = key_of(cfg)
        meta = instance_metadata.get((cfg.get('problem'),cfg['instance']),{})
        cfg.setdefault('group',meta.get('group','unknown'))
        if key in planned:
            issues.append(f'duplicate planned key: {key}')
        planned[key] = cfg
    if not planned:
        issues.append('no formal planned runs found; completeness is undefined')
    return planned, issues


def read_results(directory, planned, instance_metadata):
    records, errors = defaultdict(list), []
    for path in sorted(directory.rglob('*.json')):
        try:
            raw = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,json.JSONDecodeError) as exc:
            errors.append({'path':str(path),'error':f'{type(exc).__name__}: {exc}'})
            continue
        if not isinstance(raw,dict):
            continue
        cfg = config_of(raw)
        if cfg.get('experiment') not in FORMAL:
            continue
        if not any(name in raw for name in ('status','validation','routes','best_objective')):
            continue
        key = key_of(cfg)
        meta = instance_metadata.get((cfg.get('problem'),cfg['instance']),{})
        validation = raw.get('validation') or {}
        objective = finite(validation.get('distance'))
        verified = validation.get('valid') is True
        valid = raw.get('status') == 'ok' and verified and objective is not None
        errors_in_run = validation.get('errors', [])
        row = {'experiment':key[0], 'problem':key[1], 'instance':key[2], 'method':key[3], 'seed':key[4],
               'group':cfg.get('group',raw.get('group',meta.get('group','unknown'))),
               'run_id':cfg.get('run_id',raw.get('run_id',path.stem)),
               'status':raw.get('status','unknown'), 'validation_valid':verified,
               'analysis_eligible':valid, 'planned':key in planned, 'duplicate':False,
               'objective':objective, 'solver_objective':finite(raw.get('best_objective')),
               'n_routes':finite(validation.get('n_routes',raw.get('route_count'))),
               'elapsed_seconds':finite(raw.get('elapsed_seconds')),
               'overshoot_seconds':finite(raw.get('overshoot_seconds')),
               'iterations':finite(raw.get('iterations')), 'updates':finite(raw.get('updates')),
               'budget_mode':cfg.get('budget_mode',raw.get('budget_mode')),
               'budget_value':finite(raw.get('budget_value',cfg.get('budget_value',cfg.get('budget_seconds',cfg.get('num_steps'))))),
               'checkpoint_sha256':raw.get('checkpoint_sha256',cfg.get('checkpoint_sha256','')),
               'validation_errors':json.dumps(errors_in_run,ensure_ascii=False),
               'failure_reason':str(raw.get('error',raw.get('failure_reason',''))),
               'source_file':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        records[key].append((row,raw))
    for rows in records.values():
        if len(rows)>1:
            for row,_ in rows:
                row['duplicate'] = True
                row['analysis_eligible'] = False
    return records, errors


def aggregate(planned, records):
    by_instance = defaultdict(list)
    for key,cfg in planned.items():
        by_instance[(key[0],key[1],cfg['group'],key[2],key[3])].append(key)
    instances = []
    for key,run_keys in sorted(by_instance.items()):
        rows = [records[rk][0][0] for rk in run_keys if len(records.get(rk,[]))==1 and records[rk][0][0]['analysis_eligible']]
        all_present = [row for rk in run_keys for row,_ in records.get(rk,[])]
        complete = len(rows)==len(run_keys)
        instances.append({'experiment':key[0],'problem':key[1],'group':key[2],'instance':key[3],'method':key[4],
                          'expected_runs':len(run_keys),'valid_runs':len(rows),
                          'failed_or_unverified_files':sum(not r['analysis_eligible'] for r in all_present),
                          'missing_runs':sum(rk not in records for rk in run_keys),
                          'status':'complete' if complete else 'partial',
                          'planned_seeds':','.join(str(rk[-1]) for rk in sorted(run_keys)),
                          'objective_mean':mean([r['objective'] for r in rows]),
                          'objective_sd':sd([r['objective'] for r in rows]),
                          'objective_best':min((r['objective'] for r in rows),default=None),
                          'n_routes_mean':mean([r['n_routes'] for r in rows]),
                          'elapsed_mean':mean([r['elapsed_seconds'] for r in rows]),
                          'iterations_mean':mean([r['iterations'] for r in rows]),
                          'updates_mean':mean([r['updates'] for r in rows])})
    groups = []
    grouped = defaultdict(list)
    for row in instances:
        grouped[(row['experiment'],row['problem'],row['group'],row['method'])].append(row)
    for key,items in sorted(grouped.items()):
        complete = [row for row in items if row['status']=='complete']
        groups.append({'experiment':key[0],'problem':key[1],'group':key[2],'method':key[3],
                       'status':'complete' if len(complete)==len(items) else 'partial',
                       'expected_instances':len(items),'complete_instances':len(complete),
                       'expected_runs':sum(r['expected_runs'] for r in items),
                       'valid_runs':sum(r['valid_runs'] for r in items),
                       'objective_mean_complete_instances':mean([r['objective_mean'] for r in complete]),
                       'objective_sd_between_instances':sd([r['objective_mean'] for r in complete]),
                       'n_routes_mean_complete_instances':mean([r['n_routes_mean'] for r in complete]),
                       'elapsed_mean_complete_instances':mean([r['elapsed_mean'] for r in complete]),
                       'iterations_mean_complete_instances':mean([r['iterations_mean'] for r in complete]),
                       'updates_mean_complete_instances':mean([r['updates_mean'] for r in complete])})
    return instances, groups


def paired_test(baseline, target):
    baseline, target = np.asarray(baseline,dtype=float),np.asarray(target,dtype=float)
    differences = baseline-target
    relative = 100*differences/baseline
    nonzero = differences[differences!=0.0]
    if len(nonzero):
        result = stats.wilcoxon(differences,zero_method='wilcox',correction=False,
                                alternative='two-sided',method='auto')
        pvalue, statistic = float(result.pvalue), float(result.statistic)
        ranks = stats.rankdata(np.abs(nonzero),method='average')
        rank_biserial = float(np.sum(ranks*np.sign(nonzero))/np.sum(ranks))
    else:
        pvalue, statistic, rank_biserial = 1.0, 0.0, 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.mean(relative[rng.integers(0,len(relative),(BOOTSTRAP_REPEATS,len(relative)))],axis=1)
    lower,upper = np.percentile(bootstrap,[2.5,97.5])
    return {'p_raw':pvalue,'wilcoxon_statistic':statistic,
            'baseline_mean':float(np.mean(baseline)),'target_mean':float(np.mean(target)),
            'mean_raw_improvement':float(np.mean(differences)),
            'mean_relative_improvement_pct':float(np.mean(relative)),
            'median_relative_improvement_pct':float(np.median(relative)),
            'improvement_ci95_low':float(lower),'improvement_ci95_high':float(upper),
            'matched_rank_biserial':rank_biserial,
            'wins':int(np.sum(differences>0)),'ties':int(np.sum(differences==0)),'losses':int(np.sum(differences<0)),
            'nonzero_pairs':len(nonzero)}


def comparisons(instances, metadata):
    lookup = {(r['experiment'],r['problem'],r['instance'],r['method']):r for r in instances}
    specs = []
    def add(family,experiment,baseline,target,problem,group=None,panel=False,baseline_experiment=None):
        specs.append(dict(family=family,experiment=experiment,baseline=baseline,target=target,
                          problem=problem,group=group,panel=panel,baseline_experiment=baseline_experiment or experiment))
    for problem in ('CVRP','VRPTW'):
        add('E1_primary', 'E1','DAC-Pre','DAC-Partial',problem)
        for baseline in ('ALNS','AC'):
            add('E1_practical','E1',baseline,'DAC-Partial',problem)
        add('E2_mechanism_diagnostic','E2','DAC-Pre','DAC-Partial',problem,panel=True)
        add('E3_matched_checkpoint','E3','Frozen','Adapt',problem)
        add('E5_credit_diagnostic','E5','DAC-Partial','DAC-Credit',problem,panel=True,baseline_experiment='E1')
    for group in ('CVRP-S','CVRP-M1','CVRP-M2','CVRP-L','C','R','RC'):
        add('E1_groups_secondary','E1','DAC-Pre','DAC-Partial','CVRP' if group.startswith('CVRP') else 'VRPTW',group)
    output = []
    for spec in specs:
        selected = [m for m in metadata.values() if m['problem']==spec['problem']
                    and (spec['group'] is None or m['group']==spec['group'])
                    and (not spec['panel'] or m['panel'])]
        left,right,missing = [],[],[]
        for meta in sorted(selected,key=lambda r:r['id']):
            a=lookup.get((spec['baseline_experiment'],spec['problem'],meta['id'],spec['baseline']))
            b=lookup.get((spec['experiment'],spec['problem'],meta['id'],spec['target']))
            if not a or not b or a['status']!='complete' or b['status']!='complete' or a['planned_seeds']!=b['planned_seeds']:
                missing.append(meta['id'])
                continue
            left.append(a['objective_mean']); right.append(b['objective_mean'])
        row={**spec,'expected_pairs':len(selected),'complete_pairs':len(left),
             'status':'complete' if selected and not missing else 'partial',
             'missing_instances':','.join(missing),'p_raw':None,'p_holm':None}
        if row['status']=='complete':
            row.update(paired_test(left,right))
        output.append(row)
    families=defaultdict(list)
    for row in output:
        families[row['family']].append(row)
    for family,rows in families.items():
        complete=all(r['status']=='complete' for r in rows)
        for row in rows:
            row['family_complete']=complete
            row['family_hypotheses']=len(rows)
        if complete:
            running=0.0
            for index,row in enumerate(sorted(rows,key=lambda r:r['p_raw'])):
                running=max(running,(len(rows)-index)*row['p_raw'])
                row['p_holm']=min(1.0,running)
    return output


def friedman_tests(instances, metadata):
    lookup={(r['experiment'],r['problem'],r['instance'],r['method']):r for r in instances}
    methods=['ALNS','AC','DAC-Pre','DAC-Partial']
    output=[]
    for problem in ('CVRP','VRPTW'):
        ids=sorted(m['id'] for m in metadata.values() if m['problem']==problem)
        blocks=[]
        for inst in ids:
            rows=[lookup.get(('E1',problem,inst,method)) for method in methods]
            if all(r and r['status']=='complete' for r in rows):
                blocks.append([r['objective_mean'] for r in rows])
        row={'experiment':'E1','problem':problem,'methods':','.join(methods),
             'expected_blocks':len(ids),'complete_blocks':len(blocks),
             'status':'complete' if ids and len(blocks)==len(ids) else 'partial',
             'statistic':None,'p_raw':None,'interpretation':'secondary omnibus test; paired families are the registered inferential endpoints'}
        if row['status']=='complete':
            matrix=np.asarray(blocks)
            if np.all(matrix==matrix[:,:1]):
                row.update(statistic=0.0,p_raw=1.0)
            else:
                result=stats.friedmanchisquare(*matrix.T)
                row.update(statistic=finite(result.statistic),p_raw=finite(result.pvalue))
        output.append(row)
    return output


def protocol_groups(planned):
    groups=defaultdict(list)
    for key,cfg in planned.items():
        groups[(key[0],key[1],cfg['group'],key[3])].append(cfg)
    return [{'experiment':key[0],'problem':key[1],'group':key[2],'method':key[3],
             'seeds':sorted({int(r['seed']) for r in rows}),
             'budget_modes':sorted({str(r.get('budget_mode','unknown')) for r in rows}),
             'budget_values':sorted({float(r['budget_value']) for r in rows if r.get('budget_value') is not None}),
             'checkpoint_paths':sorted({str(r['checkpoint_path']) for r in rows if r.get('checkpoint_path')}),
             'expected_runs':len(rows)} for key,rows in sorted(groups.items())]


def time_profiles(groups, records):
    output=[]
    for group in groups:
        matching=[(row,raw) for items in records.values() for row,raw in items
                  if row['analysis_eligible'] and row['planned'] and
                  all(row[k]==group[k] for k in ('experiment','problem','group','method'))]
        with_times=[(row,raw['time_breakdown_seconds']) for row,raw in matching
                    if isinstance(raw.get('time_breakdown_seconds'),dict)]
        parts=sorted({part for _,times in with_times for part in times})
        complete=group['status']=='complete' and len(with_times)==group['expected_runs']
        profile={k:group[k] for k in ('experiment','problem','group','method')}
        profile.update(status='complete' if complete else 'partial',expected_runs=group['expected_runs'],
                       runs_with_profile=len(with_times),components={})
        if complete:
            for part in parts:
                values=[finite(times.get(part)) for _,times in with_times]
                shares=[100*float(times[part])/row['elapsed_seconds'] for row,times in with_times
                        if finite(times.get(part)) is not None and row['elapsed_seconds'] and row['elapsed_seconds']>0]
                profile['components'][part]={'seconds_mean':mean(values),'share_pct_mean':mean(shares)}
        output.append(profile)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=BASE/'results')
    parser.add_argument('--planned',type=Path,default=BASE/'configs'/'planned_runs.json')
    parser.add_argument('--output',type=Path,default=BASE/'reports'/'statistics')
    args=parser.parse_args()
    instance_rows=json.loads((BASE/'configs'/'instances.json').read_text(encoding='utf-8-sig'))
    metadata={(r['problem'],r['id']):r for r in instance_rows}
    planned,plan_issues=read_plan(args.planned,metadata)
    records,read_errors=read_results(args.results,planned,metadata)
    instances,groups=aggregate(planned,records)
    paired=comparisons(instances,metadata)
    friedman=friedman_tests(instances,metadata)
    profiles=time_profiles(groups,records)
    runs=[row for items in records.values() for row,_ in items]
    failures=[r for r in runs if not r['analysis_eligible']]
    missing=[{'experiment':key[0],'problem':key[1],'instance':key[2],'method':key[3],'seed':key[4],
              'run_id':cfg.get('run_id','')} for key,cfg in planned.items() if key not in records]
    stages=[]
    for stage in sorted(FORMAL):
        keys=[key for key in planned if key[0]==stage]
        valid=sum(len(records.get(key,[]))==1 and records[key][0][0]['analysis_eligible'] for key in keys)
        stages.append({'experiment':stage,'expected_runs':len(keys),'valid_runs':valid,
                       'missing_runs':sum(key not in records for key in keys),
                       'failed_or_duplicate_keys':sum(key in records and (len(records[key])!=1 or not records[key][0][0]['analysis_eligible']) for key in keys),
                       'status':'complete' if keys and valid==len(keys) else 'partial' if keys else 'not_planned'})
    write_csv(args.output/'per_run.csv',runs,['experiment','problem','instance','method','seed','group','run_id','status','validation_valid','analysis_eligible','planned','duplicate','objective','solver_objective','n_routes','elapsed_seconds','overshoot_seconds','iterations','updates','budget_mode','budget_value','checkpoint_sha256','validation_errors','failure_reason','source_file','source_sha256'])
    write_csv(args.output/'per_instance.csv',instances)
    write_csv(args.output/'per_group.csv',groups)
    write_csv(args.output/'paired_comparisons.csv',paired)
    write_csv(args.output/'friedman.csv',friedman)
    write_csv(args.output/'failures.csv',failures)
    write_csv(args.output/'missing_runs.csv',missing)
    valid_total=sum(s['valid_runs'] for s in stages)
    summary={'schema_version':1,'status':'complete' if planned and valid_total==len(planned) and not plan_issues else 'partial',
             'expected_runs':len(planned),'valid_runs':valid_total,'observed_files':len(runs),
             'failed_or_unverified_files':len(failures),'missing_runs':len(missing),
             'unexpected_files':sum(not r['planned'] for r in runs),
             'stages':stages,'groups':groups,'paired_comparisons':paired,'friedman':friedman,
             'protocol_groups':protocol_groups(planned),'time_profiles':profiles,
             'plan_issues':plan_issues,'read_errors':read_errors,
             'statistical_protocol':{'unit':'instance mean over all planned paired seeds',
                 'objective':'independent validation.distance, raw float64 Euclidean',
                 'positive_improvement':'100*(baseline-target)/baseline',
                 'wilcoxon':'two-sided on raw instance mean differences, zero_method=wilcox, correction=False, method=auto, exact floating zero differences removed',
                 'holm':'separate registered families; adjusted p withheld until all hypotheses in that family are complete',
                 'bootstrap':'paired instance percentile bootstrap of mean relative improvement',
                 'bootstrap_repeats':BOOTSTRAP_REPEATS,'bootstrap_seed':BOOTSTRAP_SEED,
                 'ties':'exact equality of float64 instance means',
                 'scipy_version':scipy.__version__,'numpy_version':np.__version__,
                 'incomplete_policy':'descriptive aggregates only; no paired p/CI/effect until all runs in the relevant comparison are complete',
                 'checkpoint_scope':'conditional on the single checkpoint used; solve seeds are not source-training replicates'}}
    dump(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ['status','expected_runs','valid_runs','missing_runs','failed_or_unverified_files']},ensure_ascii=False))


if __name__=='__main__':
    main()
