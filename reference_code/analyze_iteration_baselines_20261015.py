"""Standalone, independently validated ALNS/AC/DAC equal-iteration results."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy import stats
from experiment_io import ROOT, atomic_json, read_json, sha256, timestamp
from validate_routes import validate_routes
from analyze_results import paired_test

BATCH=ROOT/'iteration_baselines_20261015'
OUT=BATCH/'reports'
METHODS=('ALNS','AC','DAC-Partial')
GROUPS=[('CVRP','CVRP-S','CVRP 小型（n<100）',54),('CVRP','CVRP-M1','CVRP 中型1（100≤n<200）',22),
        ('CVRP','CVRP-M2','CVRP 中型2（200≤n<350）',30),('CVRP','CVRP-L','CVRP 大型（350≤n<500）',16),
        ('VRPTW','C','VRPTW C',17),('VRPTW','R','VRPTW R',23),('VRPTW','RC','VRPTW RC',16)]
PAIRS=(('ALNS','AC'),('ALNS','DAC-Partial'),('AC','DAC-Partial'))


def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def average(values):
    return float(np.mean(values)) if values else None


def adjust_holm(rows):
    for family in sorted({r['family'] for r in rows}):
        selected=[r for r in rows if r['family']==family]
        complete=all(r['status']=='complete' and r.get('p_raw') is not None for r in selected)
        for row in selected:
            row.update(family_hypotheses=len(selected),family_complete=complete,p_holm=None)
        if complete:
            running=0.0
            for i,row in enumerate(sorted(selected,key=lambda r:r['p_raw'])):
                running=max(running,(len(selected)-i)*row['p_raw']);row['p_holm']=min(1.0,running)


def analyze(final=False):
    statsdir=OUT/'statistics';statsdir.mkdir(parents=True,exist_ok=True)
    protocol=read_json(BATCH/'configs/frozen_protocol.json')
    problems=[]
    if sha256(__file__)!=protocol['analysis_source_sha256']:problems.append('Frozen analysis source changed')
    if sha256(ROOT/'src/analyze_results.py')!=protocol['statistic_core_sha256']:problems.append('Frozen statistic core changed')
    for name,digest in protocol['manifest_sha256'].items():
        if sha256(BATCH/'configs'/name)!=digest:problems.append('Frozen manifest changed: '+name)
    for name,digest in protocol['scientific_source_sha256'].items():
        if sha256(ROOT/name)!=digest:problems.append('Scientific source changed: '+name)
    if problems:raise RuntimeError('; '.join(problems))
    plan=read_json(BATCH/'configs/planned_runs.json')
    new=read_json(BATCH/'configs/new_runs.json')
    references=read_json(BATCH/'configs/reused_runs.json')
    metadata=read_json(BATCH/'configs/instances.json')
    assert len(plan)==5340 and len(new)==3560 and len(references)==1780
    new_ids={r['run_id'] for r in new};reuse={r['config']['run_id']:r for r in references}
    meta={(r['problem'],r['id']):r for r in metadata}
    assert len(meta)==178 and len({(r['problem'],r['instance_id'],r['method'],r['seed']) for r in plan})==5340
    files={p.stem:p for p in (BATCH/'results/E2').glob('*.json')}
    unexpected=sorted(set(files)-new_ids)
    data_hashes={key:sha256(row['path']) for key,row in meta.items()}
    checkpoints={c['checkpoint_path']:sha256(c['checkpoint_path']) for c in plan if c.get('checkpoint_path')}
    cache_path=statsdir/'validation_cache.json'
    cache=read_json(cache_path).get('records',{}) if cache_path.exists() else {}
    validator_sha=sha256(ROOT/'src/validate_routes.py')
    runs=[];missing=[];failures=[]
    for cfg in plan:
        run_id=cfg['run_id'];reference=reuse.get(run_id)
        path=Path(reference['source_file']) if reference else files.get(run_id)
        if path is None or not path.exists():
            missing.append(dict(run_id=run_id,problem=cfg['problem'],instance=cfg['instance_id'],method=cfg['method'],seed=cfg['seed']))
            continue
        digest=sha256(path);errors=[]
        try:
            raw=read_json(path)
            if raw.get('config')!=cfg:errors.append('configuration_mismatch')
            if raw.get('status')!='ok' or raw.get('validation',{}).get('valid') is not True:errors.append('solver_or_stored_validation_failed')
            if raw.get('iterations')!=cfg['budget_value']:errors.append('iteration_count_mismatch')
            if not isinstance(raw.get('elapsed_seconds'),(int,float)) or not math.isfinite(raw['elapsed_seconds']) or raw['elapsed_seconds']<=0:errors.append('invalid_elapsed_seconds')
            if cfg['budget_mode']!='iterations' or cfg['budget_value']!=max(2000,50*int(meta[(cfg['problem'],cfg['instance_id'])]['n_customers'])):errors.append('invalid_fixed_iteration_budget')
            if data_hashes[(cfg['problem'],cfg['instance_id'])]!=cfg['data_sha256']:errors.append('input_hash_changed')
            if cfg.get('checkpoint_path') and checkpoints[cfg['checkpoint_path']]!=cfg['checkpoint_sha256']:errors.append('checkpoint_hash_changed')
            if reference and reference['source_sha256']!=digest:errors.append('reused_result_changed')
            for name,expected in protocol['scientific_source_sha256'].items():
                if raw.get('source_hashes',{}).get(name)!=expected:errors.append('scientific_source_hash_mismatch:'+name)
            signature=digest+':'+cfg['data_sha256']+':'+validator_sha
            saved=cache.get(str(path))
            if saved and saved['signature']==signature:
                independent=saved['validation']
            else:
                v=validate_routes(cfg['instance_path'],cfg['problem'],{'routes':raw.get('routes',[]),'id_space':raw.get('route_id_space','internal')})
                independent={k:v[k] for k in ('valid','distance','n_routes','errors')}
                cache[str(path)]=dict(signature=signature,validation=independent)
            if independent['valid'] is not True:errors.append('independent_route_validation_failed')
            objective=float(independent['distance'])
            if not math.isfinite(objective) or objective<=0:errors.append('invalid_distance')
            if abs(float(raw.get('best_objective',float('inf')))-objective)>max(1e-3,objective*1e-6):errors.append('reported_distance_mismatch')
            log=raw.get('diagnostic_log')
            if log:
                logpath=Path(log)
                if not logpath.is_absolute():logpath=(ROOT if reference and str(path).startswith(str(ROOT/'results')) else ROOT/'expansion_20261012' if reference else BATCH)/logpath
                if not logpath.exists() or sha256(logpath)!=raw.get('diagnostic_log_sha256'):errors.append('diagnostic_log_hash_mismatch')
            row=dict(run_id=run_id,experiment='E2',problem=cfg['problem'],group=cfg['group'],instance=cfg['instance_id'],
                     method=cfg['method'],seed=cfg['seed'],status=raw.get('status'),analysis_eligible=not errors,
                     distance=objective,n_routes=independent['n_routes'],elapsed_seconds=raw.get('elapsed_seconds'),
                     iterations=raw.get('iterations'),updates=raw.get('updates'),budget_iterations=cfg['budget_value'],
                     checkpoint_sha256=cfg.get('checkpoint_sha256',''),reused=bool(reference),source_file=str(path),
                     source_sha256=digest,validation_errors=';'.join(errors))
        except Exception as exc:
            row=dict(run_id=run_id,experiment='E2',problem=cfg['problem'],group=cfg['group'],instance=cfg['instance_id'],
                     method=cfg['method'],seed=cfg['seed'],status='analysis_error',analysis_eligible=False,reused=bool(reference),
                     source_file=str(path),source_sha256=digest,validation_errors=f'{type(exc).__name__}: {exc}')
        runs.append(row)
        if not row['analysis_eligible']:failures.append(row)
    atomic_json(cache_path,dict(validator_sha256=validator_sha,records=cache))
    by_instance=defaultdict(list)
    for row in runs:by_instance[(row['problem'],row['instance'],row['method'])].append(row)
    instances=[];lookup={}
    for problem,instance in sorted(meta):
        for method in METHODS:
            observed=by_instance.get((problem,instance,method),[])
            valid=[r for r in observed if r['analysis_eligible']]
            complete=len(valid)==10 and sorted(r['seed'] for r in valid)==list(range(0,1000,100))
            row=dict(problem=problem,group=meta[(problem,instance)]['group'],instance=instance,method=method,
                     status='complete' if complete else 'partial',expected_runs=10,valid_runs=len(valid),
                     budget_iterations=max(2000,50*meta[(problem,instance)]['n_customers']),
                     distance_mean=average([r['distance'] for r in valid]) if complete else None,
                     distance_best=min((r['distance'] for r in valid),default=None) if complete else None,
                     n_routes_mean=average([r['n_routes'] for r in valid]) if complete else None,
                     elapsed_mean=average([r['elapsed_seconds'] for r in valid]) if complete else None,
                     iterations_mean=average([r['iterations'] for r in valid]) if complete else None)
            instances.append(row);lookup[(problem,instance,method)]=row
    scopes=[(p,'',p+'总体',count) for p,count in [('CVRP',122),('VRPTW',56)]]+GROUPS
    groups=[];pairs=[];friedman=[]
    for problem,group,label,count in scopes:
        ids=sorted(i for p,i in meta if p==problem and (not group or meta[(p,i)]['group']==group))
        assert len(ids)==count
        complete=all(lookup[(problem,i,m)]['status']=='complete' for i in ids for m in METHODS)
        for method in METHODS:
            selected=[lookup[(problem,i,method)] for i in ids]
            good=[r for r in selected if r['status']=='complete']
            groups.append(dict(problem=problem,group=group,label=label,method=method,expected_instances=count,
                               complete_instances=len(good),status='complete' if complete else 'partial',
                               distance_mean=average([r['distance_mean'] for r in good]) if complete else None,
                               distance_best_mean=average([r['distance_best'] for r in good]) if complete else None,
                               n_routes_mean=average([r['n_routes_mean'] for r in good]) if complete else None,
                               elapsed_mean=average([r['elapsed_mean'] for r in good]) if complete else None,
                               iterations_mean=average([r['iterations_mean'] for r in good]) if complete else None))
        family='overall_6' if not group else 'subgroups_21'
        for baseline,target in PAIRS:
            row=dict(family=family,problem=problem,group=group,label=label,baseline=baseline,target=target,
                     expected_pairs=count,complete_pairs=sum(lookup[(problem,i,baseline)]['status']==lookup[(problem,i,target)]['status']=='complete' for i in ids),
                     status='complete' if complete else 'partial',p_raw=None)
            if complete:
                row.update(paired_test([lookup[(problem,i,baseline)]['distance_mean'] for i in ids],
                                       [lookup[(problem,i,target)]['distance_mean'] for i in ids]))
            pairs.append(row)
        row=dict(family='friedman_overall_2' if not group else 'friedman_subgroups_7',problem=problem,group=group,label=label,
                 instances=count,status='complete' if complete else 'partial',p_raw=None,statistic=None,kendall_w=None)
        if complete:
            arrays=[np.asarray([lookup[(problem,i,m)]['distance_mean'] for i in ids]) for m in METHODS]
            if all(np.array_equal(arrays[0],a) for a in arrays[1:]):statistic,pvalue=0.0,1.0
            else:
                result=stats.friedmanchisquare(*arrays);statistic,pvalue=float(result.statistic),float(result.pvalue)
            row.update(p_raw=pvalue,statistic=statistic,kendall_w=statistic/(count*2))
        friedman.append(row)
    complete=len(runs)==5340 and not failures and not missing and not unexpected
    # Withhold inferential statistics until the frozen full family is complete.
    if not complete:
        for row in pairs:
            for name in list(row):
                if name not in ('family','problem','group','label','baseline','target','expected_pairs','complete_pairs','status','p_raw'):
                    row.pop(name)
            row['p_raw']=None
        for row in friedman:row.update(p_raw=None,statistic=None,kendall_w=None)
    adjust_holm(pairs);adjust_holm(friedman)
    protected=[]
    for path,digest in protocol['protected_files_sha256'].items():
        if not Path(path).exists() or sha256(path)!=digest:protected.append(path)
    summary=dict(updated_at=timestamp(),status='complete' if complete else 'partial',complete=complete,
                 expected_comparison_records=5340,valid_comparison_records=sum(r['analysis_eligible'] for r in runs),
                 expected_new_solver_runs=3560,valid_new_solver_runs=sum(r['analysis_eligible'] and not r['reused'] for r in runs),
                 reused_solver_results=1780,valid_reused_solver_results=sum(r['analysis_eligible'] and r['reused'] for r in runs),
                 missing_runs=len(missing),failed_or_unverified=len(failures),unexpected_files=unexpected,
                 methods=list(METHODS),instances={'CVRP':122,'VRPTW':56},groups=groups,paired_comparisons=pairs,friedman=friedman,
                 protocol=protocol['statistics'],protected_files_unchanged=not protected,observed_changed_protected_files=protected,
                 letter_updated=False,manuscript_updated=False,analysis_source_sha256=sha256(__file__),
                 statistic_core_sha256=sha256(ROOT/'src/analyze_results.py'),numpy_version=np.__version__,scipy_version=scipy.__version__)
    for name,rows in [('per_run.csv',runs),('per_instance.csv',instances),('per_group.csv',groups),('paired_comparisons.csv',pairs),
                      ('friedman.csv',friedman),('missing_runs.csv',missing),('failures.csv',failures)]:
        write_csv(statsdir/name,rows)
    atomic_json(statsdir/'summary.json',summary)
    render_report(summary)
    print(json.dumps({k:summary[k] for k in ('status','complete','valid_comparison_records','valid_new_solver_runs',
          'valid_reused_solver_results','missing_runs','failed_or_unverified','protected_files_unchanged')},ensure_ascii=False),flush=True)
    if final and not complete:return 2
    return 0


def fmt(value,digits=4):
    return '待完整核验' if value is None else format(float(value),f',.{digits}f')


def pformat(value):
    if value is None:return '—'
    return format(float(value),'.3g') if float(value)<.0001 else format(float(value),'.4f')


def render_report(summary):
    def table(title,headers,rows,note=''):
        esc=lambda x:html.escape(str(x))
        return '<section><h2>'+esc(title)+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div><p class="note">'+esc(note)+'</p></section>'
    state='已全部完成并通过独立验证' if summary['complete'] else '运行中；完整比较结果尚未形成'
    pieces=['<h1>ALNS／AC／DAC-Partial：全量等迭代实验结果</h1>',
            '<p class="meta">2026年10月15日独立补充实验 · 更新：'+summary['updated_at']+'</p>',
            '<p class="status">'+state+'。新增有效结果 '+str(summary['valid_new_solver_runs'])+' / 3,560；复用DAC-Partial有效结果 '+str(summary['valid_reused_solver_results'])+' / 1,780。</p>',
            '<p>122个CVRP、56个VRPTW实例，每种方法每例10个固定种子（0、100、…、900）。每次K=max(2,000,50×客户数)，无额外秒数上限。一轮指完整破坏—修复—评价与接受判断。各方法使用本轮已归档的5＋5算子池和共同问题环境；这是完整控制器比较，不能将AC／DAC差值单独归因于观测时机。</p>',
            '<p>新增ALNS和AC，DAC-Partial沿用已完成E2。VRPTW AC使用原固定轻预训练模型，CVRP AC逐次随机初始化；不重新训练或调参。旧DAC结果已知后追加本方案，统计规则在新增基线运行前固定。不同方法的运行日期不同，耗时作为本机描述性记录。此文件为独立实验结果，回复信和论文正文未自动更新。</p>']
    common='每例先平均10次，再对组内实例等权平均。只有距离是优化目标，路线数描述车辆使用情况；耗时不含离线训练。'
    for grouped,title in [(False,'总体三方法结果'),(True,'各规模／类型分组结果')]:
        selected=[r for r in summary['groups'] if bool(r['group'])==grouped]
        pieces.append(table(title,['范围','方法','实例数','平均距离 ↓','平均路线数','平均耗时（秒）','平均迭代数'],
            [[r['label'],r['method'],str(r['complete_instances'])+'/'+str(r['expected_instances']),fmt(r['distance_mean'],2),fmt(r['n_routes_mean'],3),fmt(r['elapsed_mean'],2),fmt(r['iterations_mean'],1)] for r in selected],common))
    for grouped,title in [(False,'总体配对比较：全部三对方法'),(True,'分组配对比较：全部三对方法')]:
        selected=[r for r in summary['paired_comparisons'] if bool(r['group'])==grouped]
        rows=[]
        for r in selected:
            infer=r['p_holm'] is not None
            rows.append([r['label'],r['target']+' 相对 '+r['baseline'],str(r['complete_pairs'])+'/'+str(r['expected_pairs']),
                         fmt(r.get('mean_relative_improvement_pct')),('['+fmt(r.get('improvement_ci95_low'))+', '+fmt(r.get('improvement_ci95_high'))+']') if infer else '—',
                         pformat(r['p_holm']),f"{r.get('wins','—')}/{r.get('ties','—')}/{r.get('losses','—')}"])
        pieces.append(table(title,['范围','目标／对照','完整配对','平均改善（%）','95%区间（%）','Holm p','胜／平／负'],rows,
            '正值表示目标方法距离较短；改善率逐例计算后平均。双侧Wilcoxon使用实例均距差；总体6项统一Holm校正，分组21项在另一个家族统一校正。95%配对bootstrap区间未作多重校正；不以区间是否跨零替代校正p值。'))
    pieces.append(table('三方法Friedman总体检验',['范围','实例数','统计量','Holm p','Kendall W'],
        [[r['label'],r['instances'],fmt(r['statistic']),pformat(r['p_holm']),fmt(r['kendall_w'])] for r in summary['friedman']],
        '问题级两项检验共同校正；七个分组检验在另一个家族共同校正。'))
    pieces.append('<p class="note">原始运行、逐实例结果、全部统计与缺失／失败清单见同目录statistics。结果以单个固定源训练模型为条件，10个种子代表求解重复。保护文件哈希核验：'+('回复信和论文文件均未变化。' if summary['protected_files_unchanged'] else '检测到保护文件有外部变化，详见summary.json；本分析程序不写这些文件。')+'</p>')
    css='body{max-width:1180px;margin:30px auto;padding:0 24px;color:#253044;background:#fff;font:15px/1.8 "Microsoft YaHei",sans-serif}h1{font-size:25px}h2{font-size:19px;margin-top:32px}.meta,.note{color:#647086;font-size:12px}.status{padding:12px;background:#edf4ff;border-left:4px solid #24538b}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{border:1px solid #dce2eb;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f8}tbody tr:nth-child(even){background:#fafbfd}@page{size:A4 landscape;margin:15mm}@media print{body{margin:0;padding:0;font-size:10pt}table{font-size:8pt}thead{display:table-header-group}tr{break-inside:avoid}.scroll{overflow:visible}h2{break-after:avoid}}'
    output='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>等迭代三方法实验结果</title><style>'+css+'</style><body>'+''.join(pieces)+'</body></html>'
    (OUT/'等迭代三方法实验结果_20261015.html').write_text(output,encoding='utf-8')
    lines=['ALNS／AC／DAC-Partial 全量等迭代结果',summary['updated_at'],state,
           f"新增有效：{summary['valid_new_solver_runs']} / 3560；复用有效：{summary['valid_reused_solver_results']} / 1780。"]
    if summary['complete']:
        for row in summary['paired_comparisons']:
            if not row['group'] and row['target']=='DAC-Partial':
                lines.append(f"{row['problem']}：DAC相对{row['baseline']}平均改善{row['mean_relative_improvement_pct']:+.4f}%，Holm p={pformat(row['p_holm'])}；胜/平/负={row['wins']}/{row['ties']}/{row['losses']}。")
    lines.append('本次只生成独立实验结果，未自动更新回复信或论文正文。')
    (OUT/'结果摘要.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true')
    sys.exit(analyze(parser.parse_args().final))
