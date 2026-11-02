"""Independently validate and report the fixed, complete time-only E4 experiment.

No effect estimate or hypothesis test is emitted before all planned runs pass.
Original experiments, manuscript and response files are read-only inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import stats

BATCH = Path(__file__).resolve().parents[1]
ROOT = BATCH.parent
sys.path.insert(0, str(ROOT / 'src'))
from validate_routes import validate_routes

SEEDS = list(range(0, 1000, 100))
GROUP_COUNTS = {'CVRP-S': 54, 'CVRP-M1': 22, 'CVRP-M2': 30, 'CVRP-L': 16,
                'C': 17, 'R': 23, 'RC': 16}
BUDGETS = {'CVRP-S': 30, 'CVRP-M1': 160, 'CVRP-M2': 1010, 'CVRP-L': 4320,
           'C': 300, 'R': 280, 'RC': 350}
BASELINES = {'CVRP': 'DR-ALNS-time-only', 'VRPTW': 'Wang-PPO-ALNS-time-only'}
TARGET = 'DAC-Partial'
BOOTSTRAP_SEED = 20261025
BOOTSTRAP_REPEATS = 2000


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def verify_frozen(batch, protocol, plan):
    issues = []
    checked = {}

    def check(path, expected, category):
        path = str(Path(path).resolve())
        if path not in checked:
            try:
                checked[path] = sha256(path)
            except OSError as exc:
                checked[path] = None
                issues.append(f'{category}: unavailable {path}: {exc}')
        if checked[path] != expected:
            issues.append(f'{category}: SHA256 mismatch: {path}')

    for name in ('source_sha256', 'protected_files_sha256', 'manifest_sha256', 'upstream_manifest_sha256'):
        mapping = protocol.get(name)
        if not isinstance(mapping, dict) or not mapping:
            issues.append(f'Frozen protocol missing nonempty {name}')
            continue
        for path, expected in mapping.items():
            check(path, expected, name)
    self_path = str(Path(__file__).resolve())
    if self_path not in protocol.get('source_sha256', {}):
        issues.append('Analyzer source is not included in frozen source hashes')
    for name in ('planned_runs.json', 'instances.json', 'resource_allocation.json', 'machine.json'):
        path = str((batch / 'configs' / name).resolve())
        if path not in protocol.get('manifest_sha256', {}):
            issues.append(f'Manifest is not frozen: {path}')
    upstream_roots = {
        'DR-ALNS_official_file_manifest.json': ROOT / 'vendor/DR-ALNS-144e48f6912ad9967968561b6b0ce90cedcc14aa',
        'ppo-alns_official_file_manifest.json': ROOT / 'vendor/ppo-alns-64c0ba11902c7db805f64175b2db2dcfb58df7ad',
    }
    for manifest in protocol.get('upstream_manifest_sha256', {}):
        root = upstream_roots.get(Path(manifest).name)
        if root is None:
            issues.append(f'Unknown upstream source manifest: {manifest}')
            continue
        try:
            for row in read_json(manifest):
                check(root / row['path'], row['sha256'], 'upstream_source')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f'Cannot read upstream manifest {manifest}: {exc}')
    if len(plan) != 3560 or protocol.get('new_solver_runs') != 3560 or protocol.get('reused_runs') != 0:
        issues.append('Expected exactly 3560 fresh solves and zero reused solves')
    if protocol.get('seeds') != SEEDS or protocol.get('budgets_seconds') != BUDGETS:
        issues.append('Frozen seeds or grouped budgets differ from the intended protocol')
    run_ids = [row.get('run_id') for row in plan]
    if len(set(run_ids)) != len(run_ids):
        issues.append('Duplicate planned run_id')
    cells = defaultdict(list)
    group_instances = defaultdict(set)
    for cfg in plan:
        try:
            group, problem, instance = cfg['group'], cfg['problem'], cfg['instance_id']
            if group not in GROUP_COUNTS or problem not in BASELINES:
                raise ValueError('Unknown group/problem')
            if (problem == 'CVRP') != group.startswith('CVRP-'):
                raise ValueError('Group/problem inconsistent')
            if cfg['method'] not in (TARGET, BASELINES[problem]):
                raise ValueError('Unknown method/problem combination')
            if cfg['budget_mode'] != 'time' or float(cfg['budget_value']) != BUDGETS[group]:
                raise ValueError('Incorrect time budget')
            if cfg.get('stopping_rule') != 'time_limit_only':
                raise ValueError('Expected time_limit_only stopping rule')
            if cfg.get('protocol_version') != 'E4-time-only-20261025-v1':
                raise ValueError('Incorrect protocol version')
            if cfg['distance_convention'] != 'unrounded_euclidean' or cfg['fleet_objective'] != 'distance_only_nonbinding_fleet':
                raise ValueError('Unexpected objective definition')
            check(cfg['instance_path'], cfg['data_sha256'], 'instance_data')
            if cfg.get('checkpoint_path'):
                check(cfg['checkpoint_path'], cfg['checkpoint_sha256'], 'checkpoint')
            cells[(problem, instance, cfg['method'])].append(cfg['seed'])
            group_instances[group].add(instance)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f'Invalid plan row {cfg.get("run_id")}: {exc}')
    if {g: len(s) for g, s in group_instances.items()} != GROUP_COUNTS:
        issues.append('Instance counts do not match the seven frozen groups')
    if len(cells) != 356 or any(sorted(s) != SEEDS for s in cells.values()):
        issues.append('Expected 356 instance/method cells, each with the ten exact seeds')
    for problem, instance, method in list(cells):
        if (problem, instance, TARGET) not in cells or (problem, instance, BASELINES[problem]) not in cells:
            issues.append(f'Missing paired method: {problem}/{instance}')
    return issues, checked


def validate_result(path, cfg, protocol):
    errors = []
    raw = read_json(path)
    if raw.get('config') != cfg:
        errors.append('config differs from frozen plan')
    if raw.get('status') != 'ok':
        errors.append(f'status={raw.get("status")}; {raw.get("error", "")}')
    if raw.get('pilot') is not False:
        errors.append('result is not explicitly a formal run')
    if raw.get('experiment_batch_id') != 'E4-time-only-20261025':
        errors.append('wrong experiment batch')
    if raw.get('source_hashes') != protocol['source_sha256']:
        errors.append('recorded source hashes differ from frozen protocol')
    for key in ('run_id', 'problem', 'method', 'group', 'seed'):
        if raw.get(key) != cfg[key]:
            errors.append(f'result field {key} differs from frozen plan')
    if cfg['method'] != TARGET:
        if raw.get('checkpoint_sha256') != cfg.get('checkpoint_sha256'):
            errors.append('recorded external checkpoint differs from frozen plan')
        if raw.get('native_iteration_limit', 'missing') is not None:
            errors.append('external native iteration limit remains enabled')
    if raw.get('stopping_reason') != 'time_limit':
        errors.append('stopping reason is not time_limit')
    budget = float(cfg['budget_value'])
    elapsed = finite(raw.get('elapsed_seconds'))
    overshoot = finite(raw.get('overshoot_seconds'))
    if elapsed is None or elapsed + 1e-6 < budget:
        errors.append('missing elapsed time or premature termination')
    if overshoot is None or overshoot < 0:
        errors.append('missing/negative overshoot')
    elif elapsed is not None and abs(overshoot - max(0., elapsed - budget)) > 1e-4:
        errors.append('overshoot does not match elapsed minus budget')
    iterations = finite(raw.get('iterations'))
    if iterations is None or iterations < 0 or int(iterations) != iterations:
        errors.append('missing/invalid iterations')
    for event in raw.get('best_events', []):
        event_time = finite(event.get('time'))
        if event_time is None or event_time < 0 or event_time > budget:
            errors.append('best event is outside time budget')
    if raw.get('diagnostic_log'):
        log = Path(raw['diagnostic_log'])
        if not log.is_file() or sha256(log) != raw.get('diagnostic_log_sha256'):
            errors.append('diagnostic log hash mismatch')
    validation = validate_routes(cfg['instance_path'], cfg['problem'],
                                 {'routes': raw.get('routes', []), 'id_space': raw.get('route_id_space', 'internal')})
    if not validation['valid']:
        errors.extend('independent route check: ' + str(e) for e in validation['errors'])
    recorded = raw.get('validation') or {}
    if recorded.get('valid') is not True:
        errors.append('worker validation not valid')
    objective = validation['distance']
    reported = finite(raw.get('best_objective'))
    if reported is None or abs(reported - objective) > max(1e-3, abs(objective) * 1e-6):
        errors.append('solver objective differs from independently validated distance')
    events = raw.get('best_events') or []
    if not events or finite(events[-1].get('objective')) is None:
        errors.append('missing final best event inside the time budget')
    elif abs(float(events[-1]['objective']) - objective) > max(1e-3, abs(objective) * 1e-6):
        errors.append('final in-budget best event differs from independently validated distance')
    saved_distance = finite(recorded.get('distance'))
    if saved_distance is None or abs(saved_distance - objective) > 1e-7 * max(1., abs(objective)):
        errors.append('stored validation distance mismatch')
    if recorded.get('n_routes') != validation['n_routes']:
        errors.append('stored route count mismatch')
    if objective <= 0 or not math.isfinite(objective):
        errors.append('nonpositive or nonfinite independently validated distance')
    row = dict(run_id=cfg['run_id'], problem=cfg['problem'], group=cfg['group'], instance=cfg['instance_id'],
               n=cfg['n'], method=cfg['method'], seed=cfg['seed'], status=raw.get('status'),
               analysis_eligible=not errors, distance=objective, n_routes=validation['n_routes'],
               time_limit_seconds=budget, elapsed_seconds=elapsed, overshoot_seconds=overshoot,
               iterations=iterations, stopping_reason=raw.get('stopping_reason'),
               native_done_signals_ignored=raw.get('native_done_signals_ignored'),
               stop_actions_ignored=raw.get('stop_actions_ignored'),
               accept_zero_actions_attempted=raw.get('accept_zero_actions_attempted'),
               accept_one_actions_attempted=raw.get('accept_one_actions_attempted'),
               started_at=raw.get('started_at'), completed_at=raw.get('completed_at'),
               checkpoint_sha256=cfg.get('checkpoint_sha256', ''), source_file=str(path.resolve()),
               source_sha256=sha256(path), errors=json.dumps(errors, ensure_ascii=False))
    return row, errors


def paired_test(baseline, target):
    baseline, target = np.asarray(baseline, dtype=float), np.asarray(target, dtype=float)
    difference = baseline - target
    relative = 100 * difference / baseline
    nonzero = difference[difference != 0.]
    if len(nonzero):
        result = stats.wilcoxon(difference, zero_method='wilcox', correction=False,
                                alternative='two-sided', method='auto')
        pvalue, statistic = float(result.pvalue), float(result.statistic)
        ranks = stats.rankdata(np.abs(nonzero), method='average')
        rank_biserial = float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks))
    else:
        pvalue, statistic, rank_biserial = 1., 0., 0.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = relative[rng.integers(0, len(relative), (BOOTSTRAP_REPEATS, len(relative)))].mean(axis=1)
    low, high = np.percentile(resamples, [2.5, 97.5])
    return dict(n_instances=len(baseline), baseline_mean=float(baseline.mean()), target_mean=float(target.mean()),
                mean_relative_improvement_pct=float(relative.mean()), median_relative_improvement_pct=float(np.median(relative)),
                improvement_ci95_low=float(low), improvement_ci95_high=float(high), p_raw=pvalue,
                wilcoxon_statistic=statistic, matched_rank_biserial=rank_biserial,
                wins=int(np.sum(difference > 0)), ties=int(np.sum(difference == 0)), losses=int(np.sum(difference < 0)),
                nonzero_pairs=len(nonzero))


def holm(rows):
    maximum = 0.
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_raw'])
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1., (len(rows) - rank) * rows[index]['p_raw']))
        rows[index]['p_holm'] = maximum
        rows[index]['significant_holm_0_05'] = maximum < .05


def aggregate(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['problem'], row['group'], row['instance'], row['method'])].append(row)
    instances = []
    for (problem, group, instance, method), items in sorted(cells.items()):
        assert len(items) == 10 and sorted(r['seed'] for r in items) == SEEDS
        row = dict(problem=problem, group=group, instance=instance, n=items[0]['n'], method=method,
                   runs=10, time_limit_seconds=items[0]['time_limit_seconds'])
        for metric in ('distance', 'n_routes', 'elapsed_seconds', 'overshoot_seconds', 'iterations'):
            values = np.asarray([r[metric] for r in items], dtype=float)
            row[metric + '_mean'] = float(values.mean())
            row[metric + '_sd'] = float(values.std(ddof=1))
        row['distance_best'] = min(r['distance'] for r in items)
        row['overshoot_seconds_max'] = max(r['overshoot_seconds'] for r in items)
        instances.append(row)
    groups = []
    for problem in BASELINES:
        for group in ['ALL'] + [g for g in GROUP_COUNTS if (problem == 'CVRP') == g.startswith('CVRP-')]:
            for method in (BASELINES[problem], TARGET):
                selected = [r for r in instances if r['problem'] == problem and r['method'] == method and (group == 'ALL' or r['group'] == group)]
                row = dict(problem=problem, group=group, method=method, instances=len(selected), runs=10 * len(selected),
                           time_limit_seconds= BUDGETS[group] if group != 'ALL' else 'varies_by_group')
                for metric in ('distance', 'n_routes', 'elapsed_seconds', 'overshoot_seconds', 'iterations'):
                    row[metric + '_mean'] = float(np.mean([r[metric + '_mean'] for r in selected]))
                row['overshoot_seconds_max'] = max(r['overshoot_seconds_max'] for r in selected)
                groups.append(row)
    comparisons = []
    for family in ('overall', 'subgroup'):
        family_rows = []
        for problem in BASELINES:
            group_names = ['ALL'] if family == 'overall' else [g for g in GROUP_COUNTS if (problem == 'CVRP') == g.startswith('CVRP-')]
            for group in group_names:
                subset = [r for r in instances if r['problem'] == problem and (group == 'ALL' or r['group'] == group)]
                lookup = {(r['instance'], r['method']): r for r in subset}
                names = sorted({r['instance'] for r in subset})
                base = [lookup[name, BASELINES[problem]] for name in names]
                target = [lookup[name, TARGET] for name in names]
                row = dict(family=family, problem=problem, group=group, baseline=BASELINES[problem], target=TARGET,
                           **paired_test([r['distance_mean'] for r in base], [r['distance_mean'] for r in target]))
                for metric in ('n_routes', 'elapsed_seconds', 'iterations', 'overshoot_seconds'):
                    row['baseline_' + metric + '_mean'] = float(np.mean([r[metric + '_mean'] for r in base]))
                    row['target_' + metric + '_mean'] = float(np.mean([r[metric + '_mean'] for r in target]))
                family_rows.append(row)
        assert len(family_rows) == (2 if family == 'overall' else 7)
        holm(family_rows)
        comparisons.extend(family_rows)
    return instances, groups, comparisons


def table(headers, rows):
    escape = lambda value: html.escape(str(value))
    return '<table><thead><tr>' + ''.join('<th>' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table>'


def render_reports(batch, summary, instances=None, groups=None, comparisons=None):
    report_dir = batch / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    complete = summary['complete']
    title = 'E4：仅按分组时间上限停止的实验结果'
    lines = [title, '生成时间：' + summary['generated_at'],
             f"独立复核有效：{summary['valid_runs']} / {summary['expected_runs']}；缺失：{summary['missing_runs']}；无效：{summary['invalid_runs']}。"]
    sections = ['<h1>' + title + '</h1>', '<p>' + html.escape(lines[1]) + '</p>', '<p>' + html.escape(lines[2]) + '</p>']
    if not complete:
        message = '本批尚未全部完成并通过复核，暂不计算最终改善幅度、显著性或置信区间。'
        lines.append(message)
        sections.append('<p class="notice">' + message + '</p>')
        sections.append(table(['检查项', '数量'], [['已找到结果', summary['result_files']], ['缺失', summary['missing_runs']],
                                                  ['无效', summary['invalid_runs']], ['完整性问题', len(summary['integrity_issues'])]]))
        for issue in summary['integrity_issues']:
            lines.append(issue)
    else:
        notes = [
            '范围为122个CVRP实例和56个VRPTW实例，每个实例、每种方法使用10个相同随机种子（0、100、…、900）。共3560次全新求解，DAC-Partial同期重跑1780次，未复用旧结果。',
            'CVRP比较DAC-Partial与DR-ALNS时间版；VRPTW比较DAC-Partial与Wang PPO-ALNS时间版。所有方法均仅在对应分组时间上限达到时结束。',
            '公开算法采用修改后的停止条件：取消DR的1000步停止、Wang的100步和连续停止动作结束规则。搜索状态持续保留，不重启求解。原进度特征保持原比例，达到原步数后固定为1；检查点不变，未重新训练或调参。此结果对应时间版，不等同于原生停止规则的公开实现。',
            '初始化和检查点加载计入时间；导入模块与JIT预热不计入。某个不可拆分操作可能使总耗时略超过上限，超过上限才完成的候选解不纳入最终最优解。',
            '距离为未四舍五入的欧氏距离。路线数仅作说明，没有最少车辆优先目标，也不作为限制条件。',
            '统计单位是每个实例的10种子平均值。距离改善率=100×(对照平均距离−DAC平均距离)/对照平均距离，再对实例取平均；正数表示DAC距离更短。胜/平/负按实例均值精确比较。',
            '总体2个双侧Wilcoxon配对检验进行一组Holm校正；7个子组检验单独进行一组Holm校正。表中p值为校正值。95%置信区间采用配对实例自助法（2000次，随机种子20261025），区间未作多重比较校正。',
            '本报告独立保存，本批未修改回复信或论文正文。旧实验结果已被观察，本批固定方案不是对所有研究假设的事前注册。',
        ]
        lines.extend(notes)
        sections.extend('<p>' + html.escape(note) + '</p>' for note in notes)
        sections.append('<h2>总体表现</h2>')
        headers = ['范围', '实例数', '对照平均距离', 'DAC平均距离', 'DAC平均改善', '95%区间', 'Holm p', '胜/平/负']
        def effect_cells(row):
            return [row['problem'] if row['group'] == 'ALL' else row['group'], row['n_instances'],
                    f"{row['baseline_mean']:.4f}", f"{row['target_mean']:.4f}", f"{row['mean_relative_improvement_pct']:+.4f}%",
                    f"[{row['improvement_ci95_low']:+.4f}%, {row['improvement_ci95_high']:+.4f}%]", f"{row['p_holm']:.6g}",
                    f"{row['wins']}/{row['ties']}/{row['losses']}"]
        sections.append(table(headers, [effect_cells(r) for r in comparisons if r['family'] == 'overall']))
        sections.append('<h2>七个分组</h2>')
        sections.append(table(headers, [effect_cells(r) for r in comparisons if r['family'] == 'subgroup']))
        for row in comparisons:
            lines.append(' | '.join(map(str, effect_cells(row))))
        sections.append('<h2>时间、迭代次数和路线数</h2>')
        timing_rows = [[r['group'], r['method'], r['instances'], r['time_limit_seconds'],
                        f"{r['elapsed_seconds_mean']:.3f}", f"{r['overshoot_seconds_mean']:.4f}", f"{r['overshoot_seconds_max']:.4f}",
                        f"{r['iterations_mean']:.1f}", f"{r['n_routes_mean']:.4f}"] for r in groups if r['group'] != 'ALL']
        sections.append(table(['组别', '方法', '实例数', '时间上限/秒', '实际平均秒', '平均超时秒', '最大超时秒', '平均迭代数', '平均路线数'], timing_rows))
        sections.append('<h2>178个实例的完整结果</h2>')
        lookup = {(r['problem'], r['instance'], r['method']): r for r in instances}
        instance_rows = []
        for row in instances:
            if row['method'] != TARGET:
                continue
            base = lookup[row['problem'], row['instance'], BASELINES[row['problem']]]
            effect = 100 * (base['distance_mean'] - row['distance_mean']) / base['distance_mean']
            instance_rows.append([row['group'], row['instance'], row['n'], f"{base['distance_mean']:.4f}",
                                  f"{row['distance_mean']:.4f}", f"{effect:+.4f}%", f"{base['n_routes_mean']:.2f}",
                                  f"{row['n_routes_mean']:.2f}", f"{base['elapsed_seconds_mean']:.3f}", f"{row['elapsed_seconds_mean']:.3f}"])
        sections.append(table(['组别', '实例', '客户数', '对照距离', 'DAC距离', 'DAC改善', '对照路线数', 'DAC路线数', '对照秒', 'DAC秒'], instance_rows))
        sections.append('<p>可复核数据：<a href="statistics/per_run.csv">每次求解</a> · <a href="statistics/per_instance.csv">实例均值</a> · <a href="statistics/per_group.csv">分组统计</a> · <a href="statistics/paired_comparisons.csv">配对检验</a> · <a href="statistics/summary.json">完整性与摘要</a></p>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#202c3b;line-height:1.7}h1{font-size:28px}h2{font-size:22px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{padding:7px 9px;border:1px solid #ccd5df;text-align:right}th{background:#eaf0f6}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f6f8fa}.notice{padding:15px;background:#fff3d5}a{color:#175a9a}@media print{body{margin:0;max-width:none;font-size:11px}table{font-size:9px}thead{display:table-header-group}tr{break-inside:avoid}}</style><body>' + ''.join(sections) + '</body></html>'
    (report_dir / '仅时间停止实验结果.html').write_text(document, encoding='utf-8')
    (report_dir / '结果摘要.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')


def analyze(batch):
    protocol = read_json(batch / 'configs/frozen_protocol.json')
    plan = read_json(batch / 'configs/planned_runs.json')
    issues, checked = verify_frozen(batch, protocol, plan)
    planned = {r['run_id']: r for r in plan}
    paths = sorted((batch / 'results/E4').glob('*.json'))
    by_id = {p.stem: p for p in paths}
    unexpected = sorted(set(by_id) - set(planned))
    if unexpected:
        issues.append('Unexpected result files: ' + repr(unexpected))
    missing = sorted(set(planned) - set(by_id))
    rows, invalid = [], []
    for run_id, cfg in planned.items():
        if run_id not in by_id:
            continue
        try:
            row, errors = validate_result(by_id[run_id], cfg, protocol)
            rows.append(row)
            if errors:
                invalid.append(dict(run_id=run_id, errors=errors))
        except Exception as exc:
            invalid.append(dict(run_id=run_id, errors=[f'{type(exc).__name__}: {exc}']))
    valid_runs = sum(r['analysis_eligible'] for r in rows)
    complete = not issues and not missing and not invalid and len(rows) == len(plan) == 3560
    summary = dict(complete=complete, status='complete' if complete else 'incomplete_or_invalid',
                   generated_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                   batch_id=protocol.get('batch_id'), expected_runs=len(plan), result_files=len(paths),
                   valid_runs=valid_runs, missing_runs=len(missing), invalid_runs=len(invalid),
                   missing_run_ids=missing, invalid_results=invalid, unexpected_results=unexpected,
                   integrity_issues=issues, verified_file_count=len(checked),
                   result_counts_by_method=dict(Counter(r['method'] for r in rows if r['analysis_eligible'])),
                   protocol_path=str((batch / 'configs/frozen_protocol.json').resolve()),
                   protocol_sha256=sha256(batch / 'configs/frozen_protocol.json'),
                   letter_updated=False, manuscript_updated=False, fresh_runs_only=True,
                   numpy_version=np.__version__, scipy_version=scipy.__version__,
                   statistics=protocol.get('statistics'), overall_comparisons=[])
    statistics_dir = batch / 'reports/statistics'
    write_csv(statistics_dir / 'per_run.csv', rows, fields=None if rows else ['run_id', 'status', 'analysis_eligible'])
    instances = groups = comparisons = None
    if complete:
        instances, groups, comparisons = aggregate(rows)
        write_csv(statistics_dir / 'per_instance.csv', instances)
        write_csv(statistics_dir / 'per_group.csv', groups)
        write_csv(statistics_dir / 'paired_comparisons.csv', comparisons)
        summary['overall_comparisons'] = [r for r in comparisons if r['family'] == 'overall']
        summary['group_comparisons'] = [r for r in comparisons if r['family'] == 'subgroup']
        summary['completed_at'] = max(r['completed_at'] for r in rows)
        summary['started_at'] = min(r['started_at'] for r in rows)
        summary['validated_instances'] = len({(r['problem'], r['instance']) for r in rows})
        summary['reported_route_validity'] = 'All routes independently revalidated from raw instance data'
    else:
        # Make an incomplete rerun incapable of leaving apparently current effects in the output CSVs.
        for name in ('per_instance.csv', 'per_group.csv', 'paired_comparisons.csv'):
            write_csv(statistics_dir / name, [], ['status_no_final_statistics_until_complete'])
    dump(statistics_dir / 'summary.json', summary)
    render_reports(batch, summary, instances, groups, comparisons)
    print(json.dumps({k: summary[k] for k in ('complete', 'expected_runs', 'valid_runs', 'missing_runs', 'invalid_runs', 'integrity_issues')}, ensure_ascii=False))
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, default=BATCH, help='Batch root; defaults to this source file\'s parent batch')
    args = parser.parse_args()
    return analyze(args.batch_dir.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
