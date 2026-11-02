"""Analyze the authorized 178-instance expansion without altering the old archive.

The merged plan contains comparison records, not independent solver executions.
E4 references to historical E1 DAC runs retain the original file and hash. No
result JSON is copied, rewritten, or manufactured. Inferential comparisons are
withheld until all planned instances and seeds in that comparison are complete.
The original nineteen hypotheses and Holm families are retained, with E2/E5
expanded from the historical panel to all instances. E4 remains descriptive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

import analyze_results as original
import supplement_reports as supplement
from experiment_io import atomic_json, sha256, timestamp

BASE = Path(__file__).resolve().parents[1]
DEFAULT_EXPANSION = BASE / 'expansion_20261012'
EXPECTED_STAGES = {'E1': 7660, 'E2': 3560, 'E3': 3560, 'E4': 3560, 'E5': 1780}
EXPECTED_NEW = {'E2': 2860, 'E4': 1430, 'E5': 1430}
REUSE_FIELDS = ('reused_from_experiment', 'reused_from_run_id', 'reuse_kind')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return default


def entries_of(document):
    if isinstance(document, dict):
        for field in ('runs', 'tasks', 'planned_runs', 'reused_runs', 'reuse'):
            if isinstance(document.get(field), list):
                return document[field]
        return []
    return document if isinstance(document, list) else []


def expected_keys(metadata):
    keys = set()
    for row in metadata.values():
        methods = {'E1': ['ALNS', 'AC', 'DAC-Pre', 'DAC-Partial'],
                   'E2': ['DAC-Pre', 'DAC-Partial'], 'E3': ['Frozen', 'Adapt'],
                   'E4': ['DAC-Partial', 'DR-ALNS-original' if row['problem'] == 'CVRP'
                          else 'PPO-ALNS-original'], 'E5': ['DAC-Credit']}
        if row['group'] == 'CVRP-S':
            methods['E1'].append('PPO-adapted')
        for stage, stage_methods in methods.items():
            for method in stage_methods:
                for seed in range(0, 1000, 100):
                    keys.add((stage, row['problem'], row['id'], method, seed))
    return keys


def comparisons_all(instances, metadata):
    """Retain the exact nineteen comparison definitions, expanding E2 and E5."""
    # The original panel flag only controls E2/E5 in comparisons(). Supplying a
    # view in which all rows are selected reuses its checked test/seed logic.
    all_metadata = {key: {**row, 'panel': True} for key, row in metadata.items()}
    output = original.comparisons(instances, all_metadata)
    for row in output:
        if row['experiment'] in {'E2', 'E5'}:
            row['panel'] = False
            row['sample_scope'] = 'full_178_instance_expansion'
        else:
            row['sample_scope'] = 'historical_full_scope_unchanged'
    return output


def merge_records(directories, planned, metadata):
    records, errors = defaultdict(list), []
    for directory in directories:
        incoming, issues = original.read_results(directory, planned, metadata)
        errors.extend(issues)
        for key, values in incoming.items():
            records[key].extend(values)
    for values in records.values():
        if len(values) > 1:
            for row, _ in values:
                row['duplicate'] = True
                row['analysis_eligible'] = False
    return records, errors


def reuse_target_id(entry):
    return entry.get('target_run_id') or entry.get('run_id')


def apply_reuse_manifest(records, planned, manifest):
    """Validate reuse mappings, then create explicit in-memory E4 references."""
    audit, issues = [], []
    by_id = {cfg.get('run_id'): (key, cfg) for key, cfg in planned.items()}
    source_by_id = defaultdict(list)
    for key, values in records.items():
        for row, raw in values:
            source_by_id[row['run_id']].append((key, row, raw))
    declared = set()
    for entry in entries_of(manifest):
        target_id = reuse_target_id(entry)
        if not target_id or target_id not in by_id:
            issues.append(f'reuse target not in merged plan: {target_id!r}')
            continue
        if target_id in declared:
            issues.append(f'duplicate reuse declaration: {target_id}')
            continue
        declared.add(target_id)
        target_key, target_cfg = by_id[target_id]
        source_id = entry.get('source_run_id') or target_id
        candidates = source_by_id.get(source_id, [])
        item = {'target_run_id': target_id, 'source_run_id': source_id,
                'reuse_kind': entry.get('reuse_kind', 'historical_result'),
                'status': 'missing', 'error': '', 'source_file': '', 'source_sha256': ''}
        audit.append(item)
        if len(candidates) != 1:
            item['error'] = 'source missing' if not candidates else 'source duplicated'
            continue
        source_key, source_row, source_raw = candidates[0]
        source_cfg = original.config_of(source_raw)
        item.update(source_file=source_row['source_file'], source_sha256=source_row['source_sha256'])
        errors = []
        expected_hash = entry.get('source_sha256')
        if not expected_hash:
            errors.append('reuse manifest must pin source_sha256')
        elif source_row['source_sha256'] != expected_hash:
            errors.append('source SHA256 differs from reuse manifest')
        declared_path = entry.get('source_file')
        if declared_path and Path(declared_path).resolve() != Path(source_row['source_file']).resolve():
            errors.append('source_file differs from reuse manifest')
        if not source_row['analysis_eligible']:
            errors.append('source is not independently valid and unique')
        is_alias = target_id != source_id
        if is_alias:
            if not (target_key[0] == 'E4' and source_key[0] == 'E1'
                    and target_key[1:] == source_key[1:]
                    and target_key[3] == 'DAC-Partial'):
                errors.append('only same-instance/method/seed E1 DAC-Partial to E4 reuse is allowed')
            if target_key in records:
                errors.append('E4 reuse target already has a physical result')
            for field in ('budget_mode', 'budget_value', 'checkpoint_sha256', 'data_sha256',
                          'initialization', 'distance_convention', 'fleet_objective',
                          'device', 'rollout_size', 'protocol_version'):
                # Materialized old/new plans preserve these values. A field
                # absent on one side cannot silently establish compatibility.
                left, right = source_cfg.get(field), target_cfg.get(field)
                if left != right:
                    errors.append(f'incompatible {field}: source={left!r}, target={right!r}')
        elif target_key != source_key:
            errors.append('historical reuse key differs from planned key')
        else:
            for field, value in target_cfg.items():
                if field == 'instance':
                    continue  # config_of supplies this normalized alias.
                if field not in source_cfg or source_cfg[field] != value:
                    errors.append(f'historical source differs from frozen plan field {field}')
        if errors:
            item.update(status='invalid', error='; '.join(errors))
            issues.append(f'invalid reuse {target_id}: {item["error"]}')
            # A changed historical result must never remain analysis-eligible.
            if not is_alias:
                source_row['analysis_eligible'] = False
                source_row['failure_reason'] = item['error']
            continue
        item['status'] = 'valid'
        if is_alias:
            row = dict(source_row)
            row.update(experiment='E4', run_id=target_id, planned=True,
                       reused_from_experiment='E1', reused_from_run_id=source_id,
                       reuse_kind=item['reuse_kind'])
            # Keep source_raw intact so profile/log provenance remains the
            # original source result; the CSV row identifies the comparison.
            records[target_key].append((row, source_raw))
        else:
            source_row['reuse_kind'] = item['reuse_kind']
    # Eligibility of aliases must not depend on manifest ordering. A source
    # invalidated by its own historical declaration invalidates its references.
    for item in audit:
        if item['status'] != 'valid' or item['target_run_id'] == item['source_run_id']:
            continue
        source = source_by_id.get(item['source_run_id'], [])
        if len(source) != 1 or not source[0][1]['analysis_eligible']:
            key = by_id[item['target_run_id']][0]
            for row, _ in records.get(key, []):
                row['analysis_eligible'] = False
                row['failure_reason'] = 'Referenced historical source is ineligible'
            item.update(status='invalid', error='referenced historical source is ineligible')
            issues.append(f'invalid reuse {item["target_run_id"]}: {item["error"]}')
    return audit, issues


def route_distance_rows(instances):
    """Give equally weighted instance means for both CVRP and VRPTW."""
    buckets = defaultdict(list)
    for row in instances:
        buckets[(row['experiment'], row['problem'], row['group'], row['method'])].append(row)
        buckets[(row['experiment'], row['problem'], 'ALL', row['method'])].append(row)
    output = []
    for key, values in sorted(buckets.items()):
        good = [row for row in values if row['status'] == 'complete'
                and row['n_routes_mean'] is not None and row['objective_mean'] is not None]
        complete = len(good) == len(values)
        output.append({'experiment': key[0], 'problem': key[1], 'group': key[2], 'method': key[3],
                       'expected_instances': len(values), 'complete_instances': len(good),
                       'status': 'complete' if complete else 'partial',
                       'distance_mean': original.mean([r['objective_mean'] for r in good]) if complete else None,
                       'routes_mean': original.mean([r['n_routes_mean'] for r in good]) if complete else None,
                       'elapsed_mean': original.mean([r['elapsed_mean'] for r in good]) if complete else None,
                       'measurement': 'mean over instances of validated final-solution seed means',
                       'fleet_objective': 'distance_only_nonbinding_fleet'})
    return output


def external_full(instances, metadata, runs):
    """E4 full-scope paired descriptions, without an added significance test."""
    lookup = {(r['experiment'], r['problem'], r['instance'], r['method']): r for r in instances}
    output = []
    for problem, baseline in [('CVRP', 'DR-ALNS-original'), ('VRPTW', 'PPO-ALNS-original')]:
        ids = sorted(r['id'] for r in metadata.values() if r['problem'] == problem)
        pairs = []
        for inst in ids:
            left = lookup.get(('E4', problem, inst, baseline))
            right = lookup.get(('E4', problem, inst, 'DAC-Partial'))
            if (left and right and left['status'] == right['status'] == 'complete'
                    and left['planned_seeds'] == right['planned_seeds']):
                pairs.append((left, right))
        row = {'experiment': 'E4', 'problem': problem, 'baseline': baseline, 'target': 'DAC-Partial',
               'sample_scope': 'full_178_instance_expansion', 'expected_pairs': len(ids),
               'complete_pairs': len(pairs), 'status': 'complete' if ids and len(pairs) == len(ids) else 'partial',
               'statistics': None}
        if row['status'] == 'complete':
            aa = np.asarray([a['objective_mean'] for a, _ in pairs], dtype=float)
            bb = np.asarray([b['objective_mean'] for _, b in pairs], dtype=float)
            difference = aa - bb
            relative = 100 * difference / aa
            rng = np.random.default_rng(original.BOOTSTRAP_SEED)
            draws = np.mean(relative[rng.integers(0, len(ids), (original.BOOTSTRAP_REPEATS, len(ids)))], axis=1)
            low, high = np.percentile(draws, [2.5, 97.5])
            row['statistics'] = {'baseline_mean': float(aa.mean()), 'target_mean': float(bb.mean()),
                                 'mean_relative_improvement_pct': float(relative.mean()),
                                 'ci95_low': float(low), 'ci95_high': float(high),
                                 'wins': int((difference > 0).sum()), 'ties': int((difference == 0).sum()),
                                 'losses': int((difference < 0).sum()),
                                 'baseline_elapsed_mean': original.mean([a['elapsed_mean'] for a, _ in pairs]),
                                 'target_elapsed_mean': original.mean([b['elapsed_mean'] for _, b in pairs]),
                                 'baseline_routes_mean': original.mean([a['n_routes_mean'] for a, _ in pairs]),
                                 'target_routes_mean': original.mean([b['n_routes_mean'] for _, b in pairs]),
                                 'inference_scope': 'descriptive paired bootstrap interval; no added significance test or p value'}
        observed = [r['compatibility'] for r in runs if r['experiment'] == 'E4'
                    and r['problem'] == problem and r['method'] == baseline]
        row['observed_compatibility'] = {
            key: sorted({json.dumps(r[key], ensure_ascii=False, sort_keys=True)
                         for r in observed if r.get(key) is not None})
            for key in ('upstream_commit', 'compatibility_changes', 'operator_library', 'restart_policy',
                        'checkpoint_sha256', 'checkpoint_training_customers')}
        output.append(row)
    return output


def generate_supplement(main_path, main, instances, metadata, per_run, out):
    """Reuse recorded diagnostic aggregation while changing only sample scope."""
    out.mkdir(parents=True, exist_ok=True)
    input_hash = sha256(main_path)
    code_hash = sha256(Path(supplement.__file__))
    cached = read_json(out / 'run_cache.json', {})
    if cached.get('aggregation_source_sha256') != code_hash:
        cached = read_json(BASE / 'reports' / 'supplement' / 'run_cache.json', {})
    cache = cached.get('runs', {}) if cached.get('aggregation_source_sha256') == code_hash else {}
    selected = [r for r in per_run if r['analysis_eligible'] and r['planned']]
    summaries = [supplement.summarize_run(row, cache) for row in selected]
    invalid = [r for r in summaries if not r['input_valid']]
    runs = [r for r in summaries if r['input_valid']]
    consistent = len(selected) == main['valid_runs'] and not invalid and sha256(main_path) == input_hash
    diagnostics = supplement.diagnostic_groups(main, runs)
    anytime = supplement.anytime_groups(main, runs)
    figures = supplement.plot_anytime(anytime, out) if consistent else []
    training = supplement.training_costs()
    parameters, provenance = supplement.parameter_provenance()
    external = external_full(instances, metadata, runs)
    routes = route_distance_rows(instances)
    payload = {'schema_version': 2, 'updated_at': timestamp(), 'input_summary_sha256': input_hash,
               'input_consistent': consistent, 'input_main_status': main['status'],
               'input_valid_runs': main['valid_runs'], 'expansion_complete': main['expansion_complete'],
               'status': 'complete' if consistent and main['expansion_complete'] else 'partial',
               'invalid_sources': invalid, 'parameter_provenance': provenance, 'parameter_table': parameters,
               'diagnostics': diagnostics, 'anytime': anytime, 'anytime_figures': figures,
               'training_costs': training, 'external_comparisons': external, 'distance_and_routes': routes,
               'external_source_audit': read_json(BASE / 'reports' / 'external_audit.json', {}).get('official_repositories', []),
               'measurement_notes': {
                   'diagnostics_unit': 'Pooled recorded updates/transitions are descriptive, not independent statistical samples.',
                   'raw_advantage_rms': 'Reconstructed from logged within-batch population mean/std and batch size; E5 destroy raw advantage is V(partial)-V(full).',
                   'clipping_measurement': 'Logged abs(value)>=1 is boundary occupancy, not the exact fraction changed by clipping; singleton uncentered batches are reported separately.',
                   'grad_norm_measurement': 'clip_grad_norm_ returns pre-clipping norm; fraction >1 measures threshold exceedance.',
                   'anytime_measurement': 'Solver-recorded incumbent distances; only final endpoint distances are independently validated.',
                   'not_recorded': ['per-transition minimum policy probability', 'individual TD samples/absolute TD errors',
                                    'exact pre-vs-post advantage clipping change count', 'process CPU time',
                                    'historical grid-search instance list'],
                   'e4_measurement': 'Full 122 CVRP / 56 VRPTW scope; paired descriptive bootstrap interval, no added significance test.',
                   'e4_reuse': 'Historical 35-instance E4 runs retained; other DAC-Partial controls are explicitly referenced E1 runs with pinned source hashes.',
                   'fleet_measurement': 'Distance and route count describe the same validated final solution; distance is the sole optimization objective and fleet is nonbinding.',
                   'incomplete_policy': 'Full-scope external statistics remain null until all expected paired seeds and instances are independently valid.',
                   'training_measurement': 'Recorded elapsed wall time; published checkpoint historical training compute remains unknown.'},
               'aggregation_source_sha256': code_hash, 'expansion_analysis_source_sha256': sha256(Path(__file__))}
    if not consistent:
        payload['consistency_note'] = 'Source hash mismatch or main snapshot changed; numerical supplement results must not be used.'
    for filename, rows in [('parameters_and_provenance.csv', parameters), ('diagnostic_groups.csv', diagnostics),
                           ('anytime_groups.csv', anytime), ('training_costs.csv', training),
                           ('external_paired_descriptive.csv', external), ('distance_and_routes.csv', routes)]:
        supplement.write_csv(out / filename, rows)
    payload['derived_payload_sha256'] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                                 allow_nan=False).encode('utf-8')).hexdigest()
    atomic_json(out / 'run_cache.json', {'aggregation_source_sha256': code_hash, 'runs': cache})
    atomic_json(out / 'summary.json', payload)
    atomic_json(out / 'summary_manifest.json', {'input_summary_sha256': input_hash,
                'summary_sha256': sha256(out / 'summary.json'), 'derived_payload_sha256': payload['derived_payload_sha256'],
                'aggregation_source_sha256': code_hash, 'updated_at': payload['updated_at']})
    return payload


def analyze(expansion, skip_supplement=False):
    expansion = Path(expansion).resolve()
    if expansion == BASE.resolve() or expansion in ((BASE / 'reports').resolve(), (BASE / 'results').resolve()):
        raise ValueError('Use a separate expansion directory; the historical archive is read-only input.')
    config_dir = expansion / 'configs'
    out = expansion / 'reports' / 'statistics'
    out.mkdir(parents=True, exist_ok=True)
    instance_path = config_dir / 'instances.json'
    if not instance_path.exists():
        instance_path = BASE / 'configs' / 'instances.json'
    metadata_rows = read_json(instance_path, [])
    metadata = {(r['problem'], r['id']): r for r in metadata_rows}
    planned_path = config_dir / 'planned_runs.json'
    planned, plan_issues = original.read_plan(planned_path, metadata)
    new_path = config_dir / 'new_runs.json'
    reuse_path = config_dir / 'reused_runs.json'
    new_entries = entries_of(read_json(new_path, []))
    new_keys = {original.key_of(original.config_of(cfg)) for cfg in new_entries}
    if not new_path.exists():
        plan_issues.append('new_runs manifest missing')
    if len(new_keys) != len(new_entries):
        plan_issues.append('duplicate new-run keys')
    actual_stages = Counter(k[0] for k in planned)
    if dict(actual_stages) != EXPECTED_STAGES:
        plan_issues.append(f'merged stage counts differ from authorized full-scope plan: {dict(actual_stages)}')
    actual_new = Counter(k[0] for k in new_keys)
    if dict(actual_new) != EXPECTED_NEW:
        plan_issues.append(f'new-run counts differ from authorized expansion: {dict(actual_new)}')
    if Counter(r['problem'] for r in metadata_rows) != Counter({'CVRP': 122, 'VRPTW': 56}):
        plan_issues.append('instance metadata must contain 122 CVRP and 56 VRPTW unique instances')
    if len(metadata_rows) != len(metadata):
        plan_issues.append('duplicate instance metadata keys')
    intended_keys = expected_keys(metadata)
    if set(planned) != intended_keys:
        plan_issues.append(f'merged plan differs from full methods/instances/ten-seed design; missing={len(intended_keys-set(planned))}, extra={len(set(planned)-intended_keys)}')
    for key in new_keys:
        if key not in planned:
            plan_issues.append(f'new-run key absent from merged plan: {key}')
    records, read_errors = merge_records([BASE / 'results', expansion / 'results'], planned, metadata)
    if not reuse_path.exists():
        plan_issues.append('reuse manifest missing')
    reuse_audit, reuse_issues = apply_reuse_manifest(records, planned, read_json(reuse_path, []))
    plan_issues.extend(reuse_issues)
    declared_targets = {item['target_run_id'] for item in reuse_audit}
    planned_reuse_targets = {cfg.get('run_id') for key, cfg in planned.items() if key not in new_keys}
    if declared_targets != planned_reuse_targets:
        plan_issues.append(f'reuse declarations differ from all non-new comparison records; missing={len(planned_reuse_targets-declared_targets)}, extra={len(declared_targets-planned_reuse_targets)}')
    instances, groups = original.aggregate(planned, records)
    paired = comparisons_all(instances, metadata)
    if set(planned) != intended_keys:
        # A malformed manifest can otherwise make a smaller seed set appear
        # complete. Do not issue inference from an unauthorized design.
        for row in paired:
            for field in list(row):
                if field not in {'family', 'experiment', 'baseline', 'target', 'problem', 'group',
                                 'panel', 'baseline_experiment', 'expected_pairs', 'complete_pairs',
                                 'status', 'missing_instances', 'family_complete', 'family_hypotheses',
                                 'sample_scope', 'p_raw', 'p_holm'}:
                    row.pop(field)
            row.update(status='partial', family_complete=False, p_raw=None, p_holm=None,
                       integrity_note='Merged plan differs from authorized full-scope seed/method design.')
    friedman = original.friedman_tests(instances, metadata)
    profiles = original.time_profiles(groups, records)
    runs = [row for key in sorted(records) for row, _ in records[key]]
    failures = [row for row in runs if not row['analysis_eligible']]
    missing = [{'experiment': key[0], 'problem': key[1], 'instance': key[2], 'method': key[3],
                'seed': key[4], 'run_id': cfg.get('run_id', '')}
               for key, cfg in planned.items() if key not in records]
    stages = []
    for stage in sorted(original.FORMAL):
        keys = [key for key in planned if key[0] == stage]
        valid = sum(len(records.get(key, [])) == 1 and records[key][0][0]['analysis_eligible'] for key in keys)
        new_valid = sum(key in new_keys and len(records.get(key, [])) == 1
                        and records[key][0][0]['analysis_eligible'] for key in keys)
        alias_keys = [key for key in keys if records.get(key) and records[key][0][0].get('reused_from_run_id')]
        stages.append({'experiment': stage, 'expected_runs': len(keys), 'valid_runs': valid,
                       'expected_new_solver_runs': sum(key in new_keys for key in keys),
                       'valid_new_solver_runs': new_valid, 'observed_alias_records': len(alias_keys),
                       'missing_runs': sum(key not in records for key in keys),
                       'failed_or_duplicate_keys': sum(key in records and (len(records[key]) != 1
                            or not records[key][0][0]['analysis_eligible']) for key in keys),
                       'status': 'complete' if keys and valid == len(keys) else 'partial' if keys else 'not_planned'})
    valid_total = sum(s['valid_runs'] for s in stages)
    physical = [row for row in runs if row['planned'] and not row.get('reused_from_run_id')]
    unique_physical = {(row['source_file'], row['source_sha256']) for row in physical}
    valid_physical = {(row['source_file'], row['source_sha256']) for row in physical if row['analysis_eligible']}
    expansion_complete = bool(planned and valid_total == len(planned) and not plan_issues and not read_errors)
    summary = {'schema_version': 2, 'updated_at': timestamp(),
               'status': 'complete' if expansion_complete else 'partial', 'expansion_complete': expansion_complete,
               'expected_runs': len(planned), 'valid_runs': valid_total, 'observed_files': len(unique_physical),
               'observed_comparison_records': len(runs), 'missing_runs': len(missing),
               'failed_or_unverified_files': len({r['source_file'] for r in failures}),
               'unexpected_files': sum(not r['planned'] for r in runs),
               'expected_unique_solver_runs': 18690, 'observed_unique_solver_runs': len(unique_physical),
               'valid_unique_solver_runs': len(valid_physical), 'expected_additional_solver_runs': 5720,
               'valid_additional_solver_runs': sum(s['valid_new_solver_runs'] for s in stages),
               'expected_alias_records': 1430,
               'valid_alias_records': sum(bool(r.get('reused_from_run_id')) and r['analysis_eligible'] for r in runs),
               'stages': stages, 'groups': groups, 'paired_comparisons': paired, 'friedman': friedman,
               'protocol_groups': original.protocol_groups(planned), 'time_profiles': profiles,
               'plan_issues': plan_issues, 'read_errors': read_errors,
               'reuse_audit': {'declared_records': len(reuse_audit),
                              'valid_records': sum(r['status'] == 'valid' for r in reuse_audit),
                              'missing_records': sum(r['status'] == 'missing' for r in reuse_audit),
                              'invalid_records': sum(r['status'] == 'invalid' for r in reuse_audit)},
               'input_manifests': {str(p): sha256(p) for p in (planned_path, new_path, reuse_path, instance_path) if p.exists()},
               'analysis_source_sha256': sha256(Path(__file__)),
               'statistical_protocol': {
                   'unit': 'instance mean over all planned paired seeds',
                   'objective': 'independent validation.distance, raw float64 Euclidean',
                   'positive_improvement': '100*(baseline-target)/baseline',
                   'wilcoxon': 'two-sided raw instance mean differences, zero_method=wilcox, correction=False, method=auto',
                   'holm': 'same six historical families and nineteen hypotheses; adjustment withheld until whole family complete',
                   'bootstrap': 'paired instance percentile bootstrap of mean relative improvement',
                   'bootstrap_repeats': original.BOOTSTRAP_REPEATS, 'bootstrap_seed': original.BOOTSTRAP_SEED,
                   'ties': 'exact equality of float64 instance means', 'scipy_version': scipy.__version__,
                   'numpy_version': np.__version__,
                   'incomplete_policy': 'descriptive aggregates only; p/CI/effect withheld until all expected paired runs complete',
                   'expansion_scope': 'E2/E4/E5 all 122 CVRP + 56 VRPTW; historical panel retained as provenance only',
                   'expansion_timing': 'Scope expansion authorized after historical 35-instance results were observed; not a prospectively registered full-scope experiment.',
                   'e4_inference': 'descriptive paired bootstrap only; no added significance hypothesis',
                   'checkpoint_scope': 'conditional on the single source checkpoint, not source-training replicates',
                   'run_count_definition': 'expected_runs/valid_runs count comparison records; unique_solver_runs exclude E4 aliases referencing E1 files'}}
    fields = ['experiment', 'problem', 'instance', 'method', 'seed', 'group', 'run_id', 'status',
              'validation_valid', 'analysis_eligible', 'planned', 'duplicate', 'objective', 'solver_objective',
              'n_routes', 'elapsed_seconds', 'overshoot_seconds', 'iterations', 'updates', 'budget_mode',
              'budget_value', 'checkpoint_sha256', 'validation_errors', 'failure_reason', 'source_file',
              'source_sha256', *REUSE_FIELDS]
    for filename, rows, columns in [('per_run.csv', runs, fields), ('per_instance.csv', instances, None),
                                   ('per_group.csv', groups, None), ('paired_comparisons.csv', paired, None),
                                   ('friedman.csv', friedman, None), ('failures.csv', failures, None),
                                   ('missing_runs.csv', missing, None), ('reuse_audit.csv', reuse_audit, None),
                                   ('distance_and_routes.csv', route_distance_rows(instances), None),
                                   ('external_paired_descriptive.csv', external_full(instances, metadata, []), None)]:
        supplement.write_csv(out / filename, rows, columns)
    atomic_json(out / 'summary.json', summary)
    if not skip_supplement:
        generate_supplement(out / 'summary.json', summary, instances, metadata, runs,
                            expansion / 'reports' / 'supplement')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion', type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument('--skip-supplement', action='store_true',
                        help='Fast progress snapshot; diagnostics/plots updated in a later full analysis.')
    args = parser.parse_args()
    summary = analyze(args.expansion, args.skip_supplement)
    print(json.dumps({key: summary[key] for key in ('status', 'expansion_complete', 'expected_runs', 'valid_runs',
          'missing_runs', 'valid_additional_solver_runs', 'observed_unique_solver_runs', 'valid_alias_records')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
