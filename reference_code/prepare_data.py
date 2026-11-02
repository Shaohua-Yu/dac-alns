"""Archive inputs and generate independent, reproducible experiment splits.

Run from any directory. Never reads benchmark results to choose a generated
instance. The generator function is compiled verbatim from the source snapshot.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from validate_routes import read_instance, validate_routes

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
SNAPSHOT = ROOT / 'original_source'
VRPTW_SOURCE = Path('C:/Users/Administrator/Desktop/DAC-ALNS-CVRP--update-7-15-2/CAPP/pro_data/VRPTW-Solomn')
MODEL_SOURCES = [
    ('VRPTW_online', 'DAC_VRPTW_online.py', Path('C:/Users/Administrator/Desktop/VRPTW_RD/DAC_VRPTW_new_v4/dac_vrptw_pretrained.pth')),
    ('VRPTW_transfer', 'DAC_VRPTW_transfer.py', Path('C:/Users/Administrator/Desktop/VRPTW_RD/DAC_VRPTW_new_v1_2/trained_model_v1_2.pth')),
    ('CVRP_transfer', 'DAC_CVRP_transfer.py', Path('C:/Users/Administrator/Desktop/CVRP_RD/X/DAC_CVRP_new_v1_9/trained_model_v1_9.pth')),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def manifest(name, rows):
    dump(ROOT / 'configs' / f'{name}.json', rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / 'configs' / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cvrp_group(n):
    return 'CVRP-S' if n < 100 else 'CVRP-M1' if n < 200 else 'CVRP-M2' if n < 350 else 'CVRP-L'


def make_row(path, problem, split, source, *, panel=False, group=None, **extra):
    parsed = read_instance(path, problem)
    digest = sha256(path)
    return {'problem': problem, 'id': path.stem, 'instance_id': path.stem,
            'group': group or cvrp_group(parsed['n']), 'n': parsed['n'],
            'n_customers': parsed['n'], 'path': str(path.resolve()),
            'relative_path': path.relative_to(ROOT).as_posix(), 'hash': digest,
            'sha256': digest, 'panel': panel, 'split': split, 'source': str(source), **extra}


def copy_benchmarks():
    panel_file = ROOT / 'configs' / 'DAC_ALNS_35例诊断面板_20261008.csv'
    with panel_file.open(encoding='utf-8-sig', newline='') as handle:
        panel_rows = list(csv.DictReader(handle))
    panel = {(r['problem'], r['instance_id']): r for r in panel_rows}
    sources = [('CVRP', p) for p in sorted((WORK / '算例集' / 'CVRP').glob('*/*.csv'))]
    sources += [('VRPTW', p) for p in sorted(VRPTW_SOURCE.glob('*.csv'))]
    rows = []
    for problem, source in sources:
        target = ROOT / 'data' / 'test' / problem / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        key = (problem, source.stem)
        if key in panel and sha256(target) != panel[key]['sha256']:
            raise RuntimeError(f'Preselected panel input hash has changed: {source}')
        group = None if problem == 'CVRP' else ('RC' if source.stem.startswith('RC') else source.stem[0])
        rows.append(make_row(target, problem, 'test', source, panel=key in panel, group=group))
    assert Counter(row['problem'] for row in rows) == {'CVRP': 122, 'VRPTW': 56}
    assert sum(row['panel'] for row in rows) == 35
    manifest('instances', rows)
    return rows


def load_generator(problem):
    path = SNAPSHOT / f'run_DAC_{problem}_transfer.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'generate_single_dynamic_instance')
    module = ast.Module(body=[function], type_ignores=[])
    environment = {'random': random, 'np': np, 'csv': csv, 'math': math}
    exec(compile(module, str(path), 'exec'), environment)
    return environment[function.name], path


def generated_instance(problem, split, index, seed, n, dist_type=None):
    generator, source = load_generator(problem)
    if problem == 'CVRP':
        probe = random.Random(seed)
        probe.randint(n, n)
        dist_type = probe.choice(['R', 'C', 'RC'])
    name = f'{split}_{problem}_{dist_type}_n{n}_{index:03d}_seed{seed}.csv'
    path = ROOT / 'data' / split / problem / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if problem == 'VRPTW':
        generator(path, node_range=(n, n), dist_type=dist_type, seed=seed)
    else:
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            generator(path, node_range=(n, n))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
    return make_row(path, problem, split, source, group=cvrp_group(n) if problem == 'CVRP' else dist_type,
                    generator_sha256=sha256(source), seed=seed, dist_type=dist_type,
                    node_min=n, node_max=n)


def generate_splits():
    calibration = []
    index = 0
    for sizes in [(20, 40, 60, 80, 99), (100, 125, 150, 175, 199),
                  (200, 237, 275, 312, 349), (350, 387, 425, 462, 499)]:
        for n in sizes:
            index += 1
            calibration.append(generated_instance('CVRP', 'calibration', index, 2026100800+index, n))
    for type_index, kind in enumerate(('C', 'R', 'RC')):
        for i in range(1, 6):
            calibration.append(generated_instance('VRPTW', 'calibration', i,
                                2026101800 + 100 * type_index + i, 100, kind))
    pretrain = []
    validation = []
    for type_index, kind in enumerate(('C', 'R', 'RC')):
        for i in range(1, 11):
            pretrain.append(generated_instance('VRPTW', 'pretrain', i,
                            2026110000 + 100 * type_index + i, 100, kind))
            validation.append(generated_instance('VRPTW', 'validation', i,
                              2026120000 + 100 * type_index + i, 100, kind))
    # Independent CVRP fixed validation pool, same strata as calibration.
    for i, row in enumerate([r for r in calibration if r['problem'] == 'CVRP'], start=1):
        validation.append(generated_instance('CVRP', 'validation', i, 2026130000+i, row['n']))
    manifest('calibration_instances', calibration)
    manifest('pretrain_instances', pretrain)
    manifest('validation_instances', validation)
    return calibration, pretrain, validation


def generate_transfer_sources():
    """Full source pools from the original distributions; no dropped tail batch."""
    rows = []
    for problem, count, node_range, base_seed in [
        ('CVRP', 1000, (100, 150), 2026140000),
        ('VRPTW', 500, (100, 100), 2026150000),
    ]:
        generator, source = load_generator(problem)
        for index in range(1, count+1):
            seed = base_seed + index
            probe = random.Random(seed)
            n = probe.randint(*node_range)
            dist_type = probe.choice(['R', 'C', 'RC'])
            path = ROOT / 'data' / 'transfer_source' / problem / f'transfer_{problem}_{index:04d}_seed{seed}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            if problem == 'VRPTW':
                # None preserves the original generator's internal type draw.
                generator(path, node_range=node_range, dist_type=None, seed=seed)
            else:
                py_state, np_state = random.getstate(), np.random.get_state()
                try:
                    random.seed(seed)
                    np.random.seed(seed)
                    generator(path, node_range=node_range)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
            rows.append(make_row(path, problem, 'transfer_source', source,
                        group=cvrp_group(n) if problem == 'CVRP' else dist_type,
                        generator_sha256=sha256(source), seed=seed, dist_type=dist_type,
                        node_min=node_range[0], node_max=node_range[1],
                        original_steps_per_epoch=max(100, max(50*n, 2000)//10)))
    manifest('transfer_source_instances', rows)
    manifest('source_training_instances', rows)
    return rows


def shape_dictionary(state):
    return {key: list(value.shape) if hasattr(value, 'shape') else type(value).__name__
            for key, value in state.items()}


def audit_models():
    import torch
    output = []
    for name, script, source in MODEL_SOURCES:
        destination = ROOT / 'models' / 'archive' / f'{name}_{source.name}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        item = {'name': name, 'source': str(source), 'archive_path': str(destination),
                'sha256': sha256(destination), 'size_bytes': destination.stat().st_size,
                'source_file_location_known': True, 'training_provenance': 'unknown',
                'usable_as_verified_transfer_source': False,
                'provenance_note': 'File and tensor metadata are recoverable. Training/validation inputs, seeds, selection history, and exact generating source version are not established by a weight file alone.',
                'current_architecture_script': str(SNAPSHOT / script),
                'current_architecture_script_sha256': sha256(SNAPSHOT / script)}
        if name == 'CVRP_transfer':
            item['global_step_caution'] = ('Current CVRP transfer training overwrites global_step with the current batch count instead of adding resume_global_step. The archived metadata value cannot establish lifetime source-training transitions.')
        try:
            checkpoint = torch.load(destination, map_location='cpu', weights_only=True)
            state = checkpoint.get('agent_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            item['safe_weights_only_load'] = True
            item['container_keys'] = list(checkpoint.keys())
            item['state_shapes'] = shape_dictionary(state)
            item['metadata'] = {key: value for key, value in checkpoint.items()
                                if key not in state and key not in {'agent_state_dict'}
                                and isinstance(value, (str, int, float, bool, type(None), list, tuple))}
            try:
                sys.path.insert(0, str(SNAPSHOT))
                spec = importlib.util.spec_from_file_location(f'audit_{name}', SNAPSHOT / script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                agent = module.DACAgent()
                expected = shape_dictionary(agent.state_dict())
                item['missing_keys'] = sorted(set(expected) - set(state))
                item['unexpected_keys'] = sorted(set(state) - set(expected))
                item['shape_mismatches'] = {key: {'archive': item['state_shapes'][key], 'current': expected[key]}
                                          for key in set(expected) & set(state)
                                          if item['state_shapes'][key] != expected[key]}
                item['key_and_shape_compatible'] = not any(item[key] for key in ('missing_keys', 'unexpected_keys', 'shape_mismatches'))
                if item['key_and_shape_compatible']:
                    agent.load_state_dict(state, strict=True)
                    item['strict_state_dict_load'] = True
                else:
                    item['strict_state_dict_load'] = False
            except Exception as exc:
                item['compatibility_check_error'] = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            item['safe_weights_only_load'] = False
            item['load_error'] = f'{type(exc).__name__}: {exc}'
        output.append(item)
    dump(ROOT / 'reports' / 'checkpoint_audit.json', output)
    return output


def audit_old_training_evidence():
    findings=[]
    for name,script,path in MODEL_SOURCES:
        directory=path.parent
        entry={'model':name,'model_directory':str(directory),
               'validation_run_directories':sorted(p.name for p in directory.glob('val_run_*') if p.is_dir()),
               'remaining_temporary_training_instances':len(list((directory/'_temp_train_instances').glob('*.csv'))),
               'pretrain_wang_instances':len(list((directory/'训练集wang').glob('*.csv'))),
               'source_training_seed_manifest_found':False,
               'original_training_pool_reconstructable':False}
        candidates=[p for p in directory.rglob('*') if p.is_file() and
                    (p.suffix.lower() in {'.log','.txt'} or 'manifest' in p.name.lower() or 'config' in p.name.lower())]
        entry['candidate_provenance_files']=[{'path':str(p),'sha256':sha256(p)} for p in candidates]
        findings.append(entry)
    report={'findings':findings,
            'CVRP_current_runner_validation_round_upper_bound':20,
            'CVRP_archived_validation_directories':35,
            'CVRP_validation_count_interpretation':'35 recorded val_run directories exceed floor(1000/16)/3=20 checks under the current runner; files alone cannot identify which source version and training cycle generated the selected weights.',
            'old_generator_retention':'Current runners delete/overwrite temporary generated training instances between batches. Existing final temporary data cannot reconstruct earlier batches.',
            'old_truncation':'Current runner integer division schedules 992 of 1000 CVRP and 496 of 500 VRPTW source instances. New source manifests retain all 1000 and 500.',
            'new_source_protocol':'Use independently generated archived source pools and explicitly report the frozen final-checkpoint training protocol. Do not claim reconstructed historical training provenance.'}
    dump(ROOT/'reports'/'training_provenance_audit.json',report)
    return report


def main():
    tests = copy_benchmarks()
    calibration, pretrain, validation = generate_splits()
    transfer = generate_transfer_sources()
    all_rows = tests + calibration + pretrain + validation + transfer
    hashes = [row['sha256'] for row in all_rows]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('Duplicate instance byte hashes across splits')
    # Verify generated input feasibility with one independent route per customer.
    # Fleet count is intentionally not a hard constraint in this protocol.
    generated_checks = []
    for row in calibration + pretrain + validation + transfer:
        parsed = read_instance(row['path'], row['problem'])
        result = validate_routes(row['path'], row['problem'], [[node] for node in parsed['nodes'] if node])
        generated_checks.append({'id': row['id'], 'valid': result['valid'], 'errors': result['errors']})
    assert all(row['valid'] for row in generated_checks)
    models = audit_models()
    audit_old_training_evidence()
    report = {
        'test_counts': dict(Counter(row['problem'] for row in tests)),
        'test_group_counts': dict(Counter(row['group'] for row in tests)),
        'panel_count': sum(row['panel'] for row in tests),
        'split_counts': dict(Counter(row['split'] for row in all_rows)),
        'all_instance_byte_hashes_unique': True,
        'generator_rule': 'Verbatim AST-extracted generate_single_dynamic_instance functions from archived run_DAC_CVRP_transfer.py and run_DAC_VRPTW_transfer.py. CVRP global Python/NumPy streams are seeded and restored; VRPTW uses its local seed parameter. No benchmark perturbation or result-dependent generation.',
        'calibration_seed_rule': 'CVRP 2026100801..2026100820; VRPTW base 2026101800 + C/R/RC offset 0/100/200 + 1..5.',
        'pretrain_seed_rule': 'VRPTW base 2026110000 + C/R/RC offset 0/100/200 + 1..10; same 30 files for all compared controllers.',
        'validation_seed_rule': 'VRPTW base 2026120000 + C/R/RC offset 0/100/200 + 1..10; CVRP 2026130001..2026130020.',
        'transfer_source_seed_rule': 'CVRP 2026140001..2026141000; VRPTW 2026150001..2026150500. Original generator chooses n and C/R/RC; no truncation to multiples of batch 16.',
        'transfer_source_counts': dict(Counter(row['problem'] for row in transfer)),
        'original_transfer_source_transitions_if_two_epochs': {problem: 2*sum(row['original_steps_per_epoch'] for row in transfer if row['problem'] == problem) for problem in ('CVRP','VRPTW')},
        'calibration_customers': [[20,40,60,80,99],[100,125,150,175,199],[200,237,275,312,349],[350,387,425,462,499]],
        'validation_use': 'Independent validation only. Test and diagnostic panel are never used to choose hyperparameters, time budgets, or checkpoints.',
        'generated_singleton_feasibility': generated_checks,
        'checkpoint_summary': [{k: v for k, v in row.items() if k not in {'state_shapes', 'container_keys'}} for row in models],
        'limits': ['Copied benchmark CSV provenance is the local archived file; upstream conversion history is not newly inferred.',
                   'Byte-hash distinctness establishes nonidentical files; generated data are additionally independent by construction and seed.',
                   'Recovered checkpoint training provenance remains unknown unless separately established. Do not describe these as verified source-trained checkpoints.'],
    }
    dump(ROOT / 'reports' / 'data_audit.json', report)
    print(json.dumps({key: report[key] for key in ['test_counts','test_group_counts','panel_count','split_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
