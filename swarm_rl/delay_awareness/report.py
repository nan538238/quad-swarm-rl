"""Verify completed pilot artifacts and emit descriptive tables; stdlib only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def verify(root):
    cases, commits, configs, checks = [], set(), [], {}
    for variant, count in [('rules', 560), ('mlp', 80), ('gru', 80)]:
        directory = root / variant
        read = lambda name: json.loads((directory / name).read_text())
        metadata = read('metadata.json')
        assert metadata['status'] == 'complete' and not metadata['smoke']
        assert not metadata['source_status']
        commits.add(metadata['source_commit'])
        spec = read('effective_config.json')
        configs.append(spec)
        hashes = read('artifact_hashes.json')
        for name, expected in hashes.items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
        rows = read('all_cases.json')
        policies = spec['policies'] if variant == 'rules' else [variant]
        expected = set(itertools.product(policies, spec['geometries'], spec['delay_steps'], spec['scenario_seeds']))
        keys = [(r['policy'], r['geometry'], r['delay_steps'], r['scenario_seed']) for r in rows]
        assert len(rows) == count and len(set(keys)) == count and set(keys) == expected
        assert all(r['status'] == 'complete' for r in rows)
        if variant != 'rules':
            training = read('training_summary.json')
            assert training['steps'] == spec['training_steps'] == 16384 and training['reload_identical']
            assert training['initial_sha256'] != training['final_sha256']
        checks[variant] = {'cases': len(rows), 'verified_hashes': len(hashes)}
        cases.extend(rows)
    assert len(commits) == 1 and configs[0] == configs[1] == configs[2]
    index = {(r['policy'], r['geometry'], r['delay_steps'], r['scenario_seed']): r for r in cases}
    fields = ['success', 'collision', 'failure_inclusive_completion_cost']
    paired = {}
    for r in cases:
        k = (r['geometry'], r['scenario_seed'])
        state = json.dumps(r['initial_state'], sort_keys=True)
        assert paired.setdefault(k, state) == state
        if r['policy'] in ['nominal', 'reactive_oracle', 'predictive_oracle']:
            base = index[r['policy'], r['geometry'], 0, r['scenario_seed']]
            assert all(r[f] == base[f] for f in fields)
        if r['delay_steps'] == 0 and r['policy'].startswith(('reactive_', 'predictive_')):
            base = index[r['policy'].split('_')[0] + '_raw', r['geometry'], 0, r['scenario_seed']]
            assert all(r[f] == base[f] for f in fields)
    rows = []
    for p, g, d in sorted({(r['policy'], r['geometry'], r['delay_steps']) for r in cases}):
        group = [r for r in cases if (r['policy'], r['geometry'], r['delay_steps']) == (p, g, d)]
        rows.append({'policy': p, 'geometry': g, 'delay_ms': d * 100, 'n': len(group),
                     'successes': sum(r['success'] for r in group), 'collisions': sum(r['collision'] for r in group),
                     'mean_cost_s': sum(r['failure_inclusive_completion_cost'] for r in group) / len(group)})
    result = {'source_commit': next(iter(commits)), 'checks': checks,
              'paired_initial_conditions': True, 'zero_delay_equivalence': True,
              'oracle_and_nominal_delay_invariance': True, 'rows': rows,
              'gates': json.loads((root / 'rules/gates.json').read_text())}
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    table = ['# Verified descriptive results', '', f"Source: `{result['source_commit']}`. 720 cases; 8 scenario seeds per cell.",
             'Failure-inclusive cost assigns 20 seconds to collision or timeout; it is not successful flight time.', '',
             '| Geometry | Policy | Delay (ms) | Success / 8 | Collisions / 8 | Mean cost (s) |',
             '|---|---|---:|---:|---:|---:|']
    table += [f"| {r['geometry']} | {r['policy']} | {r['delay_ms']} | {r['successes']} | {r['collisions']} | {r['mean_cost_s']:.3f} |" for r in rows]
    (root / 'verified_tables.md').write_text('\n'.join(table) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'gates']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root)
