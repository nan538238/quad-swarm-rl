"""Verify completed pilot artifacts and summarize paired cases without tuning."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def case_key(row):
    return row['scenario_seed'], row['geometry'], row['profile']


def summarize(root):
    all_cases, initialization, commits, configs = [], [], [], []
    for variant in ('blind', 'flag', 'history'):
        run = root / variant
        metadata = read(run / 'metadata.json')
        config = read(run / 'effective_config.json')
        training = read(run / 'training_summary.json')
        if metadata['status'] != 'complete' or metadata['variant'] != variant or metadata['engineering_smoke']:
            raise ValueError(f'Incomplete/incorrect run: {variant}')
        for name, digest in read(run / 'artifact_hashes.json').items():
            if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Artifact hash mismatch: {variant}/{name}')
        if training['steps'] != config['training_steps'] or not training['reload_identical']:
            raise ValueError('Wrong training budget or reload failure')
        initialization.append(training['initial_sha256'])
        commits.append(metadata['source_commit'])
        configs.append(config)
        cases = read(run / 'evaluation_cases.json')
        expected = {(s, g, p) for s in config['evaluation_scenario_seeds']
                    for g in config['geometries'] for p in config['evaluation_profiles']}
        policies = (variant, 'nominal', 'reactive') if variant == 'blind' else (variant,)
        if len(cases) != len(expected) * len(policies):
            raise ValueError('Missing or unexpected evaluation cases')
        for policy in policies:
            selected = [c for c in cases if c['policy'] == policy]
            if len(selected) != len(expected) or {case_key(c) for c in selected} != expected:
                raise ValueError('Unpaired or duplicated cases')
        if any(c['split'] != 'pilot_evaluation' for c in cases):
            raise ValueError('Incorrect split label')
        all_cases.extend(cases)
    if len(set(initialization)) != 1 or len(set(commits)) != 1 or any(c != configs[0] for c in configs):
        raise ValueError('Different initial weights, source commits or protocols')
    rows = []
    for geometry in configs[0]['geometries']:
        for policy in ('nominal', 'reactive', 'blind', 'flag', 'history'):
            cases = [c for c in all_cases if c['geometry'] == geometry and c['policy'] == policy]
            rows.append({'geometry': geometry, 'policy': policy, 'cases': len(cases),
                'successes': sum(c['scenario_success'] for c in cases),
                'collisions': sum(c['collision'] for c in cases),
                'takeover_collisions': sum(c['collision_phase'] == 'takeover' for c in cases),
                'handback_collisions': sum(c['collision_phase'] == 'handback' for c in cases),
                'inspection_successes': sum(c['check_completed'] for c in cases),
                'recovery_successes': sum(c['recovery_success'] for c in cases),
                'failure_inclusive_recovery_cost': mean(c['failure_inclusive_recovery_cost'] for c in cases)})
    comparisons = []
    baseline = {case_key(c): c for c in all_cases if c['policy'] == 'blind'}
    for policy in ('flag', 'history'):
        for geometry in configs[0]['geometries']:
            cases = [c for c in all_cases if c['policy'] == policy and c['geometry'] == geometry]
            common = [(c, baseline[case_key(c)]) for c in cases
                      if c['scenario_success'] and baseline[case_key(c)]['scenario_success']]
            comparisons.append({'policy': policy, 'geometry': geometry, 'reference': 'blind',
                'common_success_cases': len(common),
                'mean_lag_delta_m_s': mean(c['integrated_patrol_lag'] - b['integrated_patrol_lag']
                                         for c, b in common) if common else None,
                'mean_recovery_delta_s': mean(c['recovery_time'] - b['recovery_time']
                                            for c, b in common) if common else None})
    return {'source_commit': commits[0], 'training_steps_per_variant': configs[0]['training_steps'],
        'training_seed_count': 1, 'artifact_hashes_verified': True,
        'initialization_identical': True, 'paired_cases_verified': True,
        'split': 'pilot_evaluation', 'rows': rows, 'common_success_comparisons': comparisons,
        'limits': 'Single-seed engineering pilot, four scenario seeds and scripted operators. '
                  'No convergence, real-human, statistical superiority or novelty claim.'}


def markdown(summary):
    lines = ['# 接管适应工程筛查', '',
        f"源码提交：`{summary['source_commit']}`。每组训练 {summary['training_steps_per_variant']} 步，训练种子 1 个。",
        '各几何包含 4 个配对场景种子 × 5 种操作者风格；其中两种风格未用于训练。',
        '这是开发用工程筛查，不是最终测试，也未采集真实人工操作结果。', '',
        '| 场景 | 策略 | 成功/总数 | 碰撞 | 接管中碰撞 | 交还后碰撞 | 巡检完成 | 恢复完成 | 失败计10秒的恢复代价 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['rows']:
        lines.append(f"| {row['geometry']} | {row['policy']} | {row['successes']}/{row['cases']} | "
            f"{row['collisions']} | {row['takeover_collisions']} | {row['handback_collisions']} | "
            f"{row['inspection_successes']} | {row['recovery_successes']} | {row['failure_inclusive_recovery_cost']:.3f} |")
    lines.extend(['', '滞后积分仅在与 blind 配对、双方均任务成功的案例上比较，避免提前碰撞造成虚假低滞后。', '',
        '| 场景 | 策略 vs blind | 共同成功数 | 滞后积分差 m·s | 恢复时间差 s |',
        '|---|---|---:|---:|---:|'])
    for row in summary['common_success_comparisons']:
        fmt = lambda x: '无可比案例' if x is None else f'{x:.3f}'
        lines.append(f"| {row['geometry']} | {row['policy']} | {row['common_success_cases']} | "
                     f"{fmt(row['mean_lag_delta_m_s'])} | {fmt(row['mean_recovery_delta_s'])} |")
    lines.extend(['', '负差值表示该指标较低；单种子差异不能用于宣称统计显著或算法普遍有效。',
        '已核查：三组初始权重相同、训练预算相同、评估案例完整配对、模型重载一致、原始文件哈希完整。', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.output / 'RESULTS.md').write_text(markdown(summary))
    print(markdown(summary))


if __name__ == '__main__':
    main()
