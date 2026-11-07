"""Collect a warmed Nsight timeline and a bounded hardware-counter diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Keep this CLI's name from shadowing Python's stdlib profile module when
# torch imports cProfile. Import shared scripts from the repository namespace.
sys.path = [str(ROOT), *[p for p in sys.path if Path(p).resolve() != ROOT / 'scripts']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--model', default='qwen3_5')
    parser.add_argument('--output', type=Path, help='New evidence directory; defaults to <directory>/profile')
    parser.add_argument('--collect', choices=['all', 'timeline', 'counters'], default='all')
    parser.add_argument('--counter-sampling', choices=['prefix', 'configs'], default='configs')
    parser.add_argument('--counter-phase', choices=['prefill', 'decode', 'unknown'])
    parser.add_argument('--worker', choices=['timeline', 'counters'])
    args = parser.parse_args()
    output = (args.output or args.directory.resolve() / 'profile').resolve()
    output.mkdir(exist_ok=True)
    if args.worker:
        from kernel_workbench.remote_entry import Trace
        from scripts.profile_workload import run
        result = run(args.directory, Trace(), model_name=args.model, output=output,
                     single_decode=args.worker == 'counters', counter_phase=args.counter_phase)
        result['counter_phase'] = args.counter_phase
        name = f'counters-{args.counter_phase}' if args.worker == 'counters' and args.counter_phase else args.worker
        (output / f'{name}.json').write_text(json.dumps(result, indent=2))
        return
    if (output / 'status.json').exists():
        raise FileExistsError('Profile evidence already exists; use a new evaluation directory')
    files = [Path(__file__), Path(__file__).with_name('profile_workload.py'), ROOT / 'kernel_workbench/analysis.py',
             ROOT / 'kernel_workbench/remote_entry.py']
    (output / 'source_manifest.json').write_text(json.dumps({
        'files': {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    }, indent=2))
    status = {'commands': [], 'warnings': [], 'timing_source': 'separate_nsight_collection'}
    def call(command, name):
        print('+ ' + ' '.join(command), flush=True)
        with (output / name).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status['commands'].append({'args': command, 'returncode': process.returncode, 'log': name})
        if process.returncode:
            status['warnings'].append(f'{name}: command failed ({process.returncode})')
        return process.returncode
    entry = [sys.executable, str(Path(__file__).resolve()), str(args.directory.resolve()),
             '--model', args.model, '--output', str(output)]
    timeline_ok = False
    for tool in ('nsys', 'ncu'):
        if shutil.which(tool):
            call([tool, '--version'], tool + '-version.txt')
        else:
            status['warnings'].append(f'{tool} unavailable')
    try:
        if shutil.which('nsys') and args.collect in ('all', 'timeline'):
            timeline_ok = call(['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                  '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--kill=none',
                  '-o', str(output / 'trace'), *entry, '--worker', 'timeline'], 'nsys.log') == 0
            if (output / 'trace.nsys-rep').is_file():
                call(['nsys', 'export', '--type=sqlite', '--output=' + str(output / 'trace.sqlite'),
                      str(output / 'trace.nsys-rep')], 'nsys-export.log')
        if shutil.which('ncu') and args.collect in ('all', 'counters'):
            from kernel_workbench.analysis import flop_metric_plan
            call(['ncu', '--query-metrics', '--query-metrics-mode', 'base'], 'available-metrics.txt')
            environment = json.loads((args.directory / 'results/environment.json').read_text())
            plan = flop_metric_plan((output / 'available-metrics.txt').read_text(), environment['gpu'])
            status['flop_metric_plan'] = plan
            wanted = ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum']
            wanted += [name for name in plan['required'] if name not in plan['missing']]
            phases = ['prefill', 'decode', 'unknown'] if args.counter_sampling == 'configs' else [None]
            for phase in phases:
                name = f'counters-{phase}' if phase else 'counters'
                status.setdefault('phase_sources', {})[name] = {
                    str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
                filters = (['--filter-mode', 'per-launch-config',
                            '--kernel-name-base', 'demangled', '--kernel-id', ':::1']
                           if phase else ['--launch-count', '32'])
                call(['ncu', '--profile-from-start', 'off', '--target-processes', 'all',
                      *filters, '--cache-control', 'none', '--clock-control', 'none',
                      '--metrics', ','.join(wanted), '-o', str(output / name),
                      *entry, '--worker', 'counters', *(['--counter-phase', phase] if phase else [])], name + '.log')
                if (output / f'{name}.ncu-rep').is_file():
                    call(['ncu', '--import', str(output / f'{name}.ncu-rep'), '--page', 'raw', '--csv',
                          '--print-units', 'base', '--rename-kernels', 'off'], name + '.csv')
    finally:
        status['timeline_complete'] = timeline_ok and all((output / n).is_file() for n in ('timeline.json', 'trace.sqlite'))
        status['counter_report_available'] = any(output.glob('counters*.csv'))
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    if args.collect != 'counters' and not status['timeline_complete']:
        raise RuntimeError(f'No completed timeline; inspect {output}/status.json and nsys.log')
    if args.collect == 'counters' and not status['counter_report_available']:
        raise RuntimeError(f'No counter report; inspect {output}/status.json and ncu.log')


if __name__ == '__main__':
    main()
