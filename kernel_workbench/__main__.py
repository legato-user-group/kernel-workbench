"""Agent-side file tools and a passive results server, without remote execution."""
import argparse
import json
from pathlib import Path
from .paths import ROOT


def main():
    parser = argparse.ArgumentParser(description='Kernel Workbench: agent-collected result storage and visualization')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('api', 'serve'):
        server = sub.add_parser(name)
        server.add_argument('--port', type=int, default=8765)
    prepare = sub.add_parser('prepare', help='Create source bundle and remote script, without running them')
    prepare.add_argument('--runner', default=str(ROOT / 'runners/tiny_decoder.py'))
    prepare.add_argument('--config')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--remote-dir')
    analyze = sub.add_parser('analyze', help='Normalize artifacts collected by the agent')
    analyze.add_argument('--artifacts', required=True)
    analyze.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command in ('api', 'serve'):
        import uvicorn
        uvicorn.run('kernel_workbench.app:app', host='127.0.0.1', port=args.port)
        return
    try:
        from .collection import prepare as prepare_files, analyze_artifacts
        if args.command == 'prepare':
            config = json.loads(Path(args.config).read_text()) if args.config else {}
            config.setdefault('runner', Path(args.runner).stem)
            value = prepare_files(args.runner, config, args.output, args.remote_dir)
        else:
            value = analyze_artifacts(args.artifacts)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
            value = {'result': str(output.resolve()), 'coverage': value['coverage'], 'timings': value['timings']}
        print(json.dumps(value, ensure_ascii=False))
    except (ValueError, KeyError, FileNotFoundError, OSError) as error:
        print(json.dumps({'error': str(error)}))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
