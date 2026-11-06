"""Run a shared model evaluation from an experiment config; never allocate a GPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_workbench.evaluation_runner import prepare_evaluation, execute_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: experiment directory / current date')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--new-tokens', type=int)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    plan = prepare_evaluation(args.experiment, args.models, args.output, warmup=args.warmup,
                              repeats=args.repeats, new_tokens=args.new_tokens, resume=args.resume)
    if args.prepare_only:
        print(json.dumps(plan))
    else:
        execute_evaluation(plan, resume=args.resume, timeout=args.timeout)


if __name__ == '__main__':
    main()
