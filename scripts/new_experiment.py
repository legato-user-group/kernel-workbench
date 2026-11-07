"""Create experiment settings and an empty dated directory; never run a GPU job."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_workbench.configuration import ExperimentConfig, load_experiment
from kernel_workbench.paths import ROOT

TEMPLATE = ROOT / 'templates/experiment/config.json'


def create_experiment(name, directory, run_date):
    if date.fromisoformat(run_date).isoformat() != run_date:
        raise ValueError('Use a YYYY-MM-DD date')
    # The raw reader adds the date and length to the folder-derived run ID.
    if len(f'{name}-{run_date}') > 40:
        raise ValueError('Experiment name must be at most 29 characters')
    config = ExperimentConfig(**{**json.loads(TEMPLATE.read_text()), 'name': name})
    directory = Path(directory).resolve()
    settings = directory / name / 'config.json'
    output = settings.parent / run_date
    if settings.exists():
        config = load_experiment(settings)
        if config.name != name:
            raise ValueError('Existing config name differs from its experiment folder')
    if output.exists():
        raise FileExistsError(f'Run directory already exists: {output}')
    settings.parent.mkdir(parents=True, exist_ok=True)
    if not settings.exists():
        with settings.open('x') as stream:
            stream.write(json.dumps(config.model_dump(), ensure_ascii=False, indent=2) + '\n')
    output.mkdir()
    return {'config': str(settings), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='Experiment folder name, at most 29 ASCII letters/digits/_/-')
    parser.add_argument('--date', default=date.today().isoformat(), help='YYYY-MM-DD')
    parser.add_argument('--directory', type=Path, default=ROOT / 'experiments')
    args = parser.parse_args()
    try:
        result = create_experiment(args.name, args.directory, args.date)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
