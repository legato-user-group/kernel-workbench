"""Extract and execute a prepared source bundle in an existing GPU environment."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        archive.extractall(args.directory, filter='data')
    subprocess.run([sys.executable, str(args.directory / 'bootstrap.py'), str(args.directory)], check=True)


if __name__ == '__main__':
    main()
