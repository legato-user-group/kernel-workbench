"""Inspect dashboard data from experiment evidence without writing files."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_workbench.raw_experiments import RawExperiments
from kernel_workbench.paths import DATA

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=DATA)
    args=parser.parse_args()
    source=RawExperiments(args.directory)
    print(json.dumps({'measurements':len(source.records()),'experiments':len(source.attempts()),
                      'issues':source.issues},ensure_ascii=False,indent=2))
