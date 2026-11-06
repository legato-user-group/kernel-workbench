"""Summarize an existing experiment directory; reuse this command for all runs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_workbench.evaluation_summary import summarize
from kernel_workbench.evaluation_report import render_markdown

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    output = args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'report.md').write_text(render_markdown(report))
    print(json.dumps({'complete':report['complete'],'timing_rows':len(report['rows']),
                      'checks':len(report['correctness']['checks'])}))
