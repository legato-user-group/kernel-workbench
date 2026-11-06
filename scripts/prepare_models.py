"""Prepare a checkpoint configuration; model code lives in models/qwen3_5/."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def prepare(destination=None, config=None):
    source=Path(config) if config else ROOT/'models/qwen3_5/config.json'
    destination=Path(destination or ROOT/'models/checkpoints/qwen35_27b')
    value=json.loads(source.read_text())
    if value['text_config']['model_type']!='qwen3_5_text':
        raise ValueError('Expected a Qwen3.5 text model configuration')
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'config.json'
    if target.resolve()!=source.resolve():shutil.copyfile(source,target)
    return {'directory':str(destination),'config_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.config)))
