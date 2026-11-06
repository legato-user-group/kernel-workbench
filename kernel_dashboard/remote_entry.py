"""Executed as a script inside the GPU runtime, under nsys or ncu."""
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

class Trace:
    @contextmanager
    def range(self, label):
        import torch
        torch.cuda.nvtx.range_push(label)
        try: yield
        finally: torch.cuda.nvtx.range_pop()
    def phase(self, name): return self.range('phase:'+name)
    def module(self, name): return self.range('module:'+name)
    def step(self, number): return self.range('step:'+str(number))

if __name__ == '__main__':
    config=json.loads(Path(sys.argv[1]).read_text())
    spec=importlib.util.spec_from_file_location('runner',sys.argv[2])
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    result=runner.run(config,Trace())
    result['model']=runner.describe(config)
    Path(sys.argv[3]).write_text(json.dumps(result))
