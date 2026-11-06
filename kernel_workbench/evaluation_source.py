"""Load raw evaluation evidence and its recorded correctness gates once."""
import json
from pathlib import Path

from models.common.timing import summarize_samples

PHASES = ('prefill_forward', 'engine_ttft', 'decode', 'generation')
PLAIN = 'int8_plain_cudagraph'
MTP = 'int8_mtp3_cudagraph'
VERIFY = 'int8_mtp_parallel_verify4'
ENGINES = (PLAIN, MTP)


def normalize_timing(timing):
    result = dict(timing)
    for phase in PHASES:
        stage = timing[phase]
        if stage.get('sample_count', len(stage['samples'])) != len(stage['samples']):
            raise ValueError(f'Raw sample count mismatch: {phase}')
        result[phase] = {**stage, **summarize_samples(stage['samples'], stage.get('source', 'unavailable'))}
    return result


class EvaluationSource:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.registration = self.read('registration.json', {})
        self.contract = self.read('results/contract.json')
        if self.contract is None:
            raise ValueError('No results/contract.json found')
        self.correctness = self.read('results/correctness.json', {'checks': []})
        self.checks = self.correctness['checks']
        self.engines = tuple(self.contract.get('engines', ENGINES))
        self.check_engines = tuple(self.contract.get('check_engines', (PLAIN, VERIFY)))
        self.reset_checks = [c for c in self.checks if c.get('reset_repeat')]
        self.reset_ok = (set(self.check_engines) <= {c['engine'] for c in self.reset_checks}
                         and all(c['passed'] and c.get('bitwise_equal_to_first') for c in self.reset_checks))
        self.parity = {p['input_tokens']: p for p in self.read('recovered_parity.json', [])}
        self.timings = {}
        engines = (*self.engines, 'hf_bf16') if self.contract.get('hf_timing_requested') else self.engines
        for length in self.contract['lengths']:
            parity = self.read(f'results/mtp_parity_{length}.json')
            if parity is not None:
                self.parity[length] = parity
            for engine in engines:
                timing = self.read(self.timing_path(engine, length))
                if timing and all(phase in timing for phase in PHASES):
                    self.timings[engine, length] = normalize_timing(timing)

    def read(self, name, default=None):
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Evidence paths must remain inside the experiment directory')
        return json.loads(path.read_text()) if path.is_file() else default

    @staticmethod
    def timing_path(engine, length):
        return f'results/timing_{engine}_{length}.json'

    def check(self, engine, length):
        target = VERIFY if engine == MTP else engine
        return next((c for c in self.checks if c['engine'] == target and c['input_tokens'] == length
                     and not c.get('reset_repeat')), None)

    def stable(self, engine, length):
        timing = self.timings.get((engine, length))
        return bool(timing) and len({s['output_ids_sha256'] for phase in ('decode', 'generation')
                                     for s in timing[phase]['samples']}) == 1

    def quality_ok(self, engine, length):
        check = self.check(engine, length)
        parity_ok = engine != MTP or self.parity.get(length, {}).get('matches_plain_int8', False)
        return bool(check and check['passed'] and self.reset_ok and self.stable(engine, length) and parity_ok)

    def correctness_status(self, length):
        if all(self.quality_ok(engine, length) for engine in self.engines):
            return 'passed'
        checks = [self.check(engine, length) for engine in self.engines]
        failed = (any(c is not None and not c['passed'] for c in checks)
                  or any(not c['passed'] or not c.get('bitwise_equal_to_first') for c in self.reset_checks)
                  or any(not self.stable(engine, length) for engine in self.engines if (engine, length) in self.timings)
                  or (MTP in self.engines and self.parity.get(length, {}).get('matches_plain_int8') is False))
        return 'failed' if failed else 'not_checked'
