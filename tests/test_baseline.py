from types import SimpleNamespace
import pytest
from kernel_dashboard.remote_entry import collect_baseline, NoTrace
from kernel_dashboard.analysis import analyze


def test_baseline_runs_without_trace_and_uses_median():
    totals = iter([100, 5, 7, 6, 8])
    seen = []
    def run(config, trace):
        assert isinstance(trace, NoTrace)
        with trace.range('inference'), trace.phase('prefill'), trace.module('layer'), trace.step(0):
            seen.append(config)
        value = next(totals)
        return {'total_ms':value,'phase_ms':{'prefill':value/4,'decode':value*3/4},'gpu':{'name':'A100-SXM4-40GB'}}
    baseline = collect_baseline(SimpleNamespace(run=run), {'warmup':1,'baseline_repeats':5})
    assert len(seen)==5 and baseline['sample_count']==5
    assert baseline['total_ms']==7 and baseline['phase_ms']['prefill']==1.75
    assert baseline['source']=='unprofiled' and baseline['runner_reinitialized_per_sample']


def metadata():
    return {'gpu':{'name':'A100-SXM4-40GB'},'model':{'name':'test','children':[]},
            'config':{'output_tokens':2},'total_ms':20,'phase_ms':{'prefill':8,'decode':12},
            'estimates':[{'phase':'prefill','module':'test','flops':{'fp32':4000},'bytes':100},
                         {'phase':'decode','module':'test','flops':{'fp32':2000},'bytes':200}]}


def baseline(meta):
    sample={'total_ms':10,'phase_ms':{'prefill':4,'decode':6}}
    return {'source':'unprofiled','statistic':'median','sample_count':5,'samples':[sample]*5,
            **sample,'config':meta['config'],'gpu':meta['gpu']}


def test_analysis_uses_independent_time_and_preserves_nsys_time():
    meta=metadata(); result=analyze(meta,[],baseline=baseline(meta))
    assert result['timings']['total_ms']==10 and result['timings']['source']=='unprofiled'
    assert result['timings']['profiled']['total_ms']==20
    assert result['point']['tflops']==pytest.approx(6000/.01/1e12)
    assert result['phase_points']['decode']['tflops']==pytest.approx(2000/.006/1e12)
    assert result['phase_points']['decode']['timing_source']=='unprofiled'


def test_mismatched_baseline_keeps_profiled_time_with_warning():
    meta=metadata(); other=baseline(meta); other['config']={'output_tokens':16}
    result=analyze(meta,[],baseline=other)
    assert result['timings']['total_ms']==20 and result['timings']['source']=='nsys'
    assert any('baseline rejected' in warning for warning in result['warnings'])


def test_baseline_rejects_nonfinite_timings():
    runner=SimpleNamespace(run=lambda c,t:{'total_ms':float('nan'),'phase_ms':{'prefill':1,'decode':1},'gpu':{'name':'A100'}})
    with pytest.raises(ValueError):collect_baseline(runner,{'baseline_repeats':1})
