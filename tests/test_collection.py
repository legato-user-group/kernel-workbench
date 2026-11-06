import json
import tarfile
from pathlib import Path
import pytest
from kernel_workbench.collection import prepare, analyze_artifacts


@pytest.mark.parametrize('remote',['/content/test-measure','/srv/gpu/jobs/test-measure'])
def test_prepare_does_not_import_or_execute_runner(tmp_path,remote):
    runner=tmp_path/'custom.py';runner.write_text("raise RuntimeError('must not execute during prepare')")
    result=prepare(runner,{'runner':'custom','output_tokens':16},tmp_path/'prepared',remote)
    with tarfile.open(result['source_archive']) as archive:
        assert sorted(archive.getnames())==['bootstrap.py','config.json','remote_entry.py','runner.py']
    assert result['config']['baseline_repeats']==5
    assert result['remote_artifacts']==remote+'/artifacts.tar.gz'
    compile(Path(result['collect_script']).read_text(),result['collect_script'],'exec')
    with pytest.raises(FileExistsError):prepare(runner,{},tmp_path/'prepared')


def test_prepare_default_remote_path_is_transport_independent(tmp_path):
    runner=tmp_path/'runner.py';runner.write_text('')
    result=prepare(runner,{},tmp_path/'prepared')
    assert result['remote_directory'].startswith('/tmp/kernel-workbench-')
    with pytest.raises(ValueError):prepare(runner,{},tmp_path/'bad','relative/work')
    assert not (tmp_path/'bad').exists()


def test_gpu_host_collector_streams_logs_without_installing_tools(tmp_path,monkeypatch,capsys):
    import glob
    import runpy
    import shutil
    import sys
    bootstrap=Path('kernel_workbench/bootstrap.py').resolve()
    (tmp_path/'remote_entry.py').write_text(
        "import pathlib,sys\nprint('runner output',flush=True)\n"
        "print('runner diagnostic',file=sys.stderr,flush=True)\n"
        "pathlib.Path(sys.argv[3]).write_text('{}')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',[str(bootstrap),str(tmp_path)])
    monkeypatch.setattr(shutil,'which',lambda _:None)
    monkeypatch.setattr(glob,'glob',lambda _:[])
    runpy.run_path(str(bootstrap),run_name='__main__')
    output=capsys.readouterr().out
    assert 'runner output' in output and 'runner diagnostic' in output
    assert 'apt-get' not in output
    status=json.loads((tmp_path/'status.json').read_text())
    assert len(status['warnings'])==2
    assert (tmp_path/'baseline.json').exists()
    assert 'runner output' in (tmp_path/'baseline.log').read_text()
    with tarfile.open(tmp_path/'artifacts.tar.gz') as archive:
        assert 'baseline.log' in archive.getnames() and 'status.json' in archive.getnames()


def test_agent_side_analysis_preserves_independent_timing(tmp_path):
    fixture=json.load(open('kernel_workbench/static/example.json'))
    config=fixture['config'];gpu={'name':fixture['hardware']['name']}
    meta={'config':config,'gpu':gpu,'model':fixture['model'],'total_ms':20,
          'phase_ms':{'prefill':5,'decode':15},'estimates':fixture['estimates'],'versions':{}}
    sample={'total_ms':10,'phase_ms':{'prefill':2,'decode':8}}
    baseline={'config':config,'gpu':gpu,'source':'unprofiled','statistic':'median',
              'sample_count':5,'samples':[sample]*5,**sample}
    (tmp_path/'meta.json').write_text(json.dumps(meta))
    (tmp_path/'baseline.json').write_text(json.dumps(baseline))
    result=analyze_artifacts(tmp_path)
    assert result['timings']['source']=='unprofiled' and result['timings']['total_ms']==10
    assert result['timings']['profiled']['total_ms']==20
    assert result['phase_points']['decode']
    archive_path=tmp_path/'artifacts.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        archive.add(tmp_path/'meta.json',arcname='meta.json')
        archive.add(tmp_path/'baseline.json',arcname='baseline.json')
    assert analyze_artifacts(archive_path)['timings']==result['timings']
