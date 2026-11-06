import json
import pytest
from kernel_workbench import agents as module


def test_legacy_agent_history_is_read_only(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    history=module.AgentHistory()
    assert history.records()==[]
    assert not history.directory.exists()
    work=history.directory/'oldtask';work.mkdir(parents=True)
    record={'id':'oldtask','status':'failed','created_at':1,'response':'saved response'}
    (work/'task.json').write_text(json.dumps(record))
    assert history.records()==[record]
    assert history.get('oldtask')==record
    with pytest.raises(ValueError):history.get('../other')
    assert not hasattr(history,'start') and not hasattr(history,'execute')
