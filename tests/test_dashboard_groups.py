import subprocess


def test_experiments_follow_folders_and_keep_dated_attempts_together():
    script = """
import assert from 'node:assert/strict';
import {experimentGroups} from './kernel_workbench/static/experiment-groups.js';
const runs = [
  {id:'old', data_directory:'decoder/2026-11-06', runner_id:'fp16', created_at:100},
  {id:'new', data_directory:'decoder/2026-11-07', runner_id:'int8', created_at:50},
  {id:'other', data_directory:'other/2026-11-07', runner_id:'int8', created_at:50},
];
const attempts = [
  {id:'old-attempt', data_directory:'decoder/2026-11-06', candidate_run_id:'old', created_at:100},
  {id:'new-attempt', data_directory:'decoder/2026-11-07', candidate_run_id:'new', created_at:50},
  {id:'other-attempt', data_directory:'other/2026-11-07', candidate_run_id:'other', created_at:50},
  {id:'failed', data_directory:'decoder/2026-11-08', status:'failed', created_at:60},
];
const groups=experimentGroups(runs, attempts);
assert.deepEqual(groups.map(g=>g.id), ['decoder','other']);
assert.deepEqual(groups[0].runs.map(r=>r.id), ['new','old']);
assert.deepEqual(groups[0].attempts.map(a=>a.id), ['failed','new-attempt','old-attempt']);
assert.deepEqual(groups[1].attempts.map(a=>a.id), ['other-attempt']);
assert.equal(runs[0].id,'old');
// Uploaded comparisons keep their baseline and candidate in one history.
const uploads=experimentGroups([
 {id:'baseline', config:{runner:'plain'}, created_at:1},
 {id:'candidate', config:{runner:'optimized'}, created_at:2},
], [{id:'trial',series:'decode',baseline_run_id:'baseline',candidate_run_id:'candidate',created_at:2},
    {id:'failed-trial',series:'decode',status:'failed',created_at:3}]);
assert.equal(uploads.length,1);
assert.equal(uploads[0].id,'decode');
assert.equal(uploads[0].runs.length,2);
assert.equal(uploads[0].attempts[0].id,'failed-trial');
assert.deepEqual(experimentGroups([],[]),[]);
"""
    subprocess.run(['node', '--input-type=module'], input=script, text=True, check=True, capture_output=True)
