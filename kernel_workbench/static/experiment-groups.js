// Folder identity stays stable when a dated attempt changes its runner or engine.
export function experimentGroups(runs, attempts) {
  const groups = new Map();
  const key = record => record.data_directory?.split('/')[0] || record.series;
  function group(id) {
    if (!groups.has(id)) groups.set(id, {id, runs: [], attempts: []});
    return groups.get(id);
  }
  for (const attempt of attempts) group(key(attempt)).attempts.push(attempt);
  for (const run of runs) {
    const linked = attempts.filter(a => a.candidate_run_id === run.id || a.baseline_run_id === run.id);
    const ids = run.data_directory ? [key(run)] : linked.length ? linked.map(key) : [run.runner_id || run.config.runner];
    for (const id of new Set(ids)) group(id).runs.push(run);
  }
  for (const entry of groups.values()) {
    const newest = (a, b) => (b.data_directory || '').localeCompare(a.data_directory || '') || b.created_at - a.created_at;
    entry.runs.sort(newest);
    entry.attempts.sort(newest);
  }
  return [...groups.values()].sort((a, b) => a.id.localeCompare(b.id));
}
