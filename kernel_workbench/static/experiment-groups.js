// A runner fixes the input workload; baseline/candidate engines belong to its experiment.
export const runnerIdentity = measurement => measurement?.runner_id || measurement?.config?.runner;
export const inputLength = experiment => (experiment.comparison.candidate || experiment.comparison.baseline)?.config.input_tokens;

export function selectableResults(runs, experiments) {
  const baselines = new Set(experiments.map(e => e.baseline_run_id).filter(Boolean));
  const candidates = new Set(experiments.map(e => e.candidate_run_id).filter(Boolean));
  return runs.filter(run => candidates.has(run.id) || !baselines.has(run.id));
}

export function runnerExperiment(records, runnerId, series = '') {
  const cases = records.filter(e => (!series || e.series === series) &&
    ['baseline', 'candidate'].some(role => runnerIdentity(e.comparison[role]) === runnerId))
    .sort((a, b) => (inputLength(a) ?? Infinity) - (inputLength(b) ?? Infinity));
  if (!cases.length) return null;
  return {
    runnerId, cases,
    status: cases.some(e => e.status === 'inconclusive') ? 'inconclusive' :
      cases.every(e => e.status === 'failed') ? 'failed' : 'completed',
    correctness: cases.some(e => e.correctness.status === 'failed') ? 'failed' :
      cases.every(e => e.correctness.status === 'passed') ? 'passed' : 'not_checked',
    validatedCases: cases.filter(e => Object.keys(e.comparison.speedup).length).length,
    created_at: Math.max(...cases.map(e => e.created_at)),
  };
}
