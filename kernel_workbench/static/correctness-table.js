// Show the recorded logit comparison metrics.
const LABELS = {
  prefill_kl: 'Prefill KL', decode_mean_kl: 'Decode 평균 KL', decode_max_kl: 'Decode 최대 KL',
  top1_agreement: 'Top-1 일치율', strong_mismatches: 'Strong mismatch',
};
const show = value => typeof value === 'boolean' ? (value ? '예' : '아니오')
  : typeof value === 'number' && !Number.isInteger(value) ? value.toPrecision(4) : String(value);

function criteria(gate) {
  const kl = gate.max_mean_kl == null ? '기준 미기록' : `≤ ${gate.max_mean_kl}`;
  const allowed = gate.strong_mismatches_allowed;
  return {
    prefill_kl: kl, decode_mean_kl: kl,
    strong_mismatches: allowed == null ? '기준 미기록' : `${allowed}개 이하 · margin > ${gate.margin_threshold ?? '미기록'}`,
  };
}

export function correctnessTable({checks = [], gate = {}, details = ''} = {}) {
  if (!checks.length) return {columns: ['기록'], rows: [{item: '검증 기록', criterion: '—', values: [details || '미수집']}]};
  const rules = criteria(gate);
  const keys = Object.keys(LABELS).filter(key => checks.some(check => key in check));
  const rows = keys.map(key => ({
    item: LABELS[key], criterion: rules[key] || '—',
    values: checks.map(check => key in check ? show(check[key]) : '미수집'),
  }));
  return {columns: checks.map(check => checks.length === 1 ? '기록' : check.engine), rows};
}
