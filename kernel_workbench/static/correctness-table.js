// Show recorded checks as collected; a field without a label keeps its raw key.
const LABELS = {
  positions: '검사 위치 수', prefill_kl: 'Prefill KL', decode_mean_kl: 'Decode 평균 KL', decode_max_kl: 'Decode 최대 KL',
  top1_agreement: 'Top-1 일치율', strong_mismatches: 'Strong mismatch', cache_position: 'Cache 위치',
  expected_cache_position: '예상 Cache 위치', plain_token_agreement: 'Plain 대비 생성 토큰 일치율',
  reset_bitwise_equal: '재실행 bitwise 일치', passed: '통과',
};
const IDENTITY = new Set(['engine', 'input_tokens']);
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

export function correctnessTable({checks = [], gate = {}, details = '', state_comparison: state} = {}) {
  if (!checks.length) return {columns: ['기록'], rows: [{item: '검증 기록', criterion: '—', values: [details || '미수집']}]};
  const rules = criteria(gate);
  const keys = [...new Set(checks.flatMap(Object.keys))].filter(key => !IDENTITY.has(key));
  const rows = keys.map(key => ({
    item: LABELS[key] || key, criterion: rules[key] || '—',
    values: checks.map(check => key in check ? show(check[key]) : '미수집'),
  }));
  if (state) rows.push({item: 'State 비교', criterion: '—', values: checks.map(() => state)});
  return {columns: checks.map(check => check.engine), rows};
}
