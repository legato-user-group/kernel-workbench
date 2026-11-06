// Present recorded metrics without changing the experiment's stored verdict.
export function correctnessRows(details = '') {
  const number = '([+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:e[+-]?\\d+)?)';
  const pattern = new RegExp(`([\\w-]+): prefill KL=${number}, decode mean KL=${number}, strong mismatches=(\\d+)`, 'gi');
  const paths = [...details.matchAll(pattern)];
  if (!paths.length) return [{item:'검증 기록', criterion:'기록된 기준', baseline:'—', candidate:details || '미수집'}];
  const plain = paths.find(p => p[1] === 'int8_plain_cudagraph');
  const mtp = paths.find(p => p[1] === 'int8_mtp_parallel_verify4');
  const candidate = mtp || (paths.length === 1 && !plain ? paths[0] : undefined);
  const kl = details.match(new RegExp(`Gate: KL 각각 ≤ ${number}`, 'i'))?.[1];
  const margin = details.match(new RegExp(`HF margin > ${number} mismatch (\\d+)`, 'i'));
  const recorded = (path,index) => path?.[index] ?? (mtp || plain ? '미수집' : '—');
  const rows = [
    {item:'Prefill KL', criterion:kl ? `≤ ${kl}` : '기준 미기록', baseline:recorded(plain,2), candidate:recorded(candidate,2)},
    {item:'Decode 평균 KL', criterion:kl ? `≤ ${kl}` : '기준 미기록', baseline:recorded(plain,3), candidate:recorded(candidate,3)},
    {item:'Strong mismatch', criterion:margin ? `${margin[2]}개 · HF margin > ${margin[1]}` : '기준 미기록',
      baseline:recorded(plain,4), candidate:recorded(candidate,4)},
  ];
  const parity = details.match(/Plain\/MTP 출력 일치 ([\d.]+)% \((\d+)토큰\)/);
  if (mtp || plain) rows.push({item:'생성 토큰 일치율', criterion:'Plain 대비 100%', baseline:'기준 출력',
    candidate:parity ? `${parity[1]}% · ${parity[2]}토큰` : '미검증'});
  if (details.includes('cache 위치 검사')) rows.push({item:'Cache 위치', criterion:'예상 위치와 비교',
    baseline:plain ? '검사 수행' : '—', candidate:candidate ? '검사 수행' : '미수집'});
  if (details.includes('state 값 미비교')) rows.push({item:'KV / recurrent state 값', criterion:'보조 검사',
    baseline:'미비교', candidate:'미비교'});
  return rows;
}
