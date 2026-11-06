// A readable hint does not turn an unassigned launch into a mapped model module.
export function moduleLabel(kernel, config = {}) {
  if (kernel.module !== 'unattributed') return kernel.module;
  const name = kernel.name || '';
  let operation;
  if (/CatArrayBatchedCopy|cat_kernel/i.test(name)) {
    operation = config.runner === 'tiny_decoder' && kernel.phase === 'decode'
      ? 'KV cache 연결·복사 (cat)' : '텐서 연결·복사 (cat)';
  } else if (/ArgMaxOps/i.test(name)) {
    operation = '최댓값 위치 선택 (argmax)';
  } else if (/GeluCUDA|gelu/i.test(name)) {
    operation = 'GELU';
  } else if (/CUDAFunctor_add/i.test(name)) {
    operation = '텐서 덧셈 (add)';
  } else if (/layer_norm|rms_norm/i.test(name)) {
    operation = '정규화 (norm)';
  } else if (/memcpy|copy_kernel/i.test(name)) {
    operation = '메모리 복사 (copy)';
  } else if (/indexSelect|gather/i.test(name)) {
    operation = '인덱스 조회 (gather)';
  } else if (/reduce_kernel|reduction/i.test(name)) {
    operation = '축소 연산 (reduction)';
  } else if (/vectorized_elementwise_kernel|elementwise/i.test(name)) {
    operation = '요소별 연산 (elementwise)';
  }
  return operation ? `unattributed · ${operation} · 추정` : 'unattributed · 연산 식별 불가';
}
