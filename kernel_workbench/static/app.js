import { experimentGroups } from './experiment-groups.js?v=history-2';
import { correctnessRows } from './correctness-table.js';
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, phase='all',runId=null,follow=true,runRecords=[],selectedExperiment=null;
let experimentId=null,experimentRecords=[],experimentFingerprint='',loadSequence=0,refreshing=false;
const groups=()=>experimentGroups(runRecords,experimentRecords);
const currentGroup=()=>groups().find(g=>g.id===selectedExperiment);
const num=(v,d=2)=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d});
const time=v=>v==null?'—':v<.001?`${num(v*1e6,1)} ns`:v<1?`${num(v*1000,2)} µs`:`${num(v,3)} ms`;
async function api(path,options){const r=await fetch(path,options);if(!r.ok){let e=await r.json();throw Error(e.detail||r.statusText)}return r.json()}
function fail(e){$('#status').textContent=e.message}
function metric(label,value,detail,bar){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-detail">${detail}</div>${bar==null?'':`<div class="bar"><span style="width:${bar}%"></span></div>`}</div>`}
function render(){
 const t=data.timings;
 const independent=t.source==='unprofiled';
 const samplesLabel=key=>{const count=t.sample_counts?.[key]??t.sample_count;return count==null?'':`${count}${t.expected_samples?'/'+t.expected_samples:''}회`};
 const timingLabel=independent?`프로파일러 OFF · 중앙값 ${samplesLabel('total_ms')}`:(t.source==='runner_single'?'단일 실행 · 반복 기준 측정 없음':'Nsight 수집 중 · 독립 기준 측정 없음');
 const profileTime=key=>t.profiled?`<br>${t.profiled.source==='nsys'?'Nsight 수집':'수집 실행'}: ${time(t.profiled[key])}`:'';
 const decodeTokens=Math.max(0,data.config.output_tokens-1);
 const decodePerToken=decodeTokens>0?(t.decode_ms_per_token??t.decode/decodeTokens):null;
 $('#metrics').innerHTML=
  metric('END-TO-END',`${num(t.total_ms)} <small>ms</small>`,`${timingLabel}${profileTime('total_ms')}`)+
  metric('PREFILL',`${num(t.prefill)} <small>ms</small>`,`${data.config.input_tokens} input tokens · ${samplesLabel('prefill')}${profileTime('prefill')}`,t.total_ms?t.prefill/t.total_ms*100:0)+
  metric('DECODE TOTAL',`${num(t.decode)} <small>ms</small>`,`${decodeTokens} autoregressive tokens · ${samplesLabel('decode')}${profileTime('decode')}`,t.total_ms?t.decode/t.total_ms*100:0)+
  metric('DECODE / TOKEN',`${num(decodePerToken,3)} <small>ms/token</small>`,decodeTokens>0?`Decode 전체 ÷ ${decodeTokens}토큰 · ${samplesLabel('decode')}`:'Decode 토큰 없음');
 $('#notice').textContent=runId==='example'?'화면 검증용 예제 데이터입니다. 실제 GPU 측정 결과가 아닙니다.':'';
 $('#gpu-name').textContent=data.hardware.name;
 renderLogicalBounds();roofline();renderExperiments();
}
function renderLogicalBounds(){
 const labels={compute:'Compute 제한',memory:'Memory 제한',unknown:'계산 불가'};
 $('#logical-bounds-cards').innerHTML=['prefill','decode'].map(phase=>{
  const cost=data.logical_bounds?.[phase],decode=phase==='decode';
  const title=decode?'Decode':'Prefill',unit=decode?' / token':' / 회';
  if(!cost)return `<article class="logical-bound-card"><h3>${title}</h3><p>${decode&&data.config.output_tokens===1?'Decode 토큰 없음':esc(data.phase_estimate_reasons?.[phase]||'연산 비용 모형 없음')}</p></article>`;
  const values=decode?cost.per_token:cost;
  return `<article class="logical-bound-card" data-logical-phase="${phase}"><div class="logical-bound-head"><h3>${title}</h3><span class="bound ${esc(cost.bound)}">${labels[cost.bound]}</span></div><div class="logical-bound-value">${time(values.lower_ms)}<small>${unit}</small></div><div class="logical-bound-details"><span>Compute 하한 <b>${time(values.compute_ms)}</b></span><span>Memory 하한 <b>${time(values.memory_ms)}</b></span><span>연산 집약도 <b>${num(cost.intensity)} FLOP/byte</b></span></div><p>${decode?`${num(cost.steps,0)}토큰 합계 하한 ${time(cost.lower_ms)}`:`입력 ${num(data.config.input_tokens,0)}토큰`}</p></article>`;
 }).join('');
 const projectionOnly=data.estimate_scope?.startsWith('projection GEMMs only');
 $('#logical-bounds-scope').textContent=(projectionOnly?'Projection 연산 기준 · attention·DeltaNet·정규화·elementwise 제외. ':'모델 연산 비용 모형 기준. ')+'단계별 연산량과 논리적 텐서 이동량을 합산하고 GPU 이론 peak를 적용합니다. 하한 = max(Compute, Memory). 캐시 재사용은 생략하며 실제 실행 지연은 포함하지 않습니다.';
}
function roofline(){
 const labels={all:'전체 실행',prefill:'Prefill',decode:'Decode'};
 const colors={all:'#ffb36b',prefill:'#88bceb',decode:'#8dd5b2'};
 const hw=data.hardware, points={all:data.point,...data.phase_points};
 const keys=phase==='all'?['all','prefill','decode']:[phase];
 const visible=keys.filter(key=>points[key]?.intensity>0&&points[key]?.tflops>0);
 const sources=new Set(visible.map(key=>points[key].source));
 $('#point-source').textContent=!visible.length?'INSUFFICIENT DATA':sources.size>1?'MEASURED + ESTIMATED':sources.has('measured')?'MEASURED':'ESTIMATED · MODELED';
 $('#point-scope').innerHTML=visible.map(key=>{const p=points[key];return `<div data-point-summary="${key}"><b style="color:${colors[key]}">${labels[key]} · ${p.source==='measured'?'실측':'추정'}</b><span>${num(p.intensity)} FLOP/byte · ${num(p.tflops,5)} TFLOP/s</span><small>${time(key==='all'?data.timings.total_ms:data.timings[key])} (${data.timings.source==='unprofiled'?'독립 측정':'수집 중 측정'}) · ${p.source==='estimated'?'논리적 텐서 트래픽 추정':'DRAM 카운터'}<br>${esc(p.scope||'혼합 연산 경로 집계')}</small></div>`}).join('');
 const messages={not_executed:'출력 1토큰은 prefill에서 생성되므로 실제 decode 호출이 없습니다. 출력 토큰 수를 2 이상으로 수집하세요.',missing_data:'이 실행에는 해당 구간의 커널 또는 비용 추정 데이터가 없습니다.',missing_duration:'해당 구간의 실행 시간을 수집하지 못했습니다.',missing_counters_or_nonpositive_work:'연산량·메모리 트래픽이 부족하거나 0이어서 로그 축 위치를 표시할 수 없습니다.'};
 const missing=keys.filter(key=>key!=='all'&&!points[key]);
 $('#phase-notice').textContent=missing.map(key=>`${labels[key]}: ${messages[data.phase_point_reasons?.[key]]||data.phase_point_reasons?.[key]||messages.missing_data}`).join(' ');
 if(!hw.bandwidth){$('#roofline').innerHTML='<div class="empty">GPU 사양을 확인할 수 없어 roofline을 계산할 수 없습니다.</div>';return}
 const W=850,H=350,L=65,R=30,T=28,B=47;
 let xmin=-2,xmax=5,ymin=-5,ymax=3;
 // Keep axes stable across phase switches for direct visual comparison.
 for(const p of Object.values(points)){if(p?.intensity>0){xmin=Math.min(xmin,Math.floor(Math.log10(p.intensity))-1);xmax=Math.max(xmax,Math.ceil(Math.log10(p.intensity))+1)}if(p?.tflops>0){ymin=Math.min(ymin,Math.floor(Math.log10(p.tflops))-1);ymax=Math.max(ymax,Math.ceil(Math.log10(p.tflops))+1)}}
 const x=v=>L+(Math.log10(v)-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(Math.log10(v)-ymin)/(ymax-ymin)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${phase==='all'?'전체·prefill·decode 비교':labels[phase]} roofline. 가로축 연산 집약도, 세로축 TFLOP/s. 구간별 수치는 그래프 아래 표시됩니다.">`;
 for(let i=xmin;i<=xmax;i++)s+=`<line x1="${x(10**i)}" y1="${T}" x2="${x(10**i)}" y2="${H-B}" stroke="#2b343b"/><text x="${x(10**i)}" y="${H-B+20}" text-anchor="middle">10<tspan baseline-shift="super" font-size="8">${i}</tspan></text>`;
 for(let i=ymin;i<=ymax;i++)s+=`<line x1="${L}" y1="${y(10**i)}" x2="${W-R}" y2="${y(10**i)}" stroke="#2b343b"/><text x="${L-12}" y="${y(10**i)+4}" text-anchor="end">10<tspan baseline-shift="super" font-size="8">${i}</tspan></text>`;
 const roofs=[['BF16 Tensor',hw.peaks?.bf16_tensor/1e12,'#8dd5b2'],['FP16 Tensor',hw.peaks?.fp16_tensor/1e12,'#8dd5b2']].filter((r,i,rows)=>r[1]>0&&(i!==1||r[1]!==rows[0][1]));
 if(hw.peaks?.bf16_tensor>0&&hw.peaks.bf16_tensor===hw.peaks.fp16_tensor)roofs[0][0]='BF16 / FP16 Tensor';
 roofs.forEach(([name,peak,color],i)=>{const ridge=peak*1e12/hw.bandwidth;const xa=10**xmin;s+=`<path d="M${x(xa)},${y(xa*hw.bandwidth/1e12)} L${x(ridge)},${y(peak)} L${W-R},${y(peak)}" fill="none" stroke="${color}" stroke-width="1.8"/><text x="${W-R-5}" y="${y(peak)-(i===1?-14:7)}" text-anchor="end" style="fill:${color};font-size:10px">${name} · ${peak} T</text>`});
 s+=`<text x="${L+15}" y="${T+18}">HBM ${num(hw.bandwidth/1e12,3)} TB/s</text>`;
 const offsets={all:5,prefill:-15,decode:24};
 visible.forEach(key=>{const p=points[key],px=x(p.intensity),py=y(p.tflops),color=colors[key];
 let marker=key==='prefill'?`<rect x="${px-5}" y="${py-5}" width="10" height="10"/>`:key==='decode'?`<path d="M${px},${py-7} l7,7 l-7,7 l-7,-7 Z"/>`:`<circle cx="${px}" cy="${py}" r="5"/>`;
 s+=`<g data-point="${key}" fill="${color}" ${p.source==='estimated'?'stroke="white" stroke-dasharray="2 2"':''}><title>${labels[key]} · ${p.source==='measured'?'실측':'추정'}: ${num(p.intensity)} FLOP/byte, ${num(p.tflops,5)} TFLOP/s</title><line x1="${px}" x2="${px}" y1="${py}" y2="${H-B}" stroke="${color}" opacity=".3" stroke-dasharray="4 4"/><circle cx="${px}" cy="${py}" r="13" opacity=".1"/>${marker}</g><text x="${px>W/2?px-15:px+15}" y="${Math.max(T+12,Math.min(H-B-5,py+offsets[key]))}" text-anchor="${px>W/2?'end':'start'}" style="fill:${color}">${labels[key]} ${p.source==='estimated'?'(추정)':''}</text>`;
 });
 s+=`<text x="${W/2}" y="${H-4}" text-anchor="middle">Arithmetic intensity [FLOP / byte]</text><text transform="translate(14,${H/2}) rotate(-90)" text-anchor="middle">Performance [TFLOP/s]</text></svg>`;
 $('#roofline').innerHTML=s;
}
function setPhase(next){
 phase=next;
 document.querySelectorAll('[data-roof-phase]').forEach(button=>button.classList.toggle('active',button.dataset.roofPhase===phase));
 roofline();
}

function renderExperimentSelector(){
 const entries=groups();
 $('#experiment').innerHTML=entries.length?entries.map(g=>`<option value="${esc(g.id)}">${esc(g.id)}</option>`).join(''):'<option value="example">예제 · 화면 검증</option>';
 $('#experiment').value=selectedExperiment||'example';
}
async function load(id,attemptId=null){
 const sequence=++loadSequence;
 const result=await api(id==='example'?'/example.json':`/api/runs/${id}/result`);
 if(sequence!==loadSequence)return;
 data=result;runId=id;experimentId=attemptId;
 if(id==='example')selectedExperiment='example';
 $('#analysis-body').hidden=false;$('#result-unavailable').hidden=true;
 renderExperimentSelector();render();
}
async function showExperiment(id,attemptId=null){
 selectedExperiment=id;
 const group=currentGroup();
 const attempt=group?.attempts.find(a=>a.id===attemptId)||(!attemptId?group?.attempts[0]:null);
 experimentId=attempt?.id||null;
 const selected=attempt?group.runs.find(r=>r.id===attempt.candidate_run_id):group?.runs[0];
 ++loadSequence;runId=null;data=null;
 $('#analysis-body').hidden=true;$('#result-unavailable').hidden=!!selected;
 $('#notice').textContent='';
 renderExperimentSelector();renderExperiments();
 if(selected){await load(selected.id,experimentId);return}
 $('#result-unavailable').textContent='이 시도의 측정 결과는 아직 수집되지 않았습니다.';
}
async function refresh(){
 if(refreshing)return;refreshing=true;
 try{
  const [runs,experiments]=await Promise.all([api('/api/runs'),api('/api/experiments')]);
  runRecords=runs.filter(r=>r.status==='completed');experimentRecords=experiments;
  const entries=groups(),group=currentGroup();
  const selected=group?.attempts.find(a=>a.id===experimentId);
  const candidate=group?.runs.find(r=>r.id===selected?.candidate_run_id);
  if(!entries.length){if(!data||runId!=='example')await load('example');else{renderExperimentSelector();renderExperiments()}}
  else if(!group)await showExperiment(entries[0].id);
  else if((follow&&(group.attempts[0]?.id||null)!==experimentId)||
          (experimentId&&!group.attempts.some(a=>a.id===experimentId)))await showExperiment(group.id);
  else if(selected&&(candidate?.id||null)!==runId)await showExperiment(group.id,experimentId);
  else if(runId&&runRecords.some(r=>r.id===runId&&r.evidence_revision!==data?.evidence_revision))await load(runId,experimentId);
  else {renderExperimentSelector();renderExperiments()}
  $('#status').textContent=runRecords.find(r=>r.id===runId)?.cleanup_error||'';
 }finally{refreshing=false}
}
$('#experiment').onchange=e=>{follow=false;showExperiment(e.target.value).catch(fail)};
$('#roof-phases').querySelectorAll('button').forEach(b=>b.onclick=()=>setPhase(b.dataset.roofPhase));

function renderExperiments(){
 const group=currentGroup(),attempts=group?.attempts||[];
 const linked=new Set(attempts.flatMap(a=>[a.candidate_run_id,a.baseline_run_id]));
 const unlinked=(group?.runs||[]).filter(r=>!linked.has(r.id));
 const fingerprint=JSON.stringify([group,experimentId,runId]);
 if(fingerprint===experimentFingerprint)return;experimentFingerprint=fingerprint;
 const date=record=>record.data_directory?.split('/').slice(1).join('/')||record.title||new Date(record.created_at*1000).toLocaleDateString('sv');
 const options=attempts.map(e=>`<option value="attempt:${esc(e.id)}" ${e.id===experimentId?'selected':''}>${esc(date(e))}</option>`)
  .concat(unlinked.map(r=>`<option value="run:${esc(r.id)}" ${r.id===runId?'selected':''}>${esc(date(r))}</option>`));
 const e=attempts.find(e=>e.id===experimentId),detail=$('#experiment-detail');
 $('#experiment-list').innerHTML=options.length?'':'<div class="empty">이 실험의 시도가 없습니다.</div>';
 const field=(label,value)=>`<div><h3>${label}</h3><p>${esc(value||'기록 없음')}</p></div>`;
 detail.hidden=!options.length;
 detail.innerHTML=options.length?`<div class="experiment-detail-head"><h3>${esc(group.id)}</h3><select id="experiment-case" aria-label="실험 시도 선택">${options.join('')}</select></div>${e?`<div class="experiment-fields">${field('가설',e.hypothesis)}${field('결과 해석',e.conclusion)}</div>`:''}`:'';
 if($('#experiment-case'))$('#experiment-case').onchange=event=>{
  follow=false;const [kind,id]=event.target.value.split(':');
  (kind==='attempt'?showExperiment(selectedExperiment,id):load(id)).catch(fail);
 };
 $('#experiment-correctness').hidden=!e;
 if(!e)return;
 $('#correctness-status').textContent=({passed:'성공',failed:'실패',not_checked:'미검증'})[e.correctness.status];
 $('#correctness-status').dataset.status=e.correctness.status;
 const mtp=e.correctness.details.includes('int8_mtp_parallel_verify4:');
 const candidateLabel=mtp?'INT8 + MTP':(e.comparison.candidate?.config.dtype?.toUpperCase()||'평가 경로');
 $('#correctness-details').innerHTML=`<div class="table-wrap"><table><thead><tr><th>검증 항목</th><th>통과 기준</th><th>${mtp?'INT8 plain':'기준 경로'}</th><th>${esc(candidateLabel)}</th></tr></thead><tbody>`+correctnessRows(e.correctness.details).map(row=>`<tr><th scope="row">${esc(row.item)}</th><td>${esc(row.criterion)}</td><td>${esc(row.baseline)}</td><td>${esc(row.candidate)}</td></tr>`).join('')+'</tbody></table></div>';
}
refresh().catch(fail);
setInterval(()=>refresh().catch(fail),2000);
