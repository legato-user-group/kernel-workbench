import { moduleLabel } from './kernel-labels.js';
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, phase='all',moduleId=null,runId='example', active=null,follow=true;
const num=(v,d=2)=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d});
const time=v=>v==null?'—':v<.001?`${num(v*1e6,1)} ns`:v<1?`${num(v*1000,2)} µs`:`${num(v,3)} ms`;
const quantity=v=>v==null?'—':v>=1e9?`${num(v/1e9)} G`:v>=1e6?`${num(v/1e6)} M`:num(v);
async function api(path,options){const r=await fetch(path,options);if(!r.ok){let e=await r.json();throw Error(e.detail||r.statusText)}return r.json()}
function fail(e){$('#status').textContent=e.message}
function metric(label,value,detail,bar){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-detail">${detail}</div>${bar==null?'':`<div class="bar"><span style="width:${bar}%"></span></div>`}</div>`}
function render(){
 $('#model-name').textContent=data.model.name;
 $('#report-summary-text').textContent=data.agent_summary||'';
 $('#report-summary').hidden=!data.agent_summary;
 const t=data.timings,p=data.point;
 const independent=t.source==='unprofiled';
 const timingLabel=independent?`프로파일러 OFF · 중앙값 ${t.sample_count}회`:(t.source==='runner_single'?'단일 실행 · 반복 기준 측정 없음':'Nsight 수집 중 · 독립 기준 측정 없음');
 const profileTime=key=>t.profiled?`<br>${t.profiled.source==='nsys'?'Nsight 수집':'수집 실행'}: ${time(t.profiled[key])}`:'';
 $('#metrics').innerHTML=metric('END-TO-END',`${num(t.total_ms)} <small>ms</small>`,`${timingLabel}${profileTime('total_ms')}`)+metric('PREFILL',`${num(t.prefill)} <small>ms</small>`,`${data.config.input_tokens} input tokens${profileTime('prefill')}`,t.total_ms?t.prefill/t.total_ms*100:0)+metric('DECODE',`${num(t.decode)} <small>ms</small>`,`${Math.max(0,data.config.output_tokens-1)} autoregressive steps${profileTime('decode')}`,t.total_ms?t.decode/t.total_ms*100:0)+metric('COUNTER COVERAGE',`${data.coverage.measured}<small> / ${data.coverage.total} kernels</small>`,`${esc(data.hardware.name)} · ${esc(data.config.dtype.toUpperCase())}`);
 $('#notice').textContent=runId==='example'?'화면 검증용 예제 데이터입니다. 실제 GPU 측정 결과가 아닙니다.':p?.source==='estimated'?'GPU 카운터가 불완전합니다. 전체 점은 모델 연산·논리적 텐서 트래픽 추정치이며 실제 DRAM 측정값과 다릅니다.':'';
 function node(n){return n.children?`<details class="node" open><summary>${esc(n.label)}</summary>${n.children.map(node).join('')}</details>`:`<div class="node"><button data-module="${esc(n.id)}" class="${moduleId===n.id?'selected':''}">◇ ${esc(n.label)}</button></div>`}
 $('#tree').innerHTML=data.model.children.map(node).join('');
 $('#tree').querySelectorAll('button').forEach(b=>b.onclick=()=>{moduleId=b.dataset.module;render()});
 $('#gpu-name').textContent=data.hardware.name;
 roofline();table();
 $('#step').innerHTML='<option value="all">모든 decode 단계</option>'+[...new Set(data.kernels.map(k=>k.step).filter(k=>k!=null))].map(s=>`<option value="${s}">Decode ${s}</option>`).join('');
 $('#diagnostics').textContent=JSON.stringify({config:data.config,coverage:data.coverage,versions:data.versions,timings:data.timings,warnings:data.warnings,hardware:data.hardware},null,2);
 $('#estimates').innerHTML='<div class="table-wrap"><table><thead><tr><th>모델 연산</th><th>단계</th><th>FLOPs (추정)</th><th>논리적 bytes (추정)</th><th>모델 하한 (추정)</th></tr></thead><tbody>'+data.estimates.map(o=>`<tr><td>${esc(o.module)}</td><td>${esc(o.phase)}</td><td>${quantity(Object.values(o.flops).reduce((a,b)=>a+b,0))}</td><td>${quantity(o.bytes)}</td><td>${time(o.lower_ms)}</td></tr>`).join('')+'</tbody></table></div>';
 $('#downloads').innerHTML=runId==='example'?'':(data.artifacts||['result.json','artifacts.tar.gz','commands.log']).map(n=>`<a href="/api/runs/${runId}/artifacts/${n}">${n} ↓</a>`).join('');
}
function roofline(){
 const labels={all:'전체 실행',prefill:'Prefill',decode:'Decode'};
 const colors={all:'#ffb36b',prefill:'#88bceb',decode:'#8dd5b2'};
 const hw=data.hardware, points={all:data.point,...data.phase_points};
 const keys=phase==='all'?['all','prefill','decode']:[phase];
 const visible=keys.filter(key=>points[key]?.intensity>0&&points[key]?.tflops>0);
 const sources=new Set(visible.map(key=>points[key].source));
 $('#point-source').textContent=!visible.length?'INSUFFICIENT DATA':sources.size>1?'MEASURED + ESTIMATED':sources.has('measured')?'MEASURED':'ESTIMATED · MODELED';
 $('#point-scope').innerHTML=visible.map(key=>{const p=points[key];return `<div data-point-summary="${key}"><b style="color:${colors[key]}">${labels[key]} · ${p.source==='measured'?'실측':'추정'}</b><span>${num(p.intensity)} FLOP/byte · ${num(p.tflops,5)} TFLOP/s</span><small>${time(key==='all'?data.timings.total_ms:data.timings[key])} (${data.timings.source==='unprofiled'?'독립 측정':'수집 중 측정'}) · ${p.source==='estimated'?'논리적 텐서 트래픽 추정':'DRAM 카운터'} · 혼합 연산 경로 집계</small></div>`}).join('');
 const messages={not_executed:'출력 1토큰은 prefill에서 생성되므로 실제 decode 호출이 없습니다. 출력 토큰 수를 2 이상으로 수집하세요.',missing_data:'이 실행에는 해당 구간의 커널 또는 비용 추정 데이터가 없습니다.',missing_duration:'해당 구간의 실행 시간을 수집하지 못했습니다.',missing_counters_or_nonpositive_work:'연산량·메모리 트래픽이 부족하거나 0이어서 로그 축 위치를 표시할 수 없습니다.'};
 const missing=keys.filter(key=>key!=='all'&&!points[key]);
 $('#phase-notice').textContent=missing.map(key=>`${labels[key]}: ${messages[data.phase_point_reasons?.[key]]||messages.missing_data}`).join(' ');
 if(!hw.bandwidth){$('#roofline').innerHTML='<div class="empty">GPU 사양을 확인할 수 없어 roofline을 계산할 수 없습니다.</div>';return}
 const W=850,H=350,L=65,R=30,T=28,B=47;
 let xmin=-2,xmax=5,ymin=-5,ymax=3;
 // Keep axes stable across phase switches for direct visual comparison.
 for(const p of Object.values(points)){if(p?.intensity>0){xmin=Math.min(xmin,Math.floor(Math.log10(p.intensity))-1);xmax=Math.max(xmax,Math.ceil(Math.log10(p.intensity))+1)}if(p?.tflops>0){ymin=Math.min(ymin,Math.floor(Math.log10(p.tflops))-1);ymax=Math.max(ymax,Math.ceil(Math.log10(p.tflops))+1)}}
 const x=v=>L+(Math.log10(v)-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(Math.log10(v)-ymin)/(ymax-ymin)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${phase==='all'?'전체·prefill·decode 비교':labels[phase]} roofline. 가로축 연산 집약도, 세로축 TFLOP/s. 구간별 수치는 그래프 아래 표시됩니다.">`;
 for(let i=xmin;i<=xmax;i++)s+=`<line x1="${x(10**i)}" y1="${T}" x2="${x(10**i)}" y2="${H-B}" stroke="#2b343b"/><text x="${x(10**i)}" y="${H-B+20}" text-anchor="middle">10<tspan baseline-shift="super" font-size="8">${i}</tspan></text>`;
 for(let i=ymin;i<=ymax;i++)s+=`<line x1="${L}" y1="${y(10**i)}" x2="${W-R}" y2="${y(10**i)}" stroke="#2b343b"/><text x="${L-12}" y="${y(10**i)+4}" text-anchor="end">10<tspan baseline-shift="super" font-size="8">${i}</tspan></text>`;
 const roofs=[['BF16 Tensor',hw.peaks?.bf16_tensor/1e12,'#8dd5b2'],['FP16 Tensor',hw.peaks?.fp16_tensor/1e12,'#8dd5b2'],['TF32 Tensor',hw.peaks?.tf32_tensor/1e12,'#88bceb'],['FP32',hw.peaks?.fp32/1e12,'#b9a0d8']].filter((r,i,rows)=>r[1]>0&&(i!==1||r[1]!==rows[0][1]));
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
 document.querySelectorAll('[data-phase]').forEach(button=>button.classList.toggle('active',button.dataset.phase===phase));
 document.querySelectorAll('[data-roof-phase]').forEach(button=>button.classList.toggle('active',button.dataset.roofPhase===phase));
 roofline();table();
}

function table(){
 const q=$('#search').value.toLowerCase(),bound=$('#bound').value,step=$('#step').value,groups=new Map();
 for(const k of data.kernels){if(phase!=='all'&&k.phase!==phase||moduleId&&k.module!==moduleId||bound!=='all'&&k.bound!==bound||step!=='all'&&String(k.step)!==step||!`${k.name} ${moduleLabel(k,data.config)}`.toLowerCase().includes(q))continue;
 const key=JSON.stringify([k.name,k.phase,k.module,k.grid,k.block,k.bound]);if(!groups.has(key))groups.set(key,{...k,calls:[],duration_ms:0});const g=groups.get(key);g.calls.push(k);g.duration_ms+=k.duration_ms;}
 const sorted=[...groups.values()].sort((a,b)=>b.duration_ms-a.duration_ms);$('#count').textContent=`${sorted.length} groups`;
 $('#rows').innerHTML=sorted.length?sorted.map((g,i)=>`<tr class="expandable" tabindex="0" aria-expanded="false" data-group="${i}"><td><b title="${esc(g.name)}">▸ ${esc(g.name.length>95?g.name.slice(0,95)+'…':g.name)}</b><small class="operation-hint" title="${g.module==='unattributed'?'모델 연산 구간에 연결되지 않은 커널입니다. 설명은 커널명과 runner 코드에서 추정했습니다.':'모델 연산 구간'}">${esc(moduleLabel(g,data.config))}</small></td><td>${esc(g.phase)}</td><td>${g.calls.length}</td><td>${time(g.duration_ms)}</td><td><span class="bound ${g.bound}">${g.bound==='unknown'?'미확인':g.bound+'-bound'}</span></td><td>${time(g.compute_ms)}<small> / call</small></td><td>${time(g.memory_ms)}<small> / call</small></td><td>${time(g.lower_ms)}<small> / call</small></td></tr>`).join(''):'<tr><td colspan="8" class="empty">표시할 커널이 없습니다. 추정 데이터는 아래 연산 단위 비용에서 확인하세요.</td></tr>';
 $('#rows').querySelectorAll('[data-group]').forEach(row=>{const toggle=()=>{if(row.nextElementSibling?.classList.contains('detail')){row.nextElementSibling.remove();row.setAttribute('aria-expanded','false');return}const g=sorted[+row.dataset.group];const tr=document.createElement('tr');tr.className='detail';tr.innerHTML=`<td colspan="8">하한은 첫 호출 기준입니다. 호출별 세부 정보:<pre>${esc(JSON.stringify(g.calls.map(k=>({name:k.name,step:k.step,duration_ms:k.duration_ms,flops:k.flops,dram_bytes:k.bytes,source:k.source,compute_ms:k.compute_ms,memory_ms:k.memory_ms,lower_ms:k.lower_ms,grid:k.grid,block:k.block})),null,2))}</pre></td>`;row.after(tr);row.setAttribute('aria-expanded','true')};row.onclick=toggle;row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle()}}});
}
async function load(id){runId=id;if([...$('#runs').options].some(o=>o.value===id))$('#runs').value=id;moduleId=null;data=await api(id==='example'?'/example.json':`/api/runs/${id}/result`);render()}
async function refresh(){const list=await api('/api/runs');const select=$('#runs');select.innerHTML='<option value="example">예제 · 화면 검증 데이터</option>'+list.filter(r=>r.status==='completed').map(r=>`<option value="${r.id}">${r.id} · ${esc(r.config.runner)}</option>`).join('');select.value=runId;active=list.find(r=>!['completed','failed','cancelled'].includes(r.status));const latest=list[0];if(active){$('#status').textContent=`수집 ${active.id} · ${active.status}`}else if(latest&&latest.id!==runId){$('#status').textContent=`${latest.id} · ${latest.status}${latest.error?' · '+latest.error:''}${latest.cleanup_error?' · 세션 종료 실패: '+latest.cleanup_error:''}`;if(follow&&latest.status==='completed')await load(latest.id)}else $('#status').textContent=latest?.cleanup_error?'세션 종료 실패: '+latest.cleanup_error:''}
$('#runs').onchange=e=>{follow=false;load(e.target.value).catch(fail)};$('#refresh').onclick=()=>refresh().catch(fail);$('#clear').onclick=()=>{moduleId=null;render()};
$('#phases').querySelectorAll('button').forEach(b=>b.onclick=()=>setPhase(b.dataset.phase));
$('#roof-phases').querySelectorAll('button').forEach(b=>b.onclick=()=>setPhase(b.dataset.roofPhase));
['search','bound','step'].forEach(id=>$('#'+id).addEventListener(id==='search'?'input':'change',table));
$('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=`${runId}.json`;a.click();URL.revokeObjectURL(a.href)};
(async()=>{await load('example');await refresh();setInterval(()=>refresh().catch(fail),2000)})().catch(fail);

let experimentId=null,experimentRecords=[],experimentFingerprint='';
const experimentStatuses={completed:'완료',failed:'실패',inconclusive:'판단 보류'};
const correctnessStatuses={passed:'통과',failed:'실패',not_checked:'미검증'};
const comparisonReasons={baseline_missing:'기준 실행 없음',candidate_missing:'후보 실행 없음',baseline_unavailable:'기준 결과를 찾을 수 없음',candidate_unavailable:'후보 결과를 찾을 수 없음',baseline_not_independent:'기준 실행의 독립 측정 없음',candidate_not_independent:'후보 실행의 독립 측정 없음',gpu_mismatch:'GPU가 다름',model_mismatch:'모델이 다름',workload_mismatch:'측정 조건이 다름',correctness_not_passed:'정확성 검증 통과 필요',attempt_not_completed:'완료되지 않은 시도'};
function renderExperiments(){
 const series=$('#experiment-series').value;
 const filtered=experimentRecords.filter(e=>!series||e.series===series);
 if(!filtered.some(e=>e.id===experimentId))experimentId=filtered[0]?.id||null;
 const fingerprint=JSON.stringify([experimentRecords,series,experimentId]);
 if(fingerprint===experimentFingerprint)return;experimentFingerprint=fingerprint;
 $('#experiment-series').innerHTML='<option value="">모든 시리즈</option>'+[...new Set(experimentRecords.map(e=>e.series))].map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
 $('#experiment-series').value=series;
 const improvement=e=>{const v=e.comparison.latency_reduction_pct.total_ms;return v==null?'비교 보류':`${v>=0?'시간 단축':'시간 증가'} ${num(Math.abs(v),1)}%`};
 $('#experiment-list').innerHTML=filtered.length?'<div class="table-wrap"><table class="experiment-table"><thead><tr><th>시도 · 시리즈</th><th>상태</th><th>정확성</th><th>전체 시간 변화</th><th>기록 시각</th></tr></thead><tbody>'+filtered.map(e=>`<tr class="${e.id===experimentId?'selected':''}"><td><button class="experiment-open" data-experiment="${esc(e.id)}" aria-pressed="${e.id===experimentId}">${esc(e.title)}</button><small>${esc(e.series)} · ${esc(e.id)}</small></td><td>${experimentStatuses[e.status]}</td><td>${correctnessStatuses[e.correctness.status]}</td><td>${improvement(e)}</td><td>${esc(new Date(e.created_at*1000).toLocaleString('ko-KR'))}</td></tr>`).join('')+'</tbody></table></div>':'<div class="empty">아직 기록된 실험이 없습니다. 에이전트가 POST /api/experiments로 시도별 결과를 기록할 수 있습니다.</div>';
 $('#experiment-list').querySelectorAll('[data-experiment]').forEach(b=>b.onclick=()=>{experimentId=b.dataset.experiment;renderExperiments()});
 const e=filtered.find(e=>e.id===experimentId),detail=$('#experiment-detail');detail.hidden=!e;if(!e)return;
 const c=e.comparison;
 const field=(label,value)=>`<div><h3>${label}</h3><p>${esc(value||'기록 없음')}</p></div>`;
 const runButton=(role,label)=>c[role]?`<button data-experiment-run="${esc(c[role].run_id)}">${label} ${esc(c[role].run_id)} ↗</button>`:'';
 const comparisonTable=c.baseline||c.candidate?'<div class="table-wrap"><table><thead><tr><th>구간</th><th>기준 시간</th><th>후보 시간</th><th>배속 · 기준 / 후보</th></tr></thead><tbody>'+[['total_ms','전체'],['prefill','Prefill'],['decode','Decode']].map(([key,label])=>`<tr><td>${label}</td><td>${time(c.baseline?.timings[key])}</td><td>${time(c.candidate?.timings[key])}</td><td>${c.speedup[key]==null?'—':num(c.speedup[key],3)+'×'}</td></tr>`).join('')+'</tbody></table></div>':'';
 const conditions=['baseline','candidate'].filter(role=>c[role]).map(role=>({role,run_id:c[role].run_id,model:c[role].model_name,hardware:c[role].hardware.name,timing_source:c[role].timings.source,config:c[role].config}));
 detail.innerHTML=`<div class="experiment-detail-head"><h3>${esc(e.title)}</h3><span class="badge">${experimentStatuses[e.status]} · 정확성 ${correctnessStatuses[e.correctness.status]}</span></div><div class="experiment-fields">${field('변경 내용',e.changes)}${field('가설',e.hypothesis)}${field('코드 버전 · 커밋 또는 해시',e.code_revision)}${field('실행 환경',e.environment)}${field('정확성 검증',e.correctness.details)}${field('결과 해석',e.conclusion)}${field('다음 시도',e.next_steps)}</div><div class="experiment-run-links">${runButton('baseline','기준 결과')}${runButton('candidate','후보 결과')}</div>${comparisonTable}<p class="comparison-note">${c.eligible?'같은 모델·GPU·측정 조건의 독립 실행 시간으로 비교했습니다. 정확성 검증은 에이전트가 기록한 결과입니다.':`개선율 비교 보류: ${c.reasons.map(r=>comparisonReasons[r]||r).join(' · ')}`}</p>${conditions.length?`<details class="experiment-conditions"><summary>측정 조건 및 출처</summary><pre>${esc(JSON.stringify(conditions,null,2))}</pre></details>`:''}`;
 detail.querySelectorAll('[data-experiment-run]').forEach(b=>b.onclick=async()=>{try{follow=false;await load(b.dataset.experimentRun);$('#metrics').scrollIntoView({behavior:'smooth',block:'start'})}catch(error){fail(error)}});
}
async function refreshExperiments(){experimentRecords=await api('/api/experiments');renderExperiments()}
$('#experiment-series').onchange=renderExperiments;
refreshExperiments().catch(fail);setInterval(()=>refreshExperiments().catch(fail),2000);
