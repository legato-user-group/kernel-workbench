import { moduleLabel } from './kernel-labels.js';
import { runnerExperiment, inputLength, selectableResults } from './experiment-groups.js';
import { correctnessRows } from './correctness-table.js';
import { kernelTree } from './kernel-tree.js';
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, phase='all',moduleId=null,runId='example', active=null,follow=true,runRecords=[],selectedRunner='example';
const kernelExpansion=new Map();
const currentRunner=()=>selectedRunner;
const num=(v,d=2)=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d});
const time=v=>v==null?'—':v<.001?`${num(v*1e6,1)} ns`:v<1?`${num(v*1000,2)} µs`:`${num(v,3)} ms`;
const quantity=v=>v==null?'—':v>=1e9?`${num(v/1e9)} G`:v>=1e6?`${num(v/1e6)} M`:num(v);
async function api(path,options){const r=await fetch(path,options);if(!r.ok){let e=await r.json();throw Error(e.detail||r.statusText)}return r.json()}
function fail(e){$('#status').textContent=e.message}
function metric(label,value,detail,bar){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-detail">${detail}</div>${bar==null?'':`<div class="bar"><span style="width:${bar}%"></span></div>`}</div>`}
function render(){
 $('#model-name').textContent=data.model.name;
 const t=data.timings,p=data.point;
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
 $('#notice').textContent=runId==='example'?'화면 검증용 예제 데이터입니다. 실제 GPU 측정 결과가 아닙니다.':p?.source==='estimated'?'GPU 카운터가 불완전합니다. 전체 점은 모델 연산·논리적 텐서 트래픽 추정치이며 실제 DRAM 측정값과 다릅니다.':'';
 function node(n){return n.children?`<details class="node" ${n.id?.startsWith('layer.')?'':'open'}><summary>${esc(n.label)}</summary>${n.children.map(node).join('')}</details>`:`<div class="node"><button data-module="${esc(n.id)}" class="${moduleId===n.id?'selected':''}">◇ ${esc(n.label)}</button></div>`}
 $('#tree').innerHTML=data.model.children.map(node).join('');
 $('#tree').querySelectorAll('button').forEach(b=>b.onclick=()=>{moduleId=b.dataset.module;render()});
 $('#gpu-name').textContent=data.hardware.name;
 roofline();table();renderEstimates();renderExperiments();
 $('#step').innerHTML='<option value="all">모든 decode 단계</option>'+[...new Set(data.kernels.flatMap(k=>k.steps?.length?k.steps.map(s=>s.step):k.step==null?[]:[k.step]))].sort((a,b)=>a-b).map(s=>`<option value="${s}">Decode ${s}</option>`).join('');
 $('#diagnostics').textContent=JSON.stringify({config:data.config,coverage:data.coverage,profiling:data.profiling,versions:data.versions,timings:data.timings,warnings:data.warnings,hardware:data.hardware},null,2);
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
 document.querySelectorAll('[data-phase]').forEach(button=>button.classList.toggle('active',button.dataset.phase===phase));
 document.querySelectorAll('[data-roof-phase]').forEach(button=>button.classList.toggle('active',button.dataset.roofPhase===phase));
 roofline();table();renderEstimates();
}

function table(){
 $('#bound').disabled=!data.kernels.length;$('#step').disabled=!data.kernels.length;
 if(!data.kernels.length&&data.kernel_plan?.length){implementationTable();return}
 const profile=data.profiling;
 $('#kernel-evidence').textContent=data.kernels.length?profile?`실측 GPU 커널 · ${profile.decode_steps}회 decode · 토큰당 ${num(profile.phases.decode.launches/profile.decode_steps,0)} launch · ${profile.module_count?`${profile.module_count}개 레이어·연산 범위 · NVTX + graph node로 귀속 확인`:'모듈 귀속 미확인은 unattributed'}. ${profile.bounds_samples?`하한: ${profile.bounds_groups}/${data.kernels.length}그룹에 형상별 카운터 표본 연결(레이어 간 공유) · decode 표본은 0번째 단계 · 전체 트래픽으로 환산하지 않음`:`전체 FLOPs·DRAM 카운터 미수집${profile.counter_sample?` · 별도 ${profile.counter_sample.launches}-launch 카운터 표본`:''}`}`:'실측 GPU 커널 · 수집 프로파일 기준':'커널 프로파일 미수집 · 실측 시간과 호출 수 없음';
 const q=$('#search').value.toLowerCase(),bound=$('#bound').value,step=$('#step').value,groups=new Map();
 for(let k of data.kernels){
 if(step!=='all'&&k.steps?.length){const selected=k.steps.find(s=>String(s.step)===step);if(!selected)continue;k={...k,...selected,steps:[]};if(k.bounds_source==='measured_sample'&&k.bounds_reference_step!=null&&String(k.bounds_reference_step)!==step)k={...k,bound:'unknown',compute_ms:null,memory_ms:null,lower_ms:null,bounds_reason:`선택한 단계의 카운터 미수집 · 표본은 decode ${k.bounds_reference_step} 단계`}}
 if(phase!=='all'&&k.phase!==phase||moduleId&&!matchesModule(k.module)||bound!=='all'&&k.bound!==bound||step!=='all'&&String(k.step)!==step||!`${k.name} ${moduleLabel(k,data.config)}`.toLowerCase().includes(q))continue;
 const key=JSON.stringify([k.name,k.phase,k.module,k.grid,k.block,k.bound]);if(!groups.has(key))groups.set(key,{...k,calls:[],call_count:0,duration_ms:0});const g=groups.get(key);g.calls.push(k);g.call_count+=k.calls??1;g.duration_ms+=k.duration_ms;}
 const sorted=[...groups.values()];$('#count').textContent=`${sorted.length} 커널 그룹 · 모델 구조 순서`;
 const lowerCell=(g,key)=>`<td title="${esc(g.bounds_reason||'')}" >${g[key]==null?`<span class="muted">${g.bounds_reason?.startsWith('선택한 단계')?'선택 단계 미수집':key==='compute_ms'?'FLOPs 미수집':key==='memory_ms'?'DRAM 미수집':'산출 불가'}</span>`:`${time(g[key])}<small> / ${g.bounds_source==='measured_sample'?'표본':'call'}</small>`}</td>`;
 renderKernelTree(sorted,(g,i,depth)=>`<tr class="expandable kernel-tree-leaf" tabindex="0" aria-expanded="false" aria-level="${depth+1}" data-group="${i}"><td style="padding-left:${18+depth*16}px"><b title="${esc(g.name)}">▸ ${esc(g.name.length>95?g.name.slice(0,95)+'…':g.name)}</b>${g.module==='unattributed'?`<small class="operation-hint">${esc(moduleLabel(g,data.config))}</small>`:''}</td><td>${esc(g.phase)}</td><td>${num(g.call_count,0)}</td><td>${time(g.duration_ms)}</td><td><span title="${esc(g.bounds_reason||'')}" class="bound ${g.bound}">${g.bound==='unknown'?(g.bounds_source==='measured_sample'?'미분류':'카운터 미수집'):g.bound+'-bound'}</span>${g.bound!=='unknown'&&g.bounds_source==='measured_sample'?'<small>표본 · roofline</small>':''}</td>${lowerCell(g,'compute_ms')}${lowerCell(g,'memory_ms')}${lowerCell(g,'lower_ms')}</tr>`);
 $('#rows').querySelectorAll('[data-group]').forEach(row=>{const toggle=()=>{if(row.nextElementSibling?.classList.contains('detail')){row.nextElementSibling.remove();row.setAttribute('aria-expanded','false');return}const g=sorted[+row.dataset.group];const tr=document.createElement('tr');tr.className='detail';tr.innerHTML=`<td colspan="8">시간·호출 수는 전체 타임라인 집계이며, 하한은 별도 카운터 표본 기준입니다.<pre>${esc(JSON.stringify(g.calls.map(k=>({name:k.name,module:k.module,module_source:k.module_source,step:k.step,calls:k.calls??1,duration_ms:k.duration_ms,steps:k.steps,flops:k.flops,dram_bytes:k.bytes,source:k.source,compute_ms:k.compute_ms,memory_ms:k.memory_ms,lower_ms:k.lower_ms,bounds_reason:k.bounds_reason,counter_samples:k.counter_samples,grid:k.grid,block:k.block})),null,2))}</pre></td>`;row.after(tr);row.setAttribute('aria-expanded','true')};row.onclick=toggle;row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle()}}});
}
function renderKernelTree(rows,leaf){
 if(!rows.length){$('#rows').innerHTML='<tr><td colspan="8" class="empty">선택한 조건에 해당하는 커널 또는 구현 경로가 없습니다.</td></tr>';return}
 const forced=!!moduleId||!!$('#search').value.trim();
 function branch(node,depth){
  const expansionKey=forced?`${moduleId||''}|${$('#search').value.trim()}|${node.id}`:node.id;
  const open=kernelExpansion.get(expansionKey)??(forced||!/^layer\.\d+$/.test(node.id)||node.id==='layer.0');
  const idLabel=node.id==='$model'?'':`<small>${esc(node.id)}</small>`;
  let html=`<tr class="kernel-tree-branch" aria-level="${depth+1}" aria-expanded="${open}" data-tree-id="${esc(node.id)}"><td colspan="8" style="padding-left:${18+depth*16}px"><button class="kernel-branch" data-kernel-node="${esc(node.id)}" data-expansion-key="${esc(expansionKey)}" aria-expanded="${open}"><span aria-hidden="true">${open?'▾':'▸'}</span><b>${esc(node.label)}</b>${idLabel}</button></td></tr>`;
  if(open){html+=node.children.map(child=>branch(child,depth+1)).join('');html+=node.entries.map(e=>leaf(e.row,e.index,depth+1)).join('')}
  return html;
 }
 $('#rows').innerHTML=branch(kernelTree(rows,data.model),0);
 $('#rows').querySelectorAll('[data-kernel-node]').forEach(button=>button.onclick=()=>{kernelExpansion.set(button.dataset.expansionKey,button.getAttribute('aria-expanded')!=='true');table()});
}
function renderRunSelectors(){
 const family=currentRunner();
 const families=[...new Set([...runRecords].sort((a,b)=>a.config.input_tokens-b.config.input_tokens).map(r=>r.runner_id||r.config.runner))];
 $('#runner').innerHTML='<option value="example">예제 · 화면 검증</option>'+families.map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join('');
 $('#runner').value=family;
 const rows=selectableResults(runRecords,experimentRecords).filter(r=>(r.runner_id||r.config.runner)===family);
 $('#runs').innerHTML=family==='example'?'<option value="example">예제</option>':rows.map(r=>`<option value="${esc(r.id)}">${r.config.input_tokens} → ${r.config.output_tokens} tokens · ${esc(r.config.runner)}${r.data_directory?` · ${esc(r.data_directory.split('/').at(-1))}`:''}</option>`).join('');
 $('#runs').value=runId||'';
 $('#runs').hidden=rows.length<=1;
 $('#runs').disabled=!rows.length;
}
function matchesModule(name){return !moduleId||name===moduleId||name.startsWith(moduleId+'.')}
function renderEstimates(){
 const rows=data.estimates.filter(o=>(phase==='all'||o.phase===phase)&&matchesModule(o.module));
 $('#estimate-scope').textContent=data.estimate_scope||'Runner가 제공한 논리적 연산·텐서 트래픽 추정';
 $('#estimates').innerHTML='<div class="table-wrap"><table><thead><tr><th>모델 연산</th><th>단계</th><th>FLOPs (추정)</th><th>논리적 bytes (추정)</th><th>모델 하한 (추정)</th></tr></thead><tbody>'+rows.map(o=>`<tr><td>${esc(o.module)}<small>${esc(o.shape||'')}</small></td><td>${esc(o.phase)}</td><td>${quantity(Object.values(o.flops).reduce((a,b)=>a+b,0))}</td><td>${quantity(o.bytes)}</td><td>${time(o.lower_ms)}</td></tr>`).join('')+(rows.length?'':'<tr><td colspan="5" class="empty">선택한 구간·연산의 비용 추정이 없습니다.</td></tr>')+'</tbody></table></div>';
}
function implementationTable(){
 const q=$('#search').value.toLowerCase();
 const rows=data.kernel_plan.filter(o=>(phase==='all'||o.phase===phase)&&matchesModule(o.module)&&`${o.module} ${o.implementation}`.toLowerCase().includes(q));
 $('#kernel-evidence').textContent='구현 경로 · 소스에서 확인한 연산이며 실측 GPU 커널 목록이 아닙니다. 시간·실제 호출 수·DRAM 카운터는 프로파일 미수집입니다.';
 $('#count').textContent=`${rows.length} 구현 경로 · 실측 미수집`;
 renderKernelTree(rows,(o,i,depth)=>`<tr class="kernel-tree-leaf" aria-level="${depth+1}" data-plan-module="${esc(o.module)}"><td style="padding-left:${18+depth*16}px"><b>${esc(o.implementation)}</b><small>${esc(o.shape)}</small></td><td>${esc(o.phase)}</td><td>${o.calls==null?'미수집':num(o.calls)+' (추정)'}</td><td>—</td><td>미수집</td><td>—</td><td>—</td><td>—</td></tr>`);
}
async function load(id){
 runId=id;moduleId=null;
 data=await api(id==='example'?'/example.json':`/api/runs/${id}/result`);
 selectedRunner=id==='example'?'example':data.runner_id||data.config.runner;
 $('#analysis-body').hidden=false;$('#result-unavailable').hidden=true;$('#export').disabled=false;
 experimentId=null;renderRunSelectors();render();
}
async function showRunner(family, attemptId=null){
 if(family==='example'){await load('example');return}
 const candidates=selectableResults(runRecords,experimentRecords).filter(r=>(r.runner_id||r.config.runner)===family);
 const attempt=attemptId?experimentRecords.find(e=>e.id===attemptId):null;
 const selected=attempt?candidates.find(r=>r.id===attempt.candidate_run_id):candidates[0];
 if(selected){await load(selected.id);return}
 selectedRunner=family;runId=null;data=null;moduleId=null;experimentId=attemptId;
 $('#analysis-body').hidden=true;$('#result-unavailable').hidden=false;$('#export').disabled=true;
 $('#notice').textContent='';
 $('#result-unavailable').textContent='이 runner의 실험 결과는 아직 수집되지 않았습니다. 비교 기준은 실험 기록에 명시되어 있습니다.';
 renderRunSelectors();renderExperiments();
}
async function refresh(){
 const [runs,experiments]=await Promise.all([api('/api/runs'),api('/api/experiments')]);
 runRecords=runs.filter(r=>r.status==='completed');experimentRecords=experiments;
 const latest=selectableResults(runRecords,experimentRecords)[0];
 if(follow&&latest&&latest.id!==runId)await load(latest.id);
 else if(runId!=='example'&&runRecords.some(r=>r.id===runId&&r.evidence_revision!==data?.evidence_revision))await load(runId);
 else if(follow&&!latest&&runRecords.length)await showRunner(runRecords[0].runner_id||runRecords[0].config.runner);
 else {renderRunSelectors();renderExperiments()}
 $('#status').textContent=latest?.cleanup_error?'세션 종료 실패: '+latest.cleanup_error:'';
}
$('#runner').onchange=e=>{follow=false;showRunner(e.target.value).catch(fail)};
$('#runs').onchange=e=>{follow=false;load(e.target.value).catch(fail)};$('#clear').onclick=()=>{moduleId=null;render()};
$('#phases').querySelectorAll('button').forEach(b=>b.onclick=()=>setPhase(b.dataset.phase));
$('#roof-phases').querySelectorAll('button').forEach(b=>b.onclick=()=>setPhase(b.dataset.roofPhase));
['search','bound','step'].forEach(id=>$('#'+id).addEventListener(id==='search'?'input':'change',()=>{table();renderEstimates()}));
$('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=`${runId}.json`;a.click();URL.revokeObjectURL(a.href)};
(async()=>{await load('example');await refresh();setInterval(()=>refresh().catch(fail),2000)})().catch(fail);

let experimentId=null,experimentRecords=[],experimentFingerprint='';
const experimentStatuses={completed:'완료',failed:'실패',inconclusive:'판단 보류'};
const correctnessStatuses={passed:'통과',failed:'실패',not_checked:'미검증'};
function renderExperiments(){
 const series=$('#experiment-series').value;
 const group=runnerExperiment(experimentRecords,currentRunner(),series);
 const filtered=group?.cases||[];
 if(!filtered.some(e=>e.id===experimentId))experimentId=filtered.find(e=>e.baseline_run_id===runId||e.candidate_run_id===runId)?.id||filtered[0]?.id||null;
 const fingerprint=JSON.stringify([experimentRecords,series,experimentId,currentRunner()]);
 if(fingerprint===experimentFingerprint)return;experimentFingerprint=fingerprint;
 $('#experiment-series').innerHTML='<option value="">모든 시리즈</option>'+[...new Set(experimentRecords.map(e=>e.series))].map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
 $('#experiment-series').value=series;
 $('#experiment-list').innerHTML=group?'':'<div class="empty">선택한 runner의 최적화 실험이 없습니다.</div>';
 const e=filtered.find(e=>e.id===experimentId),detail=$('#experiment-detail');
 detail.hidden=!e;$('#experiment-correctness').hidden=!e;
 if(!e)return;
 $('#correctness-status').textContent=({passed:'성공',failed:'실패',not_checked:'미검증'})[e.correctness.status];
 $('#correctness-status').dataset.status=e.correctness.status;
 const mtp=e.correctness.details.includes('int8_mtp_parallel_verify4:');
 const candidateLabel=mtp?'INT8 + MTP':(e.comparison.candidate?.config.dtype?.toUpperCase()||'평가 경로');
 $('#correctness-details').innerHTML=`<div class="table-wrap"><table><thead><tr><th>검증 항목</th><th>통과 기준</th><th>${mtp?'INT8 plain':'기준 경로'}</th><th>${esc(candidateLabel)}</th></tr></thead><tbody>`+correctnessRows(e.correctness.details).map(row=>`<tr><th scope="row">${esc(row.item)}</th><td>${esc(row.criterion)}</td><td>${esc(row.baseline)}</td><td>${esc(row.candidate)}</td></tr>`).join('')+'</tbody></table></div>';
 const field=(label,value)=>`<div><h3>${label}</h3><p>${esc(value||'기록 없음')}</p></div>`;
 detail.innerHTML=`<div class="experiment-detail-head"><h3>${esc(group.runnerId)}</h3>${filtered.length>1?`<select id="experiment-case" aria-label="최적화 시도">${filtered.map(item=>`<option value="${esc(item.id)}" ${item.id===e.id?'selected':''}>${esc(item.title)}</option>`).join('')}</select>`:''}</div><div class="experiment-fields">${field('가설',e.hypothesis)}${field('결과 해석',e.conclusion)}</div>`;
 if($('#experiment-case'))$('#experiment-case').onchange=event=>{follow=false;showRunner(currentRunner(),event.target.value).catch(fail)};
}
$('#experiment-series').onchange=renderExperiments;
