let currentResults = [];
const $ = (id) => document.getElementById(id);
async function checkHealth(){
  try{const r=await fetch('/api/health');const d=await r.json();$('serviceStatus').textContent=d.model_ready?'Ready':'Train model';}
  catch{$('serviceStatus').textContent='Offline';}
}
function message(text,error=false){const el=$('message');el.textContent=text;el.className='message'+(error?' error':'');}
function clearMessage(){$('message').className='message hidden';}
async function loadMetrics(){
  try{const r=await fetch('/api/metrics');if(!r.ok)throw new Error();const m=await r.json();
    $('f1').textContent=(m.macro_f1*100).toFixed(1)+'%';
    $('metrics').innerHTML=[['Accuracy',m.accuracy],['Macro precision',m.macro_precision],['Macro recall',m.macro_recall],['Macro F1',m.macro_f1]].map(([k,v])=>`<div class="metric"><span>${k}</span><b>${(v*100).toFixed(1)}%</b></div>`).join('');
  }catch{$('metrics').innerHTML='<p class="empty">Train the model to view evaluation metrics.</p>';}
}
async function runFile(file){
  if(!file)return;clearMessage();message('Analyzing test cases with the ML model…');
  const form=new FormData();form.append('file',file);
  try{const r=await fetch('/api/prioritize',{method:'POST',body:form});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Prediction failed');
    currentResults=d.results;render(d);message(`Successfully prioritized ${d.total} test cases.`);
  }catch(e){message(e.message,true);currentResults=[];$('exportBtn').disabled=true;}
}
function render(data){
  $('total').textContent=data.total;$('critical').textContent=data.priority_counts.Critical||0;$('high').textContent=data.priority_counts.High||0;
  $('resultsBody').innerHTML=data.results.map(x=>`<tr><td class="score">#${x.rank}</td><td><span class="testid">${esc(x.test_case_id)}</span><span class="testname">${esc(x.test_name)}</span></td><td>${esc(x.module)}</td><td><span class="pill ${x.predicted_priority}">${x.predicted_priority}</span></td><td class="score">${x.risk_score}/100</td><td>${x.confidence}%</td><td>${esc(x.last_result)}</td></tr>`).join('');
  const order=['Critical','High','Medium','Low'];const total=data.total||1;
  $('distribution').innerHTML=order.map(p=>{const n=data.priority_counts[p]||0;return `<div class="bar-row"><span>${p}</span><div class="track"><div class="fill" style="width:${n/total*100}%"></div></div><b>${n}</b></div>`}).join('');
  $('exportBtn').disabled=false;
}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
$('csvFile').addEventListener('change',e=>runFile(e.target.files[0]));
$('sampleBtn').addEventListener('click',async()=>{try{const r=await fetch('/api/sample');if(!r.ok)throw new Error('Sample file unavailable. Generate it first.');const blob=await r.blob();runFile(new File([blob],'sample_test_cases.csv',{type:'text/csv'}));}catch(e){message(e.message,true);}});
$('exportBtn').addEventListener('click',()=>{if(!currentResults.length)return;const cols=Object.keys(currentResults[0]);const csv=[cols.join(','),...currentResults.map(row=>cols.map(c=>'"'+String(row[c]??'').replace(/"/g,'""')+'"').join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='prioritized_test_cases.csv';a.click();URL.revokeObjectURL(a.href);});
checkHealth();loadMetrics();
