"""A self-contained local table preview for the upload-ready dataset artifact."""

import json
from .storage import project_path


def write_preview(path, rows):
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    project_path(path).write_text(HTML.replace("__DATASET__", payload))


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDE dataset preview</title>
<style>
:root{color-scheme:light;--border:#e1e5ec;--muted:#657084;--blue:#4b55d4}
*{box-sizing:border-box}body{margin:0;padding:28px;background:#f8f9fc;color:#18202f;font:15px system-ui,-apple-system,sans-serif}
main{max-width:1800px;margin:auto;background:white;border:1px solid var(--border);border-radius:13px;overflow:hidden}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px 24px;border-bottom:1px solid var(--border)}
h1{font-size:24px;margin:0;font-weight:650}.tag{background:#eef0ff;color:#414aac;padding:6px 11px;border-radius:20px;font-size:12px}
.controls{display:flex;align-items:end;gap:22px;padding:18px 24px;border-bottom:1px solid var(--border);flex-wrap:wrap}
label{display:grid;gap:7px;color:var(--muted);font-size:13px}select,input{height:39px;border:1px solid #d5dbe6;border-radius:6px;background:white;padding:0 11px;color:#18202f;font:inherit}
select{min-width:200px}input{min-width:230px}.count{margin-left:auto;color:var(--muted);font-size:13px;padding-bottom:10px}
.note{padding:12px 24px;background:#f6f7fb;font-size:13px;line-height:1.7;color:var(--muted);border-bottom:1px solid var(--border)}
.scroll{overflow:auto}table{border-collapse:collapse;table-layout:fixed;width:2100px;text-align:left}th,td{padding:18px 17px;border-bottom:1px solid var(--border);vertical-align:top}
th{font:600 15px ui-monospace,SFMono-Regular,monospace;background:white}th small{display:block;color:var(--muted);font-weight:400;font-size:12px;margin-top:7px}
td{border-right:1px solid #edf0f5;font:13px ui-monospace,SFMono-Regular,monospace}tbody tr:nth-child(odd){background:#f8f9fc}
th:nth-child(1){width:270px}th:nth-child(2){width:310px}th:nth-child(3){width:570px}th:nth-child(4){width:440px}th:nth-child(5){width:310px}th:nth-child(6){width:200px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;margin:8px 0 0;font:inherit}summary{cursor:pointer;line-height:1.65;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}summary::before{content:'▸ ';color:var(--muted)}details[open] summary::before{content:'▾ '}summary:hover{color:var(--blue)}
.case{border:0;padding:0;background:transparent;text-align:left;font:inherit;color:#253050;cursor:pointer}.case:hover{text-decoration:underline;color:var(--blue)}
.footer{display:flex;gap:15px;flex-wrap:wrap;padding:15px 24px;font-size:12px;color:var(--muted)}button.export{margin-left:auto;border:1px solid #d5dbe6;border-radius:6px;padding:7px 12px;background:white;color:#283046;cursor:pointer}
dialog{border:1px solid #d5dbe6;border-radius:12px;width:min(1100px,92vw);height:85vh;padding:0;box-shadow:0 18px 70px #18202f30}dialog::backdrop{background:#18202f55}
.dialog-head{display:flex;justify-content:space-between;gap:15px;padding:18px 22px;border-bottom:1px solid var(--border);position:sticky;top:0;background:white}.dialog-head button{border:0;border-radius:6px;background:#eef0f5;padding:8px 12px;cursor:pointer}
#detail{padding:20px;font:13px/1.6 ui-monospace,monospace}a{color:var(--blue)}
@media(max-width:700px){body{padding:10px}header{padding:16px}h1{font-size:20px}.controls{padding:16px;gap:14px}label,select,input{width:100%}.count{margin-left:0;padding:0}.tag{display:none}}
</style>
</head>
<body><main>
<header><h1>Dataset Preview</h1><span class="tag">48 public cases · 6 PDE families · typed Parquet</span></header>
<div class="controls">
<label>Configuration<select id="family" aria-label="PDE family"></select></label>
<label>Split<select id="split"><option value="train">train</option><option value="validation">validation</option></select></label>
<label>Filter by problem ID<input id="filter" placeholder="e.g. v0" type="search"></label>
<span id="count" class="count" aria-live="polite"></span>
</div>
<div class="note">Each row is a PDE problem specification. Expand a cell to inspect its fields, or select a problem ID for the complete record.
This is a local preview; the package has not been published to Hugging Face.</div>
<div class="scroll"><table>
<thead><tr><th>id<small>string</small></th><th>pde_classification<small>struct</small></th><th>oracle_config<small>struct</small></th><th>evaluation_config<small>struct</small></th><th>solver_config<small>struct</small></th><th>provenance<small>struct</small></th></tr></thead>
<tbody id="body"></tbody></table></div>
<div class="footer"><span>Train = evolution · Validation = development · All reserved final cases remain private.</span><button class="export" id="download">Download selected rows as JSONL</button></div>
</main>
<dialog id="dialog"><div class="dialog-head"><strong id="detail-id"></strong><button id="close">Close</button></div><pre id="detail"></pre></dialog>
<script id="dataset" type="application/json">__DATASET__</script>
<script>
const rows=JSON.parse(document.getElementById('dataset').textContent);
const labels={poisson:'Poisson',heat:'Heat',convection_diffusion:'Convection–diffusion',burgers:'Viscous Burgers',caputo:'Caputo subdiffusion',multi_term:'Multi-term fractional diffusion'};
const family=document.getElementById('family'),split=document.getElementById('split'),filter=document.getElementById('filter'),body=document.getElementById('body'),dialog=document.getElementById('dialog');
for(const [value,label] of Object.entries(labels)){const option=document.createElement('option');option.value=value;option.textContent=label;family.append(option)}
let selected=[];
function draw(){
 selected=rows.filter(r=>r.pde_classification.equation_type===family.value&&(r.provenance.source_split==='evolution'?'train':'validation')===split.value&&r.id.toLowerCase().includes(filter.value.toLowerCase()));
 body.replaceChildren();document.getElementById('count').textContent=`${selected.length} rows · ${family.value} / ${split.value}`;
 for(const row of selected){const tr=document.createElement('tr');const id=document.createElement('td'),button=document.createElement('button');button.className='case';button.textContent=row.id;button.addEventListener('click',()=>{document.getElementById('detail-id').textContent=row.id;document.getElementById('detail').textContent=JSON.stringify(row,null,2);dialog.showModal()});id.append(button);tr.append(id);
  for(const column of ['pde_classification','oracle_config','evaluation_config','solver_config','provenance']){const td=document.createElement('td'),details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');const compact=JSON.stringify(row[column],null,1).replace(/\n\s*/g,' ');summary.textContent=compact.slice(0,160)+(compact.length>160?' …':'');pre.textContent=JSON.stringify(row[column],null,2);details.append(summary,pre);td.append(details);tr.append(td)}body.append(tr)}
}
family.addEventListener('change',draw);split.addEventListener('change',draw);filter.addEventListener('input',draw);
document.getElementById('close').addEventListener('click',()=>dialog.close());
document.getElementById('download').addEventListener('click',()=>{const blob=new Blob([selected.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${family.value}-${split.value}.jsonl`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
const initial=new URLSearchParams(location.search);
if(Object.hasOwn(labels,initial.get('family')))family.value=initial.get('family');
if(['train','validation'].includes(initial.get('split')))split.value=initial.get('split');
draw();
</script></body></html>'''
