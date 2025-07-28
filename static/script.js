(function(){
  // Add global loadMeta for visualization.html and other pages that expect it
  if(typeof window.loadMeta !== "function"){
    window.loadMeta = function loadMeta() {
      const chip = document.getElementById("active-file-chip");
      const metaBox = document.getElementById("viz-dataset-meta");
      const filename = localStorage.getItem("filename");
      if (chip) chip.textContent = filename ? `Active: ${filename}` : "Active: —";
      if (metaBox) metaBox.textContent = filename ? filename : "No active dataset.";
    };
  }
})();
/* =========================================================
   Global Frontend Script – Data Mining & Visualization Suite (v4.6 FINAL)
   =========================================================
   • Auth persists (SameSite=Lax), credentials: 'include'
   • Active dataset sync across pages (/api/files)
   • Preview via /api/preview_json
   • AutoExplore bundle caching in LS
   • AI insight prompt trimmed + STRICT JSON parse & fence strip
   • Correlation: HTML TABLE + hover/copy + color scale (+ optional canvas heatmap)
   • Chart.js (bar/pie/line/scatter/matrix) loaded once
   • All legacy window.* API names preserved
========================================================= */

/* ---------------- Config & LS helpers ---------------- */

/* -------- BASE_URL auto-detect -------- */
let BASE_URL = window.BASE_URL;
if (!BASE_URL) {
  const origin = window.location.origin;
  if (origin.includes("onrender.com")) {
    BASE_URL = origin;
  } else {
    BASE_URL = "http://127.0.0.1:5050"; // local dev
  }
// End of renderPCA
window.BASE_URL = BASE_URL;

const LS_KEYS_TO_CLEAR = [
  "summary","correlation","valueCounts","pca","kmeans","assoc",
  "autoBundle","autoAI","lastAI","colTypesCache","primaryCategorical"
];

const $      = id => document.getElementById(id);
const lsGet  = k => { try{return JSON.parse(localStorage.getItem(k));}catch{return null;} };
const lsSet  = (k,v)=>{ try{localStorage.setItem(k, typeof v==="string"?v:JSON.stringify(v));}catch(e){console.warn("lsSet",k,e);} };
const lsDel  = k => localStorage.removeItem(k);

/* ---------------- Toast ---------------- */
function toast(msg,type="info",timeout=3000){

function renderPCA() {
  const pca = lsGet("pca") || lsGet("autoBundle")?.pca;
  const km = lsGet("kmeans") || lsGet("autoBundle")?.kmeans;
  const box = $("pca-box") || $("pca-container");
  if (!box) return;
  const comps = pca?.components_2d || pca?.components;
  if (!comps || !Array.isArray(comps) || comps.length === 0) {
    box.innerHTML = "<p class='text-small text-dim'>No PCA data.</p>";
    return;
  }

  // Find min/max for scaling
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  comps.forEach(([x, y]) => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  // Add 5% padding
  const padX = (maxX - minX) * 0.05;
  const padY = (maxY - minY) * 0.05;
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;

  // Try to color by cluster if available
  const labels = km?.labels_preview || km?.labels || [];
  const hasClusters = labels.length === comps.length;
  const colors = ["#02b2ff", "#b136ff", "#ffb86c", "#10b981", "#ef4444", "#f59e42", "#7a4bff", "#ff7a7a"];
  const datasets = hasClusters
    ? [...new Set(labels)].map((cl, idx) => ({
        label: "Cluster " + cl,
        data: comps.map((p, i) => labels[i] === cl ? { x: p[0], y: p[1] } : null).filter(Boolean),
        backgroundColor: colors[cl % colors.length] + "cc",
        pointRadius: 4,
        pointHoverRadius: 7,
      }))
    : [{
        label: "PCA",
        data: comps.map(p => ({ x: p[0], y: p[1] })),
        backgroundColor: "#02b2ffcc",
        pointRadius: 4,
        pointHoverRadius: 7,
      }];

  box.innerHTML = "<canvas id='pca-canvas' style='width:100%;height:100%'></canvas>";
  const ctx = $("pca-canvas").getContext("2d");
  if (VizCharts.pca) VizCharts.pca.destroy();

  const exp = pca?.explained || pca?.explained_variance || [];
  VizCharts.pca = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              let txt = `(${d.x.toFixed(2)}, ${d.y.toFixed(2)})`;
              if (hasClusters && ctx.dataset.label) txt += ` | ${ctx.dataset.label}`;
              return txt;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: `PC1${exp[0] ? ` (${(exp[0] * 100).toFixed(1)}%)` : ""}`,
            color: "#b8c9d6"
          },
          min: minX,
          max: maxX,
          ticks: { color: getCss('--text-dim') }
        },
        y: {
          title: {
            display: true,
            text: `PC2${exp[1] ? ` (${(exp[1] * 100).toFixed(1)}%)` : ""}`,
            color: "#b8c9d6"
          },
          min: minY,
          max: maxY,
          ticks: { color: getCss('--text-dim') }
        }
      }
    }
  });
  syncExportButtons();
}
}
function resetAnalysisCacheOnDatasetChange(newName){
  const prev=localStorage.getItem("filename");
  if(prev && prev!==newName){ LS_KEYS_TO_CLEAR.forEach(lsDel); }
  localStorage.setItem("filename",newName);
  localStorage.setItem("activeFile",newName);
}

/* ---------------- Preview ---------------- */
async function previewDataset(){
  const box=$("preview-content")||$("data-preview")||$("preview-box");
  if(!box) return;
  const active=await syncActiveFile(true);
  if(!active){ box.innerHTML="<p class='text-small text-dim'>No active dataset.</p>"; return; }
  try{
    const res=await handleApi("/api/preview_json",{method:"POST",body:{filename:active}});
    const cols=res.columns||[];
    const rows=res.rows||[];
    const thead="<thead><tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead>";
    const tbody="<tbody>"+rows.map(r=>"<tr>"+cols.map(c=>`<td>${r[c]??""}</td>`).join("")+"</tr>").join("")+"</tbody>";
    box.innerHTML=`<table class="data-table">${thead}${tbody}</table>`;
  }catch(e){
    box.innerHTML=`<p class='text-small text-danger'>Preview error: ${e.message}</p>`;
  }
}

/* ---------------- Upload / Fetch ---------------- */
async function uploadDataset(){
  const inp=$("file-input"), st=$("upload-status");
  if(!inp||!inp.files.length){ toast("Select a file","warn"); st&&(st.textContent="No file"); return; }
  const fd=new FormData(); fd.append("file",inp.files[0]);
  try{
    st&&(st.textContent="Uploading...");
    const res=await fetch(`${BASE_URL}/api/upload`,{method:"POST",body:fd,credentials:"include"});
    const data=await res.json(); if(data.status==="error") throw new Error(data.error);
    resetAnalysisCacheOnDatasetChange(data.filename);
    st&&(st.textContent="Uploaded ✓"); toast("File uploaded","success");
    await previewDataset(); inferColumnTypes();
  }catch(e){ st&&(st.textContent="Upload failed"); toast("Upload error: "+e.message,"error"); }
}
async function fetchFromInternet(url){
  const st=$("fetch-status"); if(!url) return;
  try{
    st&&(st.textContent="Fetching...");
    const data=await handleApi("/api/fetch-url",{method:"POST",body:{url}});
    resetAnalysisCacheOnDatasetChange(data.filename);
    toast("Fetched remote CSV","success"); st&&(st.textContent="Fetched ✓");
    await previewDataset(); inferColumnTypes();
  }catch(e){
    st&&(st.textContent="Fetch failed");
    // Show backend error if available
    toast("Fetch error: "+(e.message||e.error||"Unknown error"),"error");
  }
}
async function smartSearch(){
  const q=$("search-input"), st=$("search-status"), resBox=$("search-results");
  if(!q||!q.value.trim()){ toast("Enter a search term","warn"); return; }
  try{
    st&&(st.textContent="Searching...");
    const data=await handleApi("/api/smartsearch",{method:"POST",body:{query:q.value.trim()}});
    if(resBox){
      resBox.innerHTML=(data.links||[]).length
        ? data.links.map(l=>`<div><a href="#" onclick="fetchFromInternet('${l}')">${l}</a></div>`).join("")
        : "No results found.";
    }
    st&&(st.textContent="Done");
  }catch(e){
    st&&(st.textContent="Search failed");
    toast("Search error: "+(e.message||"Unknown error"),"error");
  }
}

function renderKMeans() {
  const km = lsGet("kmeans") || lsGet("autoBundle")?.kmeans;
  const pca = lsGet("pca") || lsGet("autoBundle")?.pca;
  const box = $("kmeans-box") || $("kmeans-container");
  if (!box) return;
  const labels = km?.labels_preview || km?.labels;
  if (!labels || !Array.isArray(labels) || labels.length === 0) {
    box.innerHTML = "<p class='text-small text-dim'>No clustering data.</p>";
    return;
  }

  // Use PCA for 2D plotting if available
  const comps = pca?.components_2d;
  if (!comps || comps.length !== labels.length) {
    // fallback: 1D cluster plot
    const pts = labels.map((lab, i) => ({ x: i, y: lab }));
    // Find min/max for scaling
    let minX = 0, maxX = labels.length - 1, minY = Math.min(...labels), maxY = Math.max(...labels);
    const padX = (maxX - minX) * 0.05;
    const padY = (maxY - minY) * 0.05;
    minX -= padX; maxX += padX; minY -= padY; maxY += padY;
    box.innerHTML = "<canvas id='kmeans-canvas' style='width:100%;height:100%'></canvas>";
    const ctx = $("kmeans-canvas").getContext("2d");
    if (VizCharts.kmeans) VizCharts.kmeans.destroy();
    VizCharts.kmeans = new Chart(ctx, {
      type: "scatter",
      data: { datasets: [{ label: "Clusters", data: pts, backgroundColor: "#b136ffcc" }] },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: minX, max: maxX, title: { display: true, text: "Index", color: "#b8c9d6" }, ticks: { color: getCss('--text-dim') } },
          y: { min: minY, max: maxY, title: { display: true, text: "Cluster", color: "#b8c9d6" }, ticks: { color: getCss('--text-dim') } }
        }
      }
    });
    syncExportButtons();
    return;
  }

  // 2D cluster plot in PCA space
  // Find min/max for scaling
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  comps.forEach(([x, y]) => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  const padX = (maxX - minX) * 0.05;
  const padY = (maxY - minY) * 0.05;
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;

  const colors = ["#02b2ff", "#b136ff", "#ffb86c", "#10b981", "#ef4444", "#f59e42", "#7a4bff", "#ff7a7a"];
  const clusters = [...new Set(labels)];
  const datasets = clusters.map((cl, idx) => ({
    label: "Cluster " + cl,
    data: comps.map((p, i) => labels[i] === cl ? { x: p[0], y: p[1] } : null).filter(Boolean),
    backgroundColor: colors[cl % colors.length] + "cc",
    pointRadius: 4,
    pointHoverRadius: 7,
  }));

  // Cluster centers (if available)
  let centers = [];
  if (km?.centers && km.centers[0].length >= 2) {
    centers = km.centers.map((c, i) => ({
      x: c[0], y: c[1], cluster: i
    }));
    datasets.push({
      label: "Centers",
      data: centers,
      backgroundColor: "#fff",
      borderColor: "#000",
      pointRadius: 8,
      pointStyle: "rectRot",
      showLine: false
    });
  }

  box.innerHTML = "<canvas id='kmeans-canvas' style='width:100%;height:100%'></canvas>";
  const ctx = $("kmeans-canvas").getContext("2d");
  if (VizCharts.kmeans) VizCharts.kmeans.destroy();

  VizCharts.kmeans = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              let txt = `(${d.x.toFixed(2)}, ${d.y.toFixed(2)})`;
              if (ctx.dataset.label && ctx.dataset.label.startsWith("Cluster")) txt += ` | ${ctx.dataset.label}`;
              if (ctx.dataset.label === "Centers") txt += " (center)";
              return txt;
            }
          }
        }
      },
      scales: {
        x: { min: minX, max: maxX, title: { display: true, text: "PC1", color: "#b8c9d6" }, ticks: { color: getCss('--text-dim') } },
        y: { min: minY, max: maxY, title: { display: true, text: "PC2", color: "#b8c9d6" }, ticks: { color: getCss('--text-dim') } }
      }
    }
  });
  syncExportButtons();
}


function cleanAIBlock(objOrStr){
  if(typeof objOrStr === "string"){
    const s = stripFences(objOrStr);
    try { return JSON.parse(s); } catch { return {summary:s}; }
  }
  if(!objOrStr || typeof objOrStr !== "object") return objOrStr;
  return {
    summary:        stripFences(objOrStr.summary        ?? objOrStr.overview ?? ""),
    key_points:     Array.isArray(objOrStr.key_points||objOrStr.key_findings)
                      ? (objOrStr.key_points||objOrStr.key_findings).map(stripFences) : [],
    anomalies:      Array.isArray(objOrStr.anomalies) ? objOrStr.anomalies.map(stripFences) : [],
    recommendation: stripFences(objOrStr.recommendation ?? ""),
    next_steps:     Array.isArray(objOrStr.next_steps) ? objOrStr.next_steps.map(stripFences) : []
  };
}

async function generateAISummary(){
  const chartType=$("chart-type")?.value || $("qi-context")?.value || "overview";
  const descBox=$("chart-description")? "chart-description":"qi-desc";
  const description=($(descBox)?.value||"Key findings & anomalies").trim();
  const out=$("ai-summary")||$("qi-output")||$("ai-narrative-box");
  const st=$("qi-status")||$("ai-status");
  out&&(out.innerHTML="<em>Generating...</em>"); st&&(st.textContent="…");
  try{
    const snippet=await buildAISnippet(20);
    const richDesc = `
DATA BRIEF (JSON):
${JSON.stringify(snippet)}

USER CONTEXT: ${chartType}
USER PROMPT: ${description}

TASK: Act as a senior data analyst. Using ONLY the data above, provide:
- 2 sentence high-level summary
- 4–6 concise bullet key findings (use numbers/columns)
- Any notable anomalies/outliers (array)
- 1 actionable recommendation

Return STRICT JSON with keys exactly:
{"summary":"","key_points":[],"anomalies":[],"recommendation":""}
`;
    let data=await handleApi("/api/ai_summary",{method:"POST",body:{chart_type:chartType,description:richDesc}});

    if(data?.status==="ok"){
      data={summary:data.summary,key_points:data.key_points,anomalies:data.anomalies,recommendation:data.recommendation};
    }
    if(typeof data==="string"){
      const cleaned=stripFences(data);
      try{ data=JSON.parse(cleaned); }catch{ data={summary:cleaned}; }
    }
    const safe={
      summary:stripFences(data?.summary||""),
      key_points:Array.isArray(data?.key_points)?data.key_points.map(stripFences):[],
      anomalies:Array.isArray(data?.anomalies)?data.anomalies.map(stripFences):[],
      recommendation:stripFences(data?.recommendation||"")
    };
    lsSet("lastAI",safe);

    let html="";
    if(safe.summary) html+=`<p>${safe.summary}</p>`;
    if(safe.key_points.length){
      html+=`<strong style="font-size:.6rem;">Key Points</strong><ul style="margin:.25rem 0 .6rem 1rem;">${safe.key_points.map(k=>`<li>${k}</li>`).join("")}</ul>`;
    }
    if(safe.anomalies.length){
      html+=`<strong style="font-size:.6rem;">Anomalies</strong><ul style="margin:.25rem 0 .6rem 1rem;">${safe.anomalies.map(a=>`<li>${a}</li>`).join("")}</ul>`;
    }
    if(safe.recommendation){
      html+=`<p style="font-size:.6rem;"><strong>Recommendation:</strong> ${safe.recommendation}</p>`;
    }
    out&&(out.innerHTML=html||"<em>No structured output.</em>");
    st&&(st.textContent="✓");
    toast("AI summary ready","success");
  }catch(e){
    out&&(out.innerHTML=`<span style="color:#ef4444;font-size:.62rem;">AI Error: ${e.message}</span>`);
    st&&(st.textContent="Error");
    toast("AI error: "+e.message,"error");
  }
}

/* ---------------- Auto Explore ---------------- */
async function autoExplore(){
  const prog=$("auto-explore-progress")||$("viz-auto-status");
  prog&&(prog.textContent="Running auto exploration...");
  try{
    const res=await handleApi("/api/auto_explore",{method:"POST"});
    storeAutoBundle(res);
    prog&&(prog.textContent="Complete ✓");
    toast("Auto Explore complete","success");
    inferColumnTypes();
    if(document.body.getAttribute("data-page")==="visualization"){
      updateVizDatasetMeta();
      ensureCorrelation();
      renderOverview();
      renderAINarrative();
      syncExportButtons();
    }
  }catch(e){
    prog&&(prog.textContent="Error");
    toast("Auto explore failed: "+e.message,"error");
  }
function updateVizDatasetMeta() {
  const metaBox = document.getElementById("viz-dataset-meta");
  const filename = localStorage.getItem("filename");
  if (metaBox) {
    metaBox.textContent = filename ? filename : "No active dataset.";
  }
}

function renderOverview() {
  const box = document.getElementById("overview-meta");
  const bundle = lsGet("autoBundle");
  if (!box) return;
  if (!bundle || !bundle.profile || !bundle.profile.basic) {
    box.innerHTML = "<p class='text-dim'>No overview available. Run Auto Explore.</p>";
    return;
  }
  const meta = bundle.profile.basic;
  box.innerHTML = `
    <div>
      <strong>Rows:</strong> ${meta.rows ?? "?"} &nbsp; 
      <strong>Columns:</strong> ${meta.columns ?? "?"} &nbsp; 
      <strong>Numeric:</strong> ${meta.numeric_cols ?? "?"} &nbsp; 
      <strong>Categorical:</strong> ${meta.categorical_cols ?? "?"}
    </div>
    <div style="margin-top:.7em;">
      <strong>Sample columns:</strong> ${meta.columns_list ? meta.columns_list.join(", ") : ""}
    </div>
  `;
}

function renderAINarrative() {
  const box = document.getElementById("ai-narrative-box");
  const ai = lsGet("lastAI") || lsGet("autoAI");
  if (!box) return;
  if (!ai) {
    box.innerHTML = "<p class='text-dim'>No AI insight generated yet.</p>";
    return;
  }
  let html = "";
  if (ai.summary) html += `<p><strong>Summary:</strong> ${ai.summary}</p>`;
  if (ai.key_points && ai.key_points.length) {
    html += `<p><strong>Key Points</strong></p><ul>${ai.key_points.map(k => `<li>${k}</li>`).join("")}</ul>`;
  }
  if (ai.anomalies && ai.anomalies.length) {
    html += `<p><strong>Anomalies</strong></p><ul>${ai.anomalies.map(a => `<li>${a}</li>`).join("")}</ul>`;
  }
  if (ai.recommendation) html += `<p><strong>Recommendation:</strong> ${ai.recommendation}</p>`;
  box.innerHTML = html;
}
(function(){
  if(document.body.getAttribute("data-page")==="visualization"){
    document.addEventListener("DOMContentLoaded",function(){
      updateVizDatasetMeta();
      renderOverview();
      renderAINarrative();
    });
  }
})();
}
function storeAutoBundle(result){
  const b=result.bundle, ai=result.ai;
  lsSet("autoBundle",b);
  if(b?.summary)           lsSet("summary",b.summary);
  if(b?.correlation_matrix)lsSet("correlation",compressCorr(b.correlation_matrix));
  if(b?.categorical){
    const first=Object.keys(b.categorical)[0];
    if(first){
      lsSet("valueCounts",{
        labels:b.categorical[first].map(o=>o.value),
        values:b.categorical[first].map(o=>o.count),
        title:`Top ${first}`});
    }
  }
  if(b?.pca)     lsSet("pca",{components_2d:b.pca.components_2d,explained:b.pca.explained_variance});
  if(b?.kmeans)  lsSet("kmeans",{labels_preview:b.kmeans.labels_preview,centers:b.kmeans.centers});
  if(b?.assoc_rules) lsSet("assoc",b.assoc_rules);
  if(ai)         lsSet("autoAI", cleanAIBlock(ai));
}

/* ---------------- Reports / Exports ---------------- */
async function downloadMarkdownReport(){
  const st=$("report-status")||$("viz-export-status");
  st&&(st.textContent="Generating markdown...");
  try{
    const r=await handleApi("/api/report/markdown");
    downloadBlob(new Blob([r.markdown],{type:"text/markdown"}),(r.filename||"report")+"_report.md");
    st&&(st.textContent="Markdown ready");
    toast("Markdown ready","success");
  }catch(e){ st&&(st.textContent="Error"); toast("Report error: "+e.message,"error"); }
}
async function downloadPdfReport(){
  const st=$("report-status")||$("viz-export-status");
  st&&(st.textContent="Generating PDF...");
  try{
    const res=await fetch(`${BASE_URL}/api/report/pdf`,{credentials:"include"});
    if(!res.ok) throw new Error("PDF failed");
    const blob=await res.blob();
    downloadBlob(blob,"dataset_report.pdf");
    st&&(st.textContent="PDF ready");
    toast("PDF ready","success");
  }catch(e){ st&&(st.textContent="Error"); toast("PDF error: "+e.message,"error"); }
}
function downloadBlob(blob,filename){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download=filename; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1500);
}
function dataURLtoBlob(dataurl){
  const arr=dataurl.split(','), mime=arr[0].match(/:(.*?);/)[1];
  const bstr=atob(arr[1]); let n=bstr.length; const u8=new Uint8Array(n);
  while(n--){ u8[n]=bstr.charCodeAt(n); }
  return new Blob([u8],{type:mime});
}
function downloadChart(canvasId="main-chart",fileName="chart.png"){
  const c=$(canvasId); if(!c){ toast("Chart not found","warn"); return; }
  downloadBlob(dataURLtoBlob(c.toDataURL("image/png")),fileName);
}
function exportDataCsv(){
  const vc=lsGet("valueCounts");
  if(vc){
    const lines=["label,value",...vc.labels.map((l,i)=>`"${l.replace(/\"/g,'\"\"')}",${vc.values[i]}`)];
    downloadBlob(new Blob([lines.join("\n")],{type:"text/csv"}),"value_counts.csv"); return;
  }
  const corr=getCorrelationMatrix();
  if(corr){
    const cols=Object.keys(corr);
    const lines=[','+cols.join(',')];
    cols.forEach(r=>lines.push(r+','+cols.map(c=>(+corr[r][c]).toFixed(6)).join(',')));
    downloadBlob(new Blob([lines.join("\n")],{type:"text/csv"}),"correlation_matrix.csv");
  }
}
async function downloadCorrelationCSV(){
  try{
    const res=await fetch(BASE_URL+'/api/correlation/export?format=csv',{credentials:'include'});
    if(!res.ok) throw new Error("HTTP "+res.status);
    downloadBlob(await res.blob(),"correlation_matrix.csv");
  }catch(e){ toast("Corr CSV error: "+e.message,"error"); }
}
async function downloadCorrelationPNG(){
  try{
    const res=await fetch(BASE_URL+'/api/correlation/png',{credentials:'include'});
    if(!res.ok) throw new Error("HTTP "+res.status);
    downloadBlob(await res.blob(),"correlation_heatmap.png");
  }catch(e){ toast("Corr PNG error: "+e.message,"error"); }
}
function syncExportButtons(){
  const hasVC  = !!lsGet("valueCounts");
  const hasCorr= !!getCorrelationMatrix();
  const assoc  = lsGet("assoc");
  const t=(id,c)=>{const el=$(id); if(!el) return; el.disabled=!c; el.classList.toggle("disabled-btn",!c);};
  t("btn-chart-png",hasVC||hasCorr);
  t("btn-data-csv",hasVC||hasCorr);
  t("btn-corr-export-csv",hasCorr); t("btn-corr-export-png",hasCorr);
  t("btn-corr-export-csv-2",hasCorr); t("btn-corr-export-png-2",hasCorr);
  t("btn-export-rules",!!(assoc&&assoc.length));
}
function clearAnalysis(){
  LS_KEYS_TO_CLEAR.forEach(lsDel);
  toast("Analysis cache cleared","success");
  syncExportButtons();
}

/* ---------------- Admin ---------------- */
async function adminLoadUsers(){
  const box=$("admin-users-list"); if(!box) return;
  box.innerHTML="<p class='text-small text-dim'>Loading users...</p>";
  try{
    const data=await handleApi("/api/admin/users");
    let h="<table class='data-table'><thead><tr><th>User</th><th>Role</th></tr></thead><tbody>";
    data.users.forEach(u=>{ h+=`<tr><td>${u.username}</td><td>${u.role}</td></tr>`;});
    box.innerHTML=h+"</tbody></table>";
  }catch(e){ box.innerHTML=`<p class='text-small text-danger'>${e.message}</p>`; }
}

/* ---------------- Visualization helpers ---------------- */
(async function ensureChartJS(){
  if(window.Chart && Chart.controllers?.matrix) return;
  const load = src=>new Promise((r,j)=>{const s=document.createElement("script");s.src=src;s.onload=r;s.onerror=j;document.head.appendChild(s);});
  if(!window.Chart) await load("https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js");
  if(!Chart.controllers?.matrix) await load("https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.4.0/dist/chartjs-chart-matrix.min.js");
})();
const VizCharts={ valueCounts:null, pca:null, kmeans:null };
function getCss(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim()||"#94a3b8"; }

/* ---------- Value Counts ---------- */
function renderValueCounts(mode){
  const data=lsGet("valueCounts");
  const status=$("vc-status")||$("chart-status");
  const canvas=$("vc-canvas")||$("main-chart");
  if(!canvas) return;
  if(!data||!data.labels){ status&&(status.textContent="No value counts"); return; }
  mode = ['pie','line','scatter','bar'].includes(mode)?mode:(data.mode||'bar');
  data.mode=mode; lsSet("valueCounts",data);
  if(VizCharts.valueCounts) VizCharts.valueCounts.destroy();
  const ctx=canvas.getContext("2d");
  const dataset = mode==="scatter"
    ? {label:data.title||"Value Counts",data:data.labels.map((_,i)=>({x:i,y:data.values[i]}))}
    : {label:data.title||"Value Counts",data:data.values};
  VizCharts.valueCounts=new Chart(ctx,{
    type: mode==="scatter"?"scatter":mode,
    data:{labels:mode==="scatter"?data.labels.map((_,i)=>i):data.labels,datasets:[dataset]},
    options:{
      responsive:true,
      plugins:{legend:{display:mode==="pie"}},
      scales: mode==="pie"?{}:{x:{ticks:{color:getCss("--text-dim")}},y:{ticks:{color:getCss("--text-dim")}}}
    }
  });
  status&&(status.textContent="");
  syncExportButtons();
}

/* ---------- Correlation MATRIX/TABLE ---------- */
function getCorrelationMatrix(){ return lsGet("correlation") || lsGet("autoBundle")?.correlation_matrix || null; }
function ensureCorrelation(){ renderCorrTable(); }
function buildCorrMeta(corr){
  const cols=Object.keys(corr);
  let min=1,max=-1;
  cols.forEach(r=>cols.forEach(c=>{
    const v=corr[r][c]; if(typeof v==="number"){ if(v<min)min=v; if(v>max)max=v; }
  }));
  return {cols,min,max};
}
function colorForCorr(v,range){
  const rRange=range||1;
  const n=Math.max(-rRange,Math.min(rRange,v))/rRange;
  let r,g,b;
  if(n>=0){
    const t=n; r=Math.round(29+(14-29)*t); g=Math.round(49+(165-49)*t); b=Math.round(68+(233-68)*t);
  }else{
    const t=-n; r=Math.round(29+(239-29)*t); g=Math.round(49+(68-49)*t);  b=Math.round(68+(68-68)*t);
  }
  return `rgb(${r},${g},${b})`;
}

/* TABLE renderer */
function renderCorrTable(){
  const wrap = $("corr-table-wrap")||$("corr-wrap");
  if(!wrap) return;
  const corr = getCorrelationMatrix();
  if(!corr){
    wrap.innerHTML="<p class='text-small text-dim' style='padding:.5rem;'>No correlation available – run Correlation Matrix or Auto Explore.</p>";
    syncExportButtons(); return;
  }
  const {cols,min,max} = buildCorrMeta(corr);
  const scaleSel = $("corr-scale"); let rng=1;
  if(scaleSel && scaleSel.value!=="auto") rng=parseFloat(scaleSel.value)||1;
  else{
    const absMax=Math.max(Math.abs(min),Math.abs(max));
    rng=absMax<0.2?0.2:absMax;
  }

  let thead="<thead><tr><th></th>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead>";
  let tbody="<tbody>";
  cols.forEach(r=>{
    tbody+="<tr><th>"+r+"</th>";
    cols.forEach(c=>{
      const v=corr[r][c];
      const bg=typeof v==="number"?colorForCorr(v,rng):"transparent";
      const text=typeof v==="number"?v.toFixed(2):"";
      tbody+=`<td data-r="${r}" data-c="${c}" data-v="${v}" style="background:${bg};">${text}</td>`;
    });
    tbody+="</tr>";
  });
  tbody+="</tbody>";
  wrap.innerHTML=`<div id="corr-table-scroll" style="overflow:auto;position:relative;">
      <table id="corr-table">${thead}${tbody}</table>
      <div id="corr-cell-outline" style="position:absolute;border:1px solid #fff3;pointer-events:none;display:none;"></div>
    </div>
    <div id="corr-tooltip" style="position:fixed;z-index:99999;background:#0f1c26;border:1px solid #2c425a;border-radius:6px;padding:.35rem .5rem;font-size:.6rem;color:#dbe8f2;display:none;box-shadow:0 6px 20px -6px #000a;"></div>
    <div class="text-small text-dim" id="corr-hint" style="margin-top:.35rem;">Click a cell to copy pair + r</div>
    <div id="corr-scale-strip" style="height:8px;border-radius:4px;margin:.6rem 0 .25rem;position:relative;overflow:hidden;">
      <canvas id="corr-scale-canvas" width="300" height="8" style="width:100%;height:100%;"></canvas>
    </div>
    <div id="corr-ticks" class="text-xxs" style="display:flex;justify-content:space-between;"></div>
    <div id="corr-summary" class="text-xxs text-dim" style="margin-top:.35rem;"></div>`;

  buildCorrelationLegend(rng);
  $("corr-summary") && ($("corr-summary").textContent=`(${cols.length}×${cols.length}) min ${min.toFixed(2)} / max ${max.toFixed(2)}`);

  const tooltip = $("corr-tooltip");
  const outline = $("corr-cell-outline");
  const scroll  = $("corr-table-scroll");

  wrap.addEventListener("mousemove",e=>{
    const td=e.target.closest("td[data-v]");
    if(!td){ tooltip.style.display="none"; outline.style.display="none"; return; }
    const r=td.dataset.r,c=td.dataset.c,v=Number(td.dataset.v);
    tooltip.innerHTML=`<strong>${r}</strong> vs <strong>${c}</strong><br>r = ${v.toFixed(4)}`;
    tooltip.style.display="block";
    tooltip.style.left=(e.pageX+12)+"px";
    tooltip.style.top =(e.pageY+12)+"px";

    const rect=td.getBoundingClientRect(), rootRect=scroll.getBoundingClientRect();
    outline.style.display="block";
    outline.style.width = rect.width+"px";
    outline.style.height= rect.height+"px";
    outline.style.left  = (rect.left-rootRect.left+scroll.scrollLeft)+"px";
    outline.style.top   = (rect.top-rootRect.top+scroll.scrollTop)+"px";
  });
  wrap.addEventListener("mouseleave",()=>{
    tooltip.style.display="none"; outline.style.display="none";
  });
  wrap.addEventListener("click",e=>{
    const td=e.target.closest("td[data-v]");
    if(!td) return;
    const txt=`${td.dataset.r},${td.dataset.c},${Number(td.dataset.v).toFixed(4)}`;
    copyToClipboard(txt); toast("Copied correlation value","success");
    const hint=$("corr-hint");
    hint&&(hint.textContent=`Copied: ${txt}`);
    setTimeout(()=>{ if(hint && hint.textContent.startsWith("Copied")) hint.textContent="Click a cell to copy pair + r"; },2200);
  });

  syncExportButtons();
}

/* Optional CANVAS heatmap */
let corrChart=null;
async function renderInteractiveCorrelation(){
  const wrap=$("corr-wrap");
  if(!wrap) return renderCorrTable();
  if(!window.Chart || !Chart.controllers?.matrix){ setTimeout(renderInteractiveCorrelation,150); return; }

  if(!getCorrelationMatrix()){
    try{
      wrap.classList.add("loading");
      const data=await handleApi("/api/correlation/export?format=json");
      lsSet("correlation",compressCorr(data.correlation));
    }catch(e){
      wrap.innerHTML="<p class='text-small text-dim' style='padding:.5rem;'>No correlation available.</p>"; return;
    }finally{ wrap.classList.remove("loading"); }
  }

  const corr=getCorrelationMatrix(); const {cols,min,max}=buildCorrMeta(corr);
  const points=[]; cols.forEach((r,i)=>cols.forEach((c,j)=>{ const v=corr[r][c]; if(typeof v==="number") points.push({x:j,y:i,v}); }));
  const scaleSel=$("corr-scale"); let rng=1;
  if(scaleSel && scaleSel.value!=="auto") rng=parseFloat(scaleSel.value)||1;
  else{ const absMax=Math.max(Math.abs(min),Math.abs(max)); rng=absMax<0.2?0.2:absMax; }

  let canvas=$("corr-matrix-canvas");
  if(!canvas){ wrap.innerHTML="<canvas id='corr-matrix-canvas'></canvas>"; canvas=$("corr-matrix-canvas"); }
  const size=Math.max(14,Math.min(42,Math.floor(640/cols.length))); const full=size*cols.length;
  canvas.width=full; canvas.height=full;

  if(corrChart) corrChart.destroy();
  corrChart=new Chart(canvas.getContext("2d"),{
    type:"matrix",
    data:{datasets:[{label:"Correlation",data:points,width:()=>size-2,height:()=>size-2,
                     backgroundColor:ctx=>colorForCorr(ctx.raw.v,rng),borderColor:"rgba(255,255,255,.08)",borderWidth:1,
                     hoverBackgroundColor:"#ffffff33",hoverBorderColor:"#fff"}]},
    options:{
      animation:false,maintainAspectRatio:false,
      plugins:{tooltip:{callbacks:{title:i=>{if(!i.length)return"";const {x,y}=i[0].raw;return cols[y]+" vs "+cols[x];},
                                      label:i=>"r = "+i.raw.v.toFixed(4)}},legend:{display:false}},
      scales:{
        x:{type:"linear",position:"top",min:-.5,max:cols.length-.5,
           ticks:{callback:v=>cols[v]||"",font:{size:10},color:getCss("--text-dim")},grid:{display:false}},
        y:{type:"linear",reverse:true,min:-.5,max:cols.length-.5,
           ticks:{callback:v=>cols[v]||"",font:{size:10},color:getCss("--text-dim")},grid:{display:false}}
      },
      onClick:(e,els)=>{
        if(!els.length) return;
        const el=els[0], v=el.raw.v, colX=cols[el.raw.x], colY=cols[el.raw.y];
        copyToClipboard(`${colY},${colX},${v}`); toast("Copied correlation value","success");
        const hint=$("corr-hint"); hint&&(hint.textContent=`Copied: ${colY},${colX},${v.toFixed(4)}`);
        setTimeout(()=>{ if(hint && hint.textContent.startsWith("Copied")) hint.textContent=""; },2500);
      }
    }
  });
  buildCorrelationLegend(rng);
  $("corr-summary") && ($("corr-summary").textContent=`(${cols.length}×${cols.length}) min ${min.toFixed(2)} / max ${max.toFixed(2)}`);
  syncExportButtons();
}
function buildCorrelationLegend(range){
  const strip=$("corr-scale-strip"), cvs=$("corr-scale-canvas");
  if(strip && cvs){
    const ctx=cvs.getContext("2d");
    cvs.width=strip.clientWidth*2; cvs.height=strip.clientHeight*2; ctx.scale(2,2);
    const g=ctx.createLinearGradient(0,0,strip.clientWidth,0);
    g.addColorStop(0,colorForCorr(-range,range));
    g.addColorStop(.5,colorForCorr(0,range));
    g.addColorStop(1,colorForCorr(range,range));
    ctx.fillStyle=g; ctx.fillRect(0,0,strip.clientWidth,strip.clientHeight);
  }
  const ticks=$("corr-ticks");
  if(ticks){ ticks.innerHTML=""; [-range,-range/2,0,range/2,range].forEach(v=>{const s=document.createElement("span");s.textContent=v.toFixed(2);ticks.appendChild(s);}); }
}
function copyToClipboard(txt){
  if(navigator.clipboard){ navigator.clipboard.writeText(txt).catch(()=>{}); }
  else{ const ta=document.createElement("textarea"); ta.value=txt; document.body.appendChild(ta); ta.select(); try{document.execCommand("copy");}catch{} ta.remove(); }
}

/* ---------- PCA ---------- */

function renderPCA() {
  const pca = lsGet("pca") || lsGet("autoBundle")?.pca;
  const km = lsGet("kmeans") || lsGet("autoBundle")?.kmeans;
  const box = $("pca-box") || $("pca-container");
  if (!box) return;
  const comps = pca?.components_2d || pca?.components;
  if (!comps || !Array.isArray(comps) || comps.length === 0) {
    box.innerHTML = "<p class='text-small text-dim'>No PCA data.</p>";
    return;
  }

  // Find min/max for scaling
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  comps.forEach(([x, y]) => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  // Add 5% padding
  const padX = (maxX - minX) * 0.05;
  const padY = (maxY - minY) * 0.05;
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;

  // Try to color by cluster if available
  const labels = km?.labels_preview || [];
  const hasClusters = labels.length === comps.length;
  const colors = ["#02b2ff","#b136ff","#ffb86c","#10b981","#ef4444","#f59e42","#7a4bff","#ff7a7a"];
  const datasets = hasClusters
    ? [...new Set(labels)].map((cl,idx)=>({
        label: "Cluster "+cl,
        data: comps.map((p,i)=>labels[i]===cl?{x:p[0],y:p[1]}:null).filter(Boolean),
        backgroundColor: colors[cl%colors.length]+"cc",
        pointRadius: 4,
        pointHoverRadius: 7,
      }))
    : [{
        label:"PCA",
        data: comps.map(p=>({x:p[0],y:p[1]})),
        backgroundColor:"#02b2ffcc",
        pointRadius: 4,
        pointHoverRadius: 7,
      }];

  box.innerHTML = "<canvas id='pca-canvas' style='width:100%;height:100%'></canvas>";
  const ctx = $("pca-canvas").getContext("2d");
  if (VizCharts.pca) VizCharts.pca.destroy();

  const exp = pca?.explained || pca?.explained_variance || [];
  VizCharts.pca = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              let txt = `(${d.x.toFixed(2)}, ${d.y.toFixed(2)})`;
              if (hasClusters && ctx.dataset.label) txt += ` | ${ctx.dataset.label}`;
              return txt;
            }
          }
        }
      },
      scales: {
        x: {
          min: minX,
          max: maxX,
          title: {
            display: true,
            text: `PC1${exp[0] ? ` (${(exp[0]*100).toFixed(1)}%)` : ""}`,
            color: "#b8c9d6"
          },
          ticks: { color: getCss('--text-dim') }
        },
        y: {
          min: minY,
          max: maxY,
          title: {
            display: true,
            text: `PC2${exp[1] ? ` (${(exp[1]*100).toFixed(1)}%)` : ""}`,
            color: "#b8c9d6"
          },
          ticks: { color: getCss('--text-dim') }
        }
      }
    }
  });
  syncExportButtons();
}

/* ---------- KMeans ---------- */

function renderKMeans() {
  const km = lsGet("kmeans") || lsGet("autoBundle")?.kmeans;
  const pca = lsGet("pca") || lsGet("autoBundle")?.pca;
  const box = $("kmeans-box") || $("kmeans-container");
  if (!box) return;
  const labels = km?.labels_preview || km?.labels;
  if (!labels) {
    box.innerHTML = "<p class='text-small text-dim'>No clustering data.</p>";
    return;
  }

  // Use PCA for 2D plotting if available
  const comps = pca?.components_2d;
  if (!comps || comps.length !== labels.length) {
    // fallback: 1D cluster plot
    const pts = labels.map((lab, i) => ({ x: i, y: lab }));
    // Find min/max for scaling
    let minX = 0, maxX = labels.length - 1, minY = Math.min(...labels), maxY = Math.max(...labels);
    const padX = (maxX - minX) * 0.05;
    const padY = (maxY - minY) * 0.05;
    minX -= padX; maxX += padX; minY -= padY; maxY += padY;
    box.innerHTML = "<canvas id='kmeans-canvas' style='width:100%;height:100%'></canvas>";
    const ctx = $("kmeans-canvas").getContext("2d");
    if (VizCharts.kmeans) VizCharts.kmeans.destroy();
    VizCharts.kmeans = new Chart(ctx, {
      type: "scatter",
      data: { datasets: [{ label: "Clusters", data: pts, backgroundColor: "#b136ffcc" }] },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: minX, max: maxX, ticks: { color: getCss('--text-dim') } },
          y: { min: minY, max: maxY, ticks: { color: getCss('--text-dim') } }
        }
      }
    });
    syncExportButtons();
    return;
  }

  // 2D cluster plot in PCA space
  // Find min/max for scaling
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  comps.forEach(([x, y]) => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  const padX = (maxX - minX) * 0.05;
  const padY = (maxY - minY) * 0.05;
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;

  const colors = ["#02b2ff","#b136ff","#ffb86c","#10b981","#ef4444","#f59e42","#7a4bff","#ff7a7a"];
  const clusters = [...new Set(labels)];
  const datasets = clusters.map((cl, idx) => ({
    label: "Cluster " + cl,
    data: comps.map((p, i) => labels[i] === cl ? { x: p[0], y: p[1] } : null).filter(Boolean),
    backgroundColor: colors[cl % colors.length] + "cc",
    pointRadius: 4,
    pointHoverRadius: 7,
  }));

  // Cluster centers (if available)
  let centers = [];
  if (km?.centers && km.centers[0].length >= 2) {
    centers = km.centers.map((c, i) => ({
      x: c[0], y: c[1], cluster: i
    }));
    datasets.push({
      label: "Centers",
      data: centers,
      backgroundColor: "#fff",
      borderColor: "#000",
      pointRadius: 8,
      pointStyle: "rectRot",
      showLine: false
    });
  }

  box.innerHTML = "<canvas id='kmeans-canvas' style='width:100%;height:100%'></canvas>";
  const ctx = $("kmeans-canvas").getContext("2d");
  if (VizCharts.kmeans) VizCharts.kmeans.destroy();

  VizCharts.kmeans = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              let txt = `(${d.x.toFixed(2)}, ${d.y.toFixed(2)})`;
              if (ctx.dataset.label && ctx.dataset.label.startsWith("Cluster")) txt += ` | ${ctx.dataset.label}`;
              if (ctx.dataset.label === "Centers") txt += " (center)";
              return txt;
            }
          }
        }
      },
      scales: {
        x: { min: minX, max: maxX, title: { display: true, text: "PC1", color: "#b8c9d6" }, ticks: { color: getCss('--text-dim') } },
        y: { min: minY, max: maxY, title: { display: true, text: "PC2", color: "#b8c9d6" }, ticks: { color: getCss('--text-dim') } }
      }
    }
  });
  syncExportButtons();
}

/* ---------------- Column Type Inference ---------------- */
async function inferColumnTypes() {
  const file = localStorage.getItem("filename");
  if (!file) return;
  try {
    const res = await handleApi("/api/coltypes");
    if (res.types) lsSet("colTypesCache", res.types);
  } catch (e) {
    console.error("Error inferring column types:", e);
  }
}
}
