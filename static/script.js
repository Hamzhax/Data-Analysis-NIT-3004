// --- Redirect to login if not authenticated (on any page load) ---
document.addEventListener("DOMContentLoaded", async () => {
  try {
    await ensureAuthForProtectedPages();
  } catch (e) {
    window.location.href = "login.html";
  }
});

// --- Global helpers for dashboard/visualization ---
if (!window.$) window.$ = (id) => document.getElementById(id);
if (!window.toast) window.toast = (msg, type = "info") => console.log("[toast]", type, msg);
if (!window.lsGet) window.lsGet = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } };
if (!window.lsSet) window.lsSet = (k, v) => { try { localStorage.setItem(k, typeof v === "string" ? v : JSON.stringify(v)); } catch {} };
if (!window.lsDel) window.lsDel = (k) => localStorage.removeItem(k);
if (!window.logout) window.logout = function logout() { localStorage.clear(); window.location.href = "login.html"; };
if (!window.downloadBlob) window.downloadBlob = function downloadBlob(blob, filename) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1500); };
if (!window.cleanAIBlock) window.cleanAIBlock = function cleanAIBlock(objOrStr) { if (typeof objOrStr === "string") { try { return JSON.parse(objOrStr); } catch { return { summary: objOrStr }; } } if (!objOrStr || typeof objOrStr !== "object") return objOrStr; return { summary: objOrStr.summary ?? objOrStr.overview ?? "", key_points: Array.isArray(objOrStr.key_points || objOrStr.key_findings) ? (objOrStr.key_points || objOrStr.key_findings) : [], anomalies: Array.isArray(objOrStr.anomalies) ? objOrStr.anomalies : [], recommendation: objOrStr.recommendation ?? "", next_steps: Array.isArray(objOrStr.next_steps) ? objOrStr.next_steps : [] }; };

// --- Make syncActiveFile globally available ---
window.syncActiveFile = syncActiveFile;
window.handleApi = handleApi;
window.ensureAuthForProtectedPages = ensureAuthForProtectedPages;
window.uploadDataset = uploadDataset;
window.fetchFromInternet = fetchFromInternet;
window.smartSearch = smartSearch;
window.applyCleaning = applyCleaning;
window.inferColumnTypes = inferColumnTypes;
window.downloadMarkdownReport = downloadMarkdownReport;
window.downloadPdfReport = downloadPdfReport;
window.downloadChart = downloadChart;
window.exportDataCsv = exportDataCsv;
window.renderOverview = renderOverview;
window.renderAINarrative = renderAINarrative;
window.renderLinearRegression = renderLinearRegression;
window.renderKMeans = renderKMeans;
window.renderValueCounts = renderValueCounts;
window.renderCorrTable = renderCorrTable;
window.renderAssoc = typeof renderAssoc !== 'undefined' ? renderAssoc : function(){};
window.renderSummary = typeof renderSummary !== 'undefined' ? renderSummary : function(){};
window.ensureCorrelation = ensureCorrelation;
window.generateAISummary = generateAISummary;
window.clearAnalysis = clearAnalysis;
window.adminLoadUsers = adminLoadUsers;
window.updateVizDatasetMeta = updateVizDatasetMeta;
window.loadMeta = typeof loadMeta !== 'undefined' ? loadMeta : function(){};

// --- Fallback for BASE_URL if not set ---
if (!window.BASE_URL) {
  const origin = window.location.origin;
  if (origin.includes("onrender.com")) {
    window.BASE_URL = origin;
  } else {
    window.BASE_URL = "http://127.0.0.1:5050";
  }
}
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
}
window.BASE_URL = BASE_URL;

const LS_KEYS_TO_CLEAR = [
  "summary","correlation","valueCounts","linear_regression","kmeans","assoc",
  "autoBundle","autoAI","lastAI","colTypesCache","primaryCategorical"
];

const $      = id => document.getElementById(id);
const lsGet  = k => { try{return JSON.parse(localStorage.getItem(k));}catch{return null;} };
const lsSet  = (k,v)=>{ try{localStorage.setItem(k, typeof v==="string"?v:JSON.stringify(v));}catch(e){console.warn("lsSet",k,e);} };
const lsDel  = k => localStorage.removeItem(k);

/* ---------------- Toast ---------------- */
function toast(msg,type="info",timeout=3000){
  let wrap=$("toast-wrap");
  if(!wrap){
    wrap=document.createElement("div");
    wrap.id="toast-wrap";
    wrap.style.cssText="position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:.5rem;align-items:flex-end;";
    document.body.appendChild(wrap);
  }
  const icons = {
    error: "⛔️", success: "✅", warn: "⚠️", info: "ℹ️"
  };
  const card=document.createElement("div");
  card.innerHTML = `<span style="margin-right:.5em">${icons[type]||""}</span>${msg}`;
  card.style.cssText=`
    font:500 .7rem/1.35 Inter,system-ui,sans-serif;
    background:${type==="error"?"#3b1217":type==="success"?"#123b2a":type==="warn"?"#3b2f12":"#162235"};
    border:1px solid ${type==="error"?"#ef4444":type==="success"?"#10b981":type==="warn"?"#f59e0b":"#2c425a"};
    color:#e2e8f0;padding:.55rem .75rem;border-radius:10px;
    box-shadow:0 4px 20px -6px rgba(0,0,0,.55);
    backdrop-filter:blur(6px);opacity:0;transform:translateY(6px);
    transition:opacity .35s,transform .35s;`;
  wrap.appendChild(card);
  requestAnimationFrame(()=>{card.style.opacity=1;card.style.transform="translateY(0)";});
  setTimeout(()=>{card.style.opacity=0;card.style.transform="translateY(6px)";setTimeout(()=>wrap.removeChild(card),350);},timeout);
}

/* ---------------- Fetch wrapper ---------------- */
async function handleApi(path,opts={}){
  const init={
    method:(opts.method||"GET").toUpperCase(),
    credentials:"include",
    headers:{"Content-Type":"application/json",...(opts.headers||{})}
  };
  if(opts.body) init.body=typeof opts.body==="string"?opts.body:JSON.stringify(opts.body);
  const res=await fetch(BASE_URL+path,init);
  let js={};
  try{ js=await res.json(); }catch{}
  if(!res.ok || js.status==="error") throw new Error(js.error||`HTTP ${res.status}`);
  return js;
}

/* ---------------- Auth ---------------- */
async function logout(){
  try{await handleApi("/api/logout",{method:"POST"});}catch{}
  localStorage.clear(); window.location.href="login.html";
}
async function ensureAuthForProtectedPages(){
  const pages=["dashboard","analysis","visualization","admin","upload","preview"];
  const page=document.body.getAttribute("data-page");
  if(!pages.includes(page)) return;
  try{
    const me=await handleApi("/api/me");
    lsSet("currentUser",me.user); lsSet("role",me.role);
    if(me.role!=="admin"){
      document.querySelectorAll(".admin-link, a[href='admin.html']").forEach(a=>a.style.display="none");
    }
    $("role-badge") && ($("role-badge").textContent=me.role.toUpperCase());
  }catch(e){
    toast("Session expired. Please login.","warn");
    setTimeout(()=>window.location.href="login.html",600);
  }
}

/* ---------------- Active dataset sync ---------------- */
async function syncActiveFile(force=false){
  if(!force && localStorage.getItem("filename")) return localStorage.getItem("filename");
  try{
    const r=await handleApi("/api/files");
    const active=r.active||localStorage.getItem("filename");
    if(active){
      localStorage.setItem("filename",active);
      localStorage.setItem("activeFile",active);
    }
    return active;
  }catch(e){ console.warn("syncActiveFile",e); return null; }
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
        : "<em>No CSV links found.</em>";
    }
    st&&(st.textContent="Done");
  }catch(e){ st&&(st.textContent="Error"); toast("Search error: "+e.message,"error"); }
}

/* ---------------- Cleaning ---------------- */
async function applyCleaning(){
  const f=localStorage.getItem("filename"); if(!f){ toast("No dataset","warn"); return; }
  const body={
    filename:f,
    remove_duplicates:$("remove-duplicates")?.checked,
    drop_na:$("drop-na")?.checked,
    fill_value:$("fill-value")?.value||null
  };
  try{
    $("clean-status")&&( $("clean-status").textContent="Cleaning...");
    const res = await handleApi("/api/clean",{method:"POST",body});
    if(res.message === "already_clean"){
      $("clean-status")&&( $("clean-status").textContent="Already clean");
      toast("Dataset is already clean","info");
    } else {
      $("clean-status")&&( $("clean-status").textContent="Cleaned ✓");
      toast("Cleaning applied","success");
    }
    await previewDataset(); inferColumnTypes();
  }catch(e){
    $("clean-status")&&( $("clean-status").textContent="Failed");
    toast("Clean error: "+e.message,"error");
  }
}

/* ---------------- Analyses ---------------- */
async function runAnalysis(){
  const m=$("analysis-method")?.value;
  const column=$("column-name")?.value?.trim() || $("column-select")?.value?.trim();
  const k=parseInt($("cluster-k")?.value||"3",10);
  if(!m){ toast("Choose a method","warn"); return; }
  try{
    $("analysis-status")&&( $("analysis-status").textContent="Running...");
    const payload={method:m}; if(m==="value_counts") payload.column=column; if(m==="kmeans") payload.k=k;
    const data=await handleApi("/api/analyze",{method:"POST",body:payload});
    switch(m){
      case "summary":      lsSet("summary",data.summary); inferColumnTypes(); break;
      case "correlation":  lsSet("correlation",compressCorr(data.correlation)); break;
      case "value_counts": lsSet("valueCounts",{labels:data.labels,values:data.values,title:data.title}); break;
      case "time_series": lsSet("time_series",{
        values: data.values,
        moving_average: data.moving_average,
        indices: data.indices,
        column: data.column,
        trend_direction: data.trend_direction
      }); break;
      case "kmeans":       lsSet("kmeans",{
        labels_preview:data.labels,
        labels:data.labels,
        centers:data.centers,
        columns:data.columns,
        components_2d:data.components_2d
      }); break;
      case "assoc_rules":  lsSet("assoc",data.rules); break;
    }
    $("analysis-status")&&( $("analysis-status").textContent="Done ✓");
    toast("Analysis stored – open Visualize","success");
  }catch(e){
    $("analysis-status")&&( $("analysis-status").textContent="Error");
    toast("Analysis error: "+e.message,"error");
  }
}
function compressCorr(obj){
  const out={}; const cols=Object.keys(obj||{});
  cols.forEach(r=>{
    out[r]={};
    Object.keys(obj[r]||{}).forEach(c=>{
      const v=obj[r][c]; out[r][c]=typeof v==="number"?Number(v.toFixed(4)):v;
    });
  });
  return out;
}

/* ---------------- AI Insight ---------------- */
async function buildAISnippet(maxRows=20){
  const bundle=lsGet("autoBundle");
  const file=localStorage.getItem("filename");
  let preview=null;
  let genericDesc = "";
  try{
    const res=await handleApi("/api/preview_json",{method:"POST",body:{filename:file}});
    preview={columns:res.columns,rows:(res.rows||[]).slice(0,maxRows)};
    if (res.columns && res.rows) {
      genericDesc = `This dataset contains ${res.rows.length} rows and ${res.columns.length} columns. The columns are: ${res.columns.join(", ")}.`;
    }
  }catch(_){}
  return {
    meta: bundle?.profile?.basic || {},
    top_corr: (bundle?.top_correlations||[]).slice(0,8),
    value_counts: bundle?.categorical ? Object.fromEntries(Object.entries(bundle.categorical).slice(0,2)) : null,
    summary_stats: bundle?.summary ? Object.fromEntries(Object.entries(bundle.summary).slice(0,6)) : null,
    preview,
    genericDesc
  };
}
const stripFences = str => {
  if (typeof str !== "string") return str;
  
  // Remove markdown fences and JSON labels
  let cleaned = str
    .replace(/```json|```/gi, "")
    .replace(/^\s*"{?\s*json"?\s*[:{]/i, "{")
    .replace(/^\s*json\s*[:{]/i, "{")
    .trim();
  
  // If it starts with { "summary": remove the leading { "summary":
  cleaned = cleaned.replace(/^\{\s*"summary"\s*:\s*"/, "");
  
  // If it ends with closing quotes and brace, remove them
  cleaned = cleaned.replace(/"\s*\}\s*$/, "");
  
  // Clean up extra quotes around the content
  if (cleaned.startsWith('"') && cleaned.endsWith('"')) {
    cleaned = cleaned.slice(1, -1);
  }
  
  return cleaned;
};

function cleanAIBlock(objOrStr){
  if(typeof objOrStr === "string"){
    const s = stripFences(objOrStr);
    try { 
      return JSON.parse(s); 
    } catch { 
      // If parsing fails, try to extract meaningful content
      const lines = s.split('\n').filter(line => line.trim());
      return {
        summary: lines.slice(0, 2).join(' ') || s.substring(0, 200),
        key_points: lines.filter(line => line.includes('•') || line.includes('-') || line.includes('*')).slice(0, 5),
        anomalies: lines.filter(line => line.toLowerCase().includes('anomal') || line.toLowerCase().includes('outlier')).slice(0, 3),
        recommendation: lines.find(line => line.toLowerCase().includes('recommend')) || ""
      };
    }
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
    // Always prepend a generic paragraph about the data
    let generic = snippet.genericDesc || "This dataset contains tabular data.";
    const richDesc = `
DATA BRIEF (JSON):
${JSON.stringify(snippet)}

GENERIC DATA DESCRIPTION: ${generic}

USER CONTEXT: ${chartType}
USER PROMPT: ${description}

TASK: Act as a senior data analyst. Using ONLY the data above, provide:
- 2 sentence high-level summary (start with a generic description if possible)
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
      let cleaned=stripFences(data);
      
      // Enhanced cleaning for various response formats
      // Remove common prefixes that might appear
      cleaned = cleaned
        .replace(/^\s*{\s*"summary"\s*:\s*"?/i, '')  // Remove { "summary": "
        .replace(/"?\s*}\s*$/i, '')                   // Remove " }
        .replace(/^"?\s*summary\s*[:=]\s*"?/i, '')   // Remove summary: or summary =
        .replace(/^"?\s*response\s*[:=]\s*"?/i, '')  // Remove response: or response =
        .replace(/^"?\s*text\s*[:=]\s*"?/i, '')      // Remove text: or text =
        .trim();
      
      // If it looks like JSON, try to parse it
      if (cleaned.startsWith('{') && cleaned.endsWith('}')) {
        try{ 
          const parsed = JSON.parse(cleaned);
          data = parsed;
        } catch (e) { 
          console.warn("JSON parsing failed, treating as plain text:", e);
          // If JSON parsing fails, treat as plain text
          data = {summary: cleaned}; 
        }
      } else {
        // It's plain text after cleaning
        data = {summary: cleaned};
      }
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
  if (window._autoExploreRunning) return; // Prevent double run
  window._autoExploreRunning = true;
  prog&&(prog.textContent="Running auto exploration...");
  try{
    console.log("[Auto Explore] Starting API call...");
    const res=await handleApi("/api/auto_explore",{method:"POST"});
    console.log("[Auto Explore] API response:", res);
    console.log("[Auto Explore] AI data:", res.ai);
    storeAutoBundle(res);
    prog&&(prog.textContent="Complete ✓");
    // Use a more robust toast prevention mechanism
    const toastKey = `autoExploreToast_${Date.now()}`;
    if (!window._lastAutoExploreToast || Date.now() - window._lastAutoExploreToast > 3000) {
      toast("Auto Explore complete","success");
      window._lastAutoExploreToast = Date.now();
    }
    inferColumnTypes();
    if(document.body.getAttribute("data-page")==="visualization"){
      ensureCorrelation(); renderOverview(); renderAINarrative(); syncExportButtons();
    }
  }catch(e){
    console.error("[Auto Explore] Error:", e);
    prog&&(prog.textContent="Error");
    toast("Auto explore failed: "+e.message,"error");
  } finally {
    window._autoExploreRunning = false;
  }
}
function storeAutoBundle(result){
  const b=result.bundle, ai=result.ai;
  lsSet("autoBundle",b);
  
  console.log("Storing auto bundle:", b); // Debug log
  
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
  
  // Store Linear Regression data
  if(b?.linear_regression) {
    const lrData = {
      r2_score: b.linear_regression.r2_score,
      mse: b.linear_regression.mse,
      intercept: b.linear_regression.intercept,
      coefficients: b.linear_regression.coefficients,
      target_column: b.linear_regression.target_column,
      feature_columns: b.linear_regression.feature_columns
    };
    console.log("Storing Linear Regression data:", lrData); // Debug log
    console.log("Original LR from bundle:", b.linear_regression); // Debug original
    lsSet("linear_regression", lrData);
    
    // Also verify it was stored correctly
    setTimeout(() => {
      const stored = lsGet("linear_regression");
      console.log("Verified stored LR data:", stored);
    }, 100);
  } else {
    console.warn("No Linear Regression data in bundle:", b);
  }
  
  // Store K-means data
  if(b?.kmeans) {
    const kmeansData = {
      labels_preview: b.kmeans.labels_preview,
      labels: b.kmeans.labels, // Store full labels too
      centers: b.kmeans.centers,
      components_2d: b.kmeans.components_2d,
      columns: b.kmeans.columns,
      k: b.kmeans.k
    };
    console.log("Storing K-means data:", kmeansData); // Debug log
    lsSet("kmeans", kmeansData);
  }
  
  if(b?.assoc_rules) lsSet("assoc",b.assoc_rules);
  
  // Store AI data with better error handling
  if(ai) {
    console.log("Storing AI data:", ai);
    if(ai.error) {
      console.warn("AI generated with error:", ai.error);
      // Store the error but also check if there's any useful content
      if(ai.overview || ai.key_findings || ai.summary) {
        console.log("AI has error but also has content, storing both");
        lsSet("autoAI", cleanAIBlock(ai));
      } else {
        console.log("AI has only error, storing error info");
        lsSet("autoAI", {error: ai.error});
      }
    } else {
      lsSet("autoAI", cleanAIBlock(ai));
    }
  } else {
    console.warn("No AI data received in auto explore response");
    // Don't store anything, let renderAINarrative handle the missing data case
  }
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
  // Prevent multiple simultaneous downloads
  if (window._pdfDownloadInProgress) return;
  window._pdfDownloadInProgress = true;
  
  const st=$("report-status")||$("viz-export-status");
  st&&(st.textContent="Generating PDF...");
  
  // Create and show prominent loading overlay
  const overlay = document.createElement('div');
  overlay.id = 'pdf-loading-overlay';
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
    background: rgba(15, 28, 38, 0.95); z-index: 9999; 
    display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(4px);
  `;
  overlay.innerHTML = `
    <div style="
      background: #0f1c26; border: 2px solid #02b2ff; border-radius: 16px; 
      padding: 2rem; text-align: center; max-width: 400px; width: 90%;
      box-shadow: 0 20px 40px rgba(2, 178, 255, 0.3);
    ">
      <div style="
        width: 60px; height: 60px; border: 4px solid #1d3140; 
        border-top: 4px solid #02b2ff; border-radius: 50%; 
        margin: 0 auto 1.5rem; animation: spin 1s linear infinite;
      "></div>
      <h3 style="margin: 0 0 0.5rem; color: #e7f2fb; font-size: 1.1rem;">Generating PDF Report</h3>
      <p style="margin: 0; color: #a9bfd1; font-size: 0.9rem;">
        Processing your data analysis...
      </p>
    </div>
    <style>
      @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
  `;
  document.body.appendChild(overlay);
  
  try{
    const res=await fetch(`${BASE_URL}/api/report/pdf`,{credentials:"include"});
    if(!res.ok) throw new Error("PDF failed");
    const blob=await res.blob();
    // Prevent double download by removing any previous download links
    if (window._pdfDownloadLink) {
      document.body.removeChild(window._pdfDownloadLink);
      window._pdfDownloadLink = null;
    }
    const a=document.createElement("a");
    a.href=URL.createObjectURL(blob); a.download="dataset_report.pdf";
    document.body.appendChild(a);
    window._pdfDownloadLink = a;
    a.click();
    setTimeout(()=>{
      URL.revokeObjectURL(a.href);
      if (window._pdfDownloadLink) {
        document.body.removeChild(window._pdfDownloadLink);
        window._pdfDownloadLink = null;
      }
    },1500);
    st&&(st.textContent="PDF ready");
    toast("PDF ready","success");
  }catch(e){ st&&(st.textContent="Error"); toast("PDF error: "+e.message,"error"); }
  finally {
    window._pdfDownloadInProgress = false;
    // Remove loading overlay
    if (overlay && document.body.contains(overlay)) {
      document.body.removeChild(overlay);
    }
  }
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

/* ---------- AI Chart Descriptions ---------- */
async function addChartDescription(containerId, chartType, fallbackText) {
  const container = $(containerId);
  if (!container) return;
  
  // Remove existing description
  const existingDesc = container.querySelector('.ai-chart-description');
  if (existingDesc) existingDesc.remove();
  
  // Create description element
  const descDiv = document.createElement('div');
  descDiv.className = 'ai-chart-description';
  descDiv.style.cssText = 'margin-top:0.8rem;padding:0.6rem;background:#0a1520;border:1px solid #1a2f42;border-radius:6px;font-size:0.6rem;color:#a5b8c9;line-height:1.4;';
  descDiv.innerHTML = `<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.4rem;"><span style="color:#6ea8fe;">🤖</span><strong style="color:#d1e7dd;">AI Insight</strong></div><div class="ai-desc-content">${fallbackText}</div>`;
  
  // Try to get AI description
  try {
    const response = await fetch(BASE_URL + '/api/ai_chart_description', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ chart_type: chartType, context: fallbackText })
    });
    
    if (response.ok) {
      const data = await response.json();
      if (data.success && data.description) {
        const contentDiv = descDiv.querySelector('.ai-desc-content');
        if (contentDiv) {
          contentDiv.textContent = data.description;
        }
      }
    }
  } catch (error) {
    console.warn('Failed to get AI description:', error);
    // Keep fallback text
  }
  
  // Append to container
  container.appendChild(descDiv);
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
const VizCharts={ valueCounts:null, timeSeries:null, kmeans:null };
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
  
  // Add AI description for value counts
  setTimeout(() => addChartDescription("vc-canvas", "value_counts", `Value counts distribution showing frequency of categories in ${data.title || 'selected column'}`), 100);
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
  
  // Add AI description for correlation matrix
  setTimeout(() => addChartDescription("corr-wrap", "correlation", "Correlation matrix showing relationships between numeric variables"), 100);
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
  
  // Add AI description for interactive correlation
  setTimeout(() => addChartDescription("corr-wrap", "correlation", "Interactive correlation heatmap - click cells to copy values"), 100);
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

/* ---------- Linear Regression ---------- */
function renderLinearRegression(){
  const lr=lsGet("linear_regression")||lsGet("autoBundle")?.linear_regression;
  const box=$("lr-box")||$("linear-regression-box")||$("time-series-box"); if(!box) return;
  
  console.log("Linear Regression data from storage:", lr); // Debug output
  
  if(!lr){ 
    box.innerHTML="<p class='text-small text-dim'>No Linear Regression data available. Try running Auto Explore or Linear Regression analysis.</p>"; 
    return; 
  }
  
  // Create regression results display
  let html = `
    <div class="lr-results">
      <div class="lr-metrics" style="background:#0a1520;border:1px solid #1a2f42;border-radius:6px;padding:0.8rem;margin-bottom:1rem;">
        <h4 style="margin:0 0 0.5rem;color:#e7f2fb;font-size:0.8rem;">Model Performance</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;font-size:0.7rem;">
          <div>
            <span style="color:#a5b8c9;">R² Score:</span>
            <strong style="color:#6ea8fe;margin-left:0.5rem;">${(lr.r2_score * 100).toFixed(2)}%</strong>
          </div>
          <div>
            <span style="color:#a5b8c9;">MSE:</span>
            <strong style="color:#f59e0b;margin-left:0.5rem;">${lr.mse.toFixed(4)}</strong>
          </div>
        </div>
        <div style="margin-top:0.5rem;font-size:0.65rem;color:#9fb3c3;">
          <span style="color:#a5b8c9;">Target:</span> <strong>${lr.target_column}</strong>
        </div>
      </div>
      
      <div class="lr-coefficients" style="background:#0a1520;border:1px solid #1a2f42;border-radius:6px;padding:0.8rem;">
        <h4 style="margin:0 0 0.5rem;color:#e7f2fb;font-size:0.8rem;">Feature Coefficients</h4>
        <div class="coef-list" style="max-height:200px;overflow-y:auto;">
  `;
  
  // Add intercept
  html += `
    <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #1a2f42;font-size:0.65rem;">
      <span style="color:#a5b8c9;">Intercept</span>
      <strong style="color:#d1e7dd;">${lr.intercept.toFixed(4)}</strong>
    </div>
  `;
  
  // Add coefficients
  if(lr.coefficients && lr.coefficients.length > 0) {
    lr.coefficients.forEach(coef => {
      const absCoef = Math.abs(coef.coefficient);
      const color = absCoef > 0.1 ? '#6ea8fe' : absCoef > 0.01 ? '#f59e0b' : '#9fb3c3';
      html += `
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #1a2f42;font-size:0.65rem;">
          <span style="color:#a5b8c9;">${coef.feature}</span>
          <strong style="color:${color};">${coef.coefficient.toFixed(4)}</strong>
        </div>
      `;
    });
  }
  
  html += `
        </div>
      </div>
    </div>
  `;
  
  box.innerHTML = html;
  syncExportButtons();
  
  // Add AI description
  setTimeout(() => addChartDescription("lr-box", "linear_regression", "Linear regression analysis showing model performance and feature importance"), 100);
}

/* ---------- Time Series ---------- */
function renderTimeSeries(){
  const ts=lsGet("time_series")||lsGet("autoBundle")?.time_series;
  const box=$("time-series-box")||$("time-series-container"); if(!box) return;
  
  console.log("Time series data:", ts); // Debug output
  
  if(!ts || (!ts.trends && !ts.seasonal_patterns && !ts.decomposition)) { 
    box.innerHTML="<p class='text-small text-dim'>No time series data available. Try running Auto Explore or time series analysis.</p>"; 
    return; 
  }
  
  let html = "<div class='time-series-results'>";
  
  // Show trends
  if (ts.trends && ts.trends.length > 0) {
    html += "<div class='ts-section'><h4>Detected Trends</h4><ul>";
    ts.trends.forEach(trend => {
      html += `<li>${trend}</li>`;
    });
    html += "</ul></div>";
  }
  
  // Show seasonal patterns
  if (ts.seasonal_patterns && ts.seasonal_patterns.length > 0) {
    html += "<div class='ts-section'><h4>Seasonal Patterns</h4><ul>";
    ts.seasonal_patterns.forEach(pattern => {
      html += `<li>${pattern}</li>`;
    });
    html += "</ul></div>";
  }
  
  // Show decomposition results
  if (ts.decomposition) {
    html += "<div class='ts-section'><h4>Time Series Components</h4>";
    html += `<p><strong>Trend Component:</strong> ${ts.decomposition.trend_summary || 'Analyzed'}</p>`;
    html += `<p><strong>Seasonal Component:</strong> ${ts.decomposition.seasonal_summary || 'Analyzed'}</p>`;
    html += `<p><strong>Residual Component:</strong> ${ts.decomposition.residual_summary || 'Analyzed'}</p>`;
    html += "</div>";
  }
  
  // Show any statistics
  if (ts.statistics) {
    html += "<div class='ts-section'><h4>Time Series Statistics</h4>";
    Object.entries(ts.statistics).forEach(([key, value]) => {
      html += `<p><strong>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}</p>`;
    });
    html += "</div>";
  }
  
  html += "</div>";
  box.innerHTML = html;
  
  // Add AI description
  setTimeout(() => addChartDescription("time-series-box", "time_series", "Time series analysis showing trends, seasonal patterns, and temporal decomposition"), 100);
}

/* ---------- KMeans ---------- */
function renderKMeans(){
  const km=lsGet("kmeans")||lsGet("autoBundle")?.kmeans;
  const box=$("kmeans-box")||$("kmeans-container"); if(!box) return;
  
  console.log("K-means data:", km); // Debug output
  
  const labels=km?.labels_preview || km?.labels;
  const comps=km?.components_2d || km?.components;
  
  console.log("Labels:", labels, "Components:", comps); // Debug output
  
  if(!labels||!comps){ 
    box.innerHTML="<p class='text-small text-dim'>No clustering data available. Try running Auto Explore or K-means analysis.</p>"; 
    return; 
  }
  
  // Ensure we have matching array lengths
  const minLength = Math.min(labels.length, comps.length);
  const labelsSubset = labels.slice(0, minLength);
  const compsSubset = comps.slice(0, minLength);
  
  // Use 2D components for scatter, color by label
  let pts=compsSubset.map((p,i)=>({x:p[0],y:p[1],c:labelsSubset[i]}));
  
  // Outlier trimming and scaling
  function getBounds(arr, key) {
    const vals = arr.map(o=>o[key]).sort((a,b)=>a-b);
    const q = p => vals[Math.floor(p*vals.length)];
    return [q(0.01), q(0.99)];
  }
  let [minX,maxX]=getBounds(pts,'x'), [minY,maxY]=getBounds(pts,'y');
  const pad = (min,max) => { const d=max-min; return [min-0.1*d,max+0.1*d]; };
  [minX,maxX]=pad(minX,maxX); [minY,maxY]=pad(minY,maxY);
  // Group by cluster label for color
  const clusters={}; pts.forEach(p=>{if(!clusters[p.c])clusters[p.c]=[];clusters[p.c].push(p);});
  const colors=["#2563eb","#f59e42","#10b981","#ef4444","#a21caf","#eab308","#0ea5e9","#f43f5e","#14b8a6","#64748b"];
  box.innerHTML="<canvas id='kmeans-canvas' style='width:100%;height:100%'></canvas>";
  const ctx=$("kmeans-canvas").getContext("2d");
  if(VizCharts.kmeans) VizCharts.kmeans.destroy();

  const datasets = Object.keys(clusters).map((clusterId, idx) => ({
    label: `Cluster ${clusterId}`,
    data: clusters[clusterId].map(p => ({x: p.x, y: p.y})),
    backgroundColor: colors[idx % colors.length],
    borderColor: colors[idx % colors.length],
    borderWidth: 1,
    pointRadius: 3,
    pointHoverRadius: 5
  }));

  VizCharts.kmeans = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: `K-Means Clustering (k=${km.k || '?'})` },
        legend: { display: true, position: 'top' }
      },
      scales: {
        x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Component 1' }, min: minX, max: maxX },
        y: { title: { display: true, text: 'Component 2' }, min: minY, max: maxY }
      }
    }
  });
  syncExportButtons();
  
  // Add AI description
  setTimeout(() => addChartDescription("kmeans-box", "kmeans", "K-means clustering visualization showing data points grouped by similarity"), 100);
}

/* ---------- Association Rules ---------- */
function renderAssoc(){
  const assoc=lsGet("assoc")||lsGet("autoBundle")?.assoc_rules;
  const box=$("assoc-box")||$("assoc-container"); if(!box) return;
  if(!assoc||!assoc.length){ box.innerHTML="<p class='text-small text-dim'>No rules.</p>"; return; }
  let h=`<table class='data-table'><thead><tr>
  <th>Antecedents</th><th>Consequents</th><th>Support</th><th>Confidence</th><th>Lift</th>
  </tr></thead><tbody>`;
  assoc.slice(0,100).forEach(r=>{
    h+=`<tr>
      <td>${Array.isArray(r.antecedents)?r.antecedents.join(", "):r.antecedents}</td>
      <td>${Array.isArray(r.consequents)?r.consequents.join(", "):r.consequents}</td>
      <td>${(+r.support||0).toFixed(3)}</td>
      <td>${(+r.confidence||0).toFixed(3)}</td>
      <td>${(+r.lift||0).toFixed(3)}</td>
    </tr>`;
  });
  h+="</tbody></table>";
  box.innerHTML=h;
  
  // Add AI description
  setTimeout(() => addChartDescription("assoc-box", "assoc", "Association rules showing relationships between categorical variables"), 100);
}

/* ---------- Summary ---------- */
function renderSummary(){
  const sum=lsGet("summary")||lsGet("autoBundle")?.summary;
  const box=$("summary-box")||$("summary-container"); if(!box) return;
  if(!sum){ box.innerHTML="<p class='text-small text-dim'>No summary data.</p>"; return; }
  const cols=Object.keys(sum); const stats=new Set(); cols.forEach(c=>Object.keys(sum[c]).forEach(k=>stats.add(k)));
  let h="<table class='data-table'><thead><tr><th>Metric</th>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead><tbody>";
  [...stats].forEach(st=>{
    h+=`<tr><th style="background:#1a2c3a">${st}</th>`;
    cols.forEach(c=>{
      let v=sum[c][st];
      if(v==null) v=""; else if(typeof v==="number") v=(Math.abs(v)>1e6||Math.abs(v)<1e-4)?v:v.toPrecision(6);
      h+=`<td>${v}</td>`;
    });
    h+="</tr>";
  });
  h+="</tbody></table>"; box.innerHTML=h;
  
  // Add AI description
  setTimeout(() => addChartDescription("summary-box", "summary", "Summary statistics showing descriptive metrics for all columns"), 100);
}

/* ---------- AI Narrative render ---------- */
function renderAINarrative(){
  const box = $("ai-narrative-box") || $("ai-latest");
  if(!box) {
    console.warn("renderAINarrative: No AI narrative box found");
    return;
  }

  const raw = lsGet("autoAI") || lsGet("lastAI");
  console.log("renderAINarrative - raw data:", raw);
  
  // Check for null/undefined or NoneType error
  if(!raw || raw === null || raw === 'null' || 
     (typeof raw === 'string' && (raw.includes("'NoneType'") || raw.includes("None") || raw.trim() === ""))){
    // Show a more informative generic paragraph about the data if no AI
    const meta = lsGet("meta") || lsGet("autoBundle")?.meta || lsGet("autoBundle")?.profile?.basic;
    const bundle = lsGet("autoBundle");
    console.log("renderAINarrative - no valid AI data, showing generic message");
    console.log("renderAINarrative - meta:", meta);
    console.log("renderAINarrative - bundle:", bundle);
    
    let summary = "<p class='text-dim' style='font-size: 0.9rem; line-height: 1.4;'>This dataset";
    
    if (meta && (meta.n_rows || meta.rows) && (meta.n_cols || meta.columns)) {
      const rows = meta.n_rows || meta.rows;
      const cols = meta.n_cols || meta.columns;
      summary += ` contains <strong>${rows} rows</strong> and <strong>${cols} columns</strong>`;
      if (meta.file_name || meta.filename) summary += ` from <strong>${meta.file_name||meta.filename}</strong>`;
      summary += ". ";
      
      // Add insights about data characteristics
      summary += `With ${cols} features across ${rows} observations, this dataset provides `;
      if (rows > 1000) summary += "substantial ";
      else if (rows > 100) summary += "adequate ";
      summary += "information for statistical analysis. ";
      
      // Add some insights about what analyses are available
      if (bundle) {
        const analyses = [];
        if (bundle.correlation_matrix) analyses.push("correlation patterns");
        if (bundle.time_series) analyses.push("time series analysis");
        if (bundle.kmeans) analyses.push("clustering analysis");
        if (bundle.categorical) analyses.push("categorical distributions");
        if (bundle.summary) analyses.push("statistical summaries");
        
        if (analyses.length > 0) {
          summary += `Available insights include: ${analyses.join(", ")}.`;
        }
      }
    } else {
      summary += " is ready for comprehensive analysis using various statistical and machine learning techniques.";
    }
    summary += "</p>";
    box.innerHTML = summary;
    return;
  }

  // Handle error cases
  if(raw && raw.error) {
    let errorMsg = raw.error || "Unknown error";
    console.error("renderAINarrative - AI Error:", errorMsg);
    console.error("renderAINarrative - Full raw object:", raw);
    
    // Provide more helpful error messages
    if (errorMsg.includes("GOOGLE_API_KEY")) {
      errorMsg = "AI model not configured. Please set GOOGLE_API_KEY environment variable.";
    } else if (errorMsg.includes("google-generativeai")) {
      errorMsg = "AI model package not available. Please install google-generativeai.";
    } else if (errorMsg.includes("initialization failed")) {
      errorMsg = "AI model failed to initialize. Please check your configuration.";
    } else if (errorMsg.includes("'NoneType'") || errorMsg.includes("NoneType")) {
      errorMsg = "AI analysis data is incomplete. Try running Auto Explore again.";
    } else if (errorMsg.includes("object has no attribute 'get'")) {
      errorMsg = "Data structure error in AI analysis. Please try running Auto Explore again.";
    } else if (errorMsg.includes("Invalid bundle data") || errorMsg.includes("Bundle missing")) {
      errorMsg = "Dataset analysis data is incomplete or corrupted. Please run Auto Explore again.";
    }
    
    box.innerHTML = `<p class='text-small text-dim' style='color: #ef4444;'>⚠️ AI analysis encountered an error: ${errorMsg}</p>
                     <p class='text-small text-dim' style='margin-top: 8px;'>The dataset analysis was completed successfully, but AI insights are not available.</p>`;
    return;
  }

  try {
    const ai = cleanAIBlock(raw);
    console.log("renderAINarrative - cleaned AI data:", ai);

    const list = (title, arr) =>
      (arr && arr.length)
        ? `<strong style="font-size:.62rem;">${title}</strong>
           <ul style="font-size:.6rem;margin:.35rem 0 .6rem 1rem;">${arr.map(x=>`<li>${x}</li>`).join("")}</ul>`
        : "";

    let html = "";
    
    // Handle different AI response structures
    if(ai && ai.overview) html += `<p style="font-size:.66rem;line-height:1.45;margin-bottom:.6rem;"><strong>Overview:</strong> ${ai.overview}</p>`;
    else if(ai && ai.summary) html += `<p style="font-size:.66rem;line-height:1.45;margin-bottom:.6rem;"><strong>Summary:</strong> ${ai.summary}</p>`;
    
    if(ai) {
      html += list("Key Findings", ai.key_findings || ai.key_points);
      html += list("Correlations", ai.correlations_comment ? [ai.correlations_comment] : []);
      html += list("Clusters", ai.clusters_comment ? [ai.clusters_comment] : []);
      html += list("Time Series Insights", ai.time_series_comment ? [ai.time_series_comment] : []);
      html += list("Categorical Insights", ai.categorical_insights);
      html += list("Potential Issues", ai.potential_issues || ai.anomalies);
      
      if(ai.recommendation){
        html += `<p style="font-size:.6rem;"><strong>Recommendation:</strong> ${ai.recommendation}</p>`;
      }
      html += list("Next Steps", ai.next_steps);
      html += list("Chart Priorities", ai.chart_priorities);
    }

    // If still no structured content, try to display raw content
    if(!html.trim()){
      console.warn("renderAINarrative - No structured content found, trying raw display");
      if(typeof raw === "string"){
        html = `<div style="font-size:.6rem;line-height:1.4;white-space:pre-wrap;">${raw}</div>`;
      } else if(raw && typeof raw === "object") {
        // Better formatting for unstructured data
        const keys = Object.keys(raw);
        html = keys.map(key => {
          const value = raw[key];
          if(Array.isArray(value) && value.length) {
            return `<strong style="font-size:.62rem;">${key.replace(/_/g, ' ').toUpperCase()}:</strong>
                    <ul style="font-size:.6rem;margin:.35rem 0 .6rem 1rem;">${value.map(x=>`<li>${x}</li>`).join("")}</ul>`;
          } else if(typeof value === 'string' && value.trim()) {
            return `<p style="font-size:.6rem;margin:.3rem 0;"><strong>${key.replace(/_/g, ' ')}:</strong> ${value}</p>`;
          }
          return '';
        }).filter(x => x).join('') || "<em>AI narrative present but unstructured.</em>";
      }
    }

    console.log("renderAINarrative - Final HTML length:", html.length);
    box.innerHTML = html || "<em>AI narrative present but could not be parsed.</em>";
    
  } catch(err) {
    console.error("renderAINarrative - Error processing AI data:", err);
    box.innerHTML = `<p class='text-small text-dim' style='color: #ef4444;'>⚠️ Error displaying AI insights: ${err.message}</p>`;
  }
}

/* ---------- Overview ---------- */
function renderOverview(){
  const b=lsGet("autoBundle");
  const oc=$("overview-meta")||$("overview-container"); if(!oc) return;
  if(!b){
    const fn=localStorage.getItem("filename")||"(none)";
    oc.innerHTML=`<p class='text-small text-dim'>No Auto Explore yet. Active file: <strong>${fn}</strong>.</p>`;
    return;
  }
  const base=b.profile?.basic||{};
  const rec=(b.recommended_charts||[]).map(c=>c.type||c).join(", ");
  oc.innerHTML=`
    <div class="inline wrap" style="gap:.4rem;margin-bottom:.4rem;">
      <span class="badge-chip">${b.filename}</span>
      <span class="badge-chip">${base.rows} rows</span>
      <span class="badge-chip">${base.columns} cols</span>
      <span class="badge-chip">${base.numeric_cols} numeric</span>
      <span class="badge-chip">${base.categorical_cols} categorical</span>
    </div>
    <p style="font-size:.63rem;margin:.4rem 0;"><strong>Recommended charts:</strong> ${rec||"—"}</p>`;
}

/* ---------- Column type ribbon ---------- */
function inferColumnTypes(){
  const bundle=lsGet("autoBundle"), summary=bundle?.summary||lsGet("summary");
  let cols=[]; if(summary) cols=Object.keys(summary);
  const types={}, rowCount=bundle?.profile?.basic?.rows||0;
  const numR=/(_amt|_num|count|total|sum|avg|mean|price|age|score|rate|pct|perc|lat|lon|long|prob|rank)$/i;
  const dateR=/(date|day|time|timestamp|dt)$/i;
  const idR=/(id|uuid|guid|code|ref)$/i;
  const boolR=/^(is_|has_|flag_|active|enabled|valid)/i;
  cols.forEach(c=>{
    const info=summary?.[c]||{}, u=info.unique??info.Unique??null;
    let k="TEXT";
    if(info.mean!==undefined||info.std!==undefined||info.max!==undefined) k="NUM";
    if(u!==null){
      if(u<=2) k="BOOL";
      else if(u<=Math.min(20,Math.max(10,rowCount*0.05)) && k!=="NUM") k="CAT";
    }
    if(dateR.test(c)) k="DATE";
    if(idR.test(c))   k="ID";
    if(boolR.test(c)) k="BOOL";
    if(numR.test(c)&&k!=="DATE") k="NUM";
    types[c]=k;
  });
  try{localStorage.setItem("colTypesCache",JSON.stringify(types));}catch{}
  renderColumnTypeRibbon(types);
}
function renderColumnTypeRibbon(types){
  const wrap=$("viz-columns")||$("col-type-ribbon"); if(!wrap) return;
  wrap.innerHTML="";
  if(!types||!Object.keys(types).length){
    wrap.innerHTML='<span class="text-small text-dim" style="padding:.25rem 0;">No columns detected</span>';
    return;
  }
  Object.entries(types).forEach(([col,kind])=>{
    const el=document.createElement("code");
    el.className="chip-col";
    el.dataset.kind=kind; el.dataset.col=col;
    el.innerHTML=`${col}<span class="chip-kind">${kind}</span>`;
    el.addEventListener("click",()=>handleColumnTypeClick(col,kind));
    wrap.appendChild(el);
  });
}
function handleColumnTypeClick(col,kind){
  if(["CAT","BOOL","ID"].includes(kind)){
    toast(`Primary categorical: ${col}`,"info");
    localStorage.setItem("primaryCategorical",col);
    const b=lsGet("autoBundle");
    if(b?.categorical?.[col]){
      const counts=b.categorical[col];
      lsSet("valueCounts",{labels:counts.map(x=>x.value),values:counts.map(x=>x.count),title:`Top ${col}`});
      if(window.currentVizTab==="value_counts"||window.currentVizTab==="overview") renderValueCounts();
    }
  }else toast(`"${col}" is ${kind}. Use numeric analyses.`, "warn");
}

/* ---------- Meta banner ---------- */
async function loadMeta(){
  const box=$("viz-dataset-meta")||$("dataset-meta");
  const chip=$("active-file-chip");
  try{
    const js=await handleApi("/api/files");
    const active=js.active||localStorage.getItem("filename");
    if(active){
      chip&&(chip.textContent=`Active: ${active}`);
      const f=(js.files||[]).find(x=>x.filename===active)||{};
      box && (box.innerHTML=`<strong>${active}</strong><br>Rows: ${f.rows??"?"} | Cols: ${f.columns??"?"} | Size: ${f.size_bytes?(f.size_bytes/1024).toFixed(1)+" KB":"?"}`);
      const b=lsGet("autoBundle")||{};
      b.filename=active;
      b.profile=b.profile||{}; b.profile.basic=b.profile.basic||{};
      if(f.rows) b.profile.basic.rows=f.rows;
      if(f.columns) b.profile.basic.columns=f.columns;
      lsSet("autoBundle",b);
    }else{ box&&(box.textContent="No active dataset."); }
  }catch(e){ box&&(box.textContent="Metadata error: "+e.message); }
}

function updateVizDatasetMeta(){
  return loadMeta();
}

/* ---------------- Page inits ---------------- */
function initVisualizationPage(){
  loadMeta();
  renderOverview();
  const vc=lsGet("valueCounts"); if(vc) renderValueCounts(vc.mode);
  renderAINarrative();
  const cached=localStorage.getItem("colTypesCache");
  if(cached){ try{renderColumnTypeRibbon(JSON.parse(cached));}catch{inferColumnTypes();} }
  else inferColumnTypes();
  syncExportButtons();
}
function initDashboardPage(){ previewDataset(); loadMeta(); }
function initAdminPage(){ adminLoadUsers(); previewDataset(); }
function initAnalysisPage(){ previewDataset(); loadMeta(); }

/* ---------------- DOM Ready ---------------- */
document.addEventListener("DOMContentLoaded", async ()=>{
  await ensureAuthForProtectedPages();
  await syncActiveFile(true);

  const page=document.body.getAttribute("data-page");
  window.currentVizTab="overview";

  if(page==="visualization")      initVisualizationPage();
  else if(page==="dashboard")     initDashboardPage();
  else if(page==="admin")         initAdminPage();
  else if(page==="analysis")      initAnalysisPage();

  // Common listeners
  $("smart-search-btn")?.addEventListener("click",smartSearch);
  $("fetch-btn")?.addEventListener("click",()=>{ const u=$("remote-url")?.value?.trim(); if(u) fetchFromInternet(u); });
  $("upload-btn")?.addEventListener("click",uploadDataset);
  $("clean-btn")?.addEventListener("click",applyCleaning);
  $("analyze-btn")?.addEventListener("click",runAnalysis);
  $("ai-generate-btn")?.addEventListener("click",generateAISummary);
  $("qi-run")?.addEventListener("click",generateAISummary);
  $("qi-rerun")?.addEventListener("click",generateAISummary);
  $("auto-explore-btn")?.addEventListener("click",autoExplore);
  // viz-auto-explore and pdf-report-btn handled in individual page scripts to prevent double binding
  // $("md-report-btn")?.addEventListener("click",downloadMarkdownReport); // Commented out - handled in individual pages
  // PDF button bindings also handled in individual pages to prevent double downloads
  $("btn-clear-cache")?.addEventListener("click",clearAnalysis);
  $("viz-clear-cache")?.addEventListener("click",clearAnalysis);

  document.querySelectorAll("#logout-link,.logout-link,a[href='#logout']").forEach(el=>{
    el.addEventListener("click",e=>{e.preventDefault();logout();});
  });

  $("download-corr-csv")?.addEventListener("click",downloadCorrelationCSV);
  $("download-corr-png")?.addEventListener("click",downloadCorrelationPNG);
  $("btn-corr-export-csv-2")?.addEventListener("click",downloadCorrelationCSV);
  $("btn-corr-export-png-2")?.addEventListener("click",downloadCorrelationPNG);
  $("corr-scale")?.addEventListener("change",()=>{ if(window.currentVizTab==="correlation") renderCorrTable(); });

  // VC mode toggle
  document.querySelectorAll("[data-vc-mode]")?.forEach(b=>{
    b.addEventListener("click",()=>renderValueCounts(b.dataset.vcMode));
  });

  // Refresh meta
  $("btn-refresh-meta")?.addEventListener("click",loadMeta);
  $("viz-refresh-meta-2")?.addEventListener("click",loadMeta);

  // Tabs
  const tabsRoot=$("viz-tabs");
  if(tabsRoot){
    tabsRoot.addEventListener("click",e=>{
      const btn=e.target.closest("button[data-tab]"); if(!btn) return;
      const tab=btn.dataset.tab; window.currentVizTab=tab;
      tabsRoot.querySelectorAll("button").forEach(b=>{
        const on=b===btn; b.classList.toggle("active",on); b.setAttribute("aria-selected",on?"true":"false");
      });
      const secs={overview:"sec-overview",value_counts:"sec-value_counts",correlation:"sec-correlation",
                  time_series:"sec-time-series",kmeans:"sec-kmeans",assoc:"sec-assoc",summary:"sec-summary",ai:"sec-ai"};
      Object.entries(secs).forEach(([k,id])=>$(id)?.classList.toggle("active",k===tab));

      if(tab==="value_counts") renderValueCounts();
      if(tab==="correlation")  ensureCorrelation();
      if(tab==="time_series")  renderTimeSeries();
      if(tab==="kmeans")       renderKMeans();
      if(tab==="assoc")        renderAssoc();
      if(tab==="summary")      renderSummary();
      if(tab==="ai")           renderAINarrative();
      if(tab==="overview")     renderOverview();
    });
  }

  // Resize -> redraw corr
  let t=null;
  window.addEventListener("resize",()=>{
    if(window.currentVizTab==="correlation"){
      clearTimeout(t); t=setTimeout(renderCorrTable,220);
    }
  });

  // Storage sync
  window.addEventListener("storage",ev=>{
    if(["autoBundle","valueCounts","correlation","filename"].includes(ev.key)){
      if(page==="visualization") initVisualizationPage();
      if(page==="dashboard") previewDataset();
    }
  });
});

/* ---------------- Expose for console/debug ---------------- */
window.previewDataset=previewDataset;
window.generateAISummary=generateAISummary;
window.runAnalysis=runAnalysis;
window.autoExplore=autoExplore;
window.renderValueCounts=renderValueCounts;
window.renderInteractiveCorrelation=renderInteractiveCorrelation;
window.renderCorrTable=renderCorrTable;
window.ensureCorrelation=ensureCorrelation;
window.renderTimeSeries=renderTimeSeries;
window.renderKMeans=renderKMeans;
window.renderAssoc=renderAssoc;
window.renderSummary=renderSummary;
window.renderAINarrative=renderAINarrative;
window.renderOverview=renderOverview;
window.downloadCorrelationCSV=downloadCorrelationCSV;
window.downloadCorrelationPNG=downloadCorrelationPNG;
window.clearAnalysis=clearAnalysis;
window.handleApi=handleApi;
window.downloadMarkdownReport=downloadMarkdownReport;
window.downloadPdfReport=downloadPdfReport;
window.downloadChart=downloadChart;
window.exportDataCsv=exportDataCsv;
window.smartSearch=smartSearch;
window.fetchFromInternet=fetchFromInternet;
window.uploadDataset=uploadDataset;
window.applyCleaning=applyCleaning;
window.inferColumnTypes=inferColumnTypes;
