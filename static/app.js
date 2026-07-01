/* ── DataSage AI v2.0 — app.js ── */
'use strict';

const API = '';
let state = {
  sessionId: null,
  expertise: 'intermediate',
  task: null,
  selectedModel: null,
};

const TASK_ICONS = { regression: '↗', classification: '◎', clustering: '●', time_series: '~' };

// ── EXPERTISE ──────────────────────────────────────────────────────────────
function setExpertise(level) {
  state.expertise = level;
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  document.getElementById('pill-' + level).classList.add('active');
}

// ── HEALTH CHECK ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API}/api/health`);
    const dot = document.getElementById('health-dot');
    if (dot) dot.style.background = res.ok ? 'var(--brand-green)' : 'var(--red)';
  } catch { /* silent */ }
}
checkHealth();

// ── UPLOAD & DRAG-DROP ─────────────────────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

// ── DEMO DATASETS ──────────────────────────────────────────────────────────
async function loadDemo(filename) {
  showLoading('Loading demo dataset…', filename);
  try {
    const res = await fetch(`/demo_data/${filename}`);
    if (!res.ok) throw new Error('Demo file not found. Run generate_demo_data.py first.');
    const blob = await res.blob();
    const file = new File([blob], filename, { type: blob.type });
    await uploadFile(file);
  } catch (err) {
    hideLoading();
    showToast('Could not load demo: ' + err.message, 'error');
  }
}

async function uploadFile(file) {
  showLoading('Ingesting dataset and running EDA…', file.name);
  const fd = new FormData();
  fd.append('file', file);
  fd.append('expertise', state.expertise);
  try {
    const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.sessionId = data.session_id;
    hideLoading();
    showToast('Dataset loaded ✓', 'success');
    renderEDA(data);
    activateStep(1);
    show('pipeline-section');
    show('eda-panel');
    scrollTo('eda-panel');
  } catch (err) {
    hideLoading();
    showToast('Upload failed: ' + err.message, 'error');
  }
}

// ── EDA RENDER ─────────────────────────────────────────────────────────────
function renderEDA(data) {
  const { schema, eda, rows, columns, fe_suggestions } = data;

  // Filename
  const fn = document.getElementById('eda-filename');
  if (fn) fn.textContent = `${rows.toLocaleString()} rows × ${schema.n_cols} columns`;

  // Quality badge
  const qBadge = document.getElementById('quality-badge');
  const qs = eda.data_quality_score;
  qBadge.textContent = `Data Quality: ${qs}%`;
  qBadge.style.borderColor = qs >= 80 ? 'var(--green)' : qs >= 60 ? 'var(--amber)' : 'var(--red)';
  qBadge.style.color = qs >= 80 ? 'var(--green)' : qs >= 60 ? 'var(--amber)' : 'var(--red)';

  const sum = eda.summary;
  const cards = [
    { label: 'Rows', value: rows.toLocaleString(), sub: 'data points' },
    { label: 'Columns', value: schema.n_cols, sub: 'features' },
    { label: 'Numeric', value: sum.numeric_count, sub: 'columns' },
    { label: 'Categorical', value: sum.categorical_count, sub: 'columns' },
    { label: 'Missing Cells', value: sum.total_missing_cells.toLocaleString(), sub: `${((sum.total_missing_cells / (rows * schema.n_cols)) * 100).toFixed(1)}% of all` },
    { label: 'Duplicates', value: sum.duplicate_rows, sub: 'rows' },
    { label: 'Outlier Cols', value: Object.keys(eda.outliers || {}).length, sub: 'detected' },
    { label: 'Datetime Cols', value: schema.datetime_cols.length, sub: 'columns' },
  ];
  document.getElementById('eda-grid').innerHTML = cards.map(c => `
    <div class="eda-card">
      <div class="eda-card-label">${c.label}</div>
      <div class="eda-card-value">${c.value}</div>
      <div class="eda-card-sub">${c.sub}</div>
    </div>`).join('');

  // Feature engineering suggestions
  if (fe_suggestions && fe_suggestions.length > 0) {
    show('fe-section');
    const feCount = document.getElementById('fe-count');
    if (feCount) feCount.textContent = `${fe_suggestions.length} suggestions`;
    document.getElementById('fe-suggestions').innerHTML = fe_suggestions.slice(0, 8).map(s => `
      <div class="fe-item fe-priority-${s.priority}">
        <div class="fe-item-feature">${s.feature}</div>
        <span class="fe-item-transform fe-transform-${s.transform}">${s.transform.replace(/_/g,' ')}</span>
        <div class="fe-item-reason">${s.reason}</div>
      </div>`).join('');
  }

  // Target column select
  const sel = document.getElementById('target-select');
  sel.innerHTML = '<option value="">-- None (Clustering) --</option>' +
    columns.map(c => `<option value="${c}">${c}</option>`).join('');
}

// ── DETECT TASK ────────────────────────────────────────────────────────────
async function detectTask() {
  if (!state.sessionId) return;
  const target = document.getElementById('target-select').value;
  const hint = document.getElementById('user-hint').value;
  showLoading('Detecting ML task using AI…');
  try {
    const res = await fetch(`${API}/api/detect-task`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, target_column: target, user_hint: hint }),
    });
    if (!res.ok) throw new Error(await res.text());
    state.task = await res.json();
    hideLoading();
    renderTaskResult(state.task);
    activateStep(2);
    show('task-panel');
    await recommendModels();
  } catch (err) {
    hideLoading();
    showToast('Task detection failed: ' + err.message, 'error');
  }
}

function renderTaskResult(task) {
  const icon = TASK_ICONS[task.task_type] || '?';
  const conf = Math.round(task.confidence * 100);
  document.getElementById('task-result').innerHTML = `
    <div class="task-badge">
      <div class="task-badge-icon">${icon}</div>
      <div class="task-badge-type">${task.task_type.replace('_', ' ')}</div>
      <div class="task-badge-conf">via ${task.source}</div>
    </div>
    <div class="task-info">
      <h3>Task Detected: ${task.task_type.replace('_', ' ').toUpperCase()}</h3>
      <p>${task.reasoning}</p>
      <div class="confidence-bar">
        <div class="conf-track"><div class="conf-fill" style="width:${conf}%"></div></div>
        <span class="conf-label">${conf}%</span>
      </div>
      ${renderSubtaskInfo(task.subtask_info, task.task_type)}
    </div>`;
  scrollTo('task-panel');
}

function renderSubtaskInfo(info, taskType) {
  if (!info || Object.keys(info).length === 0) return '';
  let html = '<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:4px;">';
  if (taskType === 'classification') {
    html += `<span class="chip chip-good">${info.n_classes} classes</span>`;
    html += `<span class="chip">${info.is_binary ? 'Binary' : 'Multi-class'}</span>`;
  } else if (taskType === 'regression') {
    if (info.target_range) html += `<span class="chip">Range: ${info.target_range[0].toFixed(1)} – ${info.target_range[1].toFixed(1)}</span>`;
    if (info.target_mean) html += `<span class="chip">Mean: ${info.target_mean.toFixed(2)}</span>`;
  } else if (taskType === 'clustering') {
    html += `<span class="chip">Suggested k: ${info.suggested_k}</span>`;
  }
  return html + '</div>';
}

// ── RECOMMEND MODELS ───────────────────────────────────────────────────────
async function recommendModels() {
  showLoading('Ranking ML models for your dataset…');
  try {
    const res = await fetch(`${API}/api/recommend-models`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    hideLoading();
    renderModelCards(data.recommendations);
    activateStep(3);
    show('models-panel');
    scrollTo('models-panel');
  } catch (err) {
    hideLoading();
    showToast('Model recommendation failed: ' + err.message, 'error');
  }
}

function renderModelCards(models) {
  document.getElementById('models-grid').innerHTML = models.map(m => `
    <div class="model-card ${m.recommended ? 'top-pick' : ''}" id="card-${m.key}" onclick="selectModel('${m.key}')">
      <div class="model-name">${m.name}</div>
      <div class="model-score-row">
        <div class="score-bar"><div class="score-fill" style="width:${m.score}%"></div></div>
        <span class="model-score">${m.score}/100</span>
      </div>
      <div class="model-chips">
        <span class="chip">${m.complexity} complexity</span>
        <span class="chip">${m.interpretability} interpretability</span>
        ${m.pros.slice(0,2).map(p => `<span class="chip chip-good">✓ ${p}</span>`).join('')}
        ${m.cons.slice(0,1).map(c => `<span class="chip chip-warn">⚠ ${c}</span>`).join('')}
      </div>
      <div class="model-use-when">${m.use_when}</div>
      <p style="font-size:12px;color:var(--text-3);margin-bottom:14px;line-height:1.6;">${m.reasoning.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\*(.*?)\*/g,'<em>$1</em>')}</p>
      <button class="btn-run" onclick="event.stopPropagation(); runWorkflow('${m.key}')">Run This Model →</button>
    </div>`).join('');
}

function selectModel(key) {
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
  const el = document.getElementById('card-' + key);
  if (el) el.classList.add('selected');
  state.selectedModel = key;
}

// ── RUN WORKFLOW ───────────────────────────────────────────────────────────
async function runWorkflow(modelKey) {
  if (!state.sessionId) return;
  selectModel(modelKey);
  showLoading('Training model and computing results…', 'Running pipeline + drift detection');
  try {
    const res = await fetch(`${API}/api/run-workflow`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, model_key: modelKey }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    hideLoading();
    renderResults(data, modelKey);
    activateStep(4);
    markStepDone(4);
    show('results-panel');
    scrollTo('results-panel');
    showToast('Analysis complete ✓', 'success');
  } catch (err) {
    hideLoading();
    showToast('Workflow failed: ' + err.message, 'error');
  }
}

function renderResults(data, modelKey) {
  const { results, charts, explanation, drift_report } = data;
  document.getElementById('result-model-badge').textContent = results.model_name;

  // Metrics
  const mRow = document.getElementById('metrics-row');
  mRow.innerHTML = Object.entries(results.metrics)
    .filter(([k]) => !['class_names','cv_accuracy_std','cv_r2_std'].includes(k))
    .map(([k, v]) => `<div class="metric-card"><div class="metric-value">${typeof v === 'number' ? fmtMetric(k, v) : v}</div><div class="metric-label">${k.replace(/_/g,' ').toUpperCase()}</div></div>`)
    .join('');

  // Drift alert
  if (drift_report) renderDriftAlert(drift_report);

  // Explanation
  renderExplanation(explanation, 'explanation-card');

  // Charts
  renderCharts(charts, 'charts-grid');
}

function renderDriftAlert(drift) {
  const el = document.getElementById('drift-alert');
  if (!el) return;
  const sev = drift.drift_severity || 'none';
  const icons = { none: '✓', warning: '⚠', critical: '⛔' };
  const titles = { none: 'No Data Drift Detected', warning: 'Data Drift Warning', critical: 'Critical Distribution Shift' };
  el.className = `drift-alert ${sev}`;
  el.innerHTML = `
    <div class="drift-alert-icon">${icons[sev]}</div>
    <div>
      <div class="drift-alert-title">${titles[sev]}</div>
      <div class="drift-alert-text">${drift.summary || ''} ${drift.features_drifted > 0 ? `<strong>${drift.features_drifted}/${drift.total_features} features</strong> show distribution shift.` : ''}</div>
    </div>`;
  show('drift-alert');
}

function fmtMetric(key, val) {
  if (key === 'n_clusters_found') return val;
  if (['rmse', 'mae', 'aic'].includes(key)) return val.toFixed(3);
  if (key === 'silhouette_score') return val !== null ? val.toFixed(3) : 'N/A';
  return (val * 100).toFixed(1) + '%';
}

function renderExplanation(exp, containerId) {
  if (!exp) return;
  const sections = [
    { key: 'summary', label: 'Summary' },
    { key: 'performance_analysis', label: 'Performance' },
    { key: 'key_drivers', label: 'Key Drivers' },
    { key: 'recommendations', label: 'Recommendations' },
    { key: 'cautions', label: 'Cautions' },
  ];
  document.getElementById(containerId).innerHTML = sections
    .filter(s => exp[s.key])
    .map(s => `
      <div class="exp-section">
        <div class="exp-label">${s.label}</div>
        <div class="exp-text">${exp[s.key].replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\*(.*?)\*/g,'<em>$1</em>')}</div>
      </div>`).join('');
}

function renderCharts(charts, gridId) {
  const grid = document.getElementById(gridId);
  grid.innerHTML = '';
  Object.entries(charts).forEach(([key, fig]) => {
    if (!fig || key === 'error') return;
    const div = document.createElement('div');
    div.className = 'chart-card';
    const inner = document.createElement('div');
    inner.id = `chart-${gridId}-${key}`;
    inner.style.cssText = 'width:100%;height:320px;';
    div.appendChild(inner);
    grid.appendChild(div);
    try {
      Plotly.newPlot(inner, fig.data, { ...fig.layout, autosize: true }, { responsive: true, displayModeBar: false });
    } catch (e) { console.warn('Chart error:', e); }
  });
}

// ── MODEL ARENA ─────────────────────────────────────────────────────────────
async function runArena() {
  if (!state.sessionId) return;
  const btn = document.getElementById('arena-btn');
  if (btn) btn.disabled = true;
  showLoading('⚡ Running Model Arena…', 'Benchmarking all models — this may take 30-60s');
  try {
    const res = await fetch(`${API}/api/arena`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    hideLoading();
    renderArena(data);
    activateStep(5);
    show('arena-panel');
    scrollTo('arena-panel');
    showToast(`Arena complete! ${data.successful} models benchmarked ✓`, 'success');
  } catch (err) {
    hideLoading();
    showToast('Arena failed: ' + err.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderArena(data) {
  const { leaderboard, radar_data, winner, task_type, successful, total_models } = data;

  const subtitle = document.getElementById('arena-subtitle');
  if (subtitle) subtitle.textContent = `${successful}/${total_models} models benchmarked · Task: ${task_type}`;

  // Leaderboard
  const maxVal = leaderboard[0] ? leaderboard[0].primary_value : 1;
  document.getElementById('arena-leaderboard').innerHTML = leaderboard.map(e => `
    <div class="arena-row rank-${e.rank}">
      <div class="arena-rank">#${e.rank}</div>
      <div class="arena-badge">${e.badge}</div>
      <div class="arena-model-name">${e.model_name}</div>
      <div class="arena-bar-wrap">
        <div class="arena-bar" style="width:${Math.round((e.primary_value / maxVal) * 100)}%"></div>
      </div>
      <div class="arena-metric">
        <div class="arena-metric-value">${fmtMetric(e.primary_metric, e.primary_value)}</div>
        <div class="arena-metric-label">${e.primary_metric}</div>
      </div>
      <div class="arena-time">${e.training_time_sec}s</div>
    </div>`).join('');

  // Charts — leaderboard bar + radar
  const vizEngine = { leaderboard: null, radar: null };
  const arenaChartsGrid = document.getElementById('arena-charts-grid');
  arenaChartsGrid.innerHTML = '';

  // Build leaderboard bar chart inline
  if (leaderboard.length > 0) {
    const names = leaderboard.map(e => e.model_name).reverse();
    const vals = leaderboard.map(e => e.primary_value).reverse();
    const colors = leaderboard.map((e, i) => i === leaderboard.length - 1 ? 'var(--brand-green)' : i === leaderboard.length - 2 ? '#3b82f6' : 'var(--border)').reverse();
    const metric = leaderboard[0].primary_metric;
    renderPlotlyInGrid(arenaChartsGrid, 'arena-lb', [{
      type: 'bar', orientation: 'h',
      y: names, x: vals, marker: { color: colors },
      text: vals.map(v => v.toFixed(4)), textposition: 'outside',
    }], { title: `Model Leaderboard — ${metric.toUpperCase()}`, paper_bgcolor: '#fff', plot_bgcolor: '#f0f0f0', font: { color: '#1f1f1f' }, margin: { l: 180, r: 60, t: 50, b: 40 } });
  }

  // Radar chart
  if (radar_data && radar_data.models && radar_data.models.length > 0) {
    const traces = radar_data.models.slice(0, 4).map((m, i) => ({
      type: 'scatterpolar',
      r: [...m.values, m.values[0]],
      theta: [...radar_data.axis_labels, radar_data.axis_labels[0]],
      fill: 'toself', name: m.model_name, opacity: 0.6,
    }));
    renderPlotlyInGrid(arenaChartsGrid, 'arena-radar', traces, {
      title: 'Model Comparison Radar',
      paper_bgcolor: '#fff', font: { color: '#1f1f1f' },
      polar: { radialaxis: { visible: true, range: [0, 100] } },
      margin: { l: 30, r: 30, t: 50, b: 30 },
    });
  }

  // Winner explanation
  if (data.winner_explanation) {
    const expEl = document.getElementById('arena-explanation');
    show('arena-explanation');
    renderExplanation(data.winner_explanation, 'arena-explanation');
  }
}

function renderPlotlyInGrid(grid, id, traces, layout) {
  const div = document.createElement('div');
  div.className = 'chart-card';
  const inner = document.createElement('div');
  inner.id = id;
  inner.style.cssText = 'width:100%;height:360px;';
  div.appendChild(inner);
  grid.appendChild(div);
  try { Plotly.newPlot(inner, traces, { ...layout, autosize: true }, { responsive: true, displayModeBar: false }); }
  catch (e) { console.warn('Chart error:', e); }
}

// ── Q&A ─────────────────────────────────────────────────────────────────────
async function askQuestion() {
  const input = document.getElementById('qa-input');
  const q = input.value.trim();
  if (!q || !state.sessionId) return;
  const ansEl = document.getElementById('qa-answer');
  ansEl.textContent = 'Thinking…';
  ansEl.classList.add('visible');
  try {
    const res = await fetch(`${API}/api/ask`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, question: q }),
    });
    const data = await res.json();
    ansEl.textContent = data.answer;
    input.value = '';
  } catch (err) {
    ansEl.textContent = 'Could not get answer: ' + err.message;
  }
}
document.getElementById('qa-input').addEventListener('keydown', e => { if (e.key === 'Enter') askQuestion(); });

// ── STEPPER ─────────────────────────────────────────────────────────────────
function activateStep(n) {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('step-' + i);
    if (el) el.classList.toggle('active', i <= n);
  }
}
function markStepDone(n) {
  const el = document.getElementById('step-' + n);
  if (el) el.classList.add('done');
}

// ── UI HELPERS ───────────────────────────────────────────────────────────────
function show(id) { const el = document.getElementById(id); if (el) el.classList.remove('hidden'); }
function hide(id) { const el = document.getElementById(id); if (el) el.classList.add('hidden'); }
function showLoading(msg, sub) {
  document.getElementById('loading-text').textContent = msg || 'Processing…';
  const subEl = document.getElementById('loading-sub');
  if (subEl) subEl.textContent = sub || '';
  show('loading-overlay');
}
function hideLoading() { hide('loading-overlay'); }
function scrollTo(id) {
  setTimeout(() => { const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);
}
function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type}`;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 4500);
}

// Initialize background HLS video
(function() {
  const video = document.getElementById('space-video');
  if (video) {
    const streamUrl = 'https://stream.mux.com/tLkHO1qZoaaQOUeVWo8hEBeGQfySP02EPS02BmnNFyXys.m3u8';
    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
      const hls = new Hls({ enableWorker: false });
      hls.loadSource(streamUrl);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(e => console.log('Autoplay prevented:', e));
      });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = streamUrl;
      video.addEventListener('loadedmetadata', () => {
        video.play().catch(e => console.log('Autoplay prevented:', e));
      });
    }
  }
})();

// Initialize generative 3D sphere scene (Three.js)
(function() {
  const container = document.getElementById('generative-sphere-container');
  if (!container) return;

  const scene = new THREE.Scene();
  const width = container.clientWidth || 400;
  const height = container.clientHeight || 400;
  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.z = 3;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const geometry = new THREE.IcosahedronGeometry(1.2, 64);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0 },
      pointLightPos: { value: new THREE.Vector3(0, 0, 5) },
      color: { value: new THREE.Color("#5ed29c") } // Mint green color palette
    },
    vertexShader: `
      uniform float time;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      // Perlin Noise function
      vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
      vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
      float snoise(vec3 v) {
          const vec2 C = vec2(1.0/6.0, 1.0/3.0);
          const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
          vec3 i = floor(v + dot(v, C.yyy));
          vec3 x0 = v - i + dot(i, C.xxx);
          vec3 g = step(x0.yzx, x0.xyz);
          vec3 l = 1.0 - g;
          vec3 i1 = min(g.xyz, l.zxy);
          vec3 i2 = max(g.xyz, l.zxy);
          vec3 x1 = x0 - i1 + C.xxx;
          vec3 x2 = x0 - i2 + C.yyy;
          vec3 x3 = x0 - D.yyy;
          i = mod289(i);
          vec4 p = permute(permute(permute(
                      i.z + vec4(0.0, i1.z, i2.z, 1.0))
                  + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                  + i.x + vec4(0.0, i1.x, i2.x, 1.0));
          float n_ = 0.142857142857;
          vec3 ns = n_ * D.wyz - D.xzx;
          vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
          vec4 x_ = floor(j * ns.z);
          vec4 y_ = floor(j - 7.0 * x_);
          vec4 x = x_ * ns.x + ns.yyyy;
          vec4 y = y_ * ns.x + ns.yyyy;
          vec4 h = 1.0 - abs(x) - abs(y);
          vec4 b0 = vec4(x.xy, y.xy);
          vec4 b1 = vec4(x.zw, y.zw);
          vec4 s0 = floor(b0) * 2.0 + 1.0;
          vec4 s1 = floor(b1) * 2.0 + 1.0;
          vec4 sh = -step(h, vec4(0.0));
          vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
          vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
          vec3 p0 = vec3(a0.xy, h.x);
          vec3 p1 = vec3(a0.zw, h.y);
          vec3 p2 = vec3(a1.xy, h.z);
          vec3 p3 = vec3(a1.zw, h.w);
          vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
          p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
          vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
          m = m * m;
          return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
      }

      void main() {
          vNormal = normal;
          vPosition = position;
          float displacement = snoise(position * 2.0 + time * 0.5) * 0.2;
          vec3 newPosition = position + normal * displacement;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      uniform vec3 pointLightPos;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
          vec3 normal = normalize(vNormal);
          vec3 lightDir = normalize(pointLightPos - vPosition);
          float diffuse = max(dot(normal, lightDir), 0.0);
          
          // Fresnel effect for the glow
          float fresnel = 1.0 - dot(normal, vec3(0.0, 0.0, 1.0));
          fresnel = pow(fresnel, 2.0);
          
          vec3 finalColor = color * diffuse + color * fresnel * 0.5;
          
          gl_FragColor = vec4(finalColor, 1.0);
      }
    `,
    wireframe: true
  });

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  const pointLight = new THREE.PointLight(0xffffff, 1, 100);
  pointLight.position.set(0, 0, 5);
  scene.add(pointLight);

  let frameId;
  function animate(t) {
    material.uniforms.time.value = t * 0.0003;
    mesh.rotation.y += 0.0005;
    mesh.rotation.x += 0.0002;
    renderer.render(scene, camera);
    frameId = requestAnimationFrame(animate);
  }
  animate(0);

  function handleResize() {
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  function handleMouseMove(e) {
    const rect = container.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    const vec = new THREE.Vector3(x, y, 0.5).unproject(camera);
    const dir = vec.sub(camera.position).normalize();
    const dist = -camera.position.z / dir.z;
    const pos = camera.position.clone().add(dir.multiplyScalar(dist));
    pointLight.position.copy(pos);
    material.uniforms.pointLightPos.value = pos;
  }

  function handleTouchMove(e) {
    if (e.touches && e.touches[0]) {
      const rect = container.getBoundingClientRect();
      const x = ((e.touches[0].clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((e.touches[0].clientY - rect.top) / rect.height) * 2 + 1;
      const vec = new THREE.Vector3(x, y, 0.5).unproject(camera);
      const dir = vec.sub(camera.position).normalize();
      const dist = -camera.position.z / dir.z;
      const pos = camera.position.clone().add(dir.multiplyScalar(dist));
      pointLight.position.copy(pos);
      material.uniforms.pointLightPos.value = pos;
    }
  }

  window.addEventListener('resize', handleResize);
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('touchmove', handleTouchMove, { passive: true });
  setTimeout(handleResize, 150);
})();
