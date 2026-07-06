/* ══════════════════════════════════════════════════════════════
   StockSight AI — app.js
   Frontend logic: data fetching, chart rendering, tab navigation
   ══════════════════════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const State = {
  ticker:    'AAPL',
  days:      90,
  tickers:   {},
  history:   null,
  prediction: null,
  charts: {
    price:   null,
    rsi:     null,
    macd:    null,
    volume:  null,
    compare: null,
    backtest:null,
  },
};

// ── Chart Defaults ────────────────────────────────────────────────────────
Chart.defaults.color          = '#94a3b8';
Chart.defaults.borderColor    = 'rgba(99,102,241,.12)';
Chart.defaults.font.family    = 'Inter, sans-serif';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;

const CHART_COLORS = {
  price:    '#6366f1',
  sma20:    '#f59e0b',
  sma50:    '#a855f7',
  ema12:    '#06b6d4',
  lstm:     '#6366f1',
  rf:       '#10b981',
  gb:       '#f59e0b',
  ensemble: '#06b6d4',
  actual:   '#f1f5f9',
  rsi:      '#06b6d4',
  macd:     '#6366f1',
  signal:   '#f59e0b',
  volume:   'rgba(99,102,241,.55)',
  bb_upper: 'rgba(168,85,247,.6)',
  bb_lower: 'rgba(168,85,247,.6)',
};

function makeGradient(ctx, color, alpha1 = 0.3, alpha2 = 0.0) {
  const g = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
  g.addColorStop(0, color.replace(')', `, ${alpha1})`).replace('rgb', 'rgba'));
  g.addColorStop(1, color.replace(')', `, ${alpha2})`).replace('rgb', 'rgba'));
  return g;
}

// ── Utilities ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt = n => n != null ? `$${Number(n).toFixed(2)}` : '—';
const fmtPct = n => n != null ? `${n > 0 ? '+' : ''}${Number(n).toFixed(2)}%` : '—';

function showLoading(text = 'Loading…') {
  $('loading').classList.add('show');
  $('loading-text').textContent = text;
}
function hideLoading() { $('loading').classList.remove('show'); }

function showToast(msg, type = 'info') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  setTimeout(() => el.classList.remove('show'), 3500);
}

function destroyChart(key) {
  if (State.charts[key]) { State.charts[key].destroy(); State.charts[key] = null; }
}

// ── API Calls ─────────────────────────────────────────────────────────────
async function apiFetch(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

// ── Ticker Bar ────────────────────────────────────────────────────────────
async function loadTickers() {
  try {
    const data = await apiFetch('/api/tickers');
    State.tickers = data;
    renderTickerBar(data);
    renderModelsGrid(data);
  } catch (e) {
    showToast(`Failed to load tickers: ${e.message}`, 'error');
  }
}

function renderTickerBar(tickers) {
  const bar = $('ticker-bar');
  bar.innerHTML = '';
  for (const [ticker, info] of Object.entries(tickers)) {
    const btn = document.createElement('button');
    btn.className = `ticker-btn ${ticker === State.ticker ? 'active' : ''}`;
    btn.id = `tb-${ticker}`;
    btn.dataset.ticker = ticker;
    const priceHtml = info.last_close
      ? `<span class="ticker-price">${fmt(info.last_close)}</span>`
      : '';
    btn.innerHTML = `<strong>${ticker}</strong>${priceHtml}`;
    btn.addEventListener('click', () => selectTicker(ticker));
    bar.appendChild(btn);
  }
}

function renderModelsGrid(tickers) {
  const grid = $('models-grid');
  grid.innerHTML = '';
  for (const [ticker, info] of Object.entries(tickers)) {
    const m = info.metrics?.Ensemble || {};
    const card = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML = `
      <div class="mc-ticker">${ticker}</div>
      <div class="mc-name">${info.name}</div>
      <div class="mc-status ${info.trained ? 'trained' : 'untrained'}">
        ${info.trained ? '✅ Trained' : '⚠ Not Trained'}
      </div>
      ${info.trained ? `
        <div class="mc-metrics">
          MAE: <span>${m.MAE ?? '—'}</span><br>
          RMSE: <span>${m.RMSE ?? '—'}</span><br>
          R²: <span>${m.R2 ?? '—'}</span><br>
          Last: <span>${info.last_date ?? '—'}</span>
        </div>` : ''}
      <button class="mc-train-btn" id="train-btn-${ticker}"
        onclick="triggerTrain('${ticker}')">
        ${info.trained ? '🔄 Retrain' : '🚀 Train Model'}
      </button>
    `;
    grid.appendChild(card);
  }
}

// ── Ticker Selection ──────────────────────────────────────────────────────
async function selectTicker(ticker) {
  State.ticker = ticker;
  document.querySelectorAll('.ticker-btn').forEach(b => b.classList.remove('active'));
  const btn = $(`tb-${ticker}`);
  if (btn) btn.classList.add('active');

  await loadDashboardData();
}

async function loadDashboardData() {
  showLoading('Fetching stock data…');
  try {
    const hist = await apiFetch(`/api/history/${State.ticker}?days=${State.days + 60}`);
    State.history = hist;
    updatePriceKPI(hist);
    renderPriceChart(hist);
    renderRSIChart(hist);
    renderMACDChart(hist);
    renderVolumeChart(hist);

    // Try to load prediction if model exists
    if (State.tickers[State.ticker]?.trained) {
      try {
        const pred = await apiFetch(`/api/predict/${State.ticker}`);
        State.prediction = pred;
        updatePredictionKPI(pred);
        renderPredictPage(pred);
      } catch (e) {
        clearPredictionKPI();
      }
    } else {
      clearPredictionKPI();
    }
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  } finally {
    hideLoading();
  }
}

// ── KPI Updates ───────────────────────────────────────────────────────────
function updatePriceKPI(hist) {
  const prices = hist.close;
  const last   = prices[prices.length - 1];
  const prev   = prices[prices.length - 2];
  const chg    = ((last - prev) / prev * 100).toFixed(2);
  $('kv-price').textContent = fmt(last);
  $('kv-date').textContent  = hist.dates[hist.dates.length - 1];
  $('kv-price').style.color = chg >= 0 ? '#10b981' : '#ef4444';
}

function updatePredictionKPI(pred) {
  const p = pred.predictions.Ensemble;
  const c = pred.change.Ensemble;
  const r2 = pred.model_metrics?.Ensemble?.R2;
  $('kv-pred').textContent      = fmt(p);
  $('kv-pred-date').textContent = pred.next_date;
  $('kv-change').textContent    = fmtPct(c);
  $('kv-change').style.color    = c >= 0 ? '#10b981' : '#ef4444';
  $('kv-r2').textContent        = r2 != null ? r2.toFixed(4) : '—';
}

function clearPredictionKPI() {
  $('kv-pred').textContent   = '—';
  $('kv-pred-date').textContent = 'Train model first';
  $('kv-change').textContent  = '—';
  $('kv-r2').textContent      = '—';
}

// ── Charts ────────────────────────────────────────────────────────────────
function sliceHistory(hist, days) {
  const n = Math.min(days, hist.dates.length);
  const sl = {};
  for (const [k, v] of Object.entries(hist)) sl[k] = Array.isArray(v) ? v.slice(-n) : v;
  return sl;
}

function renderPriceChart(hist) {
  destroyChart('price');
  const h = sliceHistory(hist, State.days);
  const ctx = $('priceChart').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 360);
  grad.addColorStop(0, 'rgba(99,102,241,0.30)');
  grad.addColorStop(1, 'rgba(99,102,241,0.00)');

  State.charts.price = new Chart(ctx, {
    type: 'line',
    data: {
      labels: h.dates,
      datasets: [
        {
          label: 'Close', data: h.close,
          borderColor: CHART_COLORS.price, borderWidth: 2,
          pointRadius: 0, fill: true, backgroundColor: grad, tension: 0.3,
        },
        {
          label: 'SMA 20', data: h.sma20,
          borderColor: CHART_COLORS.sma20, borderWidth: 1.5,
          pointRadius: 0, fill: false, tension: 0.3, borderDash: [],
        },
        {
          label: 'SMA 50', data: h.sma50,
          borderColor: CHART_COLORS.sma50, borderWidth: 1.5,
          pointRadius: 0, fill: false, tension: 0.3,
        },
        {
          label: 'BB Upper', data: h.bb_upper,
          borderColor: CHART_COLORS.bb_upper, borderWidth: 1,
          pointRadius: 0, fill: false, tension: 0.3, borderDash: [5,5],
        },
        {
          label: 'BB Lower', data: h.bb_lower,
          borderColor: CHART_COLORS.bb_lower, borderWidth: 1,
          pointRadius: 0, fill: false, tension: 0.3, borderDash: [5,5],
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,.95)', borderColor: 'rgba(99,102,241,.4)',
          borderWidth: 1, padding: 12,
          callbacks: { label: ctx => ` ${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2)}` },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { maxTicksLimit: 8, font: { size: 11 } } },
        y: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { callback: v => `$${v}` } },
      },
    },
  });
}

function renderRSIChart(hist) {
  destroyChart('rsi');
  const h = sliceHistory(hist, State.days);
  const ctx = $('rsiChart').getContext('2d');
  State.charts.rsi = new Chart(ctx, {
    type: 'line',
    data: {
      labels: h.dates,
      datasets: [{
        label: 'RSI (14)', data: h.rsi,
        borderColor: CHART_COLORS.rsi, borderWidth: 2,
        pointRadius: 0, fill: false, tension: 0.4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(13,17,23,.95)' } },
      scales: {
        x: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { maxTicksLimit: 5, font: { size: 10 } } },
        y: {
          min: 0, max: 100,
          grid: { color: 'rgba(99,102,241,.08)' },
          ticks: { stepSize: 25 },
        },
      },
      plugins: {
        annotation: {},
        legend: { display: false },
      },
    },
    plugins: [{
      afterDraw(chart) {
        const { ctx, chartArea: { left, right }, scales: { y } } = chart;
        [30, 70].forEach(level => {
          const yp = y.getPixelForValue(level);
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = level === 70 ? 'rgba(239,68,68,.5)' : 'rgba(16,185,129,.5)';
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(left, yp); ctx.lineTo(right, yp); ctx.stroke();
          ctx.restore();
        });
      }
    }],
  });
}

function renderMACDChart(hist) {
  destroyChart('macd');
  const h = sliceHistory(hist, State.days);
  const ctx = $('macdChart').getContext('2d');
  State.charts.macd = new Chart(ctx, {
    type: 'line',
    data: {
      labels: h.dates,
      datasets: [
        {
          label: 'MACD', data: h.macd,
          borderColor: CHART_COLORS.macd, borderWidth: 2,
          pointRadius: 0, fill: false, tension: 0.4,
        },
        {
          label: 'Signal', data: h.macd_signal,
          borderColor: CHART_COLORS.signal, borderWidth: 1.5,
          pointRadius: 0, fill: false, tension: 0.4,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } }, tooltip: { backgroundColor: 'rgba(13,17,23,.95)' } },
      scales: {
        x: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { maxTicksLimit: 5, font: { size: 10 } } },
        y: { grid: { color: 'rgba(99,102,241,.08)' } },
      },
    },
  });
}

function renderVolumeChart(hist) {
  destroyChart('volume');
  const h = sliceHistory(hist, State.days);
  const ctx = $('volumeChart').getContext('2d');
  State.charts.volume = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: h.dates,
      datasets: [{
        label: 'Volume', data: h.volume,
        backgroundColor: CHART_COLORS.volume,
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8, font: { size: 10 } } },
        y: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { callback: v => v >= 1e9 ? `${(v/1e9).toFixed(1)}B` : v >= 1e6 ? `${(v/1e6).toFixed(0)}M` : v } },
      },
    },
  });
}

// ── Predict Page ──────────────────────────────────────────────────────────
function renderPredictPage(pred) {
  const cards = $('predict-cards');
  const models = [
    { key: 'LSTM',            icon: '🧠', weight: 'Ensemble weight: 50%' },
    { key: 'RandomForest',    icon: '🌲', weight: 'Ensemble weight: 30%' },
    { key: 'GradientBoosting',icon: '⚡', weight: 'Ensemble weight: 20%' },
    { key: 'Ensemble',        icon: '🎯', weight: 'Final prediction' },
  ];

  cards.innerHTML = '';
  for (const { key, icon, weight } of models) {
    const price  = pred.predictions[key];
    const change = pred.change[key];
    const up     = change >= 0;
    const card   = document.createElement('div');
    card.className = 'predict-card';
    card.innerHTML = `
      <div class="pc-model">${key}</div>
      <div class="pc-icon">${icon}</div>
      <div class="pc-price" style="color:${up ? '#10b981' : '#ef4444'}">${fmt(price)}</div>
      <div class="pc-change ${up ? 'up' : 'down'}">${fmtPct(change)}</div>
      <div class="pc-weight">${weight}</div>
    `;
    cards.appendChild(card);
  }

  // Model comparison bar chart
  renderModelCompareChart(pred);
}

function renderModelCompareChart(pred) {
  destroyChart('compare');
  const ctx = $('modelCompareChart').getContext('2d');
  const labels   = ['LSTM', 'Random Forest', 'Gradient Boosting', 'Ensemble'];
  const prices   = [pred.predictions.LSTM, pred.predictions.RandomForest, pred.predictions.GradientBoosting, pred.predictions.Ensemble];
  const actual   = Array(4).fill(pred.last_close);
  const colors   = [CHART_COLORS.lstm, CHART_COLORS.rf, CHART_COLORS.gb, CHART_COLORS.ensemble];

  State.charts.compare = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Predicted', data: prices, backgroundColor: colors,
          borderRadius: 8, borderSkipped: false,
        },
        {
          label: 'Last Close', data: actual,
          type: 'line', borderColor: '#f1f5f9', borderWidth: 2,
          pointRadius: 0, fill: false,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,.95)',
          callbacks: { label: ctx => ` ${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2)}` },
        },
      },
      scales: {
        x: { grid: { display: false } },
        y: {
          grid: { color: 'rgba(99,102,241,.08)' },
          ticks: { callback: v => `$${v}` },
          min: Math.min(...prices, pred.last_close) * 0.998,
          max: Math.max(...prices, pred.last_close) * 1.002,
        },
      },
    },
  });
}

// ── Backtest Page ──────────────────────────────────────────────────────────
async function loadBacktest() {
  const ticker = State.ticker;
  if (!State.tickers[ticker]?.trained) {
    showToast('Train the model first to see backtest results.', 'error');
    return;
  }

  showLoading('Running backtest…');
  $('backtest-status').textContent = 'Loading…';
  $('backtest-status').className   = 'status-badge';

  try {
    const data = await apiFetch(`/api/backtest/${ticker}?days=90`);
    $('backtest-status').textContent = `✅ ${data.dates.length} data points`;
    $('backtest-status').className   = 'status-badge ok';
    renderBacktestChart(data);
    renderBacktestMetrics();
  } catch (e) {
    $('backtest-status').textContent = `Error: ${e.message}`;
    $('backtest-status').className   = 'status-badge err';
    showToast(e.message, 'error');
  } finally {
    hideLoading();
  }
}

function renderBacktestChart(data) {
  destroyChart('backtest');
  const ctx = $('backtestChart').getContext('2d');

  State.charts.backtest = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [
        {
          label: 'Actual', data: data.actual,
          borderColor: CHART_COLORS.actual, borderWidth: 2.5,
          pointRadius: 0, fill: false, tension: 0.3,
        },
        {
          label: 'LSTM', data: data.predictions.LSTM,
          borderColor: CHART_COLORS.lstm, borderWidth: 1.5,
          pointRadius: 0, fill: false, tension: 0.3, borderDash: [4,3],
        },
        {
          label: 'Random Forest', data: data.predictions.RandomForest,
          borderColor: CHART_COLORS.rf, borderWidth: 1.5,
          pointRadius: 0, fill: false, tension: 0.3, borderDash: [4,3],
        },
        {
          label: 'Ensemble', data: data.predictions.Ensemble,
          borderColor: CHART_COLORS.ensemble, borderWidth: 2,
          pointRadius: 0, fill: false, tension: 0.3,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,.95)', borderColor: 'rgba(99,102,241,.4)', borderWidth: 1, padding: 12,
          callbacks: { label: ctx => ` ${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2)}` },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { maxTicksLimit: 8, font: { size: 11 } } },
        y: { grid: { color: 'rgba(99,102,241,.08)' }, ticks: { callback: v => `$${v}` } },
      },
    },
  });
}

function renderBacktestMetrics() {
  const info   = State.tickers[State.ticker];
  const metrics = info?.metrics || {};
  const grid   = $('backtest-metrics');
  grid.innerHTML = '';

  const modelLabels = { LSTM: '🧠 LSTM', RandomForest: '🌲 Random Forest', GradientBoosting: '⚡ Gradient Boosting', Ensemble: '🎯 Ensemble' };
  for (const [key, label] of Object.entries(modelLabels)) {
    const m = metrics[key] || {};
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-model">${label}</div>
      <div class="metric-stats">
        <div><div class="metric-stat-label">MAE</div><div class="metric-stat-val">${m.MAE ?? '—'}</div></div>
        <div><div class="metric-stat-label">RMSE</div><div class="metric-stat-val">${m.RMSE ?? '—'}</div></div>
        <div><div class="metric-stat-label">R²</div><div class="metric-stat-val">${m.R2 ?? '—'}</div></div>
      </div>
    `;
    grid.appendChild(card);
  }
}

// ── Train Trigger ─────────────────────────────────────────────────────────
async function triggerTrain(ticker) {
  const btn = $(`train-btn-${ticker}`);
  if (btn) { btn.textContent = '⏳ Training…'; btn.disabled = true; }
  try {
    const res = await apiFetch(`/api/train/${ticker}`, { method: 'POST' });
    showToast(`🚀 ${res.message}`, 'success');
  } catch (e) {
    showToast(`Training error: ${e.message}`, 'error');
    if (btn) { btn.textContent = '🚀 Train Model'; btn.disabled = false; }
  }
}

// Patch apiFetch to support POST
const _apiFetch = apiFetch;
window.apiFetch = async function(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
};

// ── Tab Navigation ────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  $(`tab-${tab}`).classList.add('active');
  $(`nav-${tab}`).classList.add('active');

  if (tab === 'predict') {
    if (State.prediction) renderPredictPage(State.prediction);
    else $('predict-cards').innerHTML = '<p style="color:var(--text-3);text-align:center;padding:40px;">Train a model first to see predictions.</p>';
  }
  if (tab === 'backtest') loadBacktest();
  if (tab === 'models')   renderModelsGrid(State.tickers);
}

// ── Range Buttons ──────────────────────────────────────────────────────────
document.querySelectorAll('.range-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    State.days = parseInt(btn.dataset.days);
    if (State.history) {
      renderPriceChart(State.history);
      renderRSIChart(State.history);
      renderMACDChart(State.history);
      renderVolumeChart(State.history);
    }
  });
});

// ── Nav Buttons ────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await loadTickers();
  await loadDashboardData();
})();
