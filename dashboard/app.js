/* ═══════════════════════════════════════════════════════════
   IBKR AI Trading Bot — Dashboard Frontend JS
   Fetches data from Flask API, renders charts and tables
   ═══════════════════════════════════════════════════════════ */

// ─── Chart.js Global Config ───
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 16;

// Color palette
const COLORS = {
    blue: '#3b82f6', cyan: '#06b6d4', green: '#10b981',
    red: '#ef4444', amber: '#f59e0b', purple: '#8b5cf6',
    pink: '#ec4899', slate: '#64748b',
    blueBg: 'rgba(59,130,246,0.15)', cyanBg: 'rgba(6,182,212,0.15)',
    greenBg: 'rgba(16,185,129,0.15)', redBg: 'rgba(239,68,68,0.15)',
    amberBg: 'rgba(245,158,11,0.15)', purpleBg: 'rgba(139,92,246,0.15)',
};

// ─── Utility Functions ───
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const setText = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
const setHTML = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };

function formatPnl(val) {
    if (val == null || isNaN(val)) return '--';
    const cls = val >= 0 ? 'pnl-positive' : 'pnl-negative';
    return `<span class="${cls}">$${val >= 0 ? '+' : ''}${val.toFixed(2)}</span>`;
}

function formatPct(val) {
    if (val == null || isNaN(val)) return '--';
    const cls = val >= 0 ? 'pnl-positive' : 'pnl-negative';
    return `<span class="${cls}">${val >= 0 ? '+' : ''}${val.toFixed(2)}%</span>`;
}

function formatTs(ts) {
    if (!ts) return '--';
    // Convert "20260909_202424" to "2026-09-09 20:24:24"
    if (ts.length === 15 && ts.includes('_')) {
        return `${ts.slice(0,4)}-${ts.slice(4,6)}-${ts.slice(6,8)} ${ts.slice(9,11)}:${ts.slice(11,13)}:${ts.slice(13,15)}`;
    }
    return ts;
}

async function fetchJSON(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error(`Fetch error: ${url}`, e);
        return null;
    }
}

// ─── Chart instance store (for cleanup) ───
const charts = {};
function destroyChart(key) {
    if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

// ═══════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    const tabs = $$('.nav-tab');
    const panels = $$('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            $(`#tab-${target}`).classList.add('active');

            // Load data for specific tabs on first activation
            if (target === 'chart') loadChartTab();
        });
    });

    // Start loading data
    loadOverview();
    loadDailyPlan();
    loadModelGate();
    loadDailyReview();
    loadModelData();
    loadBacktestList();
    loadPennyPicks();
    loadConfig();
    loadWatchlist();
    startClock();

    // Auto-refresh every 60s
    setInterval(() => {
        loadOverview();
        loadDailyPlan();
        loadModelGate();
        loadDailyReview();
        loadBacktestList();
    }, 60000);
});

// ═══════════════════════════════════════════════════════════
//  MARKET CLOCK
// ═══════════════════════════════════════════════════════════
function startClock() {
    function tick() {
        const now = new Date();
        const etOffset = -4; // EDT
        const utc = now.getTime() + now.getTimezoneOffset() * 60000;
        const et = new Date(utc + 3600000 * etOffset);
        const h = et.getHours(), m = et.getMinutes();
        const timeStr = et.toLocaleTimeString('en-US', { hour12: false });
        setText('clockTime', timeStr);

        const day = et.getDay();
        const isWeekday = day >= 1 && day <= 5;
        const marketOpen = h === 9 && m >= 30 || h > 9 && h < 16;
        const preMarket = h >= 4 && (h < 9 || (h === 9 && m < 30));
        const afterHours = h >= 16 && h < 20;

        let label = 'Closed';
        let isOpen = false;
        if (isWeekday) {
            if (marketOpen) { label = 'Market Open'; isOpen = true; }
            else if (preMarket) { label = 'Pre-Market'; }
            else if (afterHours) { label = 'After Hours'; }
        }

        setText('clockLabel', label);
        const dot = $('.clock-dot');
        if (dot) dot.classList.toggle('closed', !isOpen);

        setText('lastUpdate', `Last update: ${now.toLocaleTimeString()}`);
    }
    tick();
    setInterval(tick, 1000);
}

// ═══════════════════════════════════════════════════════════
//  OVERVIEW TAB
// ═══════════════════════════════════════════════════════════
async function loadOverview() {
    const data = await fetchJSON('/api/overview');
    if (!data) return;

    // KPI Cards
    setText('kpiCapital', `$${(data.capital || 0).toLocaleString()}`);
    setText('kpiAccuracy', `${((data.model?.accuracy || 0) * 100).toFixed(1)}%`);
    setText('kpiF1', (data.model?.f1 || 0).toFixed(3));
    setText('kpiPennies', data.penny_picks?.count || 0);

    // Backtest KPI
    const bt = data.backtest || {};
    const btReturn = bt.total_return || 0;
    const kpiBt = document.getElementById('kpiBacktest');
    if (kpiBt) {
        kpiBt.textContent = `${btReturn >= 0 ? '+' : ''}${btReturn.toFixed(1)}%`;
        kpiBt.style.color = btReturn >= 0 ? COLORS.green : COLORS.red;
    }

    // Health KPI
    const healthStatus = data.health?.overall_status || 'Unknown';
    const kpiH = document.getElementById('kpiHealth');
    if (kpiH) {
        const short = healthStatus.includes('POOR') ? 'POOR' :
                      healthStatus.includes('OK') ? 'OK' :
                      healthStatus.includes('GOOD') ? 'GOOD' : 'N/A';
        kpiH.textContent = short;
        kpiH.className = `kpi-value ${short === 'GOOD' ? 'health-good' : short === 'OK' ? 'health-warn' : 'health-poor'}`;
    }

    // Latest backtest details
    if (bt.timestamp) {
        setText('latestBtTimestamp', formatTs(bt.timestamp));
        setText('lbTrades', bt.total_trades || 0);
        const wrEl = document.getElementById('lbWinRate');
        if (wrEl) wrEl.innerHTML = formatPct(bt.win_rate);
        const retEl = document.getElementById('lbReturn');
        if (retEl) retEl.innerHTML = formatPct(bt.total_return);
        const pnlEl = document.getElementById('lbPnl');
        if (pnlEl) pnlEl.innerHTML = formatPnl(bt.total_pnl);
        setText('lbSharpe', (bt.sharpe || 0).toFixed(2));
        const ddEl = document.getElementById('lbDrawdown');
        if (ddEl) ddEl.innerHTML = formatPct(bt.max_drawdown);
    }

    // Model summary
    const m = data.model || {};
    setText('modelEnsemble', m.ensemble || '--');
    setText('msEnsemble', m.ensemble || '--');
    setText('msFeatures', m.n_features || '--');
    setText('msSamples', (m.n_samples || 0).toLocaleString());
    setText('msBuyWR', m.buy_win_rate ? `${(m.buy_win_rate * 100).toFixed(1)}%` : '--');
    setText('msTrained', m.trained_at ? m.trained_at.slice(0, 16).replace('T', ' ') : '--');
}

// ═══════════════════════════════════════════════════════════
//  MODEL / TRAINING TAB
// ═══════════════════════════════════════════════════════════
async function loadModelData() {
    const data = await fetchJSON('/api/model');
    if (!data || data.error) return;

    const tm = data.training_metrics || {};
    const wf = tm.walk_forward || {};

    // Training metrics cards
    const metricsHTML = [
        { label: 'Train Accuracy', value: `${((tm.train_accuracy || 0) * 100).toFixed(1)}%`, color: COLORS.blue },
        { label: 'Train F1', value: (tm.train_f1 || 0).toFixed(4), color: COLORS.cyan },
        { label: 'WF Accuracy', value: `${((wf.overall_accuracy || 0) * 100).toFixed(1)}%`, color: COLORS.purple },
        { label: 'WF F1', value: (wf.overall_f1 || 0).toFixed(4), color: COLORS.green },
        { label: 'Buy Win Rate', value: `${((wf.buy_signal_win_rate || 0) * 100).toFixed(1)}%`, color: COLORS.amber },
        { label: 'Samples', value: (tm.n_samples || 0).toLocaleString(), color: COLORS.slate },
        { label: 'Features (orig)', value: tm.n_features_original || '--', color: COLORS.slate },
        { label: 'Features (selected)', value: tm.n_features_selected || '--', color: COLORS.cyan },
        { label: 'Ensemble', value: data.ensemble_method || '--', color: COLORS.purple },
        { label: 'Trained At', value: (data.saved_at || '').slice(0, 16).replace('T', ' ') || '--', color: COLORS.slate },
    ].map(m => `
        <div class="training-metric-card">
            <span class="tm-label">${m.label}</span>
            <span class="tm-value" style="color: ${m.color}">${m.value}</span>
        </div>
    `).join('');
    setHTML('trainingMetrics', metricsHTML);

    // Walk-forward fold chart
    if (wf.fold_metrics && wf.fold_metrics.length > 0) {
        destroyChart('walkForward');
        const ctx = document.getElementById('walkForwardChart');
        charts.walkForward = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: wf.fold_metrics.map(f => `Fold ${f.fold}`),
                datasets: [
                    {
                        label: 'Accuracy',
                        data: wf.fold_metrics.map(f => (f.accuracy * 100).toFixed(1)),
                        backgroundColor: COLORS.blueBg,
                        borderColor: COLORS.blue,
                        borderWidth: 2,
                        borderRadius: 4,
                    },
                    {
                        label: 'F1 (weighted)',
                        data: wf.fold_metrics.map(f => (f.f1_weighted * 100).toFixed(1)),
                        backgroundColor: COLORS.purpleBg,
                        borderColor: COLORS.purple,
                        borderWidth: 2,
                        borderRadius: 4,
                    },
                    {
                        label: 'Buy Precision',
                        data: wf.fold_metrics.map(f => ((f.buy_precision || 0) * 100).toFixed(1)),
                        backgroundColor: COLORS.greenBg,
                        borderColor: COLORS.green,
                        borderWidth: 2,
                        borderRadius: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: v => v + '%' },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                    },
                    x: { grid: { display: false } },
                },
                plugins: {
                    tooltip: {
                        backgroundColor: 'rgba(17,24,39,0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: ctx => `${ctx.dataset.label}: ${ctx.raw}%`,
                            afterBody: (items) => {
                                const idx = items[0]?.dataIndex;
                                if (idx != null && wf.fold_metrics[idx]) {
                                    const f = wf.fold_metrics[idx];
                                    return [`Train: ${f.train_size}  Test: ${f.test_size}`];
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    // Label distribution pie chart
    if (tm.label_distribution) {
        destroyChart('labelDist');
        const ld = tm.label_distribution;
        const ctx = document.getElementById('labelDistChart');
        const labels = Object.keys(ld).map(k => k === '-1' ? 'SELL' : k === '0' ? 'HOLD' : 'BUY');
        const values = Object.values(ld);
        const colors = [COLORS.red, COLORS.amber, COLORS.green];
        const bgColors = [COLORS.redBg, COLORS.amberBg, COLORS.greenBg];

        charts.labelDist = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: bgColors,
                    borderColor: colors,
                    borderWidth: 2,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        backgroundColor: 'rgba(17,24,39,0.95)',
                        callbacks: {
                            label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString()} (${((ctx.raw / values.reduce((a,b) => a+b, 0)) * 100).toFixed(1)}%)`
                        }
                    }
                }
            }
        });
    }

    // Selected features
    const selected = data.selected_features || [];
    setText('featureCount', `${selected.length} / ${(data.feature_names || []).length}`);
    setHTML('selectedFeaturesList',
        selected.map(f => `<span class="feature-tag">${f}</span>`).join('')
    );

    // Classification report table
    if (wf.classification_report) {
        const cr = wf.classification_report;
        const classMap = { '0': 'HOLD', '1': 'SELL', '2': 'BUY' };
        let rows = '';
        for (const [key, val] of Object.entries(cr)) {
            if (typeof val === 'object' && val['f1-score'] != null) {
                const label = classMap[key] || key;
                rows += `<tr>
                    <td><strong>${label}</strong></td>
                    <td>${(val.precision * 100).toFixed(1)}%</td>
                    <td>${(val.recall * 100).toFixed(1)}%</td>
                    <td>${(val['f1-score'] * 100).toFixed(1)}%</td>
                    <td>${val.support ? val.support.toLocaleString() : '--'}</td>
                </tr>`;
            }
        }
        // Add overall accuracy row
        if (cr.accuracy != null) {
            rows += `<tr style="border-top: 2px solid rgba(255,255,255,0.1);">
                <td><strong>Overall</strong></td>
                <td colspan="2">Accuracy</td>
                <td>${(cr.accuracy * 100).toFixed(1)}%</td>
                <td>${cr['weighted avg']?.support?.toLocaleString() || '--'}</td>
            </tr>`;
        }
        setHTML('classReportBody', rows);
    }

    // Feature importance chart (from selected features — we'll use feature names as proxy)
    // Since we don't have importance values in the meta, we show the selected features with equal bars
    // The actual importance would need an API endpoint that loads the model
    if (selected.length > 0) {
        destroyChart('feature');
        const ctx = document.getElementById('featureChart');
        const top15 = selected.slice(0, 15);
        charts.feature = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: top15,
                datasets: [{
                    label: 'Selected Feature',
                    data: top15.map((_, i) => ((15 - i) / 15 * 100).toFixed(0)),
                    backgroundColor: top15.map((_, i) => {
                        const hue = 200 + i * 10;
                        return `hsla(${hue}, 70%, 55%, 0.3)`;
                    }),
                    borderColor: top15.map((_, i) => {
                        const hue = 200 + i * 10;
                        return `hsla(${hue}, 70%, 55%, 1)`;
                    }),
                    borderWidth: 1.5,
                    borderRadius: 3,
                }],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false,
                        grid: { display: false },
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            font: { family: "'JetBrains Mono', monospace", size: 11 },
                        }
                    },
                },
                plugins: {
                    legend: { display: false },
                }
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════
//  BACKTEST TAB
// ═══════════════════════════════════════════════════════════
let backtestRuns = [];

async function loadBacktestList() {
    const runs = await fetchJSON('/api/backtest/list');
    if (!runs || !runs.length) return;

    backtestRuns = runs;
    const select = document.getElementById('backtestSelector');
    select.innerHTML = runs.map(r =>
        `<option value="${r.timestamp}">${formatTs(r.timestamp)} | ${r.total_trades} trades | ${r.total_return_pct >= 0 ? '+' : ''}${r.total_return_pct}% | $${r.total_pnl.toFixed(2)}</option>`
    ).join('');

    // Load the first (latest)
    select.addEventListener('change', () => loadBacktestDetail(select.value));
    loadBacktestDetail(runs[0].timestamp);
}

async function loadBacktestDetail(timestamp) {
    if (!timestamp) return;
    const data = await fetchJSON(`/api/backtest/${timestamp}`);
    if (!data) return;

    const m = data.metrics || {};

    // KPI Row
    const kpis = [
        { label: 'Total Trades', value: m.total_trades || 0 },
        { label: 'Long / Short', value: `${m.long_trades || 0} / ${m.short_trades || 0}` },
        { label: 'Win Rate', value: `${m.win_rate_pct || 0}%`, cls: (m.win_rate_pct || 0) >= 50 ? 'pnl-positive' : 'pnl-negative' },
        { label: 'Total Return', value: `${m.total_return_pct >= 0 ? '+' : ''}${m.total_return_pct || 0}%`, cls: (m.total_return_pct || 0) >= 0 ? 'pnl-positive' : 'pnl-negative' },
        { label: 'Total PnL', value: `$${(m.total_pnl || 0).toFixed(2)}`, cls: (m.total_pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative' },
        { label: 'Profit Factor', value: (m.profit_factor || 0).toFixed(2) },
        { label: 'Sharpe Ratio', value: (m.sharpe_ratio || 0).toFixed(2) },
        { label: 'Max Drawdown', value: `${m.max_drawdown_pct || 0}%`, cls: 'pnl-negative' },
        { label: 'Avg Win', value: `$${(m.avg_win || 0).toFixed(2)}`, cls: 'pnl-positive' },
        { label: 'Avg Loss', value: `$${(m.avg_loss || 0).toFixed(2)}`, cls: 'pnl-negative' },
    ];

    setHTML('btKpiRow', kpis.map(k => `
        <div class="bt-kpi">
            <span class="bk-label">${k.label}</span>
            <span class="bk-value ${k.cls || ''}">${k.value}</span>
        </div>
    `).join(''));

    // Equity curve
    if (data.equity && data.equity.length > 0) {
        destroyChart('equity');
        const ctx = document.getElementById('equityChart');
        const labels = data.equity.map(e => {
            const d = e.Date || '';
            return d.length > 16 ? d.slice(5, 16) : d;
        });
        const values = data.equity.map(e => e.equity);
        const initial = values[0] || 1000;

        charts.equity = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Equity',
                    data: values,
                    borderColor: COLORS.blue,
                    backgroundColor: COLORS.blueBg,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 8,
                    borderWidth: 2,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        ticks: { callback: v => `$${v.toFixed(0)}` },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                    },
                    x: {
                        ticks: { maxTicksLimit: 15 },
                        grid: { display: false },
                    },
                },
                plugins: {
                    tooltip: {
                        backgroundColor: 'rgba(17,24,39,0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: ctx => `Equity: $${ctx.raw.toFixed(2)} (${((ctx.raw / initial - 1) * 100).toFixed(2)}%)`,
                        }
                    },
                    legend: { display: false },
                },
                interaction: { intersect: false, mode: 'index' },
            }
        });
    }

    // Trade log table
    if (data.trades && data.trades.length > 0) {
        const rows = data.trades.map(t => {
            const pnlCls = (t.pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative';
            return `<tr>
                <td>${(t.entry_date || '').slice(5, 16)}</td>
                <td>${(t.exit_date || '').slice(5, 16)}</td>
                <td>${t.type || '--'}</td>
                <td>$${(t.entry_price || 0).toFixed(4)}</td>
                <td>$${(t.exit_price || 0).toFixed(4)}</td>
                <td>${t.shares || 0}</td>
                <td class="${pnlCls}">$${(t.pnl || 0).toFixed(2)}</td>
                <td class="${pnlCls}">${(t.pnl_pct || 0).toFixed(2)}%</td>
                <td>${t.exit_reason || '--'}</td>
            </tr>`;
        }).join('');
        setHTML('tradeLogBody', rows);
    } else {
        setHTML('tradeLogBody', '<tr><td colspan="9" class="text-muted" style="text-align:center; padding:30px;">No trades in this backtest</td></tr>');
    }

    // Exit reasons pie chart
    if (m.exit_reasons) {
        destroyChart('exitReasons');
        const ctx = document.getElementById('exitReasonsChart');
        const labels = Object.keys(m.exit_reasons);
        const values = Object.values(m.exit_reasons);
        const palette = [COLORS.red, COLORS.amber, COLORS.green, COLORS.blue, COLORS.purple, COLORS.cyan];
        const bgPalette = [COLORS.redBg, COLORS.amberBg, COLORS.greenBg, COLORS.blueBg, COLORS.purpleBg, COLORS.cyanBg];

        charts.exitReasons = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: bgPalette.slice(0, labels.length),
                    borderColor: palette.slice(0, labels.length),
                    borderWidth: 2,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '50%',
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: { backgroundColor: 'rgba(17,24,39,0.95)' },
                }
            }
        });
    }

    // Weekly PnL
    if (m.weekly_pnl && m.weekly_pnl.weeks) {
        let html = m.weekly_pnl.weeks.map(w => {
            const pnlCls = w.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            return `<div class="weekly-pnl-item">
                <span class="text-mono">Week ${w.week} (${w.year})</span>
                <span>${w.trades} trades</span>
                <span class="${pnlCls} text-mono font-bold">$${w.pnl.toFixed(2)}</span>
            </div>`;
        }).join('');
        html += `<div class="weekly-pnl-item" style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
            <span class="text-mono"><strong>Summary</strong></span>
            <span>Avg: $${(m.weekly_pnl.avg_weekly_pnl || 0).toFixed(2)}</span>
            <span class="text-mono">Target hit: ${m.weekly_pnl.target_hit_count || 0}/${m.weekly_pnl.total_weeks || 0}</span>
        </div>`;
        setHTML('weeklyPnlContainer', html);
    }
}

// ═══════════════════════════════════════════════════════════
//  PENNY PICKS TAB
// ═══════════════════════════════════════════════════════════
let pennyPicks = [];

async function loadPennyPicks() {
    const data = await fetchJSON('/api/penny-picks');
    if (!data) return;

    pennyPicks = data.picks || [];
    setText('pennyDate', data.date || '--');
    setText('pennyPickDate', data.date || '--');

    // Quick list (overview tab)
    if (pennyPicks.length > 0) {
        setHTML('pennyQuickList', pennyPicks.slice(0, 5).map(p => {
            const scoreCls = p.total_score >= 50 ? 'score-high' : p.total_score >= 30 ? 'score-med' : 'score-low';
            return `<div class="penny-quick-item">
                <span class="penny-symbol">${p.symbol}</span>
                <span class="penny-price">$${p.price.toFixed(2)}</span>
                <span class="penny-score ${scoreCls}">${p.total_score.toFixed(1)}</span>
            </div>`;
        }).join(''));
    } else {
        setHTML('pennyQuickList', '<p class="text-muted">No penny picks available. Run: python main.py scan-pennies</p>');
    }

    // Full table
    if (pennyPicks.length > 0) {
        const rows = pennyPicks.map(p => {
            const retCls = (p.return_5d_pct || 0) >= 0 ? 'pnl-positive' : 'pnl-negative';
            return `<tr>
                <td>${p.rank || '--'}</td>
                <td><strong class="text-cyan">${p.symbol}</strong></td>
                <td>${p.name || '--'}</td>
                <td class="text-mono">$${p.price.toFixed(2)}</td>
                <td class="text-mono font-bold">${p.total_score.toFixed(1)}</td>
                <td class="text-mono">${(p.volume_surge_score || 0).toFixed(1)}</td>
                <td class="text-mono">${(p.momentum_score || 0).toFixed(1)}</td>
                <td class="text-mono">${(p.technical_score || 0).toFixed(1)}</td>
                <td class="text-mono">${(p.sentiment_score || 0).toFixed(1)}</td>
                <td class="${retCls} text-mono">${(p.return_5d_pct || 0).toFixed(1)}%</td>
                <td>${p.key_catalyst || '--'}</td>
                <td class="text-mono text-red">$${(p.suggested_stop_loss || 0).toFixed(2)}</td>
                <td class="text-mono text-green">$${(p.suggested_target || 0).toFixed(2)}</td>
            </tr>`;
        }).join('');
        setHTML('pennyTableBody', rows);

        // Populate radar selector
        const select = document.getElementById('pennyRadarSelect');
        select.innerHTML = '<option value="">Select stock...</option>' +
            pennyPicks.map(p => `<option value="${p.symbol}">${p.symbol} — $${p.price.toFixed(2)}</option>`).join('');
        select.addEventListener('change', () => renderPennyRadar(select.value));

        // Sector distribution
        renderSectorChart();

        // Auto-select first for radar
        if (pennyPicks.length > 0) {
            select.value = pennyPicks[0].symbol;
            renderPennyRadar(pennyPicks[0].symbol);
        }
    }
}

function renderPennyRadar(symbol) {
    const pick = pennyPicks.find(p => p.symbol === symbol);
    if (!pick) return;

    destroyChart('pennyRadar');
    const ctx = document.getElementById('pennyRadarChart');

    charts.pennyRadar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Volume Surge', 'Momentum', 'Technical', 'Sentiment', 'Rel. Strength'],
            datasets: [{
                label: pick.symbol,
                data: [
                    pick.volume_surge_score || 0,
                    pick.momentum_score || 0,
                    pick.technical_score || 0,
                    pick.sentiment_score || 0,
                    pick.relative_strength_score || 0,
                ],
                backgroundColor: COLORS.blueBg,
                borderColor: COLORS.blue,
                borderWidth: 2,
                pointBackgroundColor: COLORS.blue,
                pointRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 25,
                        color: 'rgba(255,255,255,0.3)',
                        backdropColor: 'transparent',
                    },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#94a3b8', font: { size: 11 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'rgba(17,24,39,0.95)' },
            }
        }
    });
}

function renderSectorChart() {
    const sectors = {};
    pennyPicks.forEach(p => {
        const s = p.sector || 'Unknown';
        sectors[s] = (sectors[s] || 0) + 1;
    });

    destroyChart('sector');
    const ctx = document.getElementById('sectorChart');
    const labels = Object.keys(sectors);
    const values = Object.values(sectors);
    const palette = [COLORS.blue, COLORS.green, COLORS.purple, COLORS.amber, COLORS.cyan, COLORS.red, COLORS.pink];
    const bgPalette = [COLORS.blueBg, COLORS.greenBg, COLORS.purpleBg, COLORS.amberBg, COLORS.cyanBg, COLORS.redBg, 'rgba(236,72,153,0.15)'];

    charts.sector = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: bgPalette.slice(0, labels.length),
                borderColor: palette.slice(0, labels.length),
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '50%',
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 11 } } },
                tooltip: { backgroundColor: 'rgba(17,24,39,0.95)' },
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════
//  CHART TAB (TradingView Lightweight Charts)
// ═══════════════════════════════════════════════════════════
let tvChart = null;
let tvVolume = null;

async function loadWatchlist() {
    const data = await fetchJSON('/api/watchlist');
    if (!data) return;

    const select = document.getElementById('chartSymbolSelect');
    const allSymbols = [...new Set([...(data.watchlist || []), ...(data.available_symbols || [])])].sort();
    select.innerHTML = allSymbols.map(s => `<option value="${s}">${s}</option>`).join('');

    // Load chart button
    document.getElementById('loadChartBtn').addEventListener('click', loadChartTab);
}

async function loadChartTab() {
    const symbol = document.getElementById('chartSymbolSelect').value;
    const interval = document.getElementById('chartIntervalSelect').value;
    if (!symbol) return;

    setText('chartTitle', `🕯️ ${symbol} — ${interval} Chart`);

    const data = await fetchJSON(`/api/price/${symbol}?interval=${interval}`);
    if (!data || !data.data || data.data.length === 0) {
        setHTML('tradingChart', '<div class="loading">No data available for this symbol/interval</div>');
        return;
    }

    // Clear existing chart
    const chartContainer = document.getElementById('tradingChart');
    const volumeContainer = document.getElementById('volumeChart');
    chartContainer.innerHTML = '';
    volumeContainer.innerHTML = '';

    // Parse candlestick data
    const candleData = data.data.map(bar => {
        let time;
        const d = bar.date;
        // Parse various date formats
        if (d.includes('T') || d.includes(' ')) {
            const dt = new Date(d);
            time = Math.floor(dt.getTime() / 1000);
        } else {
            time = Math.floor(new Date(d).getTime() / 1000);
        }
        return {
            time,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
        };
    }).filter(d => !isNaN(d.time)).sort((a, b) => a.time - b.time);

    const volumeData = data.data.map(bar => {
        let time;
        const d = bar.date;
        if (d.includes('T') || d.includes(' ')) {
            const dt = new Date(d);
            time = Math.floor(dt.getTime() / 1000);
        } else {
            time = Math.floor(new Date(d).getTime() / 1000);
        }
        return {
            time,
            value: bar.volume,
            color: bar.close >= bar.open ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)',
        };
    }).filter(d => !isNaN(d.time)).sort((a, b) => a.time - b.time);

    if (candleData.length === 0) {
        setHTML('tradingChart', '<div class="loading">Could not parse chart data</div>');
        return;
    }

    // Create price chart
    tvChart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: 500,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#94a3b8',
            fontFamily: "'Inter', sans-serif",
        },
        grid: {
            vertLines: { color: 'rgba(255,255,255,0.03)' },
            horzLines: { color: 'rgba(255,255,255,0.03)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: 'rgba(59,130,246,0.3)', width: 1, style: 2 },
            horzLine: { color: 'rgba(59,130,246,0.3)', width: 1, style: 2 },
        },
        rightPriceScale: {
            borderColor: 'rgba(255,255,255,0.06)',
        },
        timeScale: {
            borderColor: 'rgba(255,255,255,0.06)',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    const candleSeries = tvChart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });
    candleSeries.setData(candleData);

    // Create volume chart
    tvVolume = LightweightCharts.createChart(volumeContainer, {
        width: volumeContainer.clientWidth,
        height: 150,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#94a3b8',
            fontFamily: "'Inter', sans-serif",
        },
        grid: {
            vertLines: { color: 'rgba(255,255,255,0.03)' },
            horzLines: { color: 'rgba(255,255,255,0.03)' },
        },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
        timeScale: {
            borderColor: 'rgba(255,255,255,0.06)',
            timeVisible: true,
            visible: false,
        },
    });

    const volumeSeries = tvVolume.addHistogramSeries({
        priceFormat: { type: 'volume' },
    });
    volumeSeries.setData(volumeData);

    // Sync time scales
    tvChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) tvVolume.timeScale().setVisibleLogicalRange(range);
    });

    tvChart.timeScale().fitContent();

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
        tvChart.applyOptions({ width: chartContainer.clientWidth });
        tvVolume.applyOptions({ width: volumeContainer.clientWidth });
    });
    resizeObserver.observe(chartContainer);
}

// ═══════════════════════════════════════════════════════════
//  CONFIG TAB
// ═══════════════════════════════════════════════════════════
async function loadConfig() {
    const cfg = await fetchJSON('/api/config');
    if (!cfg) return;

    // IBKR
    const ibkr = cfg.ibkr || {};
    const isPaper = [7497, 4002].includes(ibkr.port);
    setHTML('cfgIbkr', statRows({
        'Host': `${ibkr.host}:${ibkr.port}`,
        'Mode': isPaper ? '📝 Paper' : '💰 Live',
        'Client ID': ibkr.client_id,
        'Account Type': ibkr.account_type || 'cash',
    }));

    // Strategy
    const strat = cfg.strategy || {};
    setHTML('cfgStrategy', statRows({
        'Timeframe': strat.timeframe,
        'Lookback': `${strat.lookback_years}y`,
        'Confidence': strat.confidence_threshold,
        'Label Mode': strat.label_mode,
        'Label Horizon': `${strat.label_horizon_days}d`,
        'Label Threshold': `${strat.label_threshold_pct}%`,
        'ATR Multiplier': strat.label_atr_multiplier,
        'Model Type': strat.model_type,
        'Retrain Interval': `${strat.retrain_interval_days}d`,
    }));

    // Model
    const model = cfg.model || {};
    setHTML('cfgModel', statRows({
        'Ensemble': model.ensemble_method,
        'Base Models': (model.base_models || []).join(', '),
        'Feature Selection': model.feature_selection ? '✅ ON' : '❌ OFF',
        'Max Features': model.max_features,
        'Purge Gap': `${model.purge_gap_bars} bars`,
        'Embargo': `${model.embargo_bars} bars`,
        'Min Prob Gap': model.min_probability_gap,
        'Sample Weights': model.use_sample_weights ? '✅' : '❌',
    }));

    // Risk
    const risk = cfg.risk || {};
    setHTML('cfgRisk', statRows({
        'Risk/Trade': `${risk.max_risk_per_trade_pct}%`,
        'Max Position': `${risk.max_position_pct}%`,
        'Max Daily Loss': `${risk.max_daily_loss_pct}%`,
        'Max Positions': risk.max_open_positions,
        'Stop Loss ATR': `${risk.stop_loss_atr_mult}x`,
        'Take Profit ATR': `${risk.take_profit_atr_mult}x`,
        'Trailing Stop': `${risk.trailing_stop_pct}%`,
        'Time Stop': `${risk.time_stop_bars} bars`,
        'Settlement': `T+${risk.settlement_days}`,
    }));

    // Targets
    const targets = cfg.targets || {};
    setHTML('cfgTargets', statRows({
        'Daily Target': `$${targets.daily_profit_target}`,
        'Scale Down At': `${targets.scale_down_at_pct}%`,
        'Stop Trading At': `${targets.stop_trading_at_pct}%`,
    }));

    // Regime
    const regime = cfg.regime || {};
    setHTML('cfgRegime', statRows({
        'Enabled': regime.enabled ? '✅ ON' : '❌ OFF',
        'Vol Lookback': `${regime.volatility_lookback} periods`,
        'Trend ADX Min': regime.trend_strength_min_adx,
        'High Vol %ile': regime.high_vol_threshold,
        'Low Vol %ile': regime.low_vol_threshold,
    }));

    // Penny Scanner
    const penny = cfg.penny_scanner || {};
    setHTML('cfgPenny', statRows({
        'Enabled': penny.enabled ? '✅ ON' : '❌ OFF',
        'Price Range': `$${penny.min_price} – $${penny.max_price}`,
        'Min Volume': (penny.min_avg_volume || 0).toLocaleString(),
        'Top N': penny.top_n_picks,
        'Exchanges': (penny.exchanges || []).join(', '),
    }));

    // Watchlist
    const symbols = cfg.watchlist?.symbols || [];
    setHTML('cfgWatchlist', symbols.map(s => `<span class="chip">${s}</span>`).join(''));
}

function statRows(obj) {
    return Object.entries(obj).map(([k, v]) =>
        `<div class="stat-row"><span>${k}</span><span>${v}</span></div>`
    ).join('');
}

// ═══════════════════════════════════════════════════════════
//  DAILY TRADE PLAN & MODEL APPROVAL GATE
// ═══════════════════════════════════════════════════════════

async function loadModelGate() {
    const data = await fetchJSON('/api/model-gate');
    if (!data) return;

    const champ = data.champion || {};
    const ver = champ.version || 'v1.0';
    setText('gateChampionVer', ver);

    const isApproved = champ.status === 'active' || champ.status === 'approved';
    const statusEl = document.getElementById('gateStatusText');
    if (statusEl) {
        statusEl.textContent = isApproved ? 'APPROVED' : (champ.status || 'PENDING').toUpperCase();
        statusEl.className = `bk-value ${isApproved ? 'text-green' : 'text-amber'}`;
    }

    const badgeEl = document.getElementById('gateApprovalBadge');
    if (badgeEl) {
        badgeEl.textContent = isApproved ? `Active: ${ver}` : 'Pending Approval';
        badgeEl.className = `card-badge ${isApproved ? 'score-high' : 'score-med'}`;
    }

    const trials = champ.total_cumulative_trials || 0;
    setText('gateCumulativeTrials', trials > 0 ? `${trials} runs` : 'Initial');

    const wfAcc = champ.walk_forward_accuracy != null ? (champ.walk_forward_accuracy * 100).toFixed(1) + '%' : '--%';
    setText('gateWfAcc', wfAcc);

    const wfF1 = champ.walk_forward_f1 != null ? champ.walk_forward_f1.toFixed(3) : '--';
    setText('gateWfF1', wfF1);

    const minAcc = (data.min_walk_forward_accuracy || 0.50) * 100;
    const minF1 = (data.min_walk_forward_f1 || 0.35).toFixed(2);
    setText('gateThresholds', `Acc: ${minAcc.toFixed(0)}% | F1: ${minF1}`);

    const lastUpdated = data.last_updated ? formatTs(data.last_updated.slice(0, 19).replace('T', ' ')) : '--';
    setText('gateEvaluatedAt', `Checked: ${lastUpdated}`);

    if (data.history && data.history.length > 0) {
        const lastEval = data.history[data.history.length - 1];
        const reasonText = lastEval.reason || (lastEval.approved ? 'Challenger exceeded benchmarks and promoted.' : 'Challenger did not beat champion.');
        setText('gateChallengerReason', `Last evaluation (${lastEval.challenger_version || 'v' + data.history.length}): ${reasonText}`);
    }
}

async function loadDailyPlan() {
    const data = await fetchJSON('/api/daily-plan');
    if (!data || !data.plan || data.plan.length === 0) {
        setText('dailyPlanDate', 'Date: Today (No Active Plan)');
        setText('dailyPlanBadge', 'Pending Scan');
        return;
    }

    setText('dailyPlanDate', `Date: ${data.date || 'Today'}`);
    const badge = document.getElementById('dailyPlanBadge');
    if (badge) {
        badge.textContent = `${data.plan.length} Setups Ready`;
        badge.className = 'card-badge score-high';
    }

    const rows = data.plan.map(p => {
        const price = (p.current_price || 0).toFixed(2);
        const entry = (p.entry_trigger || 0).toFixed(2);
        const stop = (p.stop_loss || 0).toFixed(2);
        const be = (p.target_1_breakeven || 0).toFixed(2);
        const tgt2 = (p.target_2_profit || 0).toFixed(2);
        const shares = p.position_shares || 0;
        const maxRisk = (p.max_risk_dollars || 0).toFixed(2);
        const notes = p.strategy_notes || 'Breakout with volume confirmation';

        return `<tr>
            <td class="font-bold text-cyan">${p.symbol}</td>
            <td>$${price}</td>
            <td class="text-green font-bold">$${entry}</td>
            <td class="text-red">$${stop}</td>
            <td class="text-amber">$${be}</td>
            <td class="text-green">$${tgt2}</td>
            <td>${shares} shs</td>
            <td class="text-red font-bold">-$${maxRisk}</td>
            <td class="text-muted" style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: normal;">${notes}</td>
        </tr>`;
    }).join('');

    setHTML('dailyPlanBody', rows);
}

async function loadDailyReview() {
    const data = await fetchJSON('/api/daily-review');
    if (!data || !data.audit || Object.keys(data.audit).length === 0) {
        setText('reviewTrades', '0');
        setText('reviewWinRate', '--%');
        setText('reviewPnl', '$0.00');
        setText('reviewBeSaves', '0');
        return;
    }

    const a = data.audit;
    setText('reviewTrades', `${a.trades_taken || 0} / ${a.planned_trades || 0}`);
    setText('reviewWinRate', a.win_rate_pct != null ? `${a.win_rate_pct.toFixed(1)}%` : '--%');
    
    const pnl = a.realized_pnl || 0;
    const pnlEl = document.getElementById('reviewPnl');
    if (pnlEl) {
        pnlEl.textContent = `$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`;
        pnlEl.className = `metric-value text-mono ${pnl >= 0 ? 'text-green' : 'text-red'}`;
    }

    setText('reviewBeSaves', `${a.breakeven_saves || 0}`);

    const badge = document.getElementById('dailyReviewBadge');
    if (badge) {
        badge.textContent = `Audited: ${data.date || 'Today'}`;
        badge.className = 'card-badge score-high';
    }

    if (a.learnings) {
        setText('reviewNotes', a.learnings);
    }
}

