const API_URL = "/api";

// State
let currentFilters = { clinic: '', doctor: '', condition: '' };
let currentPage = 1;
const pageSize = 50;

// Session ID for conversation memory
const sessionId = crypto.randomUUID();

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    loadFilters();
    loadDashboard();

    // Filter Event Listeners (reset to page 1 on filter change)
    ['clinic', 'doctor', 'condition'].forEach(id => {
        document.getElementById(`filter-${id}`).addEventListener('change', (e) => {
            currentFilters[id] = e.target.value;
            currentPage = 1;
            loadDashboard();
        });
    });

    // Query Form Listener
    document.getElementById('query-form').addEventListener('submit', handleQuerySubmit);
});

// --- TABS ---
function switchTab(tab) {
    const dashboardSection = document.getElementById('dashboard-section');
    const querySection = document.getElementById('query-section');
    const btnDash = document.getElementById('btn-dashboard');
    const btnQuery = document.getElementById('btn-query');

    if (tab === 'dashboard') {
        dashboardSection.classList.remove('hidden');
        querySection.classList.add('hidden');

        btnDash.className = "px-4 py-2 rounded-md text-sm font-medium bg-indigo-600 text-white transition-colors shadow-sm";
        btnQuery.className = "px-4 py-2 rounded-md text-sm font-medium text-slate-300 hover:text-white transition-colors";
    } else {
        dashboardSection.classList.add('hidden');
        querySection.classList.remove('hidden');

        btnDash.className = "px-4 py-2 rounded-md text-sm font-medium text-slate-300 hover:text-white transition-colors";
        btnQuery.className = "px-4 py-2 rounded-md text-sm font-medium bg-indigo-600 text-white transition-colors shadow-sm";
    }
}

// --- DASHBOARD LOGIC ---
async function loadFilters() {
    try {
        const res = await fetch(`${API_URL}/filters`);
        const data = await res.json();

        populateSelect('filter-clinic', data.clinics);
        populateSelect('filter-doctor', data.doctors);
        populateSelect('filter-condition', data.conditions);
    } catch (e) {
        console.error("Backend offline?", e);
    }
}

function populateSelect(id, options) {
    const select = document.getElementById(id);
    options.forEach(opt => {
        const el = document.createElement('option');
        el.value = opt;
        el.textContent = opt;
        select.appendChild(el);
    });
}

function resetFilters() {
    document.getElementById('filter-clinic').value = "";
    document.getElementById('filter-doctor').value = "";
    document.getElementById('filter-condition').value = "";
    currentFilters = { clinic: '', doctor: '', condition: '' };
    currentPage = 1;
    loadDashboard();
}

// --- PAGINATION ---
function updatePagination(pagination) {
    const info = document.getElementById('page-info');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    if (!info) return;

    info.textContent = `Page ${pagination.page} of ${pagination.total_pages} (${pagination.total_rows} records)`;
    btnPrev.disabled = pagination.page <= 1;
    btnNext.disabled = pagination.page >= pagination.total_pages;
}

function changePage(delta) {
    currentPage += delta;
    if (currentPage < 1) currentPage = 1;
    loadDashboard();
}

async function loadDashboard() {
    const tbody = document.getElementById('patient-table-body');
    tbody.innerHTML = '<tr><td colspan="5" class="text-center py-10 text-slate-400 font-medium animate-pulse">Loading data...</td></tr>';

    try {
        const res = await fetch(`${API_URL}/dashboard?page=${currentPage}&page_size=${pageSize}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentFilters)
        });
        const data = await res.json();

        const rows = data.rows || data;
        const pagination = data.pagination || null;

        renderTable(rows);
        updateMetrics(rows);

        if (pagination) {
            updatePagination(pagination);
        }

    } catch (e) {
        console.error(e);
        tbody.innerHTML = '<tr><td colspan="5" class="text-center py-10 text-red-400">Error connecting to server. Ensure backend is running.</td></tr>';
    }
}

function renderTable(data) {
    const tbody = document.getElementById('patient-table-body');
    const recordCount = document.getElementById('record-count');

    recordCount.textContent = `${data.length} records`;
    tbody.innerHTML = '';

    if (data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center py-10 text-slate-400">No records found matching filters.</td></tr>';
        return;
    }

    const statusColors = {
        'Good': 'bg-blue-100 text-blue-800',
        'Refill Due': 'bg-green-100 text-green-800',
        'Renewal Needed': 'bg-yellow-100 text-yellow-800',
        'Non-Adherent': 'bg-red-100 text-red-800'
    };

    data.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.className = "border-b border-slate-50 hover:bg-slate-50 transition-colors group";

        const rowId = `row-${index}`;

        tr.innerHTML = `
            <td class="px-6 py-4">
                <div class="font-medium text-slate-900">${row.name}</div>
                <div class="text-xs text-slate-400">${row.clinic}</div>
            </td>
            <td class="px-6 py-4">
                <div class="font-medium text-slate-600">${row.medication}</div>
                <div class="text-xs text-slate-400 mt-0.5">Refills Left: ${row.refills_left}</div>
            </td>
            <td class="px-6 py-4">
                <span class="px-2.5 py-1 rounded-full text-xs font-semibold ${statusColors[row.status] || 'bg-slate-100'} border border-transparent">
                    ${row.status}
                </span>
            </td>
            <td class="px-6 py-4">
                <button class="bg-slate-100 text-slate-600 border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-medium cursor-default shadow-sm">
                    ${row.action}
                </button>
            </td>
            <td class="px-6 py-4 text-right">
                <button onclick="toggleDetails('${rowId}')" class="text-slate-400 hover:text-indigo-600 transition-colors p-2 rounded-full hover:bg-indigo-50">
                    <i class="fa-solid fa-chevron-down"></i>
                </button>
            </td>
        `;

        const detailTr = document.createElement('tr');
        detailTr.id = rowId;
        detailTr.className = "hidden bg-indigo-50/30 shadow-inner";
        detailTr.innerHTML = `
            <td colspan="5" class="px-6 py-4">
                <div class="flex gap-6 text-sm">
                    <div class="flex-1">
                        <h4 class="font-bold text-xs uppercase mb-2 text-indigo-900 flex items-center gap-2">
                            <i class="fa-solid fa-notes-medical"></i> Clinical Note
                        </h4>
                        <div class="bg-white p-4 rounded-lg border border-indigo-100 text-slate-600 italic shadow-sm relative">
                            <i class="fa-solid fa-quote-left text-indigo-100 absolute top-2 left-2 -z-0 text-3xl"></i>
                            <span class="relative z-10">"${row.note_snippet}"</span>
                        </div>
                    </div>
                    <div class="w-1/3 border-l border-indigo-100 pl-6 flex flex-col justify-center">
                        <div class="mb-3">
                            <h4 class="font-bold text-xs uppercase text-slate-400 mb-1">Prescriber</h4>
                            <p class="text-slate-800 font-medium">${row.doctor}</p>
                        </div>
                        <div>
                            <h4 class="font-bold text-xs uppercase text-slate-400 mb-1">Condition</h4>
                            <p class="text-slate-800 font-medium">${row.condition}</p>
                        </div>
                    </div>
                </div>
            </td>
        `;

        tbody.appendChild(tr);
        tbody.appendChild(detailTr);
    });
}

function toggleDetails(id) {
    const el = document.getElementById(id);
    const isHidden = el.classList.contains('hidden');

    document.querySelectorAll('tr[id^="row-"]').forEach(row => {
        if (!row.classList.contains('hidden')) row.classList.add('hidden');
    });

    if (isHidden) {
        el.classList.remove('hidden');
    }
}

function updateMetrics(data) {
    let risk = 0, due = 0, active = 0, lost = 0;

    data.forEach(r => {
        if (r.status === 'Non-Adherent') risk++;
        else if (r.status === 'Refill Due') due++;
        else if (r.status === 'Good') active++;
        else if (r.status === 'Renewal Needed') lost++;
    });

    const total = data.length || 1;
    const totalRisk = risk + lost;

    document.getElementById('total-rx').textContent = data.length;

    animateValue("count-risk", parseInt(document.getElementById("count-risk").innerText), totalRisk, 500);
    animateValue("count-due", parseInt(document.getElementById("count-due").innerText), due, 500);
    animateValue("count-active", parseInt(document.getElementById("count-active").innerText), active, 500);

    const riskPct = (totalRisk / total) * 100;
    const duePct = (due / total) * 100;

    const endRed = riskPct;
    const endAmber = riskPct + duePct;

    const background = data.length === 0
        ? `conic-gradient(#e2e8f0 0% 100%)`
        : `conic-gradient(
            #ef4444 0% ${endRed}%,
            #f59e0b ${endRed}% ${endAmber}%,
            #22c55e ${endAmber}% 100%
          )`;

    document.getElementById('donut-chart').style.background = background;

    const secured = (active + due) * 45;
    const lostRev = (risk + lost) * 45;
    const fulfillPct = Math.round((active / total) * 100);

    document.getElementById('fulfill-rate').textContent = `${fulfillPct}%`;
    document.getElementById('fulfill-bar').style.width = `${fulfillPct}%`;

    document.getElementById('rev-secured').textContent = `$${secured}`;
    document.getElementById('rev-lost').textContent = `$${lostRev}`;
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (start === end) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// --- QUERY LOGIC ---
function fillQuery(text) {
    document.getElementById('query-input').value = text;
    document.getElementById('query-input').focus();
}

async function handleQuerySubmit(e) {
    e.preventDefault();
    const input = document.getElementById('query-input');
    const question = input.value.trim();
    if (!question) return;

    addMessage('user', question);
    input.value = '';

    const loading = document.getElementById('chat-loading');
    loading.classList.remove('hidden');

    try {
        const res = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: sessionId })
        });
        const data = await res.json();

        loading.classList.add('hidden');
        addAIMessage(data);

    } catch (e) {
        loading.classList.add('hidden');
        addMessage('error', 'Error: Could not reach the backend. Is Ollama/FastAPI running?');
    }
}

// --- QUERY TYPE BADGE ---
function getQueryTypeBadge(queryType) {
    const badges = {
        sql: { bg: 'bg-blue-100 text-blue-700 border-blue-200', icon: 'fa-database', label: 'SQL' },
        rag: { bg: 'bg-purple-100 text-purple-700 border-purple-200', icon: 'fa-book-open', label: 'RAG' },
        hybrid: { bg: 'bg-amber-100 text-amber-700 border-amber-200', icon: 'fa-code-merge', label: 'HYBRID' },
    };
    const b = badges[queryType] || badges.sql;
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${b.bg} border">
        <i class="fa-solid ${b.icon}"></i> ${b.label}
    </span>`;
}

// --- CONFIDENCE METER ---
function getConfidenceMeter(confidence, metadata) {
    const pct = Math.round((confidence || 0) * 100);
    let color, label, levelText, levelIcon;
    if (pct >= 70) {
        color = 'bg-green-500'; label = 'text-green-700';
        levelText = 'High — answer is well-supported by records';
        levelIcon = 'fa-circle-check';
    } else if (pct >= 40) {
        color = 'bg-yellow-500'; label = 'text-yellow-700';
        levelText = 'Moderate — some claims may not be in the records';
        levelIcon = 'fa-circle-info';
    } else {
        color = 'bg-red-500'; label = 'text-red-700';
        levelText = 'Low — answer may include general knowledge beyond records';
        levelIcon = 'fa-triangle-exclamation';
    }

    // Build signal breakdown if available
    let signalHtml = '';
    const detail = metadata?.confidence_detail;
    if (detail?.signals) {
        const s = detail.signals;
        const rows = [];
        if (s.retrieval_margin !== null && s.retrieval_margin !== undefined) {
            rows.push(`<div class="flex justify-between"><span>Retrieval Margin</span><span class="font-semibold">${Math.round(s.retrieval_margin * 100)}%</span></div>`);
        }
        if (s.coverage !== null && s.coverage !== undefined) {
            rows.push(`<div class="flex justify-between"><span>Source Coverage</span><span class="font-semibold">${Math.round(s.coverage * 100)}%</span></div>`);
        }
        if (s.llm_self_assessment !== null && s.llm_self_assessment !== undefined) {
            rows.push(`<div class="flex justify-between"><span>Model Self-Assessment</span><span class="font-semibold">${Math.round(s.llm_self_assessment * 100)}%</span></div>`);
        }
        if (rows.length > 0) {
            signalHtml = `
            <div class="mt-1.5 text-[10px] text-slate-400 space-y-0.5 border-t border-slate-100 pt-1.5">
                <div class="font-semibold text-slate-500 mb-0.5">Score Breakdown</div>
                ${rows.join('')}
            </div>`;
        }
    }

    return `<div class="mt-1">
        <div class="flex items-center gap-2">
            <div class="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div class="${color} h-full rounded-full transition-all" style="width: ${pct}%"></div>
            </div>
            <span class="text-[11px] font-bold ${label}">${pct}%</span>
        </div>
        <div class="flex items-center gap-1 mt-0.5">
            <i class="fa-solid ${levelIcon} text-[9px] ${label}"></i>
            <span class="text-[10px] ${label}">${levelText}</span>
        </div>
        ${signalHtml}
    </div>`;
}

// --- GROUNDING INDICATOR ---
function getGroundingIndicator(grounding) {
    if (!grounding) return '';
    const grounded = grounding.is_grounded;
    const score = Math.round((grounding.score || 0) * 100);
    const supported = grounding.supported_sentences || 0;
    const total = grounding.total_sentences || 0;

    let icon, color, mainLabel, tooltip;
    if (grounded) {
        icon = 'fa-shield-check'; color = 'text-green-600 bg-green-50 border-green-200';
        mainLabel = 'Verified';
        tooltip = `All ${total} claims verified against clinical records`;
    } else if (score >= 50) {
        icon = 'fa-shield-halved'; color = 'text-amber-600 bg-amber-50 border-amber-200';
        mainLabel = 'Partially Verified';
        tooltip = `${supported} of ${total} claims found in clinical records. Some details come from general medical knowledge.`;
    } else {
        icon = 'fa-book-medical'; color = 'text-blue-600 bg-blue-50 border-blue-200';
        mainLabel = 'General Knowledge';
        tooltip = `${supported} of ${total} claims from records. Answer supplemented with medical knowledge not in your records.`;
    }

    return `<div class="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-medium ${color} border cursor-help" title="${tooltip}">
        <i class="fa-solid ${icon}"></i>
        <span>${mainLabel}</span>
        <span class="opacity-60">${supported}/${total}</span>
    </div>`;
}

// --- SOURCE CITATIONS ---
function renderSources(sources) {
    if (!sources || sources.length === 0) return '';

    const cited = sources.filter(s => s.cited);
    const uncited = sources.filter(s => !s.cited);
    const toShow = [...cited, ...uncited].slice(0, 5);

    if (toShow.length === 0) return '';

    const sourceCards = toShow.map(s => {
        const citedBadge = s.cited
            ? '<span class="text-[9px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded font-bold">CITED</span>'
            : '';
        const relevance = Math.round((s.relevance_score || 0) * 100);
        return `<div class="bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs">
            <div class="flex items-center justify-between mb-1">
                <span class="font-semibold text-slate-700">${s.patient_name || 'Unknown'}</span>
                <div class="flex items-center gap-1.5">
                    ${citedBadge}
                    <span class="text-slate-400">${relevance}%</span>
                </div>
            </div>
            <div class="text-[10px] text-slate-400 mb-1">${s.condition || ''} ${s.visit_date ? '| ' + s.visit_date : ''}</div>
            <div class="text-slate-500 line-clamp-2">${s.text_snippet || ''}</div>
        </div>`;
    }).join('');

    return `<div class="mt-3 border-t border-slate-100 pt-3">
        <button onclick="this.nextElementSibling.classList.toggle('hidden'); this.querySelector('i').classList.toggle('fa-chevron-down'); this.querySelector('i').classList.toggle('fa-chevron-up');"
                class="text-xs text-slate-500 font-medium flex items-center gap-1 hover:text-indigo-600 transition-colors mb-2">
            <i class="fa-solid fa-book-open"></i> Sources (${toShow.length})
            <i class="fa-solid fa-chevron-down text-[8px] ml-1"></i>
        </button>
        <div class="hidden space-y-2">
            ${sourceCards}
        </div>
    </div>`;
}

// --- RESPONSE TIMING ---
function getTimingInfo(metadata) {
    if (!metadata) return '';
    const parts = [];
    if (metadata.retrieval_time_ms) parts.push(`Retrieval: ${(metadata.retrieval_time_ms / 1000).toFixed(1)}s`);
    if (metadata.generation_time_ms) parts.push(`Generation: ${(metadata.generation_time_ms / 1000).toFixed(1)}s`);
    if (metadata.grounding_time_ms) parts.push(`Grounding: ${(metadata.grounding_time_ms / 1000).toFixed(1)}s`);
    if (metadata.total_time_ms) parts.push(`Total: ${(metadata.total_time_ms / 1000).toFixed(1)}s`);
    if (parts.length === 0) return '';
    return `<div class="text-[9px] text-slate-300 mt-2 flex items-center gap-3 border-t border-slate-100 pt-2">
        <i class="fa-solid fa-clock"></i> ${parts.join(' &middot; ')}
    </div>`;
}

// --- AI MESSAGE WITH FULL RESPONSE ---
function addAIMessage(data) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = "p-4 rounded-xl text-sm mb-4 max-w-[85%] message-bubble shadow-sm bg-white border border-slate-200 text-slate-700 mr-auto rounded-bl-none";

    const queryBadge = getQueryTypeBadge(data.query_type);
    const confidenceMeter = getConfidenceMeter(data.confidence, data.metadata);
    const groundingHtml = getGroundingIndicator(data.grounding);
    const sourcesHtml = renderSources(data.sources);
    const timingHtml = getTimingInfo(data.metadata);

    const answer = data.answer || data.result || '';
    const sql = data.sql_generated;

    // Hybrid mode badge
    const hybridMode = data.hybrid_mode
        ? `<span class="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 font-medium uppercase">${data.hybrid_mode}</span>`
        : '';

    div.innerHTML = `
        <div class="flex items-center flex-wrap gap-2 mb-2">
            <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <i class="fa-solid fa-robot"></i> AI Analysis
            </div>
            ${queryBadge}
            ${hybridMode}
            <div class="ml-auto flex items-center gap-2">
                ${groundingHtml}
            </div>
        </div>
        <div class="mb-3">
            <div class="text-[10px] text-slate-400 font-medium mb-0.5">Answer Confidence</div>
            ${confidenceMeter}
        </div>
        ${sql ? `
        <div class="mb-3 bg-slate-900 rounded-lg p-3 font-mono text-xs text-green-400 overflow-x-auto border border-slate-800">
            <div class="text-slate-500 mb-1 select-none">// Generated SQL</div>
            ${escapeHtml(sql)}
        </div>` : ''}
        <div class="whitespace-pre-wrap leading-relaxed">${escapeHtml(answer)}</div>
        ${sourcesHtml}
        ${timingHtml}
    `;

    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function addMessage(type, content) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');

    if (type === 'user') {
        div.className = "p-4 rounded-xl text-sm mb-4 max-w-[85%] message-bubble shadow-sm bg-indigo-600 text-white ml-auto rounded-br-none";
        div.textContent = content;
    } else if (type === 'error') {
        div.className = "p-4 rounded-xl text-sm mb-4 max-w-[85%] message-bubble shadow-sm bg-red-50 border border-red-100 text-red-600 mr-auto";
        div.innerHTML = `<i class="fa-solid fa-triangle-exclamation mr-2"></i> ${escapeHtml(content)}`;
    }

    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
