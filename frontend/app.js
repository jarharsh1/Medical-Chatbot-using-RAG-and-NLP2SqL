const API_URL = "/api";

// State
let currentFilters = { clinic: '', doctor: '', condition: '', search: '' };
let currentPage = 1;
const pageSize = 50;
const sessionId = crypto.randomUUID();

// --- UTILITY ---
function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

function formatTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    loadFilters();
    loadDashboard();

    ['clinic', 'doctor', 'condition'].forEach(id => {
        document.getElementById(`filter-${id}`).addEventListener('change', (e) => {
            currentFilters[id] = e.target.value;
            currentPage = 1;
            loadDashboard();
        });
    });

    // Search bar with debounce
    document.getElementById('filter-search').addEventListener('input', debounce((e) => {
        currentFilters.search = e.target.value.trim();
        currentPage = 1;
        loadDashboard();
    }, 400));

    document.getElementById('query-form').addEventListener('submit', handleQuerySubmit);
});

// --- TABS (2 tabs) ---
function switchTab(tab) {
    const sections = ['dashboard', 'query'];
    sections.forEach(s => {
        const el = document.getElementById(`${s}-section`);
        const btn = document.getElementById(`btn-${s}`);
        if (s === tab) {
            el.classList.remove('hidden');
            btn.classList.add('active');
        } else {
            el.classList.add('hidden');
            btn.classList.remove('active');
        }
    });
}

// --- DASHBOARD ---
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
    document.getElementById('filter-search').value = "";
    currentFilters = { clinic: '', doctor: '', condition: '', search: '' };
    currentPage = 1;
    loadDashboard();
}

function updatePagination(pagination) {
    const info = document.getElementById('page-info');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    if (!info) return;
    info.textContent = `Page ${pagination.page} of ${pagination.total_pages} (${pagination.total_rows.toLocaleString()} records)`;
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
    tbody.innerHTML = `<tr><td colspan="5" class="text-center py-12 text-slate-400">
        <i class="fa-solid fa-spinner fa-spin mr-2"></i> Loading data...
    </td></tr>`;

    try {
        const res = await fetch(`${API_URL}/dashboard?page=${currentPage}&page_size=${pageSize}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentFilters)
        });
        const data = await res.json();
        const rows = data.rows || data;
        const pagination = data.pagination || null;
        const kpis = data.kpis || null;

        renderTable(rows);
        updateMetrics(rows, kpis);
        if (pagination) updatePagination(pagination);
    } catch (e) {
        console.error(e);
        tbody.innerHTML = `<tr><td colspan="5" class="text-center py-12 text-red-500">
            <i class="fa-solid fa-triangle-exclamation mr-2"></i> Error connecting to server
        </td></tr>`;
    }
}

function renderTable(data) {
    const tbody = document.getElementById('patient-table-body');
    const recordCount = document.getElementById('record-count');
    recordCount.textContent = `${data.length} records`;
    tbody.innerHTML = '';

    if (data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center py-12 text-slate-400">No records found</td></tr>`;
        return;
    }

    const statusClasses = {
        'Good': 'status-badge good',
        'Refill Due': 'status-badge refill',
        'Renewal Needed': 'status-badge renewal',
        'Non-Adherent': 'status-badge risk'
    };

    data.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.className = "cursor-pointer";
        const rowId = `detail-${index}`;

        tr.innerHTML = `
            <td class="px-6 py-4">
                <div class="font-medium text-slate-800">${escapeHtml(row.name)}</div>
                <div class="text-xs text-slate-400 mt-0.5">${escapeHtml(row.clinic)}</div>
            </td>
            <td class="px-6 py-4">
                <div class="text-slate-700">${escapeHtml(row.medication)}</div>
                <div class="text-xs text-slate-400 mt-0.5">${escapeHtml(row.dosage)} &bull; ${row.refills_left} refills left</div>
            </td>
            <td class="px-6 py-4">
                <span class="${statusClasses[row.status] || 'status-badge'}">${escapeHtml(row.status)}</span>
            </td>
            <td class="px-6 py-4">
                <span class="text-xs font-medium text-teal-700 bg-teal-50 px-3 py-1.5 rounded-lg">${escapeHtml(row.action)}</span>
            </td>
            <td class="px-6 py-4 text-right">
                <button onclick="event.stopPropagation(); toggleDetails('${rowId}')" class="text-slate-400 hover:text-teal-600 transition-colors p-2">
                    <i class="fa-solid fa-chevron-down text-xs"></i>
                </button>
            </td>
        `;

        tr.onclick = () => toggleDetails(rowId);

        const detailTr = document.createElement('tr');
        detailTr.id = rowId;
        detailTr.className = "hidden";

        // Build doctor notes section for expanded view
        let doctorNotesHtml = '';
        if (row.has_doctor_notes && row.doctor_notes_snippet) {
            const soapHtml = parseSOAPToHTML(row.doctor_notes_snippet);
            doctorNotesHtml = `
                <div class="mt-4">
                    <h4 class="text-xs font-semibold text-teal-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <i class="fa-solid fa-file-medical"></i> Doctor Notes
                        <span class="soap-badge"><i class="fa-solid fa-check text-[8px]"></i> SOAP</span>
                    </h4>
                    <div class="soap-container text-sm">${soapHtml}</div>
                </div>
                ${row.note_id ? `<button onclick="event.stopPropagation(); openNoteModal(${row.note_id})"
                    class="mt-3 text-xs text-teal-600 hover:text-teal-700 font-medium flex items-center gap-1 transition-colors">
                    <i class="fa-solid fa-expand"></i> View Full Note
                </button>` : ''}
            `;
        }

        detailTr.innerHTML = `
            <td colspan="5" class="px-6 py-4 bg-slate-50 border-t border-slate-100">
                <div class="grid grid-cols-3 gap-6">
                    <div class="col-span-2">
                        <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                            <i class="fa-solid fa-notes-medical mr-1"></i> Clinical Note
                        </h4>
                        <div class="bg-white p-4 rounded-lg border border-slate-200 text-sm text-slate-600 italic">
                            "${escapeHtml(row.note_snippet)}"
                        </div>
                        ${doctorNotesHtml}
                    </div>
                    <div class="space-y-4">
                        <div>
                            <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Prescriber</h4>
                            <p class="text-sm font-medium text-slate-800">${escapeHtml(row.doctor)}</p>
                        </div>
                        <div>
                            <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Condition</h4>
                            <p class="text-sm font-medium text-slate-800">${escapeHtml(row.condition)}</p>
                        </div>
                        <div>
                            <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Last Visit</h4>
                            <p class="text-sm font-medium text-slate-800">${row.last_visit || 'N/A'}</p>
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
    document.querySelectorAll('tr[id^="detail-"]').forEach(row => row.classList.add('hidden'));
    if (isHidden) el.classList.remove('hidden');
}

// --- KPI: Use backend kpis when available ---
function updateMetrics(data, kpis) {
    let risk = 0, due = 0, active = 0, lost = 0;

    if (kpis) {
        const totalRx = kpis.total_rows || 0;
        const activeRx = kpis.active_rx || 0;
        const expiredRx = kpis.expired_rx || 0;

        data.forEach(r => {
            if (r.status === 'Non-Adherent') risk++;
            else if (r.status === 'Refill Due') due++;
            else if (r.status === 'Good') active++;
            else if (r.status === 'Renewal Needed') lost++;
        });

        animateValue("total-rx", 0, totalRx, 500);
        animateValue("count-risk", 0, expiredRx, 500);
        animateValue("count-due", 0, totalRx - activeRx - expiredRx, 500);
        animateValue("count-active", 0, activeRx, 500);
    } else {
        data.forEach(r => {
            if (r.status === 'Non-Adherent') risk++;
            else if (r.status === 'Refill Due') due++;
            else if (r.status === 'Good') active++;
            else if (r.status === 'Renewal Needed') lost++;
        });

        animateValue("total-rx", 0, data.length, 500);
        animateValue("count-risk", 0, risk + lost, 500);
        animateValue("count-due", 0, due, 500);
        animateValue("count-active", 0, active, 500);
    }

    const total = data.length || 1;
    const totalRisk = risk + lost;

    document.getElementById('legend-risk').textContent = totalRisk;
    document.getElementById('legend-due').textContent = due;
    document.getElementById('legend-active').textContent = active;
    document.getElementById('donut-total').textContent = data.length;

    const riskPct = (totalRisk / total) * 100;
    const duePct = (due / total) * 100;
    const donut = document.getElementById('donut-chart');

    donut.style.background = data.length === 0
        ? `conic-gradient(#e2e8f0 0% 100%)`
        : `conic-gradient(#ef4444 0% ${riskPct}%, #f59e0b ${riskPct}% ${riskPct + duePct}%, #10b981 ${riskPct + duePct}% 100%)`;

    const fulfillPct = Math.round((active / total) * 100);
    document.getElementById('fulfill-rate').textContent = `${fulfillPct}%`;
    document.getElementById('fulfill-bar').style.width = `${fulfillPct}%`;

    const secured = (active + due) * 45;
    const lostRev = (risk + lost) * 45;
    const maxRev = Math.max(secured, lostRev, 1);

    document.getElementById('bar-secured').style.height = `${(secured / maxRev) * 100}%`;
    document.getElementById('bar-lost').style.height = `${(lostRev / maxRev) * 100}%`;
    document.getElementById('rev-secured').textContent = `$${secured.toLocaleString()}`;
    document.getElementById('rev-lost').textContent = `$${lostRev.toLocaleString()}`;
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (!obj || start === end) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.textContent = Math.floor(progress * (end - start) + start).toLocaleString();
        if (progress < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
}

// =========================================
// SOAP Parser & Note Modal
// =========================================

const SOAP_SECTIONS = {
    'CC': { label: 'Chief Complaint', icon: 'fa-comment-medical' },
    'CHIEF COMPLAINT': { label: 'Chief Complaint', icon: 'fa-comment-medical' },
    'HPI': { label: 'History of Present Illness', icon: 'fa-clock-rotate-left' },
    'HISTORY OF PRESENT ILLNESS': { label: 'History of Present Illness', icon: 'fa-clock-rotate-left' },
    'VITALS': { label: 'Vital Signs', icon: 'fa-heart-pulse' },
    'VITAL SIGNS': { label: 'Vital Signs', icon: 'fa-heart-pulse' },
    'PHYSICAL EXAM': { label: 'Physical Exam', icon: 'fa-stethoscope' },
    'PHYSICAL EXAMINATION': { label: 'Physical Exam', icon: 'fa-stethoscope' },
    'ASSESSMENT': { label: 'Assessment', icon: 'fa-clipboard-check' },
    'PLAN': { label: 'Plan', icon: 'fa-list-check' },
    'TREATMENT PLAN': { label: 'Treatment Plan', icon: 'fa-list-check' },
};

function parseSOAPToHTML(text) {
    if (!text) return '<span class="text-slate-400 italic">No content</span>';

    const sectionPattern = new RegExp(
        '(' + Object.keys(SOAP_SECTIONS).join('|') + ')\\s*[:.]\\s*',
        'gi'
    );

    const matches = [...text.matchAll(sectionPattern)];

    if (matches.length === 0) {
        return `<div class="soap-section-content">${escapeHtml(text)}</div>`;
    }

    let html = '';
    for (let i = 0; i < matches.length; i++) {
        const key = matches[i][1].toUpperCase();
        const start = matches[i].index + matches[i][0].length;
        const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
        const content = text.slice(start, end).trim();

        const section = SOAP_SECTIONS[key] || { label: key, icon: 'fa-note-medical' };

        html += `
            <div class="soap-section">
                <div class="soap-section-label">
                    <i class="fa-solid ${section.icon} text-teal-500"></i>
                    ${section.label}
                </div>
                <div class="soap-section-content">${escapeHtml(content)}</div>
            </div>
        `;
    }

    if (matches[0].index > 0) {
        const preamble = text.slice(0, matches[0].index).trim();
        if (preamble) {
            html = `<div class="soap-section"><div class="soap-section-content">${escapeHtml(preamble)}</div></div>` + html;
        }
    }

    return html;
}

async function openNoteModal(noteId) {
    const modalContainer = document.getElementById('note-modal');

    modalContainer.innerHTML = `
        <div class="modal-backdrop" onclick="closeNoteModal(event)">
            <div class="modal-content p-8" onclick="event.stopPropagation()">
                <div class="text-center py-12 text-slate-400">
                    <i class="fa-solid fa-spinner fa-spin text-xl mb-2"></i>
                    <p class="text-sm">Loading note...</p>
                </div>
            </div>
        </div>
    `;
    modalContainer.classList.remove('hidden');

    try {
        const res = await fetch(`${API_URL}/clinical-notes/${noteId}`);
        if (!res.ok) throw new Error('Note not found');
        const note = await res.json();
        renderNoteModal(note);
    } catch (e) {
        modalContainer.innerHTML = `
            <div class="modal-backdrop" onclick="closeNoteModal(event)">
                <div class="modal-content p-8" onclick="event.stopPropagation()">
                    <div class="text-center py-12 text-red-500">
                        <i class="fa-solid fa-triangle-exclamation text-xl mb-2"></i>
                        <p class="text-sm">Failed to load note</p>
                    </div>
                </div>
            </div>
        `;
    }
}

function renderNoteModal(note) {
    const modalContainer = document.getElementById('note-modal');

    const soapHtml = note.has_doctor_notes
        ? parseSOAPToHTML(note.doctor_notes)
        : '';

    modalContainer.innerHTML = `
        <div class="modal-backdrop" onclick="closeNoteModal(event)">
            <div class="modal-content" onclick="event.stopPropagation()">
                <!-- Header -->
                <div class="p-6 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-teal-50/30">
                    <div class="flex items-start justify-between">
                        <div>
                            <h3 class="text-lg font-bold text-slate-800">${escapeHtml(note.patient_name)}</h3>
                            <div class="flex items-center gap-3 mt-1 text-xs text-slate-500">
                                <span><i class="fa-solid fa-hospital mr-1"></i>${escapeHtml(note.clinic_name)}</span>
                                ${note.clinic_location ? `<span><i class="fa-solid fa-location-dot mr-1"></i>${escapeHtml(note.clinic_location)}</span>` : ''}
                            </div>
                        </div>
                        <button onclick="closeNoteModal()" class="text-slate-400 hover:text-slate-600 transition-colors p-1">
                            <i class="fa-solid fa-xmark text-lg"></i>
                        </button>
                    </div>

                    <div class="flex flex-wrap items-center gap-3 mt-4 text-xs text-slate-600">
                        <span class="px-2.5 py-1 bg-white rounded-lg border border-slate-200 font-medium">
                            <i class="fa-solid fa-user-doctor mr-1 text-teal-500"></i>${escapeHtml(note.doctor_name)}
                        </span>
                        <span class="px-2.5 py-1 bg-white rounded-lg border border-slate-200 font-medium">
                            <i class="fa-solid fa-calendar mr-1 text-teal-500"></i>${note.visit_date || 'N/A'}
                        </span>
                        <span class="px-2.5 py-1 bg-white rounded-lg border border-slate-200 font-medium">
                            <i class="fa-solid fa-stethoscope mr-1 text-teal-500"></i>${escapeHtml(note.condition_name)} (${note.diagnosis_code || 'N/A'})
                        </span>
                        ${note.gender ? `<span class="px-2.5 py-1 bg-white rounded-lg border border-slate-200 font-medium">
                            <i class="fa-solid fa-user mr-1 text-teal-500"></i>${note.gender}${note.dob ? ' &bull; DOB: ' + note.dob : ''}
                        </span>` : ''}
                        ${note.insurance_provider ? `<span class="px-2.5 py-1 bg-white rounded-lg border border-slate-200 font-medium">
                            <i class="fa-solid fa-id-card mr-1 text-teal-500"></i>${escapeHtml(note.insurance_provider)}
                        </span>` : ''}
                    </div>
                </div>

                <!-- Body -->
                <div class="p-6 space-y-5">
                    <div>
                        <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                            <i class="fa-solid fa-notes-medical mr-1"></i> Clinical Note
                        </h4>
                        <div class="bg-slate-50 p-4 rounded-lg border border-slate-200 text-sm text-slate-600 leading-relaxed">
                            ${escapeHtml(note.note_text || 'No note text available.')}
                        </div>
                    </div>

                    ${note.has_doctor_notes ? `
                    <div>
                        <h4 class="text-xs font-semibold text-teal-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <i class="fa-solid fa-file-medical"></i> Doctor Notes
                            <span class="soap-badge"><i class="fa-solid fa-check text-[8px]"></i> SOAP</span>
                        </h4>
                        <div class="soap-container">${soapHtml}</div>
                    </div>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
}

function closeNoteModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('note-modal').classList.add('hidden');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const modal = document.getElementById('note-modal');
        if (!modal.classList.contains('hidden')) {
            modal.classList.add('hidden');
        }
    }
});

// =========================================
// QUERY / AI CHAT
// =========================================
function fillQuery(text) {
    switchTab('query');
    document.getElementById('query-input').value = text;
    document.getElementById('query-input').focus();
}

function clearChat() {
    const history = document.getElementById('chat-history');
    history.innerHTML = `
        <div class="chat-message ai">
            <div class="message-content">
                <p>Chat cleared. How can I help you?</p>
            </div>
        </div>
    `;
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
        addMessage('error', 'Could not reach the backend. Is the server running?');
    }
}

function getQueryTypeBadge(queryType) {
    const types = {
        sql: { class: 'sql', label: 'SQL', icon: 'fa-database' },
        rag: { class: 'rag', label: 'RAG', icon: 'fa-book-open' },
        hybrid: { class: 'hybrid', label: 'HYBRID', icon: 'fa-code-merge' },
        orchestrated: { class: 'orchestrated', label: 'MULTI-PART', icon: 'fa-sitemap' },
    };
    const t = types[queryType] || types.sql;
    return `<span class="query-badge ${t.class}"><i class="fa-solid ${t.icon} mr-1"></i>${t.label}</span>`;
}

function getConfidenceMeter(confidence) {
    const pct = Math.round((confidence || 0) * 100);
    let level, color;
    if (pct >= 70) { level = 'high'; color = 'High confidence'; }
    else if (pct >= 40) { level = 'medium'; color = 'Moderate confidence'; }
    else { level = 'low'; color = 'Low confidence'; }

    return `
        <div class="mt-3 pt-3 border-t border-slate-100">
            <div class="flex items-center justify-between text-xs mb-1">
                <span class="text-slate-500">Confidence</span>
                <span class="font-semibold">${pct}%</span>
            </div>
            <div class="confidence-bar">
                <div class="fill ${level}" style="width: ${pct}%"></div>
            </div>
            <p class="text-[10px] text-slate-400 mt-1">${color}</p>
        </div>
    `;
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return '';
    const toShow = sources.slice(0, 4);

    const cards = toShow.map(s => `
        <div class="source-card">
            <div class="flex items-center justify-between mb-1">
                <span class="font-medium text-slate-700">${escapeHtml(s.patient_name || 'Unknown')}</span>
                ${s.cited ? '<span class="text-[9px] bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded font-bold">CITED</span>' : ''}
            </div>
            <div class="text-[10px] text-slate-400 mb-1">${escapeHtml(s.condition || '')} ${s.visit_date ? '&bull; ' + s.visit_date : ''}</div>
            <div class="text-slate-500 line-clamp-2">${escapeHtml(s.text_snippet || '')}</div>
        </div>
    `).join('');

    return `
        <div class="mt-3 pt-3 border-t border-slate-100">
            <button onclick="this.nextElementSibling.classList.toggle('hidden')"
                class="text-xs text-slate-500 font-medium flex items-center gap-1 hover:text-teal-600 mb-2">
                <i class="fa-solid fa-book-open"></i> Sources (${toShow.length})
                <i class="fa-solid fa-chevron-down text-[8px] ml-1"></i>
            </button>
            <div class="hidden space-y-2">${cards}</div>
        </div>
    `;
}

function addAIMessage(data) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = "chat-message ai";

    const badge = getQueryTypeBadge(data.query_type);
    const confidence = getConfidenceMeter(data.confidence);
    const sources = renderSources(data.sources);
    const answer = data.answer || data.result || '';
    const sql = data.sql_generated;
    const timestamp = formatTimestamp();

    let decompositionHtml = '';
    if (data.decomposition && data.decomposition.parts_count > 1) {
        const parts = data.decomposition.sub_questions.map(p =>
            `<div class="flex items-center gap-2 text-xs">
                <span class="query-badge ${p.route} inline">${p.route.toUpperCase()}</span>
                <span class="text-slate-500 truncate">${escapeHtml(p.question)}</span>
            </div>`
        ).join('');
        decompositionHtml = `
            <div class="mb-3 p-3 bg-slate-50 rounded-lg border border-slate-100">
                <div class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    Decomposed into ${data.decomposition.parts_count} parts
                </div>
                <div class="space-y-1.5">${parts}</div>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="message-content">
            <div class="flex items-center gap-2 mb-3">
                ${badge}
                ${data.hybrid_mode ? `<span class="text-[10px] px-2 py-0.5 bg-slate-100 text-slate-500 rounded-full uppercase font-medium">${escapeHtml(data.hybrid_mode)}</span>` : ''}
            </div>
            ${decompositionHtml}
            ${sql ? `<div class="sql-block mb-3"><span class="comment">// Generated SQL</span><br>${escapeHtml(sql)}</div>` : ''}
            <div class="prose prose-sm max-w-none prose-slate prose-headings:text-slate-800 prose-headings:font-semibold prose-p:text-slate-700 prose-strong:text-slate-800 prose-ul:text-slate-700 prose-ol:text-slate-700 prose-li:marker:text-teal-500">${marked.parse(answer || '')}</div>
            ${sources}
            ${confidence}
        </div>
        <div class="message-timestamp">${timestamp}</div>
    `;

    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function addMessage(type, content) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = `chat-message ${type}`;
    const timestamp = formatTimestamp();

    if (type === 'user') {
        div.innerHTML = `
            <div class="message-content">${escapeHtml(content)}</div>
            <div class="message-timestamp">${timestamp}</div>
        `;
    } else if (type === 'error') {
        div.innerHTML = `<div class="message-content bg-red-50 border-red-200 text-red-600">
            <i class="fa-solid fa-triangle-exclamation mr-2"></i>${escapeHtml(content)}
        </div>`;
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
