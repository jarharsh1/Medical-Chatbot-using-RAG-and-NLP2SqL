const API_URL = "/api";

// State
let currentFilters = { clinic: '', doctor: '', condition: '', search: '', from_date: '', to_date: '' };
let currentPage = 1;
const pageSize = 50;
let sessionId = crypto.randomUUID();
let currentRows = [];
let selectedModel = null;  // null = use server default

// Dashboard Chart.js instances
let donutChart = null;
let fulfillmentBarChart = null;

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
    loadModels();

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

    // Date range filters
    document.getElementById('filter-from-date').addEventListener('change', (e) => {
        currentFilters.from_date = e.target.value;
        currentPage = 1;
        loadDashboard();
    });
    document.getElementById('filter-to-date').addEventListener('change', (e) => {
        currentFilters.to_date = e.target.value;
        currentPage = 1;
        loadDashboard();
    });

    document.getElementById('query-form').addEventListener('submit', handleQuerySubmit);

    // Auto-resize textarea + Enter/Shift+Enter
    const queryInput = document.getElementById('query-input');
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
    });
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('query-form').requestSubmit();
        }
    });

    // Sidebar init
    _restoreSidebarState();
    loadSessions();
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
    if (tab === 'query') loadSessions();
}

async function loadModels() {
    try {
        const res = await fetch(`${API_URL}/models`);
        const data = await res.json();
        const models = data.models || [];
        selectedModel = models.includes(data.current) ? data.current : (models[0] || null);
        _renderModelOptions(models, selectedModel);
    } catch (e) {
        console.warn('Could not load models:', e);
        _renderModelOptions([], null);
    }
}

function _renderModelOptions(models, active) {
    const pill = document.getElementById('model-pill-name');
    const dropdown = document.getElementById('model-dropdown');
    if (!pill || !dropdown) return;

    pill.textContent = active || 'Select model';
    dropdown.innerHTML = '';

    models.forEach(m => {
        const item = document.createElement('div');
        item.className = 'model-option' + (m === active ? ' selected' : '');
        item.innerHTML = `<span>${m}</span><i class="fa-solid fa-check model-check"></i>`;
        item.onclick = () => {
            selectedModel = m;
            pill.textContent = m;
            dropdown.querySelectorAll('.model-option').forEach(el => el.classList.remove('selected'));
            item.classList.add('selected');
            closeModelDropdown();
        };
        dropdown.appendChild(item);
    });
}

function toggleModelDropdown(e) {
    if (e) e.stopPropagation();
    const dropdown = document.getElementById('model-dropdown');
    const chevron = document.getElementById('model-pill-chevron');
    if (!dropdown) return;
    const isOpen = !dropdown.classList.contains('hidden');
    if (isOpen) {
        dropdown.classList.add('hidden');
        chevron && chevron.classList.remove('open');
    } else {
        dropdown.classList.remove('hidden');
        chevron && chevron.classList.add('open');
    }
}

function closeModelDropdown() {
    const dropdown = document.getElementById('model-dropdown');
    const chevron = document.getElementById('model-pill-chevron');
    dropdown && dropdown.classList.add('hidden');
    chevron && chevron.classList.remove('open');
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.model-pill-container')) closeModelDropdown();
});

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
    document.getElementById('filter-from-date').value = "";
    document.getElementById('filter-to-date').value = "";
    currentFilters = { clinic: '', doctor: '', condition: '', search: '', from_date: '', to_date: '' };
    currentPage = 1;
    loadDashboard();
}

function updatePagination(pagination) {
    const info = document.getElementById('page-info');
    const container = document.getElementById('pagination-container');
    if (!info || !container) return;

    info.textContent = `Page ${pagination.page} of ${pagination.total_pages} (${pagination.total_rows.toLocaleString()} records)`;

    const { page, total_pages } = pagination;
    container.innerHTML = '';

    if (total_pages <= 1) return;

    const makeBtn = (label, pageNum, isActive, isDisabled, isIcon) => {
        const btn = document.createElement('button');
        btn.className = isActive ? 'pagination-btn pagination-active' : 'pagination-btn';
        if (isIcon) btn.innerHTML = label;
        else btn.textContent = label;
        btn.disabled = isDisabled;
        if (!isDisabled && !isActive) btn.onclick = () => goToPage(pageNum);
        return btn;
    };

    const makeEllipsis = () => {
        const span = document.createElement('span');
        span.className = 'px-2 text-slate-400 text-sm select-none';
        span.textContent = '...';
        return span;
    };

    container.appendChild(makeBtn('<i class="fa-solid fa-angles-left text-xs"></i>', 1, false, page === 1, true));
    container.appendChild(makeBtn('<i class="fa-solid fa-chevron-left text-xs"></i>', page - 1, false, page === 1, true));

    let pages = [];
    if (total_pages <= 7) {
        for (let i = 1; i <= total_pages; i++) pages.push(i);
    } else if (page <= 4) {
        pages = [1, 2, 3, 4, 5, -1, total_pages];
    } else if (page >= total_pages - 3) {
        pages = [1, -1, total_pages - 4, total_pages - 3, total_pages - 2, total_pages - 1, total_pages];
    } else {
        pages = [1, -1, page - 1, page, page + 1, -1, total_pages];
    }

    pages.forEach(p => {
        if (p === -1) container.appendChild(makeEllipsis());
        else container.appendChild(makeBtn(String(p), p, p === page, false, false));
    });

    container.appendChild(makeBtn('<i class="fa-solid fa-chevron-right text-xs"></i>', page + 1, false, page === total_pages, true));
    container.appendChild(makeBtn('<i class="fa-solid fa-angles-right text-xs"></i>', total_pages, false, page === total_pages, true));
}

function goToPage(pageNum) {
    currentPage = pageNum;
    loadDashboard();
}

async function loadDashboard() {
    const tbody = document.getElementById('patient-table-body');
    tbody.innerHTML = `<tr><td colspan="11" class="text-center py-12 text-slate-400">
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
        tbody.innerHTML = `<tr><td colspan="11" class="text-center py-12 text-red-500">
            <i class="fa-solid fa-triangle-exclamation mr-2"></i> Error connecting to server
        </td></tr>`;
    }
}

function getStatusClass(status) {
    const map = {
        'Good': 'status-badge good',
        'Refill Due': 'status-badge refill',
        'Renewal Needed': 'status-badge renewal',
        'Non-Adherent': 'status-badge risk',
        'Not Purchased': 'status-badge not-purchased'
    };
    return map[status] || 'status-badge';
}

function getNextStepsClass(steps) {
    const map = {
        'Monitor': 'next-steps-badge monitor',
        'Call for Refill': 'next-steps-badge call-refill',
        'Book Appointment': 'next-steps-badge book-appt',
        'Call Patient': 'next-steps-badge call-patient',
        'Did Not Buy': 'next-steps-badge did-not-buy'
    };
    return map[steps] || 'next-steps-badge monitor';
}

function renderTable(data) {
    const tbody = document.getElementById('patient-table-body');
    const recordCount = document.getElementById('record-count');
    currentRows = data;
    recordCount.textContent = `${data.length} records`;
    tbody.innerHTML = '';

    if (data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="11" class="text-center py-12 text-slate-400">No records found</td></tr>`;
        return;
    }

    data.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.className = "transition-colors";

        tr.innerHTML = `
            <td class="px-3 py-2.5">
                <div class="font-medium text-slate-800">${escapeHtml(row.name)}</div>
                <div class="text-[10px] text-slate-400 mt-0.5">${escapeHtml(row.clinic)}</div>
            </td>
            <td class="px-3 py-2.5 text-slate-700">${escapeHtml(row.doctor)}</td>
            <td class="px-3 py-2.5 text-slate-700">${escapeHtml(row.condition)}</td>
            <td class="px-3 py-2.5 text-slate-700">${escapeHtml(row.medication)}</td>
            <td class="px-3 py-2.5 text-slate-500">${escapeHtml(row.dosage)}</td>
            <td class="px-3 py-2.5 text-slate-500 whitespace-nowrap">${row.last_visit || 'N/A'}</td>
            <td class="px-3 py-2.5 text-slate-500 whitespace-nowrap">${row.last_filled_date || 'N/A'}</td>
            <td class="px-3 py-2.5 text-slate-500 whitespace-nowrap">${row.refill_due_date || 'N/A'}</td>
            <td class="px-3 py-2.5">
                <span class="${getStatusClass(row.status)}">${escapeHtml(row.status)}</span>
            </td>
            <td class="px-3 py-2.5">
                <span class="${getNextStepsClass(row.next_steps)}">${escapeHtml(row.next_steps)}</span>
            </td>
            <td class="px-3 py-2.5 text-center">
                <button onclick="event.stopPropagation(); openViewModal(${index})" class="view-btn">
                    <i class="fa-solid fa-eye text-[10px]"></i> View
                </button>
            </td>
        `;

        tbody.appendChild(tr);
    });
}

function openViewModal(index) {
    const row = currentRows[index];
    if (!row) return;

    const modalContainer = document.getElementById('note-modal');

    let doctorNotesSection = '';
    if (row.has_doctor_notes && row.doctor_notes_snippet) {
        const soapHtml = parseSOAPToHTML(row.doctor_notes_snippet);
        doctorNotesSection = `
            <hr class="border-slate-100">
            <div>
                <h4 class="text-xs font-semibold text-teal-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <i class="fa-solid fa-file-medical"></i> Doctor Notes
                    <span class="soap-badge"><i class="fa-solid fa-check text-[8px]"></i> SOAP</span>
                </h4>
                <div class="soap-container text-sm">${soapHtml}</div>
            </div>
            ${row.note_id ? `<button onclick="event.stopPropagation(); closeNoteModal(); openNoteModal(${row.note_id})"
                class="mt-3 text-xs text-teal-600 hover:text-teal-700 font-medium flex items-center gap-1 transition-colors">
                <i class="fa-solid fa-expand"></i> View Full Note
            </button>` : ''}
        `;
    }

    modalContainer.innerHTML = `
        <div class="modal-backdrop" onclick="closeNoteModal(event)">
            <div class="modal-content" style="max-width: 560px;" onclick="event.stopPropagation()">
                <div class="p-5 border-b border-slate-100 bg-slate-50">
                    <div class="flex items-start justify-between">
                        <div>
                            <h3 class="text-base font-bold text-slate-800">${escapeHtml(row.name)}</h3>
                            <div class="flex items-center gap-3 mt-1 text-xs text-slate-500">
                                <span><i class="fa-solid fa-hospital mr-1"></i>${escapeHtml(row.clinic)}</span>
                                <span><i class="fa-solid fa-user-doctor mr-1"></i>${escapeHtml(row.doctor)}</span>
                            </div>
                        </div>
                        <button onclick="closeNoteModal()" class="text-slate-400 hover:text-slate-600 transition-colors p-1">
                            <i class="fa-solid fa-xmark text-lg"></i>
                        </button>
                    </div>
                </div>

                <div class="p-5 space-y-4">
                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-0.5">Condition</p>
                            <p class="text-sm font-medium text-slate-800">${escapeHtml(row.condition)}</p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-0.5">Visit Date</p>
                            <p class="text-sm font-medium text-slate-800">${row.last_visit || 'N/A'}</p>
                        </div>
                    </div>

                    <hr class="border-slate-100">

                    <div>
                        <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                            <i class="fa-solid fa-prescription mr-1"></i> Prescription Details
                        </h4>
                        <div class="grid grid-cols-2 gap-3 text-sm">
                            <div>
                                <p class="text-[10px] text-slate-400">Medication</p>
                                <p class="font-medium text-slate-800">${escapeHtml(row.medication)}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Dosage</p>
                                <p class="font-medium text-slate-800">${escapeHtml(row.dosage)}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Days Supply</p>
                                <p class="font-medium text-slate-800">${row.days_supply || 'N/A'}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Refills Remaining</p>
                                <p class="font-medium text-slate-800">${row.refills_left}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Last Filled</p>
                                <p class="font-medium text-slate-800">${row.last_filled_date || 'N/A'}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Rx Status</p>
                                <p class="font-medium text-slate-800">${escapeHtml(row.rx_status || 'N/A')}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Refill Due Date</p>
                                <p class="font-medium text-slate-800">${row.refill_due_date || 'N/A'}</p>
                            </div>
                            <div>
                                <p class="text-[10px] text-slate-400">Next Steps</p>
                                <p class="mt-0.5"><span class="${getNextStepsClass(row.next_steps)}">${escapeHtml(row.next_steps)}</span></p>
                            </div>
                        </div>
                        <div class="mt-3">
                            <span class="${getStatusClass(row.status)}">${escapeHtml(row.status)}</span>
                        </div>
                    </div>

                    ${doctorNotesSection}
                </div>
            </div>
        </div>
    `;
    modalContainer.classList.remove('hidden');
}

// --- Dashboard Chart.js Init ---
function initDashboardCharts() {
    // Doughnut chart
    const donutCtx = document.getElementById('donut-chart-canvas');
    if (donutCtx && !donutChart) {
        donutChart = new Chart(donutCtx, {
            type: 'doughnut',
            data: {
                labels: ['At Risk', 'Due Soon', 'Active'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                    borderWidth: 0,
                    hoverOffset: 6,
                }]
            },
            options: {
                cutout: '62%',
                responsive: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        titleFont: { size: 12, weight: '600' },
                        bodyFont: { size: 11 },
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: (ctx) => ` ${ctx.label}: ${ctx.raw} prescriptions`
                        }
                    }
                },
                animation: { animateRotate: true, duration: 800 }
            }
        });
    }

    // Fulfillment bar chart
    const barCtx = document.getElementById('fulfillment-bar-canvas');
    if (barCtx && !fulfillmentBarChart) {
        fulfillmentBarChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Secured', 'Opportunity'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: ['#10b981', '#cbd5e1'],
                    borderRadius: 6,
                    barPercentage: 0.5,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        callbacks: {
                            label: (ctx) => ` $${ctx.raw.toLocaleString()}`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#f1f5f9' },
                        ticks: { font: { size: 10 }, color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 11, weight: '500' }, color: '#64748b' }
                    }
                }
            }
        });
    }
}

// --- KPI: Use backend kpis when available (always use distinct counts) ---
function updateMetrics(data, kpis) {
    let totalRx = 0, uniquePatients = 0, riskRx = 0, dueRx = 0, activeRx = 0;

    if (kpis) {
        totalRx = kpis.total_rows || 0;
        uniquePatients = kpis.unique_patients || 0;
        activeRx = kpis.active_rx || 0;
        const expiredRx = kpis.expired_rx || 0;
        riskRx = expiredRx;
        dueRx = totalRx - activeRx - expiredRx;
    } else {
        uniquePatients = new Set(data.map(r => r.patient_id)).size;
        totalRx = new Set(data.map(r => r.rx_id || r.patient_id + '_' + r.medication + '_' + r.dosage)).size;
        const statusCounts = {};
        data.forEach(r => {
            const status = r.status;
            const key = r.rx_id || r.patient_id + '_' + r.medication + '_' + r.dosage;
            if (!statusCounts[status]) statusCounts[status] = new Set();
            statusCounts[status].add(key);
        });
        riskRx = (statusCounts['Non-Adherent'] || new Set()).size +
                 (statusCounts['Renewal Needed'] || new Set()).size +
                 (statusCounts['Not Purchased'] || new Set()).size;
        dueRx = (statusCounts['Refill Due'] || new Set()).size;
        activeRx = (statusCounts['Good'] || new Set()).size;
    }

    // Update KPI cards
    animateValue("total-rx", 0, totalRx, 400);
    animateValue("total-patients", 0, uniquePatients, 400);
    animateValue("count-risk", 0, riskRx, 400);
    animateValue("count-due", 0, dueRx, 400);
    animateValue("count-active", 0, activeRx, 400);

    // Update legend text
    document.getElementById('legend-risk').textContent = riskRx;
    document.getElementById('legend-due').textContent = dueRx;
    document.getElementById('legend-active').textContent = activeRx;
    document.getElementById('donut-total').textContent = totalRx;

    // Initialize Chart.js instances if needed
    initDashboardCharts();

    // Update doughnut chart
    if (donutChart) {
        donutChart.data.datasets[0].data = [riskRx, dueRx, activeRx];
        donutChart.update();
    }

    // Update fulfillment bar + progress bar
    const total = totalRx || 1;
    const fulfillPct = Math.round((activeRx / total) * 100);
    document.getElementById('fulfill-rate').textContent = `${fulfillPct}%`;
    document.getElementById('fulfill-bar').style.width = `${fulfillPct}%`;

    const secured = (activeRx + dueRx) * 45;
    const lostRev = riskRx * 45;

    if (fulfillmentBarChart) {
        fulfillmentBarChart.data.datasets[0].data = [secured, lostRev];
        fulfillmentBarChart.update();
    }
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
    const input = document.getElementById('query-input');
    input.value = text;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    input.focus();
}

// =========================================
// SESSION MANAGEMENT
// =========================================
const WELCOME_HTML = `
    <div id="nb-welcome" class="nb-welcome">
        <div class="nb-welcome-icon">
            <i class="fa-solid fa-heart-pulse text-teal-400 text-3xl"></i>
        </div>
        <h2 class="text-xl font-bold text-white mt-4">Ask MediGraph</h2>
        <p class="text-sm text-slate-300 mt-1.5 max-w-md mx-auto leading-relaxed">
            Ask questions about patients, prescriptions, clinical notes, or medical knowledge.
            Everything runs locally — your data never leaves this machine.
        </p>
        <div class="nb-suggestions">
            <button onclick="fillQuery('How many active prescriptions are there?')" class="nb-suggestion-chip">
                <i class="fa-solid fa-database text-blue-400 text-xs"></i>
                How many active prescriptions?
            </button>
            <button onclick="fillQuery('What symptoms are described for diabetic patients?')" class="nb-suggestion-chip">
                <i class="fa-solid fa-book-open text-purple-400 text-xs"></i>
                Symptoms for diabetic patients?
            </button>
            <button onclick="fillQuery('Which clinic has the most diabetes patients? Who are the doctors there?')" class="nb-suggestion-chip">
                <i class="fa-solid fa-code-merge text-amber-400 text-xs"></i>
                Top diabetes clinic & doctors?
            </button>
            <button onclick="fillQuery('What is the root cause of hypertension? How many patients have it?')" class="nb-suggestion-chip">
                <i class="fa-solid fa-sitemap text-emerald-400 text-xs"></i>
                Causes & count of hypertension?
            </button>
            <button onclick="fillQuery('List all patients with non-adherent prescriptions')" class="nb-suggestion-chip">
                <i class="fa-solid fa-triangle-exclamation text-red-400 text-xs"></i>
                Non-adherent prescriptions?
            </button>
            <button onclick="fillQuery('What medications are commonly prescribed for asthma?')" class="nb-suggestion-chip">
                <i class="fa-solid fa-pills text-teal-400 text-xs"></i>
                Medications for asthma?
            </button>
        </div>
        <div class="flex items-center justify-center gap-4 mt-6 text-xs text-slate-400">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-shield-check text-emerald-400"></i> HIPAA Compliant</span>
            <span class="text-slate-500">&middot;</span>
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-server text-slate-500"></i> Local Processing</span>
            <span class="text-slate-500">&middot;</span>
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-brain text-teal-400"></i> RAG + SQL</span>
        </div>
    </div>
`;

function clearChat() {
    document.getElementById('chat-history').innerHTML = WELCOME_HTML;
}

function startNewChat() {
    sessionId = crypto.randomUUID();
    clearChat();
    renderSessionList(window._sessions || []);
}

async function loadSessions() {
    try {
        const res = await fetch(`${API_URL}/sessions`);
        const sessions = await res.json();
        window._sessions = sessions;
        renderSessionList(sessions);
    } catch (e) {
        console.error('Failed to load sessions:', e);
    }
}

function renderSessionList(sessions) {
    const list = document.getElementById('session-list');
    if (!list) return;

    if (!sessions || sessions.length === 0) {
        list.innerHTML = '<div class="text-xs text-slate-500 text-center py-8">No conversations yet</div>';
        return;
    }

    list.innerHTML = sessions.map(s => {
        const isActive = s.session_id === sessionId;
        const timeStr = _formatSessionTime(s.updated_at);
        return `
            <div class="session-item ${isActive ? 'active' : ''}" onclick="switchSession('${s.session_id}')">
                <div class="session-item-icon">
                    <i class="fa-solid fa-message"></i>
                </div>
                <div class="session-item-text">
                    <div class="session-item-title">${escapeHtml(s.title)}</div>
                    <div class="session-item-meta">${timeStr} &middot; ${s.message_count} msg</div>
                </div>
                <button class="session-item-delete" onclick="event.stopPropagation(); deleteSession('${s.session_id}')" title="Delete">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `;
    }).join('');
}

function _formatSessionTime(epoch) {
    if (!epoch) return '';
    const d = new Date(epoch * 1000);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

async function switchSession(id) {
    if (id === sessionId) return;
    sessionId = id;
    clearChat();
    renderSessionList(window._sessions || []);

    try {
        const res = await fetch(`${API_URL}/sessions/${id}/messages`);
        const messages = await res.json();
        if (!messages || messages.length === 0) return;

        hideWelcome();
        for (const msg of messages) {
            if (msg.role === 'user') {
                addMessage('user', msg.content);
            } else {
                addAIMessage({
                    answer: msg.content,
                    query_type: msg.query_type,
                    sql_generated: msg.sql_generated,
                    confidence: msg.confidence,
                    sources: msg.sources,
                    decomposition: msg.decomposition,
                    hybrid_mode: msg.hybrid_mode,
                    chart_data: msg.chart_data,
                });
            }
        }
    } catch (e) {
        console.error('Failed to load session messages:', e);
    }
}

async function deleteSession(id) {
    try {
        await fetch(`${API_URL}/sessions/${id}`, { method: 'DELETE' });
        if (id === sessionId) startNewChat();
        loadSessions();
    } catch (e) {
        console.error('Failed to delete session:', e);
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById('chat-sidebar');
    if (!sidebar) return;
    // Mobile: toggle open class
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('open');
    } else {
        sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
    }
}

// Restore sidebar state on load
function _restoreSidebarState() {
    const collapsed = localStorage.getItem('sidebar-collapsed') === 'true';
    if (collapsed) {
        const sidebar = document.getElementById('chat-sidebar');
        if (sidebar) sidebar.classList.add('collapsed');
    }
}

async function handleQuerySubmit(e) {
    e.preventDefault();
    const input = document.getElementById('query-input');
    const question = input.value.trim();
    if (!question) return;

    addMessage('user', question);
    input.value = '';
    input.style.height = 'auto';

    const loading = document.getElementById('chat-loading');
    const stageText = document.getElementById('loading-stage-text');
    loading.classList.remove('hidden');

    try {
        const res = await fetch(`${API_URL}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: sessionId, model: selectedModel })
        });

        if (!res.ok || !res.body) throw new Error('Stream failed');

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const event = JSON.parse(line.slice(6));
                    if (event.stage === 'routing' || event.stage === 'progress') {
                        if (stageText) stageText.textContent = event.message;
                    } else if (event.stage === 'complete') {
                        loading.classList.add('hidden');
                        addAIMessage(event.result);
                        loadSessions();
                    }
                } catch (_) {}
            }
        }

        loading.classList.add('hidden');
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

function renderSources(sources) {
    if (!sources || sources.length === 0) return '';
    const toShow = sources.slice(0, 5);
    const citedCount = toShow.filter(s => s.cited).length;

    const cards = toShow.map((s, i) => `
        <div class="src-card">
            <div class="src-card-header">
                <span class="src-idx">${i + 1}</span>
                <div class="src-card-info">
                    <span class="src-patient">${escapeHtml(s.patient_name || 'Unknown Patient')}</span>
                    <span class="src-meta">${escapeHtml(s.condition || '')}${s.visit_date ? ' · ' + s.visit_date : ''}</span>
                </div>
                ${s.cited ? '<span class="src-cited">✓ CITED</span>' : ''}
            </div>
            <div class="src-snippet">${escapeHtml(s.text_snippet || '')}</div>
        </div>
    `).join('');

    return `
        <details class="src-dropdown">
            <summary class="src-summary">
                <span class="src-summary-left">
                    <i class="fa-solid fa-database text-teal-400"></i>
                    <span>Retrieved Context</span>
                    <span class="src-count-badge">${toShow.length}</span>
                    ${citedCount > 0 ? `<span class="src-cited-badge">${citedCount} cited</span>` : ''}
                </span>
                <i class="fa-solid fa-chevron-down src-chevron"></i>
            </summary>
            <div class="src-list">${cards}</div>
        </details>
    `;
}

function renderMetrics(data) {
    const conf = data.confidence || 0;
    const grounding = data.grounding || {};
    const signals = (data.metadata && data.metadata.confidence_detail && data.metadata.confidence_detail.signals) || {};
    const isRAG = data.query_type === 'rag' || data.query_type === 'hybrid';

    // Recall@K: cited_sources / total_retrieved (proxy — true recall needs ground truth labels)
    const sources = data.sources || [];
    const K = sources.length;
    const cited = sources.filter(s => s.cited).length;
    const recallAtK = isRAG && K > 0 ? cited / K : null;

    const faithfulness = isRAG && grounding.score != null ? grounding.score : null;
    const answerRelevancy = signals.llm_self_assessment != null ? signals.llm_self_assessment : conf;
    const contextPrecision = signals.retrieval_margin != null ? signals.retrieval_margin : null;

    const fmt = v => v != null ? Math.round(v * 100) + '%' : '—';
    const mkBar = (v, color) => `<div class="metric-bar"><div class="metric-fill" style="width:${v != null ? Math.round(v * 100) : 0}%;background:${color}"></div></div>`;

    const overallColor = conf >= 0.7 ? '#10b981' : conf >= 0.4 ? '#f59e0b' : '#ef4444';
    const metrics = [
        {
            label: 'Faithfulness',
            val: faithfulness,
            color: '#10b981',
            desc: isRAG
                ? `${grounding.supported_sentences ?? '?'}/${grounding.total_sentences ?? '?'} sentences grounded`
                : 'N/A for SQL queries'
        },
        {
            label: 'Answer Relevancy',
            val: answerRelevancy,
            color: '#06b6d4',
            desc: 'LLM self-assessed confidence'
        },
        {
            label: 'Context Precision',
            val: contextPrecision,
            color: '#8b5cf6',
            desc: isRAG ? 'Top-doc separation (rerank margin)' : 'N/A for SQL queries'
        },
        {
            label: `Recall@${K || 'K'}`,
            val: recallAtK,
            color: '#f59e0b',
            desc: isRAG
                ? `${cited}/${K} retrieved docs cited · proxy (no ground truth)`
                : 'N/A for SQL queries'
        },
        {
            label: 'Overall Confidence',
            val: conf,
            color: overallColor,
            desc: conf >= 0.7 ? 'High — answer is reliable' : conf >= 0.4 ? 'Moderate — verify key details' : 'Low — treat with caution'
        },
    ];

    const chips = metrics.map((m, i) => `
        <div class="metric-chip${i === 4 ? ' metric-chip-full' : ''}">
            <div class="metric-chip-top">
                <span class="metric-label">${m.label}</span>
                <span class="metric-value" style="color:${m.color}">${fmt(m.val)}</span>
            </div>
            ${mkBar(m.val, m.color)}
            <div class="metric-desc">${m.desc}</div>
        </div>
    `).join('');

    return `
        <div class="metrics-panel">
            <div class="metrics-title"><i class="fa-solid fa-chart-simple mr-1.5 text-teal-400"></i>Evaluation Metrics</div>
            <div class="metrics-grid">${chips}</div>
        </div>
    `;
}

function hideWelcome() {
    const welcome = document.getElementById('nb-welcome');
    if (welcome) welcome.remove();
}

// --- Chart rendering in chat ---
let _chatChartCounter = 0;

function renderChartInBubble(chartData, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !chartData) return;

    const canvas = document.createElement('canvas');
    canvas.height = chartData.chart_type === 'doughnut' ? 220 : 200;
    container.appendChild(canvas);

    const config = {
        type: chartData.chart_type,
        data: { labels: chartData.labels, datasets: chartData.datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: chartData.chart_type === 'doughnut',
                    position: 'bottom',
                    labels: { font: { size: 10 }, padding: 12, usePointStyle: true, color: '#cbd5e1' },
                },
                title: {
                    display: true, text: chartData.title || '',
                    font: { size: 12, weight: '600' }, color: '#e2e8f0', padding: { bottom: 8 },
                },
                tooltip: { backgroundColor: '#334155', cornerRadius: 6 },
            },
        },
    };

    if (chartData.chart_type === 'doughnut') {
        config.options.cutout = '55%';
    } else {
        config.options.scales = {
            y: { beginAtZero: true, grid: { color: '#334155' }, ticks: { font: { size: 10 }, color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { font: { size: 9 }, maxRotation: 45, color: '#94a3b8' } },
        };
    }

    // Update dataset colors for dark mode
    if (chartData.datasets && chartData.datasets[0]) {
        const colors = ['#14b8a6', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899',
              '#06b6d4', '#84cc16', '#f97316', '#6366f1', '#0d9488', '#f43f5e'];
        chartData.datasets[0].backgroundColor = chartData.datasets[0].backgroundColor || colors.slice(0, chartData.labels.length);
        chartData.datasets[0].borderColor = 'transparent';
    }

    new Chart(canvas, config);
}

function addAIMessage(data) {
    const history = document.getElementById('chat-history');
    const row = document.createElement('div');
    row.className = 'gpt-msg-row';

    const badge = getQueryTypeBadge(data.query_type);
    const sources = renderSources(data.sources);
    const metrics = renderMetrics(data);
    const answer = data.answer || data.result || '';
    const sql = data.sql_generated;

    let decompositionHtml = '';
    if (data.decomposition && data.decomposition.parts_count > 1) {
        const parts = data.decomposition.sub_questions.map(p =>
            `<div class="flex items-center gap-2 text-xs">
                <span class="query-badge ${p.route} inline">${p.route.toUpperCase()}</span>
                <span class="text-slate-300 truncate">${escapeHtml(p.question)}</span>
            </div>`
        ).join('');
        decompositionHtml = `
            <div class="mb-3 p-3 bg-white/5 rounded-lg border border-white/10">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                    Decomposed into ${data.decomposition.parts_count} parts
                </div>
                <div class="space-y-1.5">${parts}</div>
            </div>
        `;
    }

    let chartHtml = '';
    let chartContainerId = null;
    if (data.chart_data) {
        _chatChartCounter++;
        chartContainerId = `chat-chart-${_chatChartCounter}`;
        chartHtml = `<div id="${chartContainerId}" class="mt-3 mb-2 p-3 bg-slate-800 rounded-lg border border-slate-700" style="height:240px;"></div>`;
    }

    row.innerHTML = `
        <div class="gpt-msg-content gpt-msg-ai">
            <div class="gpt-avatar">🩺</div>
            <div class="gpt-msg-body">
                <div class="msg-meta">
                    ${badge}
                    ${data.hybrid_mode ? `<span class="text-xs px-2 py-0.5 bg-white/10 text-slate-300 rounded-full uppercase font-medium">${escapeHtml(data.hybrid_mode)}</span>` : ''}
                </div>
                ${decompositionHtml}
                ${sql ? `<div class="sql-block mb-3"><span class="comment">// Generated SQL</span><br>${escapeHtml(sql)}</div>` : ''}
                <div class="prose prose-sm prose-invert max-w-none prose-headings:font-semibold prose-li:marker:text-teal-400">${marked.parse(answer || '')}</div>
                ${chartHtml}
                ${sources}
                ${metrics}
            </div>
        </div>
    `;

    history.appendChild(row);
    history.scrollTop = history.scrollHeight;

    if (chartContainerId && data.chart_data) {
        requestAnimationFrame(() => renderChartInBubble(data.chart_data, chartContainerId));
    }
}

function addMessage(type, content) {
    hideWelcome();
    const history = document.getElementById('chat-history');
    const row = document.createElement('div');

    if (type === 'user') {
        row.className = 'gpt-msg-row user-row';
        row.innerHTML = `
            <div class="gpt-msg-content">
                <div class="gpt-user-avatar">🧑</div>
                <div class="gpt-msg-body" style="color: #e2e8f0; font-weight: 500;">
                    ${escapeHtml(content)}
                </div>
            </div>
        `;
    } else if (type === 'error') {
        row.className = 'gpt-msg-row';
        row.innerHTML = `
            <div class="gpt-msg-content gpt-msg-ai">
                <div class="gpt-avatar" style="background: linear-gradient(135deg, #dc2626, #f97316); box-shadow: 0 0 0 2px rgba(220,38,38,0.35);">⚠️</div>
                <div class="gpt-msg-body" style="color: #f87171;">
                    <i class="fa-solid fa-triangle-exclamation mr-1"></i>${escapeHtml(content)}
                </div>
            </div>
        `;
    }

    history.appendChild(row);
    history.scrollTop = history.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
