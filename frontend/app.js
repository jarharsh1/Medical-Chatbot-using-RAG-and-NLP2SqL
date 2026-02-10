const API_URL = "/api";

// State
let currentFilters = { clinic: '', doctor: '', condition: '', search: '', from_date: '', to_date: '' };
let currentPage = 1;
const pageSize = 50;
const sessionId = crypto.randomUUID();
let currentRows = [];

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

// --- KPI: Use backend kpis when available (always use distinct counts) ---
function updateMetrics(data, kpis) {
    let totalRx = 0, uniquePatients = 0, riskRx = 0, dueRx = 0, activeRx = 0;

    if (kpis) {
        // Use backend-calculated distinct counts
        totalRx = kpis.total_rows || 0;
        uniquePatients = kpis.unique_patients || 0;
        activeRx = kpis.active_rx || 0;
        const expiredRx = kpis.expired_rx || 0;
        riskRx = expiredRx;
        dueRx = totalRx - activeRx - expiredRx;
    } else {
        // Calculate distinct counts from page data as fallback
        uniquePatients = new Set(data.map(r => r.patient_id)).size;
        totalRx = new Set(data.map(r => r.rx_id || r.patient_id + '_' + r.medication + '_' + r.dosage)).size;
        
        // Count distinct prescriptions per status
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

    // Update KPI cards with distinct counts
    animateValue("total-rx", 0, totalRx, 400);
    animateValue("total-patients", 0, uniquePatients, 400);
    animateValue("count-risk", 0, riskRx, 400);
    animateValue("count-due", 0, dueRx, 400);
    animateValue("count-active", 0, activeRx, 400);

    // Update legend and donut chart with distinct counts
    document.getElementById('legend-risk').textContent = riskRx;
    document.getElementById('legend-due').textContent = dueRx;
    document.getElementById('legend-active').textContent = activeRx;
    document.getElementById('donut-total').textContent = totalRx;

    const total = totalRx || 1;
    const riskPct = (riskRx / total) * 100;
    const duePct = (dueRx / total) * 100;
    const donut = document.getElementById('donut-chart');

    donut.style.background = totalRx === 0
        ? `conic-gradient(#e2e8f0 0% 100%)`
        : `conic-gradient(#ef4444 0% ${riskPct}%, #f59e0b ${riskPct}% ${riskPct + duePct}%, #10b981 ${riskPct + duePct}% 100%)`;

    const fulfillPct = Math.round((activeRx / total) * 100);
    document.getElementById('fulfill-rate').textContent = `${fulfillPct}%`;
    document.getElementById('fulfill-bar').style.width = `${fulfillPct}%`;

    const secured = (activeRx + dueRx) * 45;
    const lostRev = riskRx * 45;
    const maxRev = Math.max(secured, lostRev, 1);

    document.getElementById('bar-secured').style.height = `${(secured / maxRev) * 100}%`;
    document.getElementById('bar-lost').style.height = `${(lostRev / maxRev) * 100}%`;
    document.getElementById('rev-secured').textContent = `${secured.toLocaleString()}`;
    document.getElementById('rev-lost').textContent = `${lostRev.toLocaleString()}`;
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
    input.focus();
}

function clearChat() {
    const history = document.getElementById('chat-history');
    history.innerHTML = `
        <div id="nb-welcome" class="nb-welcome">
            <div class="nb-welcome-icon">
                <i class="fa-solid fa-heart-pulse text-teal-600 text-3xl"></i>
            </div>
            <h2 class="text-xl font-bold text-slate-800 mt-4">Ask MediGraph</h2>
            <p class="text-sm text-slate-500 mt-1.5 max-w-md mx-auto leading-relaxed">
                Ask questions about patients, prescriptions, clinical notes, or medical knowledge.
                Everything runs locally — your data never leaves this machine.
            </p>
            <div class="nb-suggestions">
                <button onclick="fillQuery('How many active prescriptions are there?')" class="nb-suggestion-chip">
                    <i class="fa-solid fa-database text-blue-500 text-[10px]"></i>
                    How many active prescriptions?
                </button>
                <button onclick="fillQuery('What symptoms are described for diabetic patients?')" class="nb-suggestion-chip">
                    <i class="fa-solid fa-book-open text-purple-500 text-[10px]"></i>
                    Symptoms for diabetic patients?
                </button>
                <button onclick="fillQuery('Which clinic has the most diabetes patients? Who are the doctors there?')" class="nb-suggestion-chip">
                    <i class="fa-solid fa-code-merge text-amber-500 text-[10px]"></i>
                    Top diabetes clinic & doctors?
                </button>
                <button onclick="fillQuery('What is the root cause of hypertension? How many patients have it?')" class="nb-suggestion-chip">
                    <i class="fa-solid fa-sitemap text-emerald-500 text-[10px]"></i>
                    Causes & count of hypertension?
                </button>
                <button onclick="fillQuery('List all patients with non-adherent prescriptions')" class="nb-suggestion-chip">
                    <i class="fa-solid fa-triangle-exclamation text-red-400 text-[10px]"></i>
                    Non-adherent prescriptions?
                </button>
                <button onclick="fillQuery('What medications are commonly prescribed for asthma?')" class="nb-suggestion-chip">
                    <i class="fa-solid fa-pills text-teal-500 text-[10px]"></i>
                    Medications for asthma?
                </button>
            </div>
            <div class="flex items-center justify-center gap-4 mt-6 text-[11px] text-slate-400">
                <span class="flex items-center gap-1.5"><i class="fa-solid fa-shield-check text-emerald-400"></i> HIPAA Compliant</span>
                <span class="text-slate-300">&middot;</span>
                <span class="flex items-center gap-1.5"><i class="fa-solid fa-server text-slate-400"></i> Local Processing</span>
                <span class="text-slate-300">&middot;</span>
                <span class="flex items-center gap-1.5"><i class="fa-solid fa-brain text-teal-400"></i> RAG + SQL</span>
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

function hideWelcome() {
    const welcome = document.getElementById('nb-welcome');
    if (welcome) welcome.remove();
}

function addAIMessage(data) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = "nb-msg nb-msg-ai";

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
        <div class="nb-ai-avatar-sm">
            <i class="fa-solid fa-sparkles text-white text-[10px]"></i>
        </div>
        <div class="nb-bubble" style="max-width: 85%;">
            <div class="flex items-center gap-2 mb-3">
                ${badge}
                ${data.hybrid_mode ? `<span class="text-[10px] px-2 py-0.5 bg-slate-100 text-slate-500 rounded-full uppercase font-medium">${escapeHtml(data.hybrid_mode)}</span>` : ''}
            </div>
            ${decompositionHtml}
            ${sql ? `<div class="sql-block mb-3"><span class="comment">// Generated SQL</span><br>${escapeHtml(sql)}</div>` : ''}
            <div class="prose prose-sm max-w-none prose-slate prose-headings:text-slate-800 prose-headings:font-semibold prose-p:text-slate-700 prose-strong:text-slate-800 prose-ul:text-slate-700 prose-ol:text-slate-700 prose-li:marker:text-teal-500">${marked.parse(answer || '')}</div>
            ${sources}
            ${confidence}
            <div class="nb-timestamp">${timestamp}</div>
        </div>
    `;

    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function addMessage(type, content) {
    hideWelcome();
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    const timestamp = formatTimestamp();

    if (type === 'user') {
        div.className = 'nb-msg nb-msg-user';
        div.innerHTML = `
            <div class="nb-user-avatar">You</div>
            <div>
                <div class="nb-bubble">${escapeHtml(content)}</div>
                <div class="nb-timestamp" style="text-align: right;">${timestamp}</div>
            </div>
        `;
    } else if (type === 'error') {
        div.className = 'nb-msg nb-msg-ai';
        div.innerHTML = `
            <div class="nb-ai-avatar-sm">
                <i class="fa-solid fa-sparkles text-white text-[10px]"></i>
            </div>
            <div class="nb-bubble" style="background: #fef2f2; border-color: #fecaca; color: #dc2626;">
                <i class="fa-solid fa-triangle-exclamation mr-2"></i>${escapeHtml(content)}
            </div>
        `;
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
