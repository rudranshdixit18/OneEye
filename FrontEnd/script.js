/**
 * OneEye Sentinel — AI Surveillance Operations Center
 * Frontend Logic & Telemetry Engine
 */

// ==========================================
// 1. Navigation & Section Switching
// ==========================================
const sections = document.querySelectorAll(".section");
const navItems = document.querySelectorAll(".nav-item");

navItems.forEach(item => {
    item.addEventListener("click", () => {
        navItems.forEach(n => n.classList.remove("active"));
        item.classList.add("active");
        sections.forEach(s => s.classList.remove("active-section"));
        const target = document.getElementById(item.dataset.section);
        if (target) {
            target.classList.add("active-section");
        }
    });
});

// ==========================================
// 2. Animated Numerical Counters
// ==========================================
function animateCounter(counterEl, targetValue) {
    let current = 0;
    const duration = 1000; // ms
    const frameRate = 1000 / 60;
    const totalFrames = Math.round(duration / frameRate); 
    let frame = 0;

    const timer = setInterval(() => {
        frame++;
        // Easing function (easeOutQuad)
        const progress = frame / totalFrames;
        const ease = 1 - (1 - progress) * (1 - progress);
        current = Math.round(targetValue * ease);

        counterEl.textContent = current;

        if (frame >= totalFrames) {
            counterEl.textContent = targetValue;
            clearInterval(timer);
        }
    }, frameRate);
}

document.querySelectorAll('.animated-counter').forEach(el => {
    const val = Number(el.getAttribute('data-value')) || 0;
    animateCounter(el, val);
});

// ==========================================
// 3. Mock Incident Data & Icon Provider
// ==========================================
let detections = window.location.hostname.endsWith('.vercel.app') ? [] : [
    { 
        id: "INC-8941", 
        timestamp: "2026-08-27 11:30:45", 
        type: "fight", 
        confidence: 0.94, 
        location: "Camera 1 (Main Gate)", 
        details: "Physical altercation detected near turnstile A" 
    },
    { 
        id: "INC-8938", 
        timestamp: "2026-08-27 11:28:12", 
        type: "garbage", 
        confidence: 0.93,
        location: "Camera 2 (Loading Bay)", 
        details: "Unattended waste container overflow detected" 
    },
    { 
        id: "INC-8932", 
        timestamp: "2026-08-27 11:25:33", 
        type: "garbage", 
        confidence: 0.92,
        location: "Camera 1 (Main Gate)", 
        details: "Littering event recorded on pedestrian pathway" 
    },
    { 
        id: "INC-8925", 
        timestamp: "2026-08-27 11:20:15", 
        type: "fight", 
        confidence: 0.92,
        location: "Camera 3 (North Perimeter)", 
        details: "Aggressive motion classified between 2 subjects" 
    },
    { 
        id: "INC-8919", 
        timestamp: "2026-08-27 11:15:02", 
        type: "people", 
        confidence: 0.98, 
        location: "Camera 4 (Plaza Lobby)", 
        details: "High density crowd surge threshold exceeded" 
    }
];

function getTypeIcon(type) {
    if (type === "fight") {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>`;
    }
    if (type === "garbage") {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>`;
    }
    if (type === "people") {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>`;
    }
    if (type === "fallen_person") {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="7" cy="5" r="2"/><path d="m5 9 4 3 3 5 5 2"/><path d="M3 20h18"/></svg>`;
    }
    if (type === "mobile_phone") {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="2" width="12" height="20" rx="2"/><line x1="10" x2="14" y1="18" y2="18"/></svg>`;
    }
    if (type === "accident") {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17h18l-2-7H5l-2 7Z"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/><path d="m12 2 2 4h-4l2-4Z"/></svg>`;
    }
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>`;
}

function detectionsListHTML(list) {
    if (!list || list.length === 0) {
        return `<li style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 0.88rem;">No detection events match the current filter.</li>`;
    }

    const displayNames = {
        fight: 'Physical Altercation',
        garbage: 'Garbage / Litter',
        fallen_person: 'Fallen Person',
        mobile_phone: 'Mobile Phone',
        accident: 'Accident',
        people: 'Occupancy Spike'
    };
    return list.map(d => {
        const confPercent = Math.round(d.confidence * 100);
        return `
        <li class="detection-item ${d.type}">
            <div class="icon-wrapper">
                ${getTypeIcon(d.type)}
            </div>
            <div class="detection-info">
                <div class="type-label">${displayNames[d.type] || d.type.replaceAll('_', ' ')}</div>
                <div class="detection-sub">${d.details || 'Automated vision neural classification'}</div>
            </div>
            <div class="conf-badge">
                <span class="conf">${confPercent}%</span>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width: ${confPercent}%;"></div>
                </div>
            </div>
            <span class="location-pill">${d.location}</span>
            <span class="timestamp">${d.timestamp.split(' ')[1] || d.timestamp}</span>
        </li>`;
    }).join('');
}

// Render Initial Detections
function renderAllDetections(filterType = "all", searchQuery = "") {
    let filtered = detections;

    if (filterType !== "all") {
        filtered = filtered.filter(d => d.type === filterType);
    }

    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(d => 
            d.type.toLowerCase().includes(q) ||
            d.location.toLowerCase().includes(q) ||
            (d.details && d.details.toLowerCase().includes(q))
        );
    }

    const dList = document.getElementById("detections-list");
    if (dList) dList.innerHTML = detectionsListHTML(filtered);

    const dListDetail = document.getElementById("detections-list-detail");
    if (dListDetail) dListDetail.innerHTML = detectionsListHTML(filtered);
}

renderAllDetections();

// Render Timeline in Analytics
function renderTimeline() {
    const timelineEl = document.getElementById("detections-timeline");
    if (!timelineEl) return;
    if (!detections.length) {
        timelineEl.innerHTML = `<li class="timeline-item"><span class="location">No incidents recorded in the selected period.</span></li>`;
        return;
    }
    timelineEl.innerHTML = detections.map(d => `
        <li class="timeline-item ${d.type}">
            <div class="timeline-header-row">
                <span class="type-label">${d.type.toUpperCase()}</span>
                <span class="conf">${Math.round(d.confidence * 100)}% CONF</span>
            </div>
            <span class="location">${d.location}</span>
            <span class="timestamp">${d.timestamp}</span>
        </li>
    `).join("");
}

renderTimeline();

// ==========================================
// 4. Interactive Detection Filters & Search
// ==========================================
let currentFilter = "all";
document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentFilter = btn.dataset.filter;
        renderAllDetections(currentFilter, document.getElementById("detections-search")?.value || "");
    });
});

const searchInput = document.getElementById("detections-search");
if (searchInput) {
    searchInput.addEventListener("input", (e) => {
        renderAllDetections(currentFilter, e.target.value);
    });
}

// ==========================================
// 5. Live UTC Clock & Telemetry Timers
// ==========================================
function formatUTCTime(date) {
    const pad = (n) => String(n).padStart(2, '0');
    const y = date.getUTCFullYear();
    const m = pad(date.getUTCMonth() + 1);
    const d = pad(date.getUTCDate());
    const h = pad(date.getUTCHours());
    const min = pad(date.getUTCMinutes());
    const s = pad(date.getUTCSeconds());
    return `${y}-${m}-${d} ${h}:${min}:${s} UTC`;
}

function updateClocks() {
    const now = new Date();
    const timeOnly = now.toTimeString().split(' ')[0] + ' UTC';
    const fullUTC = formatUTCTime(now);

    const headerClock = document.getElementById("header-clock");
    if (headerClock) headerClock.textContent = timeOnly;

    const hudTime = document.getElementById("hud-timestamp");
    if (hudTime) hudTime.textContent = fullUTC;

    const hudTimeLive = document.getElementById("hud-timestamp-live");
    if (hudTimeLive) hudTimeLive.textContent = fullUTC;
}

setInterval(updateClocks, 1000);
updateClocks();

// ==========================================
// 6. Camera Switching HUD Simulation
// ==========================================
const cameraNames = {
    "1": "CAM-01 // MAIN ENTRANCE",
    "2": "CAM-02 // WAREHOUSE LOADING BAY",
    "3": "CAM-03 // NORTH PERIMETER FENCE",
    "4": "CAM-04 // PLAZA LOBBY TURNSTILES"
};
let activeCameraId = "cam1";

document.querySelectorAll(".feed-cam-select").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".feed-cam-select").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const camId = tab.dataset.cam;
        activeCameraId = `cam${camId}`;
        const hudCamName = document.getElementById("hud-cam-name");
        if (hudCamName && cameraNames[camId]) {
            hudCamName.textContent = cameraNames[camId];
        }
        showToast(`Switched channel to ${cameraNames[camId] || 'CAM-' + camId}`, "info");
    });
});

// ==========================================
// 7. Live Feed Controls (Interactive UX)
// ==========================================
let isRecording = false;
let recordSeconds = 872;
let recordTimerInterval = null;

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnRecord = document.getElementById("btn-record");
const btnRecordText = document.getElementById("btn-record-text");
const volumeSlider = document.getElementById("alert-volume-slider");
const volumeReadout = document.getElementById("volume-readout");
const qualitySelector = document.querySelector(".quality-selector");

if (btnStart) {
    btnStart.addEventListener("click", async () => {
        try {
            await apiFetch(`/cameras/${activeCameraId}/start`, { method: "POST", body: JSON.stringify({}) });
            mountCameraStream(activeCameraId);
            showToast("Optical stream initialized", "info");
        } catch (error) {
            showToast(error.message, "critical");
        }
    });
}

if (btnStop) {
    btnStop.addEventListener("click", async () => {
        try {
            await apiFetch(`/cameras/${activeCameraId}/stop`, { method: "POST" });
            unmountCameraStream();
            showToast("Camera feed paused by operator", "info");
        } catch (error) {
            showToast(error.message, "critical");
        }
    });
}

if (btnRecord) {
    btnRecord.addEventListener("click", async () => {
        const requestedState = !isRecording;
        try {
            await apiFetch(`/cameras/${activeCameraId}/recording/${requestedState ? 'start' : 'stop'}`, { method: "POST" });
            isRecording = requestedState;
        } catch (error) {
            showToast(error.message, "critical");
            return;
        }
        if (isRecording) {
            btnRecord.classList.add("recording");
            if (btnRecordText) btnRecordText.textContent = "Recording Active";
            showToast("Recording session initialized — writing to storage", "critical");
        } else {
            btnRecord.classList.remove("recording");
            if (btnRecordText) btnRecordText.textContent = "Toggle Recording";
            showToast("Recording session saved", "info");
        }
    });
}

// Format Recording Timer
function updateRecTimer() {
    recordSeconds++;
    const pad = (n) => String(n).padStart(2, '0');
    const hrs = pad(Math.floor(recordSeconds / 3600));
    const mins = pad(Math.floor((recordSeconds % 3600) / 60));
    const secs = pad(recordSeconds % 60);
    const el = document.getElementById("hud-rec-timer");
    if (el) el.textContent = `${hrs}:${mins}:${secs}`;
}
setInterval(updateRecTimer, 1000);

if (volumeSlider && volumeReadout) {
    volumeSlider.addEventListener("input", (e) => {
        volumeReadout.textContent = `${e.target.value}%`;
    });
}

if (qualitySelector) {
    qualitySelector.addEventListener("change", async (e) => {
        const presets = {
            "1080p": { stream_width: 1920, stream_height: 1080, stream_fps: 60, jpeg_quality: 92 },
            "720p": { stream_width: 1280, stream_height: 720, stream_fps: 30, jpeg_quality: 88 },
            "480p": { stream_width: 854, stream_height: 480, stream_fps: 24, jpeg_quality: 82 },
        };
        const preset = presets[e.target.value];
        try {
            await apiFetch("/settings", { method: "PUT", body: JSON.stringify(preset) });
            const hudMode = document.getElementById("hud-stream-mode");
            if (hudMode) hudMode.textContent = `REQUEST ${e.target.value.toUpperCase()} • ${preset.stream_fps} FPS`;
            showToast(`Requested ${e.target.value} at ${preset.stream_fps} FPS`, "info");
        } catch (error) {
            showToast(error.message, "critical");
        }
    });
}

// ==========================================
// 8. Analytics Dynamic Telemetry SVG Chart
// ==========================================
function renderTelemetryChart() {
    const chartContainer = document.getElementById("chart-container");
    if (!chartContainer) return;

    // Rich 24h Area & Bar Chart SVG
    chartContainer.innerHTML = `
        <div class="chart-inner-header">
            <span class="chart-title-tag">24-HOUR INCIDENT FREQUENCY TELEMETRY</span>
            <div class="chart-legend">
                <div class="legend-item"><span class="legend-dot" style="background: var(--blue);"></span> People Surge</div>
                <div class="legend-item"><span class="legend-dot" style="background: var(--orange);"></span> Waste/Garbage</div>
                <div class="legend-item"><span class="legend-dot" style="background: var(--red);"></span> Fights/Alarms</div>
            </div>
        </div>
        <div class="chart-svg-container">
            <svg viewBox="0 0 1000 220" width="100%" height="100%" preserveAspectRatio="none" style="overflow: visible;">
                <defs>
                    <linearGradient id="areaGradientBlue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
                        <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
                    </linearGradient>
                    <linearGradient id="areaGradientRed" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#f43f5e" stop-opacity="0.4"/>
                        <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>

                <!-- Grid lines -->
                <line x1="40" y1="40" x2="980" y2="40" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
                <line x1="40" y1="90" x2="980" y2="90" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
                <line x1="40" y1="140" x2="980" y2="140" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
                <line x1="40" y1="190" x2="980" y2="190" stroke="rgba(255,255,255,0.12)"/>

                <!-- Y-axis labels -->
                <text x="15" y="44" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">20</text>
                <text x="15" y="94" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">12</text>
                <text x="15" y="144" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">5</text>
                <text x="15" y="194" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">0</text>

                <!-- Area Fill (People Occupancy Trend) -->
                <path d="M 60 180 Q 180 140 300 160 T 540 100 T 780 70 T 960 110 L 960 190 L 60 190 Z" fill="url(#areaGradientBlue)"/>
                <!-- Line (People Occupancy Trend) -->
                <path d="M 60 180 Q 180 140 300 160 T 540 100 T 780 70 T 960 110" fill="none" stroke="#38bdf8" stroke-width="2.5"/>

                <!-- Waste Events Bars -->
                <rect x="190" y="130" width="12" height="60" rx="3" fill="#f59e0b" opacity="0.85"/>
                <rect x="370" y="110" width="12" height="80" rx="3" fill="#f59e0b" opacity="0.85"/>
                <rect x="580" y="140" width="12" height="50" rx="3" fill="#f59e0b" opacity="0.85"/>
                <rect x="810" y="90" width="12" height="100" rx="3" fill="#f59e0b" opacity="0.85"/>

                <!-- Fight Alert Marker -->
                <circle cx="540" cy="100" r="6" fill="#f43f5e" stroke="#ffffff" stroke-width="2"/>
                <circle cx="780" cy="70" r="6" fill="#f43f5e" stroke="#ffffff" stroke-width="2"/>

                <!-- X-axis labels -->
                <text x="60" y="212" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">00:00</text>
                <text x="280" y="212" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">06:00</text>
                <text x="520" y="212" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">12:00</text>
                <text x="760" y="212" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">18:00</text>
                <text x="940" y="212" fill="#64748b" font-size="11" font-family="'JetBrains Mono', monospace">23:59</text>
            </svg>
        </div>
    `;
}

renderTelemetryChart();

// ==========================================
// 9. Settings Interactive Sliders
// ==========================================
const fightSlider = document.getElementById("fight-threshold-slider");
const fightVal = document.getElementById("fight-threshold-val");
if (fightSlider && fightVal) {
    fightSlider.addEventListener("input", (e) => {
        fightVal.textContent = `${e.target.value}%`;
    });
}

const garbageSlider = document.getElementById("garbage-threshold-slider");
const garbageVal = document.getElementById("garbage-threshold-val");
if (garbageSlider && garbageVal) {
    garbageSlider.addEventListener("input", (e) => {
        garbageVal.textContent = `${e.target.value}%`;
    });
}

// ==========================================
// 10. Toast Notification System
// ==========================================
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <svg style="width: 18px; height: 18px; flex-shrink: 0; color: ${type === 'critical' ? 'var(--red)' : 'var(--blue)'};" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

// ==========================================
// 11. OneEye Backend Integration
// ==========================================
const API_BASE = (window.ONEEYE_API_BASE || `${window.location.protocol}//${window.location.host}`).replace(/\/$/, "") + "/api/v1";
let backendOnline = false;
let lastEventId = null;

async function apiFetch(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
    const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Backend request failed (${response.status})`);
    }
    return response.json();
}

function setBackendState(online, message) {
    backendOnline = online;
    const label = document.getElementById("backend-status");
    const dot = document.querySelector(".nav-system-status .status-dot");
    if (label) label.textContent = message;
    if (dot) {
        dot.classList.toggle("pulse-emerald", online);
        dot.classList.toggle("pulse-critical", !online);
    }
}

function mountCameraStream(cameraId) {
    const url = `${API_BASE}/cameras/${cameraId}/stream?ts=${Date.now()}`;
    ["dashboard-feed-viewport", "live-feed-viewport"].forEach(id => {
        const viewport = document.getElementById(id);
        if (!viewport) return;
        let image = viewport.querySelector(".live-stream-image");
        if (!image) {
            image = document.createElement("img");
            image.className = "live-stream-image";
            image.alt = `Live annotated stream from ${cameraId}`;
            viewport.prepend(image);
        }
        image.src = url;
    });
    const standbyText = document.querySelector("#dashboard-feed-viewport .standby-title");
    if (standbyText) standbyText.textContent = "SIGNAL CONNECTED • STREAMING";
}

function unmountCameraStream() {
    document.querySelectorAll(".live-stream-image").forEach(image => image.remove());
    const standbyText = document.querySelector("#dashboard-feed-viewport .standby-title");
    if (standbyText) standbyText.textContent = "SURVEILLANCE STREAM PAUSED";
}

function normalizeEvent(event) {
    const timestamp = new Date(event.started_at);
    return {
        id: event.id,
        timestamp: Number.isNaN(timestamp.getTime()) ? event.started_at : formatUTCTime(timestamp).replace(" UTC", ""),
        type: event.event_type,
        confidence: Number(event.confidence),
        location: event.camera_id.toUpperCase(),
        details: `${event.label}${event.acknowledged ? " • acknowledged" : ""}`,
        snapshotUrl: event.snapshot_url
    };
}

function setCounter(id, value) {
    const element = document.getElementById(id);
    if (!element) return;
    element.dataset.value = String(value || 0);
    element.textContent = String(value || 0);
}

async function refreshHealth() {
    try {
        const health = await apiFetch("/health");
        setBackendState(true, health.status === "ready" ? "BACKEND READY" : "BACKEND DEGRADED");
        const camerasLabel = document.getElementById("channels-status");
        if (camerasLabel) camerasLabel.textContent = `${health.cameras.length} Channels Configured`;
        const activeCamera = health.cameras.find(camera => camera.camera_id === activeCameraId);
        const hudMode = document.getElementById("hud-stream-mode");
        if (hudMode && activeCamera?.online && activeCamera.actual_width && activeCamera.actual_height) {
            hudMode.textContent = `${activeCamera.actual_width}×${activeCamera.actual_height} • ${Number(activeCamera.actual_fps).toFixed(1)} FPS ACTUAL`;
        }
        const times = Object.values(health.pipeline).map(value => value.last_inference_ms).filter(Number.isFinite);
        const inferenceLabel = document.getElementById("inference-status");
        if (inferenceLabel) inferenceLabel.textContent = `Inference: ${times.length ? Math.round(Math.max(...times)) : "--"} ms`;
    } catch (_error) {
        setBackendState(false, "BACKEND OFFLINE");
    }
}

async function refreshEvents() {
    try {
        const events = await apiFetch("/events?limit=100&hours=24");
        detections = events.map(normalizeEvent);
        renderAllDetections(currentFilter, searchInput?.value || "");
        renderTimeline();
        if (events[0] && lastEventId && events[0].id !== lastEventId && document.getElementById("setting-notifications")?.checked) {
            showToast(`${events[0].event_type.replaceAll('_', ' ')} detected on ${events[0].camera_id}`, "critical");
        }
        lastEventId = events[0]?.id || lastEventId;
    } catch (_error) {
        // Health polling owns the visible connectivity state.
    }
}

async function refreshAnalytics() {
    try {
        const data = await apiFetch(`/analytics/summary?hours=${analyticsHours}`);
        const count = type => data.totals[type]?.count || 0;
        setCounter("stat-fights-value", count("fight"));
        setCounter("stat-garbage-value", count("garbage"));
        setCounter("stat-fallen-value", count("fallen_person"));
        setCounter("stat-phone-value", count("mobile_phone"));
        setCounter("stat-accident-value", count("accident"));
        setCounter("stat-alerts-value", Object.values(data.totals).reduce((sum, item) => sum + item.count, 0));
        setCounter("analytics-fight-count", count("fight"));
        setCounter("analytics-garbage-count", count("garbage"));
        setCounter("analytics-other-count", count("fallen_person") + count("mobile_phone") + count("accident"));
        const confidenceText = type => {
            const average = data.totals[type]?.average_confidence;
            return average == null ? "No confirmed events" : `Avg confidence: ${Math.round(average * 1000) / 10}%`;
        };
        const fightNote = document.getElementById("analytics-fight-note");
        const garbageNote = document.getElementById("analytics-garbage-note");
        if (fightNote) fightNote.textContent = confidenceText("fight");
        if (garbageNote) garbageNote.textContent = confidenceText("garbage");
    } catch (_error) {
        // Keep the last confirmed values during transient outages.
    }
}

let analyticsHours = 24;
document.querySelectorAll(".range-chip[data-hours]").forEach(chip => {
    chip.addEventListener("click", () => {
        analyticsHours = Number(chip.dataset.hours);
        document.querySelectorAll(".range-chip[data-hours]").forEach(item => item.classList.remove("active"));
        chip.classList.add("active");
        refreshAnalytics();
    });
});

const thresholdControls = {
    fight: ["fight-threshold-slider", "fight-threshold-val"],
    garbage: ["garbage-threshold-slider", "garbage-threshold-val"],
    fallen_person: ["fallen-threshold-slider", "fallen-threshold-val"],
    mobile_phone: ["phone-threshold-slider", "phone-threshold-val"],
    accident: ["accident-threshold-slider", "accident-threshold-val"]
};
let thresholdSaveTimer = null;

function configureThresholds(values = {}) {
    Object.entries(thresholdControls).forEach(([eventType, [sliderId, readoutId]]) => {
        const slider = document.getElementById(sliderId);
        const readout = document.getElementById(readoutId);
        if (!slider || !readout) return;
        if (values[eventType] != null) slider.value = Math.round(values[eventType] * 100);
        readout.textContent = `${slider.value}%`;
        if (slider.dataset.backendBound) return;
        slider.dataset.backendBound = "true";
        slider.addEventListener("input", () => {
            readout.textContent = `${slider.value}%`;
            clearTimeout(thresholdSaveTimer);
            thresholdSaveTimer = setTimeout(saveThresholds, 350);
        });
    });
}

async function saveThresholds() {
    const thresholds = {};
    Object.entries(thresholdControls).forEach(([eventType, [sliderId]]) => {
        thresholds[eventType] = Number(document.getElementById(sliderId).value) / 100;
    });
    try {
        await apiFetch("/settings", { method: "PUT", body: JSON.stringify({ thresholds }) });
        showToast("AI thresholds saved", "info");
    } catch (error) {
        showToast(error.message, "critical");
    }
}

async function initializeBackendIntegration() {
    configureThresholds();
    try {
        const settings = await apiFetch("/settings");
        configureThresholds(settings.thresholds);
    } catch (_error) {
        setBackendState(false, "BACKEND OFFLINE");
    }
    await Promise.all([refreshHealth(), refreshEvents(), refreshAnalytics()]);
    setInterval(refreshHealth, 5000);
    setInterval(refreshEvents, 3000);
    setInterval(refreshAnalytics, 5000);
}

initializeBackendIntegration();
