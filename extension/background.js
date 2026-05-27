/**
 * OpenSeek — Background Service Worker (Manifest v3)
 * Handles: context menu, result storage, badge counter, notifications
 */

const BACKEND = "https://openseek-production.up.railway.app";

/* ─── Context menu ─────────────────────────────────────────────────────────── */

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "openseek-scan-image",
        title: "Analyze for Deepfake",
        contexts: ["image"],
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    const url = info.srcUrl || info.mediaType;
    if (!url) return;

    // Only send to content script — it handles the scan + sends RESULT back.
    // Do NOT also call analyzeAndNotify here; that would scan twice and charge 2 credits.
    chrome.tabs.sendMessage(tab.id, { type: "SCAN_CONTEXT", url }).catch(async (err) => {
        // Content script not injected (e.g. non-HTML page) — fall back to background scan
        console.warn("OpenSeek: Content script unavailable, scanning from background.", err.message);
        await analyzeAndNotify(url, "image", tab.id);
    });
});

/* ─── Direct background analysis (for context menu) ──────────────────────── */

async function analyzeAndNotify(url, type, tabId) {
    try {
        const { openseek_token, openseek_backend_url = "https://openseek-production.up.railway.app" } = await chrome.storage.local.get(["openseek_token", "openseek_backend_url"]);
        const headers = { "Content-Type": "application/json" };
        if (openseek_token) {
            headers["Authorization"] = `Bearer ${openseek_token}`;
        }
        const resp = await fetch(`${openseek_backend_url}/analyze-image-data`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify({ url }),
        });
        if (!resp.ok) {
            if (resp.status === 401) {
                chrome.storage.local.remove(["openseek_token"]);
            }
            return;
        }
        const data = await resp.json();
        storeResult(data, url);
        sendNotification(data);
    } catch (_) {
        // Backend offline — silent fail
    }
}

/* ─── Notification ─────────────────────────────────────────────────────────── */

function sendNotification(result) {
    const r = result.risk_level || "Low";
    const s = result.ai_probability ? Math.round(result.ai_probability * 100) : (result.authenticity_score ?? 0);
    const icons = { Low: "✅", Medium: "⚠️", High: "🔴" };
    chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: `OpenSeek — ${icons[r]} ${r} Risk`,
        message: `Authenticity score: ${Math.round(s)}/100`,
    });
}

/* ─── Result storage ───────────────────────────────────────────────────────── */

async function storeResult(result, url) {
    const { history = [] } = await chrome.storage.local.get("history");
    history.unshift({
        url,
        type: result.type || "image",
        risk_level: result.risk_level || "Low",
        score: result.ai_probability ? Math.round(result.ai_probability * 100) : (result.authenticity_score ?? 0),
        ai_probability: result.ai_probability,
        content_type: result.content_type,
        predicted_class: result.predicted_class,
        timestamp: Date.now(),
    });
    // Keep last 50
    await chrome.storage.local.set({ history: history.slice(0, 50) });

    // Update badge on the extension icon
    const counts = { Low: 0, Medium: 0, High: 0, Uncertain: 0 };
    history.slice(0, 10).forEach(h => {
        if (h && h.risk_level) {
            counts[h.risk_level] = (counts[h.risk_level] || 0) + 1;
        }
    });
    if (counts.High > 0) {
        chrome.action.setBadgeText({ text: String(counts.High) });
        chrome.action.setBadgeBackgroundColor({ color: "#e53935" });
    } else {
        chrome.action.setBadgeText({ text: "" });
    }
}

/* ─── Messages from content script ────────────────────────────────────────── */

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === "RESULT") {
        storeResult(msg.result, msg.url);
        // Show notification when the content script finishes a context-menu scan
        if (msg.fromContextMenu) {
            sendNotification(msg.result);
        }
    }
    if (msg.type === "SHOW_DETAIL") {
        chrome.storage.session.set({ activeDetail: msg });
        chrome.action.openPopup?.().catch((err) => {
            console.warn("OpenSeek: Cannot open popup programmatically", err);
        });
    }

    // DO_SCAN_FILE: version that handles multi-part form data uploads
    if (msg.type === "DO_SCAN_FILE") {
        const { endpoint, filename, base64 } = msg.payload;

        // Convert base64 to Blob
        fetch(base64)
            .then(res => res.blob())
            .then(async (blob) => {
                const formData = new FormData();
                formData.append("file", blob, filename);

                const { openseek_token, openseek_backend_url = "https://openseek-production.up.railway.app" } = await chrome.storage.local.get(["openseek_token", "openseek_backend_url"]);
                const headers = {};
                if (openseek_token) {
                    headers["Authorization"] = `Bearer ${openseek_token}`;
                }

                return fetch(`${openseek_backend_url}/${endpoint}`, {
                    method: "POST",
                    headers: headers,
                    body: formData,
                });
            })
            .then(async (resp) => {
                if (!resp.ok) {
                    const e = await resp.json().catch(() => ({ detail: resp.statusText }));
                    sendResponse({ error: e.detail || resp.statusText });
                } else {
                    const data = await resp.json();
                    sendResponse({ data });
                }
            })
            .catch((err) => sendResponse({ error: err.message }));
        return true;
    }

    // DO_SCAN: content scripts proxy backend calls through here
    // because service workers have unrestricted localhost access.
    if (msg.type === "DO_SCAN") {
        const { endpoint, body } = msg.payload;
        chrome.storage.local.get(["openseek_token", "openseek_backend_url"]).then(({ openseek_token, openseek_backend_url = "https://openseek-production.up.railway.app" }) => {
            const headers = { "Content-Type": "application/json" };
            if (openseek_token) {
                headers["Authorization"] = `Bearer ${openseek_token}`;
            }
            return fetch(`${openseek_backend_url}/${endpoint}`, {
                method: "POST",
                headers: headers,
                body: JSON.stringify(body),
            });
        })
            .then(async (resp) => {
                if (!resp.ok) {
                    const e = await resp.json().catch(() => ({ detail: resp.statusText }));
                    sendResponse({ error: e.detail || resp.statusText });
                } else {
                    const data = await resp.json();
                    sendResponse({ data });
                }
            })
            .catch((err) => sendResponse({ error: err.message }));
        return true; // keep channel open for async response
    }
});

/* ─── Backend health check ─────────────────────────────────────────────────── */

async function checkBackend() {
    try {
        const { openseek_backend_url = "https://openseek-production.up.railway.app" } = await chrome.storage.local.get("openseek_backend_url");
        const resp = await fetch(`${openseek_backend_url}/health`);
        const data = await resp.json();
        await chrome.storage.local.set({ backendOnline: data.status === "ok" });
    } catch (_) {
        await chrome.storage.local.set({ backendOnline: false });
    }
}

// Check every 30 seconds
checkBackend();
setInterval(checkBackend, 30_000);
