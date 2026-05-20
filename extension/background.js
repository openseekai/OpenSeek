/**
 * OpenSeek — Background Service Worker (Manifest v3)
 * Handles: context menu, result storage, badge counter, notifications
 */

const BACKEND = "https://openseek-763952043156.europe-west1.run.app";

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

    let type = "image";

    // Tell the content script to trigger a scan
    chrome.tabs.sendMessage(tab.id, { type: "SCAN_CONTEXT", url });

    // Also do it from the background directly and show notification result
    await analyzeAndNotify(url, type, tab.id);
});

/* ─── Direct background analysis (for context menu) ──────────────────────── */

async function analyzeAndNotify(url, type, tabId) {
    try {
        const resp = await fetch(`${BACKEND}/analyze-${type}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });
        if (!resp.ok) return;
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
    const s = result.authenticity_score ?? 0;
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
        type: result.type,
        risk_level: result.risk_level,
        score: result.authenticity_score,
        timestamp: Date.now(),
    });
    // Keep last 50
    await chrome.storage.local.set({ history: history.slice(0, 50) });

    // Update badge on the extension icon
    const counts = { Low: 0, Medium: 0, High: 0, ...{} };
    history.slice(0, 10).forEach(h => counts[h.risk_level]++);
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
    }
    if (msg.type === "SHOW_DETAIL") {
        chrome.storage.session.set({ activeDetail: msg });
        chrome.action.openPopup?.();
    }

    // DO_SCAN_FILE: version that handles multi-part form data uploads
    if (msg.type === "DO_SCAN_FILE") {
        const { endpoint, filename, base64 } = msg.payload;

        // Convert base64 to Blob
        fetch(base64)
            .then(res => res.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append("file", blob, filename);

                return fetch(`${BACKEND}/${endpoint}`, {
                    method: "POST",
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
        fetch(`${BACKEND}/${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
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
        const resp = await fetch(`${BACKEND}/health`);
        const data = await resp.json();
        await chrome.storage.local.set({ backendOnline: data.status === "ok" });
    } catch (_) {
        await chrome.storage.local.set({ backendOnline: false });
    }
}

// Check every 30 seconds
checkBackend();
setInterval(checkBackend, 30_000);
