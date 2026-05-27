/**
 * OpenSeek — Content Script (Floating Overlay Edition)
 *
 * Works on ALL websites including Instagram Reels, TikTok, YouTube Shorts.
 * Uses position:fixed overlays — never clipped by overflow:hidden.
 * On social media: captures canvas frames (avoids auth-gated CDN URL issues).
 */

// Sync the token from the dashboard localStorage to chrome extension storage
if (window.location.origin === "https://openseek-production.up.railway.app" || 
    window.location.origin.includes(".vercel.app") ||
    window.location.origin === "http://127.0.0.1:8000" || 
    window.location.origin === "http://localhost:8000") {
    
    function syncToken() {
        const syncEl = document.getElementById("openseek-sync-data");
        let token = null;
        let backendUrl = "https://openseek-production.up.railway.app";
        
        if (window.location.origin === "http://127.0.0.1:8000" || window.location.origin === "http://localhost:8000") {
            backendUrl = window.location.origin;
        }

        if (syncEl) {
            token = syncEl.getAttribute("data-token") || null;
            backendUrl = syncEl.getAttribute("data-backend") || backendUrl;
        } else {
            try {
                token = localStorage.getItem("openseek_token") || null;
            } catch (_) {}
        }

        chrome.storage.local.get(["openseek_token", "openseek_backend_url"], (res) => {
            if (token !== res.openseek_token || backendUrl !== res.openseek_backend_url) {
                if (token) {
                    chrome.storage.local.set({ 
                        openseek_token: token,
                        openseek_backend_url: backendUrl
                    });
                } else {
                    chrome.storage.local.remove(["openseek_token", "openseek_backend_url"]);
                }
            }
        });
    }
    
    // Sync immediately on load
    syncToken();
    // Check every second for login/logout changes
    setInterval(syncToken, 1000);
}
const MIN_SIZE = 80;
const MAX_CONCURRENT = 3;

/* ─── State ─────────────────────────────────────────────────────────────── */
const seen = new WeakMap();   // el → { btn, badge }
const results = new Map();       // url → result
const videoUrls = new WeakMap();   // video el → real CDN url
let scanning = 0;

/* ─── Resolve real URL for any media element ────────────────────────────── */
function getRealUrl(el) {
    // Images
    const src = el.src || el.currentSrc || el.getAttribute("src") || "";
    return (src && src.startsWith("http")) ? src : null;
}

/* ─── Overlay creation helper ───────────────────────────────────────────── */
function createOverlayEl(cls, html) {
    const el = document.createElement("div");
    el.className = cls;
    el.innerHTML = html;
    el.dataset.ds = "overlay";
    document.body.appendChild(el);
    return el;
}



/* ─── Send scan request via background service worker ───────────────────────── */
// Content scripts can't reliably fetch localhost — route through background.
function requestScan(payload) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({ type: "DO_SCAN", payload }, (resp) => {
            if (chrome.runtime.lastError) { reject(new Error(chrome.runtime.lastError.message)); return; }
            if (resp?.error) reject(new Error(resp.error));
            else resolve(resp?.data);
        });
    });
}

async function toBase64(url) {
    const resp = await fetch(url);
    const blob = await resp.blob();
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

/* ─── Scanning logic ────────────────────────────────────────────────────────── */
async function scan(el, fromContextMenu = false) {
    const state = seen.get(el);
    if (!state) return;
    if (scanning >= MAX_CONCURRENT) { showBadge(el, "error", "⚡ Busy—retry"); return; }

    const url = getRealUrl(el);
    if (!url) {
        showBadge(el, "error", "⚡ URL not found"); return;
    }
    if (results.has(url)) { renderResult(el, results.get(url)); return; }

    scanning++;
    showBadge(el, "scanning", '<span class="ds-spinner"></span> Scanning…');
    const type = "image";
    try {
        if (type === "image") {
            const base64 = await toBase64(url);

            // Convert base64 to Blob directly
            const blobResp = await fetch(base64);
            const blob = await blobResp.blob();

            const formData = new FormData();
            formData.append("file", blob, url.split("/").pop().split("?")[0] || "scan.jpg");

            const { openseek_token, openseek_backend_url = "https://openseek-production.up.railway.app" } = await chrome.storage.local.get(["openseek_token", "openseek_backend_url"]);
            const headers = {};
            if (openseek_token) {
                headers["Authorization"] = `Bearer ${openseek_token}`;
            }

            const apiResp = await fetch(`${openseek_backend_url}/detect-image`, {
                method: "POST",
                headers: headers,
                body: formData
            });

            if (!apiResp.ok) {
                const e = await apiResp.json().catch(() => ({ detail: apiResp.statusText }));
                const errMsg = e.detail || apiResp.statusText || "";
                const lowerMsg = errMsg.toLowerCase();

                if (lowerMsg.includes("insufficient credits") || lowerMsg.includes("credit limit")) {
                    showBadge(el, "error", "❌ Credit limit reached");
                } else if (apiResp.status === 401 ||
                           lowerMsg.includes("invalid session") ||
                           lowerMsg.includes("session expired") ||
                           lowerMsg.includes("unauthorized")) {
                    // Clear stale token — don't keep sending invalid credentials
                    chrome.storage.local.remove(["openseek_token"]);
                    // Show a clickable badge that opens the dashboard login page
                    const { openseek_backend_url: bUrl = "https://openseek-production.up.railway.app" } =
                        await chrome.storage.local.get("openseek_backend_url");
                    const loginUrl = bUrl.includes("railway.app") || bUrl.includes("localhost")
                        ? bUrl
                        : "https://openseek-production.up.railway.app";
                    showLoginBadge(el, loginUrl);
                } else {
                    showBadge(el, "error", `⚠️ ${errMsg.slice(0, 60)}`);
                }
                scanning--;
                return;
            }

            const data = await apiResp.json();

            const result = data;

            results.set(url, result);
            await chrome.runtime.sendMessage({ type: "RESULT", result, url, fromContextMenu }).catch(err => {
                console.warn("OpenSeek: Could not store result in history:", err.message);
            });
            renderResult(el, result);
            scanning--;
            return;
        } else {
            const data = await requestScan({
                endpoint: `analyze-${type}`,
                body: { url },
            });
            results.set(url, data);
            await chrome.runtime.sendMessage({ type: "RESULT", result: data, url }).catch(err => {
                console.warn("OpenSeek: Could not store result in history:", err.message);
            });
            renderResult(el, data);
        }
    } catch (e) {
        const raw = e.message || "";
        const msg = raw.includes("Extension context invalidated")
            ? "🔄 Refresh Page"
            : raw.includes("fetch") || raw.includes("Could not establish")
                ? "Backend offline" : raw.slice(0, 40);
        chrome.storage.local.get(["openseek_backend_url"], (res) => {
            const bUrl = res.openseek_backend_url || "https://openseek-production.up.railway.app";
            showBadge(el, "error", `⚡ [Backend: ${bUrl}] ${msg}`);
        });
        scanning--;
    }
}

/* ─── Badge helpers ─────────────────────────────────────────────────────── */
function getBadgePos(el) {
    const r = el.getBoundingClientRect();
    return {
        top_badge: r.bottom - 36,
        left_badge: r.left + 6,
        top_btn: r.top + 6,
        left_btn: r.right - 80,       // ~80px wide button
    };
}

function showBadge(el, kind, html) {
    const state = seen.get(el);
    if (!state) return;
    state.badge?.remove();

    const cls = kind === "scanning"
        ? "ds-result-overlay ds-result-scanning"
        : "ds-result-overlay ds-result-medium";

    const badge = createOverlayEl(cls, html);
    state.badge = badge;

    const p = getBadgePos(el);
    badge.style.top = `${p.top_badge}px`;
    badge.style.left = `${p.left_badge}px`;
}

function showLoginBadge(el, dashboardUrl) {
    const state = seen.get(el);
    if (!state) return;
    state.badge?.remove();

    const badge = createOverlayEl("ds-result-overlay ds-result-medium",
        `<span style="cursor:pointer; text-decoration:underline;" title="Click to log in to OpenSeek">
            🔑 Session expired — <strong>click to log in</strong>
        </span>`
    );
    badge.style.cursor = "pointer";
    badge.addEventListener("click", () => {
        chrome.tabs.create({ url: dashboardUrl });
    });
    state.badge = badge;

    const p = getBadgePos(el);
    badge.style.top = `${p.top_badge}px`;
    badge.style.left = `${p.left_badge}px`;
}

function renderResult(el, result) {
    const state = seen.get(el);
    if (!state) return;
    state.badge?.remove();

    const r = result.risk_level || "Low";
    const s = result.ai_probability ? Math.round(result.ai_probability * 100) : Math.round(result.authenticity_score ?? 0);
    const icon = { Low: "✅", Medium: "⚠️", High: "🔴", Uncertain: "❓" }[r] || "✅";
    const cls = { Low: "ds-result-low", Medium: "ds-result-medium", High: "ds-result-high", Uncertain: "ds-result-medium" }[r];

    // Fallbacks for older cached scans
    const cType = result.content_type || "Photograph";
    const pClass = result.predicted_class || (r === "High" ? "Deepfake_AI" : "Real");
    const displayClass = pClass.includes("AI") ? "AI" : pClass.replace("_", " ");

    const badge = createOverlayEl(
        `ds-result-overlay ${cls}`,
        `
        <div style="display:flex; flex-direction:column; gap:2px; text-align:left;">
            <div>${icon} <strong>Risk: ${r}</strong> (${s}%)</div>
            <div style="font-size:0.85em; opacity:0.9;">Detected: <em>${displayClass}</em></div>
        </div>
        `
    );
    state.badge = badge;

    const p = getBadgePos(el);
    badge.style.top = `${p.top_badge}px`;
    badge.style.left = `${p.left_badge}px`;

    if (r === "High") el.classList.add("ds-high-risk");
}

/* ─── Attach scan button ────────────────────────────────────────────────── */
function attach(el) {
    const currentUrl = getRealUrl(el);
    const state = seen.get(el);
    if (state) {
        if (state.lastUrl !== currentUrl) {
            state.badge?.remove();
            state.badge = null;
            state.lastUrl = currentUrl;
        }
        return;
    }

    if (el.tagName === "IMG") {
        const src = el.getAttribute("src") || el.src || "";
        if (!src || src.startsWith("data:")) return;
    }

    // Only skip if we KNOW it's tiny (not just not-yet-rendered)
    const rect = el.getBoundingClientRect();
    const w = rect.width || el.offsetWidth || el.clientWidth;
    if (w > 0 && w < MIN_SIZE) return;

    seen.set(el, { btn: null, badge: null, lastUrl: currentUrl });

    const updatePos = () => {
        const r = el.getBoundingClientRect();
        if (!r.width) return;
        const b = seen.get(el)?.badge;
        if (b) {
            b.style.top = `${r.bottom - 36}px`;
            b.style.left = `${r.left + 6}px`;
        }
    };

    // Button removed per user request

    document.addEventListener("scroll", updatePos, { passive: true, capture: true });
    window.addEventListener("resize", updatePos, { passive: true });
    updatePos();
}

/* ─── MutationObserver ──────────────────────────────────────────────────── */
function processNode(node) {
    if (!(node instanceof Element)) return;
    if (node.tagName === "IMG") attach(node);
    node.querySelectorAll("img").forEach(attach);
}

new MutationObserver(muts => {
    muts.forEach(m => {
        m.addedNodes.forEach(processNode);
        if (m.type === "attributes") processNode(m.target);

        m.removedNodes.forEach(node => {
            if (!(node instanceof Element)) return;
            const imgs = node.tagName === "IMG" ? [node] : [];
            node.querySelectorAll("img").forEach(i => imgs.push(i));
            imgs.forEach(img => {
                const state = seen.get(img);
                if (state?.badge) {
                    state.badge.remove();
                    state.badge = null;
                }
            });
        });
    });
}).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src"],
});

document.querySelectorAll("img").forEach(attach);

window.addEventListener("load", () => {
    document.querySelectorAll("img").forEach(attach);
});

/* ─── Context menu trigger ───────────────────────────────────────────────── */
chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "SCAN_CONTEXT") {
        const el = [...document.querySelectorAll("img")]
            .find(e => getRealUrl(e) === msg.url || e.src === msg.url);
        if (el) scan(el, /* fromContextMenu= */ true);
    }
});
