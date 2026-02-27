/**
 * DeepShield — Popup Script
 * Loads history, shows stats, handles button actions.
 */

const BACKEND = "http://localhost:8000";

/* ─── DOM refs ─────────────────────────────────────────────────────────────── */
const dot = document.getElementById("dot");
const statusText = document.getElementById("statusText");
const offlineBanner = document.getElementById("offlineBanner");
const historyList = document.getElementById("historyList");
const emptyState = document.getElementById("emptyState");
const countLow = document.getElementById("countLow");
const countMedium = document.getElementById("countMedium");
const countHigh = document.getElementById("countHigh");
const clearBtn = document.getElementById("clearBtn");
const docsBtn = document.getElementById("docsBtn");

/* ─── Helpers ──────────────────────────────────────────────────────────────── */

function shortUrl(url) {
    try {
        const u = new URL(url);
        return (u.hostname + u.pathname).slice(0, 38);
    } catch (_) {
        return url.slice(0, 38);
    }
}

function timeAgo(ts) {
    const diff = Date.now() - ts;
    if (diff < 60_000) return "just now";
    if (diff < 3600_000) return `${Math.floor(diff / 60_000)}m ago`;
    return `${Math.floor(diff / 3600_000)}h ago`;
}

function typeEmoji(t) {
    return { image: "🖼", video: "🎥", audio: "🔊" }[t] || "🖼";
}

/* ─── Render backend status ────────────────────────────────────────────────── */

function setStatus(online) {
    if (online) {
        dot.className = "dot online";
        statusText.textContent = "Online";
        offlineBanner.classList.remove("visible");
    } else {
        dot.className = "dot offline";
        statusText.textContent = "Offline";
        offlineBanner.classList.add("visible");
    }
}

/* ─── Render history ───────────────────────────────────────────────────────── */

function renderHistory(history) {
    // Stats
    const counts = { Low: 0, Medium: 0, High: 0 };
    history.forEach(h => counts[h.risk_level] = (counts[h.risk_level] || 0) + 1);
    countLow.textContent = counts.Low;
    countMedium.textContent = counts.Medium;
    countHigh.textContent = counts.High;

    if (!history.length) {
        emptyState.style.display = "block";
        return;
    }
    emptyState.style.display = "none";

    // Remove old items (keep structure)
    historyList.querySelectorAll(".history-item").forEach(n => n.remove());

    history.slice(0, 20).forEach(h => {
        const item = document.createElement("div");
        item.className = "history-item";

        let cType = h.content_type || "Photograph";
        let pClass = h.predicted_class || (h.risk_level === "High" ? "AI" : "Real");
        let scoreStr = h.ai_probability ? Math.round(h.ai_probability * 100) : Math.round(h.score || 0);

        item.innerHTML = `
      <div style="display:flex; justify-content: space-between; align-items: center; width:100%; margin-bottom: 4px;">    
        <span class="risk-chip chip-${h.risk_level}">${h.risk_level} (${scoreStr}%)</span>
        <span class="history-type">${typeEmoji(h.type)} ${h.type}</span>
      </div>
      <div style="font-size:0.85em; color:#fff; opacity:0.8; text-align:left; margin-bottom: 2px;">
         [${cType}] → <strong>${pClass.replace("_", " ")}</strong>
      </div>
      <div style="display:flex; justify-content:space-between; width:100%; font-size:0.8em; opacity: 0.6;">
        <span class="history-url" title="${h.url}">${shortUrl(h.url)}</span>
      </div>
    `;
        historyList.appendChild(item);
    });
}

/* ─── Load data from storage ───────────────────────────────────────────────── */

async function loadData() {
    const { history = [], backendOnline = false } =
        await chrome.storage.local.get(["history", "backendOnline"]);

    setStatus(backendOnline);
    renderHistory(history);
}

/* ─── Button actions ───────────────────────────────────────────────────────── */

clearBtn.addEventListener("click", async () => {
    await chrome.storage.local.set({ history: [] });
    chrome.action.setBadgeText({ text: "" });
    renderHistory([]);
});

docsBtn.addEventListener("click", () => {
    chrome.tabs.create({ url: `${BACKEND}/docs` });
});

/* ─── Init ─────────────────────────────────────────────────────────────────── */

loadData();

// Live refresh if popup stays open
setInterval(loadData, 5_000);
