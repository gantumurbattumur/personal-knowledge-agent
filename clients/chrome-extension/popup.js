// popup.js — Handles the RAG Assistant popup UI

const queryInput = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const queryForm = document.getElementById("query-form");
const contextBar = document.getElementById("context-bar");
const answerDiv = document.getElementById("answer");
const sourcesDiv = document.getElementById("sources");
const errorDiv = document.getElementById("error");
const statusDot = document.getElementById("status-dot");
const tokenInput = document.getElementById("token-input");

let clipboardContext = "";

// -- Lifecycle ---------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
    // Check daemon health
    await checkDaemonHealth();

    // Load stored context from right-click
    const stored = await chrome.storage.session.get([
        "clipboard_context",
        "page_url",
    ]);
    clipboardContext = stored.clipboard_context || "";

    if (clipboardContext) {
        const preview =
            clipboardContext.length > 120
                ? clipboardContext.slice(0, 120) + "…"
                : clipboardContext;
        contextBar.textContent = `Context: "${preview}"`;
    }

    // Clear badge
    chrome.action.setBadgeText({ text: "" });

    // Load saved token
    const { rag_token } = await chrome.storage.local.get("rag_token");
    if (rag_token) tokenInput.value = rag_token;
});

// -- Token management --------------------------------------------------------

tokenInput.addEventListener("change", async () => {
    const token = tokenInput.value.trim();
    await chrome.storage.local.set({ rag_token: token });
});

// -- Health check ------------------------------------------------------------

async function checkDaemonHealth() {
    try {
        const result = await chrome.runtime.sendMessage({ type: "health" });
        if (result?.status === "ok") {
            statusDot.classList.add("online");
            statusDot.classList.remove("offline");
            statusDot.title = `Online — ${result.workspaces_loaded} workspace(s)`;
        } else {
            statusDot.classList.add("offline");
            statusDot.classList.remove("online");
            statusDot.title = "Daemon offline";
        }
    } catch {
        statusDot.classList.add("offline");
        statusDot.title = "Cannot reach daemon";
    }
}

// -- Query submission --------------------------------------------------------

queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    submitBtn.disabled = true;
    answerDiv.textContent = "";
    sourcesDiv.innerHTML = "";
    errorDiv.textContent = "";
    answerDiv.innerHTML = '<div class="loading">Thinking…</div>';

    const payload = {
        query,
        clipboard_context: clipboardContext || null,
        workspace_hint: "drive://default", // Browser → Drive context by default
    };

    try {
        const result = await chrome.runtime.sendMessage({
            type: "query",
            payload,
        });

        if (result?.error) {
            displayError(result.error);
            return;
        }

        displayAnswer(result.answer, result.sources || []);
    } catch (err) {
        displayError(err.message || "Unknown error");
    } finally {
        submitBtn.disabled = false;
    }
});

// -- Display helpers ---------------------------------------------------------

function displayAnswer(answer, sources) {
    answerDiv.textContent = answer;

    if (sources.length > 0) {
        sourcesDiv.innerHTML =
            "<strong>Sources:</strong>" +
            sources
                .map((s) => `<div class="source-item">${escapeHtml(s)}</div>`)
                .join("");
    }
}

function displayError(message) {
    answerDiv.textContent = "";
    errorDiv.textContent = message;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
