// background.js — Service worker for RAG Assistant Chrome Extension

const DAEMON_URL = "http://localhost:8741";

// Create context menu on install
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "ask-rag",
        title: "Ask RAG Assistant",
        contexts: ["selection", "page"],
    });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId !== "ask-rag") return;

    const selectedText = info.selectionText || "";
    const pageUrl = tab?.url || "";

    // Store context for popup to access
    await chrome.storage.session.set({
        clipboard_context: selectedText,
        page_url: pageUrl,
    });

    // Open popup (note: chrome.action.openPopup() may not be available in all contexts)
    try {
        await chrome.action.openPopup();
    } catch {
        // Fallback: badge the extension icon to indicate context is ready
        chrome.action.setBadgeText({ text: "!" });
        chrome.action.setBadgeBackgroundColor({ color: "#4A90D9" });
    }
});

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "query") {
        handleQuery(message.payload)
            .then(sendResponse)
            .catch((err) => sendResponse({ error: err.message }));
        return true; // Keep message channel open for async response
    }

    if (message.type === "health") {
        checkHealth()
            .then(sendResponse)
            .catch((err) => sendResponse({ error: err.message }));
        return true;
    }
});

async function getToken() {
    const result = await chrome.storage.local.get("rag_token");
    return result.rag_token || "";
}

async function handleQuery(payload) {
    const token = await getToken();

    try {
        const response = await fetch(`${DAEMON_URL}/query`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-RAG-Token": token,
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(
                errorData.detail?.hint ||
                errorData.detail?.error ||
                `HTTP ${response.status}`
            );
        }

        return await response.json();
    } catch (err) {
        if (err.message === "Failed to fetch" || err.name === "TypeError") {
            throw new Error(
                "RAG Assistant is not running. Start it with: rag daemon start"
            );
        }
        throw err;
    }
}

async function checkHealth() {
    try {
        const response = await fetch(`${DAEMON_URL}/health`, { method: "GET" });
        return await response.json();
    } catch {
        return { status: "offline" };
    }
}
