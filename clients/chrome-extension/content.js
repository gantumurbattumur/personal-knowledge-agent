// content.js — Reads selected text from the active page

// Listen for messages from the popup or background
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "getSelection") {
        const selection = window.getSelection()?.toString() || "";
        sendResponse({ selection });
    }
    return false;
});
