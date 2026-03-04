/**
 * panel.ts — WebviewPanel for displaying RAG answers in VSCode.
 */
import * as vscode from "vscode";
import type { QueryResponse } from "./ragClient";

export class ResultPanel {
    public static currentPanel: ResultPanel | undefined;
    private static readonly viewType = "ragResult";

    private constructor(
        private readonly panel: vscode.WebviewPanel,
        private readonly extensionUri: vscode.Uri
    ) {
        this.panel.onDidDispose(() => {
            ResultPanel.currentPanel = undefined;
        });
    }

    public static show(extensionUri: vscode.Uri, result: QueryResponse) {
        const column = vscode.ViewColumn.Beside;

        if (ResultPanel.currentPanel) {
            ResultPanel.currentPanel.panel.reveal(column);
            ResultPanel.currentPanel.update(result);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            ResultPanel.viewType,
            "RAG Answer",
            column,
            { enableScripts: false }
        );

        ResultPanel.currentPanel = new ResultPanel(panel, extensionUri);
        ResultPanel.currentPanel.update(result);
    }

    private update(result: QueryResponse) {
        this.panel.webview.html = this.getHtml(result);
    }

    private escapeHtml(text: string): string {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    private getHtml(result: QueryResponse): string {
        const answer = this.escapeHtml(result.answer);
        const sources = result.sources
            .map((s) => `<li>${this.escapeHtml(s)}</li>`)
            .join("\n");

        return /* html */ `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Answer</title>
  <style>
    body {
      font-family: var(--vscode-font-family, sans-serif);
      font-size: var(--vscode-font-size, 14px);
      color: var(--vscode-foreground);
      background: var(--vscode-editor-background);
      padding: 16px;
      line-height: 1.6;
    }
    h2 { margin-top: 0; font-size: 1.2em; }
    .answer { white-space: pre-wrap; margin-bottom: 16px; }
    h3 { font-size: 1em; margin-bottom: 8px; }
    ul { padding-left: 20px; }
    li {
      font-size: 0.85em;
      color: var(--vscode-descriptionForeground);
      margin-bottom: 4px;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <h2>Answer</h2>
  <div class="answer">${answer}</div>
  ${sources
                ? `<h3>Sources</h3><ul>${sources}</ul>`
                : ""
            }
</body>
</html>`;
    }
}
