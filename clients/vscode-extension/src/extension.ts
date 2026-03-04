import * as vscode from "vscode";
import { RagClient } from "./ragClient";
import { ResultPanel } from "./panel";

let client: RagClient;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration("rag");
    const daemonUrl = config.get<string>("daemonUrl", "http://localhost:8741");
    const token = config.get<string>("token", "");
    client = new RagClient(daemonUrl, token);

    // -- Main command: Ask about workspace ---

    const askCmd = vscode.commands.registerCommand(
        "rag.askWorkspace",
        async () => {
            const editor = vscode.window.activeTextEditor;
            const workspaceRoot =
                vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
            const activeFile = editor?.document.uri.fsPath;
            const selectedCode =
                editor?.document.getText(editor.selection) || undefined;

            // Prompt user for query
            const query = await vscode.window.showInputBox({
                prompt: "Ask your workspace...",
                placeHolder: "How does the auth middleware work?",
            });
            if (!query) return;

            // Show progress notification
            await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: "RAG Assistant",
                    cancellable: false,
                },
                async (progress) => {
                    progress.report({ message: "Querying..." });

                    try {
                        const result = await client.query({
                            query,
                            workspace_hint: workspaceRoot,
                            active_file: activeFile,
                            active_code_selection: selectedCode,
                        });

                        ResultPanel.show(context.extensionUri, result);
                    } catch (err: unknown) {
                        const message =
                            err instanceof Error ? err.message : String(err);

                        if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
                            const action = await vscode.window.showErrorMessage(
                                "RAG daemon is not running. Start it with: rag daemon start",
                                "Open Terminal"
                            );
                            if (action === "Open Terminal") {
                                const terminal = vscode.window.createTerminal("RAG Daemon");
                                terminal.sendText("rag daemon start");
                                terminal.show();
                            }
                        } else if (message.includes("no_workspace_found") || message.includes("workspace_not_indexed")) {
                            const action = await vscode.window.showWarningMessage(
                                `Workspace not indexed. Run 'rag init-workspace' in this directory.`,
                                "Open Terminal"
                            );
                            if (action === "Open Terminal") {
                                const terminal = vscode.window.createTerminal("RAG Init");
                                terminal.sendText(
                                    `rag init-workspace "${workspaceRoot || "."}"`
                                );
                                terminal.show();
                            }
                        } else {
                            vscode.window.showErrorMessage(`RAG query failed: ${message}`);
                        }
                    }
                }
            );
        }
    );

    // -- Health check command ---

    const healthCmd = vscode.commands.registerCommand(
        "rag.health",
        async () => {
            try {
                const health = await client.health();
                vscode.window.showInformationMessage(
                    `RAG daemon: ${health.status} — ${health.workspaces_loaded} workspace(s) loaded`
                );
            } catch {
                vscode.window.showErrorMessage(
                    "RAG daemon is not running. Start it with: rag daemon start"
                );
            }
        }
    );

    // -- React to config changes ---

    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration("rag")) {
                const cfg = vscode.workspace.getConfiguration("rag");
                client = new RagClient(
                    cfg.get<string>("daemonUrl", "http://localhost:8741"),
                    cfg.get<string>("token", "")
                );
            }
        })
    );

    context.subscriptions.push(askCmd, healthCmd);
}

export function deactivate() { }
