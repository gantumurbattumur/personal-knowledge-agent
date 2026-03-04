/**
 * ragClient.ts — HTTP client for the PKA daemon API.
 */

export interface QueryRequest {
    query: string;
    workspace_hint?: string;
    clipboard_context?: string;
    active_file?: string;
    active_code_selection?: string;
}

export interface QueryResponse {
    answer: string;
    sources: string[];
}

export interface HealthResponse {
    status: string;
    workspaces_loaded: number;
}

export class RagClient {
    constructor(
        private baseUrl: string = "http://localhost:8741",
        private token: string = ""
    ) { }

    private headers(): Record<string, string> {
        const h: Record<string, string> = {
            "Content-Type": "application/json",
        };
        if (this.token) {
            h["X-RAG-Token"] = this.token;
        }
        return h;
    }

    async query(req: QueryRequest): Promise<QueryResponse> {
        const resp = await fetch(`${this.baseUrl}/query`, {
            method: "POST",
            headers: this.headers(),
            body: JSON.stringify(req),
        });

        if (!resp.ok) {
            const body: unknown = await resp.json().catch(() => ({}));
            const detail =
                typeof body === "object" && body !== null && "detail" in body
                    ? (body as { detail?: unknown }).detail
                    : undefined;
            if (typeof detail === "object" && detail !== null) {
                const detailError =
                    "error" in detail && typeof (detail as { error?: unknown }).error === "string"
                        ? (detail as { error: string }).error
                        : undefined;
                const err = new Error(detailError || `HTTP ${resp.status}`);
                (err as any).detail = detail;
                throw err;
            }
            throw new Error(typeof detail === "string" ? detail : `HTTP ${resp.status}`);
        }

        return (await resp.json()) as QueryResponse;
    }

    async health(): Promise<HealthResponse> {
        const resp = await fetch(`${this.baseUrl}/health`);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }
        return (await resp.json()) as HealthResponse;
    }
}
