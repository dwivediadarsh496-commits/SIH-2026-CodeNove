import type { HealthResponse, ModelInfoResponse, SuperResolutionResult } from "@/types/api";

// Always use relative paths — Next.js rewrites proxy to Railway backend
export const API_BASE_URL = "";

// Direct Railway URL — used as primary for inference (avoids Vercel proxy timeout on large uploads)
export const RENDER_DIRECT_URL = "https://deep-learning-based-super-resolution-mapping-srm-production.up.railway.app";

type ApiErrorPayload = {
  detail?: string;
  message?: string;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const urls = API_BASE_URL
    ? [`${API_BASE_URL}${path}`]
    : [`${path}`, `${RENDER_DIRECT_URL}${path}`];

  let lastError: Error | null = null;

  for (const url of urls) {
    try {
      const response = await fetch(url, {
        ...init,
        headers: {
          ...(init?.headers || {}),
        },
      });

      if (!response.ok) {
        let message = `Request failed with ${response.status}`;
        try {
          const payload = (await response.json()) as ApiErrorPayload;
          message = payload.detail || payload.message || message;
        } catch {
          // The backend may return plain text for infrastructure errors.
        }
        throw new Error(message);
      }

      return (await response.json()) as T;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      // Try next endpoint in fallback list
    }
  }

  throw lastError || new Error("Failed to reach API server");
}

/**
 * Directly calls Railway backend for inference — bypasses Vercel proxy.
 * This avoids Vercel's 60s function timeout for large-file CPU inference.
 */
async function requestInferenceDirect<T>(path: string, init?: RequestInit): Promise<T> {
  // Try direct Railway URL first (fastest), then fall back to proxy
  const urls = [
    `${RENDER_DIRECT_URL}${path}`,
    ...(API_BASE_URL ? [`${API_BASE_URL}${path}`] : [`${path}`]),
  ];

  let lastError: Error | null = null;

  for (const url of urls) {
    try {
      const response = await fetch(url, {
        ...init,
        headers: {
          ...(init?.headers || {}),
        },
      });

      if (!response.ok) {
        let message = `Inference failed (${response.status})`;
        try {
          const payload = (await response.json()) as ApiErrorPayload;
          message = payload.detail || payload.message || message;
        } catch {
          // ignore
        }
        throw new Error(message);
      }

      return (await response.json()) as T;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }

  throw lastError || new Error("Backend server unreachable. Please try again in a moment.");
}


export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  return `${RENDER_DIRECT_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health", {
    cache: "no-store",
  });
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return requestJson<ModelInfoResponse>("/api/model-info", {
    cache: "no-store",
  });
}

export async function runSuperResolution(file: File, reference?: File | null): Promise<SuperResolutionResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (reference) {
    formData.append("reference", reference);
  }

  // Use direct Railway URL for inference to avoid Vercel proxy timeout
  return requestInferenceDirect<SuperResolutionResult>("/api/super-resolution", {
    method: "POST",
    body: formData,
  });
}
