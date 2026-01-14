import axios from 'axios';
import type {
    ProcessRequest,
    ProcessResponse,
    JobStatusResponse,
    JobLogsResponse,
    ValidateRequest,
    ValidateResponse,
    ValidationJobStatus,
    ValidationLogsResponse,
    ValidationStats
} from './types';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || '',
    timeout: 300000, // 5 minutes for heavy models to load on first request
    headers: { 'Content-Type': 'application/json' }
});

// =============================================================================
// Validation API (Primary - Guardrail Microservice)
// =============================================================================

export const validationApi = {
    /**
     * Validate an image (synchronous - waits for result)
     */
    validate: (request: ValidateRequest) =>
        api.post<ValidateResponse>('/api/v1/validate', request),

    /**
     * Validate an image (asynchronous - returns job ID)
     */
    validateAsync: (request: Omit<ValidateRequest, 'async_mode'>) =>
        api.post<{ job_id: string; status: string; poll_url: string }>(
            '/api/v1/validate',
            { ...request, async_mode: true }
        ),

    /**
     * Get async validation job status
     */
    getJobStatus: (jobId: string) =>
        api.get<ValidationJobStatus>(`/api/v1/validate/jobs/${jobId}`),

    /**
     * Get validation logs (audit trail)
     */
    getLogs: (page: number = 1, pageSize: number = 20, status?: string) => {
        const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
        if (status) params.append('status', status);
        return api.get<ValidationLogsResponse>(`/api/v1/validate/logs?${params}`);
    },

    /**
     * Get validation statistics
     */
    getStats: () =>
        api.get<ValidationStats>('/api/v1/validate/stats'),
};

// =============================================================================
// Legacy Pipeline API (Deprecated - will be removed)
// =============================================================================

export const pipelineApi = {
    /**
     * @deprecated Use validationApi.validate instead
     * Submit an image for full pipeline processing
     */
    process: (request: ProcessRequest) =>
        api.post<ProcessResponse>('/api/v1/process', request),

    /**
     * Get job status by ID
     */
    getStatus: (jobId: string) =>
        api.get<JobStatusResponse>(`/api/v1/status/${jobId}`),

    /**
     * Get lightweight progress (optimized for polling)
     */
    getProgress: (jobId: string) =>
        api.get<JobStatusResponse>(`/api/v1/status/${jobId}/progress`),

    /**
     * Get structured logs for a job
     */
    getLogs: (jobId: string) =>
        api.get<JobLogsResponse>(`/api/v1/status/${jobId}/logs`),
};

// =============================================================================
// Health API
// =============================================================================

export const healthApi = {
    check: () => api.get('/health'),
    ready: () => api.get('/ready'),
};

export default api;
