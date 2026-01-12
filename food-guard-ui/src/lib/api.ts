import axios from 'axios';
import type { 
    ProcessRequest, 
    ProcessResponse,
    JobStatusResponse,
    JobLogsResponse
} from './types';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || '',
    timeout: 300000, // 5 minutes for heavy models to load on first request
    headers: { 'Content-Type': 'application/json' }
});

// =============================================================================
// Pipeline API
// =============================================================================

export const pipelineApi = {
    /**
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
