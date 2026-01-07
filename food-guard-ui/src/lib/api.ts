import axios from 'axios';
import type { GuardrailRequest, GuardrailResult } from './types';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || '',
    timeout: 300000, // 5 minutes for heavy models to load on first request
    headers: { 'Content-Type': 'application/json' }
});

export const guardrailApi = {
    validate: (request: GuardrailRequest) =>
        api.post<GuardrailResult>('/v1/guardrail/validate', request),
};
