export interface GuardrailRequest {
    prompt: string;
    image_bytes?: string | null;
}

export interface GuardrailResult {
    status: 'PASS' | 'BLOCK';
    reasons: string[];
    scores: Record<string, number>;
    metadata: {
        food_type?: string;
        cache_hit?: boolean;
        processing_time_ms: number;
    };
}

export interface TestHistory {
    id: string;
    request: GuardrailRequest;
    result: GuardrailResult;
    timestamp: number;
}
