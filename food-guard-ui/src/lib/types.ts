// =============================================================================
// Pipeline Types
// =============================================================================

export type JobStatus =
    | 'PENDING'
    | 'VALIDATING'
    | 'VALIDATED'
    | 'BLOCKED'
    | 'PROCESSING'
    | 'COMPLETED'
    | 'FAILED'
    | 'CANCELLED';

export type StageStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'skipped';

export interface StageProgress {
    stage: string;
    status: StageStatus;
    duration_ms?: number;
    started_at?: string;
    completed_at?: string;
    error?: string;
}

export interface StorageUrls {
    original?: string;
    transparent?: string;
    upscaled?: string;
    generated?: string;
    final?: string;
}

export interface JobStatusResponse {
    id: string;
    status: JobStatus;
    current_stage?: string;
    stages: StageProgress[];
    storage_urls?: StorageUrls;
    guardrail?: GuardrailResult;
    error?: {
        message: string;
        stage: string;
    };
    cost_usd: number;
    processing_time_ms: number;
    created_at: string;
    completed_at?: string;
}

export interface ProcessRequest {
    prompt: string;
    image_base64: string;
    options?: {
        simulate?: boolean;
        scale?: number;
        placement?: Record<string, unknown>;
    };
}

export interface ValidationTrace {
    levels_executed: string[];
    levels_passed: string[];
    levels_failed: string[];
    levels_skipped: string[];
    timings: Record<string, number>;
    detected_foods?: Array<{
        class_id: number;
        class_name: string;
        confidence: number;
        bbox?: number[];
    }>;
}

export interface GuardrailResult {
    status: string;
    reasons?: string[];
    scores?: Record<string, number>;
    metadata?: {
        processing_time_ms: number;
        cache_hit: boolean;
        validation_trace?: ValidationTrace;
    };
}

export interface ProcessResponse {
    job_id: string;
    status: string;
    message: string;
    guardrail?: GuardrailResult;
    estimated_cost_usd: number;
    estimated_time_seconds: number;
}

// =============================================================================
// Log Types
// =============================================================================

export interface LogEntry {
    timestamp: string;
    level: 'info' | 'warning' | 'error' | 'debug';
    event: string;
    stage?: string;
    job_id?: string;
    duration_ms?: number;
    cost_usd?: number;
    vram_used_gb?: number;
    error?: string;
}

export interface JobLogsResponse {
    job_id: string;
    logs: LogEntry[];
    status: JobStatus;
}

// =============================================================================
// Pipeline Stage Definitions
// =============================================================================

export const PIPELINE_STAGES = [
    { key: 'upload', label: 'Upload', description: 'Image received' },
    { key: 'guardrail', label: 'Quality Check', description: 'Content validation (CLIP + NLP)' },
    { key: 'rembg', label: 'Background Removal', description: 'Remove background (U2-Net)' },
    { key: 'nano_banana', label: 'Smart Placement', description: 'Apply design/theme ($0.08)' },
    { key: 'generation', label: 'Generate Versions', description: 'Create 2 hero versions' },
    { key: 'realesrgan', label: 'Auto Enhancement', description: '4K upscaling (RealESRGAN)' },
] as const;

export type PipelineStageKey = typeof PIPELINE_STAGES[number]['key'];
