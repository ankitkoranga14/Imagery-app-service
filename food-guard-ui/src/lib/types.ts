// =============================================================================
// Validation Types (Primary - Guardrail Microservice)
// =============================================================================

export interface ValidateRequest {
    prompt: string;
    image_bytes: string;  // Base64 encoded
    async_mode?: boolean;
}

export interface ValidateResponse {
    status: 'PASS' | 'BLOCK';
    failure_reason?: string;
    scores: Record<string, number>;
    latency_ms: number;
    validation_id?: string;
}

export interface ValidationJobStatus {
    job_id: string;
    status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
    result?: ValidateResponse;
    error?: string;
    created_at: string;
    completed_at?: string;
}

export interface ValidationLogEntry {
    id: string;
    status: string;
    failure_reason?: string;
    failure_level?: string;
    scores: Record<string, number>;
    latency_ms: number;
    latency_breakdown: Record<string, number>;
    cache_hit: boolean;
    parallel_execution: boolean;
    created_at: string;
}

export interface ValidationLogsResponse {
    logs: ValidationLogEntry[];
    total: number;
    page: number;
    page_size: number;
}

export interface ValidationStats {
    total_validations: number;
    pass_count: number;
    block_count: number;
    pass_rate: number;
    block_rate: number;
    average_latency_ms: number;
    cache_hit_count: number;
    cache_hit_rate: number;
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

// =============================================================================
// Legacy Pipeline Types (Deprecated - will be removed)
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
// Validation Stage Definitions (New simplified pipeline)
// =============================================================================

export const VALIDATION_STAGES = [
    { key: 'cache', label: 'L0: Cache', description: 'Check Redis cache (instant)' },
    { key: 'text', label: 'L1: Text', description: 'Injection/policy/food domain check' },
    { key: 'physics', label: 'L2: Physics', description: 'Brightness/blur/contrast (OpenCV)' },
    { key: 'geometry', label: 'L3: Geometry', description: 'Food detection (YOLO)' },
    { key: 'context', label: 'L4: Context', description: 'Food/NSFW classification (CLIP)' },
] as const;

export type ValidationStageKey = typeof VALIDATION_STAGES[number]['key'];

// =============================================================================
// Legacy Pipeline Stage Definitions (Deprecated)
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
