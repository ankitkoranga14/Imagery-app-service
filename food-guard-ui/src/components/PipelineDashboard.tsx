import { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
    Upload,
    Shield,
    Eraser,
    Maximize2,
    Sparkles,
    CheckCircle2,
    XCircle,
    Clock,
    Loader2,
    DollarSign,
    Activity,
    Image as ImageIcon,
    RefreshCw,
    Terminal,
    ChevronDown,
    ChevronUp,
    AlertTriangle,
    Info,
    Zap
} from 'lucide-react';
import { pipelineApi } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import type { JobStatusResponse, StageProgress, LogEntry } from '@/lib/types';

// =============================================================================
// Pipeline Stage Configuration (matching Imagery Board PDF flow)
// =============================================================================

const PIPELINE_STAGES = [
    { key: 'upload', label: 'Upload', description: 'Image received' },
    { key: 'guardrail', label: 'Quality Check', description: 'Content validation (CLIP + NLP)' },
    { key: 'rembg', label: 'Background Removal', description: 'Remove background (U2-Net)' },
    { key: 'nano_banana', label: 'Smart Placement', description: 'Apply design/theme ($0.08)' },
    { key: 'generation', label: 'Generate Versions', description: 'Create 2 hero versions' },
    { key: 'realesrgan', label: 'Auto Enhancement', description: '4K upscaling (RealESRGAN)' },
] as const;

const stageIcons: Record<string, React.ComponentType<{ className?: string }>> = {
    upload: Upload,
    guardrail: Shield,
    rembg: Eraser,
    nano_banana: Sparkles,
    generation: ImageIcon,
    realesrgan: Maximize2,
    finalize: CheckCircle2
};

// =============================================================================
// Pipeline Stepper Component (Visual Flow)
// =============================================================================

interface StepperProps {
    stages: StageProgress[];
    currentStage?: string;
    jobStatus: string;
}

function PipelineStepper({ stages }: StepperProps) {
    const getStageState = (stage: StageProgress) => {
        if (stage.status === 'completed') return 'completed';
        if (stage.status === 'in_progress') return 'active';
        if (stage.status === 'failed') return 'failed';
        return 'pending';
    };

    const getStageInfo = (key: string) => 
        PIPELINE_STAGES.find(s => s.key === key) || { label: key, description: '' };

    return (
        <div className="flex items-center justify-between w-full py-6 px-4">
            {stages.map((stage, index) => {
                const state = getStageState(stage);
                const Icon = stageIcons[stage.stage] || CheckCircle2;
                const isLast = index === stages.length - 1;
                const info = getStageInfo(stage.stage);

                return (
                    <div key={stage.stage} className="flex items-center flex-1">
                        {/* Step circle */}
                        <div className="flex flex-col items-center gap-2 min-w-[80px]">
                            <div
                                className={`
                                    relative w-12 h-12 rounded-full flex items-center justify-center
                                    transition-all duration-500 ease-out
                                    ${state === 'completed'
                                        ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30'
                                        : state === 'active'
                                            ? 'bg-amber-500 text-white shadow-lg shadow-amber-500/30 animate-pulse'
                                            : state === 'failed'
                                                ? 'bg-red-500 text-white shadow-lg shadow-red-500/30'
                                                : 'bg-zinc-800 text-zinc-500 border border-zinc-700'
                                    }
                                `}
                            >
                                {state === 'active' ? (
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                ) : state === 'failed' ? (
                                    <XCircle className="w-5 h-5" />
                                ) : (
                                    <Icon className="w-5 h-5" />
                                )}
                            </div>
                            <span className={`
                                text-xs font-medium tracking-wide text-center
                                ${state === 'completed' ? 'text-emerald-400'
                                    : state === 'active' ? 'text-amber-400'
                                        : state === 'failed' ? 'text-red-400'
                                            : 'text-zinc-500'
                                }
                            `}>
                                {info.label}
                            </span>
                            {stage.duration_ms && (
                                <span className="text-[10px] text-zinc-600 font-mono">
                                    {(stage.duration_ms / 1000).toFixed(2)}s
                                </span>
                            )}
                        </div>

                        {/* Connector line */}
                        {!isLast && (
                            <div className="flex-1 h-0.5 mx-2 relative overflow-hidden">
                                <div className="absolute inset-0 bg-zinc-800" />
                                <div
                                    className={`
                                        absolute inset-0 origin-left transition-transform duration-700 ease-out
                                        ${state === 'completed'
                                            ? 'bg-gradient-to-r from-emerald-500 to-emerald-400 scale-x-100'
                                            : 'scale-x-0'
                                        }
                                    `}
                                />
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

// =============================================================================
// Metric Cards Component
// =============================================================================

interface MetricCardsProps {
    status: JobStatusResponse | null;
    isProcessing: boolean;
}

function MetricCards({ status, isProcessing }: MetricCardsProps) {
    const [elapsedTime, setElapsedTime] = useState(0);

    // Live timer when processing
    useEffect(() => {
        if (!isProcessing || !status?.created_at) {
            setElapsedTime(status?.processing_time_ms || 0);
            return;
        }

        const startTime = new Date(status.created_at).getTime();
        const interval = setInterval(() => {
            setElapsedTime(Date.now() - startTime);
        }, 100);

        return () => clearInterval(interval);
    }, [isProcessing, status?.created_at, status?.processing_time_ms]);

    const formatTime = (ms: number) => {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        if (minutes > 0) {
            return `${minutes}m ${remainingSeconds}s`;
        }
        return `${(ms / 1000).toFixed(1)}s`;
    };

    const completedStages = status?.stages?.filter(s => s.status === 'completed').length || 0;
    const totalStages = status?.stages?.length || 6;

    return (
        <div className="grid grid-cols-4 gap-4">
            {/* Execution Time */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-blue-500/10">
                            <Clock className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">Time</p>
                            <p className={`text-xl font-mono font-bold ${isProcessing ? 'text-amber-400' : 'text-white'}`}>
                                {formatTime(elapsedTime)}
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Estimated Cost */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-emerald-500/10">
                            <DollarSign className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">Cost</p>
                            <p className="text-xl font-mono font-bold text-white">
                                ${(status?.cost_usd || 0.08).toFixed(2)}
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Progress */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-purple-500/10">
                            <Zap className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">Progress</p>
                            <p className="text-xl font-mono font-bold text-white">
                                {completedStages}/{totalStages}
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Status */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${status?.status === 'COMPLETED' ? 'bg-emerald-500/10' :
                                status?.status === 'FAILED' ? 'bg-red-500/10' :
                                    status?.status === 'BLOCKED' ? 'bg-orange-500/10' :
                                        'bg-amber-500/10'
                            }`}>
                            <Activity className={`w-5 h-5 ${status?.status === 'COMPLETED' ? 'text-emerald-400' :
                                    status?.status === 'FAILED' ? 'text-red-400' :
                                        status?.status === 'BLOCKED' ? 'text-orange-400' :
                                            'text-amber-400'
                                }`} />
                        </div>
                        <div>
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">Status</p>
                            <p className={`text-lg font-semibold ${status?.status === 'COMPLETED' ? 'text-emerald-400' :
                                    status?.status === 'FAILED' ? 'text-red-400' :
                                        status?.status === 'BLOCKED' ? 'text-orange-400' :
                                            'text-amber-400'
                                }`}>
                                {status?.status || 'Ready'}
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}

// =============================================================================
// Enhanced Logs Terminal Component
// =============================================================================

interface LogsTerminalProps {
    jobId: string | null;
    isPolling: boolean;
    stages: StageProgress[];
}

function LogsTerminal({ jobId, isPolling, stages }: LogsTerminalProps) {
    const terminalRef = useRef<HTMLDivElement>(null);
    const [isExpanded, setIsExpanded] = useState(true);
    const [localLogs, setLocalLogs] = useState<LogEntry[]>([]);

    const { data: logsData } = useQuery({
        queryKey: ['logs', jobId],
        queryFn: () => jobId ? pipelineApi.getLogs(jobId) : null,
        enabled: !!jobId && isPolling,
        refetchInterval: isPolling ? 1000 : false,
    });

    // Generate logs from stages when API logs are not available
    useEffect(() => {
        if (logsData?.data?.logs && logsData.data.logs.length > 0) {
            setLocalLogs(logsData.data.logs);
        } else if (stages && stages.length > 0) {
            // Generate synthetic logs from stage data
            const syntheticLogs: LogEntry[] = [];
            
            stages.forEach((stage) => {
                const stageInfo = PIPELINE_STAGES.find(s => s.key === stage.stage);
                const stageName = stageInfo?.label || stage.stage;
                
                if (stage.status === 'in_progress') {
                    syntheticLogs.push({
                        timestamp: stage.started_at || new Date().toISOString(),
                        level: 'info',
                        event: `${stageName} processing...`,
                        stage: stage.stage
                    });
                } else if (stage.status === 'completed') {
                    syntheticLogs.push({
                        timestamp: stage.started_at || new Date().toISOString(),
                        level: 'info',
                        event: `${stageName} started`,
                        stage: stage.stage
                    });
                    syntheticLogs.push({
                        timestamp: stage.completed_at || new Date().toISOString(),
                        level: 'info',
                        event: `${stageName} completed`,
                        stage: stage.stage,
                        duration_ms: stage.duration_ms
                    });
                } else if (stage.status === 'failed') {
                    syntheticLogs.push({
                        timestamp: stage.started_at || new Date().toISOString(),
                        level: 'error',
                        event: `${stageName} failed`,
                        stage: stage.stage,
                        error: stage.error
                    });
                }
            });
            
            setLocalLogs(syntheticLogs);
        }
    }, [logsData, stages]);

    // Auto-scroll to bottom
    useEffect(() => {
        if (terminalRef.current && isExpanded) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [localLogs, isExpanded]);

    const formatTimestamp = (ts: string) => {
        try {
            const date = new Date(ts);
            const time = date.toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            const ms = date.getMilliseconds().toString().padStart(3, '0');
            return `${time}.${ms}`;
        } catch {
            return ts;
        }
    };

    const getLevelIcon = (level: string) => {
        switch (level) {
            case 'error': return <XCircle className="w-3 h-3 text-red-400" />;
            case 'warning': return <AlertTriangle className="w-3 h-3 text-amber-400" />;
            case 'info': return <Info className="w-3 h-3 text-emerald-400" />;
            default: return <Terminal className="w-3 h-3 text-zinc-400" />;
        }
    };

    const getStageColor = (stage: string) => {
        const colors: Record<string, string> = {
            upload: 'text-blue-400',
            guardrail: 'text-purple-400',
            rembg: 'text-cyan-400',
            nano_banana: 'text-pink-400',
            generation: 'text-orange-400',
            realesrgan: 'text-green-400'
        };
        return colors[stage] || 'text-zinc-400';
    };

    return (
        <Card className="bg-zinc-950 border-zinc-800">
            <CardHeader 
                className="py-3 border-b border-zinc-800 cursor-pointer"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="flex gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-red-500" />
                            <div className="w-3 h-3 rounded-full bg-amber-500" />
                            <div className="w-3 h-3 rounded-full bg-emerald-500" />
                        </div>
                        <Terminal className="w-4 h-4 text-zinc-500 ml-2" />
                        <span className="text-xs text-zinc-500 font-mono">
                            pipeline-logs — {jobId || 'no-job'}
                        </span>
                        {isPolling && (
                            <div className="flex items-center gap-1 ml-2">
                                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                <span className="text-[10px] text-emerald-400">LIVE</span>
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-zinc-600">{localLogs.length} entries</span>
                        {isExpanded ? (
                            <ChevronUp className="w-4 h-4 text-zinc-500" />
                        ) : (
                            <ChevronDown className="w-4 h-4 text-zinc-500" />
                        )}
                    </div>
                </div>
            </CardHeader>
            {isExpanded && (
                <CardContent className="p-0">
                    <div
                        ref={terminalRef}
                        className="h-72 overflow-y-auto font-mono text-xs p-4 space-y-1.5"
                    >
                        {localLogs.length === 0 ? (
                            <div className="flex items-center gap-2 text-zinc-600">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                <span>Waiting for pipeline to start...</span>
                            </div>
                        ) : (
                            localLogs.map((log, idx) => (
                                <div 
                                    key={idx} 
                                    className="flex gap-2 hover:bg-zinc-900/50 px-2 py-1 -mx-2 rounded group"
                                >
                                    <span className="text-zinc-600 shrink-0 w-20">
                                        {formatTimestamp(log.timestamp)}
                                    </span>
                                    <span className="shrink-0">
                                        {getLevelIcon(log.level)}
                                    </span>
                                    {log.stage && (
                                        <span className={`shrink-0 font-semibold ${getStageColor(log.stage)}`}>
                                            [{log.stage.toUpperCase()}]
                                        </span>
                                    )}
                                    <span className="text-zinc-300 flex-1">
                                        {log.event}
                                    </span>
                                    <div className="flex items-center gap-2 shrink-0">
                                        {log.duration_ms && (
                                            <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px]">
                                                {log.duration_ms}ms
                                            </span>
                                        )}
                                        {log.cost_usd && (
                                            <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">
                                                ${log.cost_usd}
                                            </span>
                                        )}
                                        {log.vram_used_gb && (
                                            <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 text-[10px]">
                                                {log.vram_used_gb.toFixed(2)}GB
                                            </span>
                                        )}
                                    </div>
                                    {log.error && (
                                        <span className="text-red-400 text-[10px]">
                                            ⚠ {log.error}
                                        </span>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                </CardContent>
            )}
        </Card>
    );
}

// =============================================================================
// Stage Details Panel
// =============================================================================

interface StageDetailsPanelProps {
    stages: StageProgress[];
    currentStage?: string;
}

function StageDetailsPanel({ stages }: StageDetailsPanelProps) {
    return (
        <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader className="py-3">
                <CardTitle className="text-sm flex items-center gap-2">
                    <Activity className="w-4 h-4 text-zinc-400" />
                    Stage Details
                </CardTitle>
            </CardHeader>
            <CardContent className="p-4 pt-0">
                <div className="space-y-2">
                    {stages.map((stage) => {
                        const info = PIPELINE_STAGES.find(s => s.key === stage.stage);
                        const Icon = stageIcons[stage.stage] || CheckCircle2;
                        
                        return (
                            <div 
                                key={stage.stage}
                                className={`
                                    flex items-center gap-3 p-2 rounded-lg transition-colors
                                    ${stage.status === 'in_progress' ? 'bg-amber-500/10 border border-amber-500/20' :
                                      stage.status === 'completed' ? 'bg-emerald-500/5 border border-emerald-500/10' :
                                      stage.status === 'failed' ? 'bg-red-500/10 border border-red-500/20' :
                                      'bg-zinc-800/30 border border-zinc-800'}
                                `}
                            >
                                <div className={`
                                    p-1.5 rounded-md
                                    ${stage.status === 'completed' ? 'bg-emerald-500/10' :
                                      stage.status === 'in_progress' ? 'bg-amber-500/10' :
                                      stage.status === 'failed' ? 'bg-red-500/10' :
                                      'bg-zinc-800'}
                                `}>
                                    {stage.status === 'in_progress' ? (
                                        <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                                    ) : (
                                        <Icon className={`w-3.5 h-3.5 ${
                                            stage.status === 'completed' ? 'text-emerald-400' :
                                            stage.status === 'failed' ? 'text-red-400' :
                                            'text-zinc-500'
                                        }`} />
                                    )}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className={`text-xs font-medium ${
                                        stage.status === 'completed' ? 'text-emerald-400' :
                                        stage.status === 'in_progress' ? 'text-amber-400' :
                                        stage.status === 'failed' ? 'text-red-400' :
                                        'text-zinc-400'
                                    }`}>
                                        {info?.label || stage.stage}
                                    </p>
                                    <p className="text-[10px] text-zinc-600 truncate">
                                        {info?.description}
                                    </p>
                                </div>
                                {stage.duration_ms && (
                                    <span className="text-[10px] font-mono text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded">
                                        {(stage.duration_ms / 1000).toFixed(2)}s
                                    </span>
                                )}
                            </div>
                        );
                    })}
                </div>
            </CardContent>
        </Card>
    );
}

// =============================================================================
// Image Comparison Slider
// =============================================================================

interface ComparisonSliderProps {
    beforeUrl?: string;
    afterUrl?: string;
}

function ComparisonSlider({ beforeUrl, afterUrl }: ComparisonSliderProps) {
    const [sliderPosition, setSliderPosition] = useState(50);
    const containerRef = useRef<HTMLDivElement>(null);
    const isDragging = useRef(false);

    const handleMove = useCallback((clientX: number) => {
        if (!containerRef.current || !isDragging.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
        setSliderPosition(percentage);
    }, []);

    const handleMouseDown = () => {
        isDragging.current = true;
    };

    const handleMouseUp = () => {
        isDragging.current = false;
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => handleMove(e.clientX);
        const handleTouchMove = (e: TouchEvent) => handleMove(e.touches[0].clientX);

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        window.addEventListener('touchmove', handleTouchMove);
        window.addEventListener('touchend', handleMouseUp);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            window.removeEventListener('touchmove', handleTouchMove);
            window.removeEventListener('touchend', handleMouseUp);
        };
    }, [handleMove]);

    if (!beforeUrl && !afterUrl) {
        return (
            <div className="aspect-video bg-zinc-900 rounded-lg flex items-center justify-center border border-zinc-800">
                <div className="text-center text-zinc-600">
                    <ImageIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>Upload an image to begin</p>
                </div>
            </div>
        );
    }

    return (
        <div
            ref={containerRef}
            className="relative aspect-video rounded-lg overflow-hidden cursor-ew-resize select-none border border-zinc-800"
            onMouseDown={handleMouseDown}
            onTouchStart={handleMouseDown}
        >
            {/* Before image (full width background) */}
            <div className="absolute inset-0">
                {beforeUrl ? (
                    <img
                        src={beforeUrl}
                        alt="Original"
                        className="w-full h-full object-contain bg-zinc-950"
                    />
                ) : (
                    <div className="w-full h-full bg-zinc-900 flex items-center justify-center">
                        <span className="text-zinc-600">Original</span>
                    </div>
                )}
            </div>

            {/* After image (clipped) */}
            <div
                className="absolute inset-0 overflow-hidden"
                style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
            >
                {afterUrl ? (
                    <img
                        src={afterUrl}
                        alt="Enhanced"
                        className="w-full h-full object-contain bg-zinc-950"
                    />
                ) : (
                    <div className="w-full h-full bg-zinc-800 flex items-center justify-center">
                        <span className="text-zinc-500">Enhanced</span>
                    </div>
                )}
            </div>

            {/* Slider handle */}
            <div
                className="absolute top-0 bottom-0 w-1 bg-white shadow-lg cursor-ew-resize"
                style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
            >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white shadow-lg flex items-center justify-center">
                    <div className="flex gap-0.5">
                        <div className="w-0.5 h-4 bg-zinc-400 rounded-full" />
                        <div className="w-0.5 h-4 bg-zinc-400 rounded-full" />
                    </div>
                </div>
            </div>

            {/* Labels */}
            <div className="absolute bottom-3 left-3 px-2 py-1 bg-black/70 rounded text-xs text-white font-medium">
                Original
            </div>
            <div className="absolute bottom-3 right-3 px-2 py-1 bg-black/70 rounded text-xs text-white font-medium">
                Enhanced
            </div>
        </div>
    );
}

// =============================================================================
// Main Pipeline Dashboard Component
// =============================================================================

export function PipelineDashboard() {
    const [prompt, setPrompt] = useState('');
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [imageBase64, setImageBase64] = useState<string | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [isPolling, setIsPolling] = useState(false);

    // Process mutation
    const processMutation = useMutation({
        mutationFn: (data: { prompt: string; image_base64: string }) =>
            pipelineApi.process({
                prompt: data.prompt,
                image_base64: data.image_base64,
                options: { simulate: true } // Use simulation for development
            }),
        onSuccess: (response) => {
            const jobId = response.data.job_id;
            setCurrentJobId(jobId);

            if (response.data.status === 'BLOCKED') {
                toast.error(`Content blocked: ${response.data.message}`);
                setIsPolling(false);
            } else {
                toast.success('Pipeline started!');
                setIsPolling(true);
            }
        },
        onError: (error: Error) => {
            toast.error(`Failed to start pipeline: ${error.message}`);
        }
    });

    // Poll job status
    const { data: statusData } = useQuery({
        queryKey: ['jobStatus', currentJobId],
        queryFn: () => currentJobId ? pipelineApi.getStatus(currentJobId) : null,
        enabled: !!currentJobId && isPolling,
        refetchInterval: isPolling ? 1500 : false,
    });

    const jobStatus = statusData?.data;

    // Stop polling when job completes
    useEffect(() => {
        if (jobStatus?.status === 'COMPLETED' ||
            jobStatus?.status === 'FAILED' ||
            jobStatus?.status === 'BLOCKED') {
            setIsPolling(false);

            if (jobStatus.status === 'COMPLETED') {
                toast.success('Pipeline completed successfully!');
            } else if (jobStatus.status === 'FAILED') {
                toast.error(`Pipeline failed: ${jobStatus.error?.message}`);
            }
        }
    }, [jobStatus?.status]);

    // File drop handler
    const onDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        // Preview
        const reader = new FileReader();
        reader.onload = () => {
            setImagePreview(reader.result as string);
        };
        reader.readAsDataURL(file);

        // Base64 for API
        const base64Reader = new FileReader();
        base64Reader.onload = () => {
            const base64 = (base64Reader.result as string).split(',')[1];
            setImageBase64(base64);
        };
        base64Reader.readAsDataURL(file);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
        maxFiles: 1,
        maxSize: 10 * 1024 * 1024 // 10MB
    });

    const handleSubmit = () => {
        if (!imageBase64) {
            toast.error('Please upload an image first');
            return;
        }

        processMutation.mutate({
            prompt: prompt || 'Food image processing',
            image_base64: imageBase64
        });
    };

    const handleReset = () => {
        setPrompt('');
        setImagePreview(null);
        setImageBase64(null);
        setCurrentJobId(null);
        setIsPolling(false);
    };

    // Default stages for display
    const defaultStages: StageProgress[] = [
        { stage: 'upload', status: 'pending' },
        { stage: 'guardrail', status: 'pending' },
        { stage: 'rembg', status: 'pending' },
        { stage: 'nano_banana', status: 'pending' },
        { stage: 'generation', status: 'pending' },
        { stage: 'realesrgan', status: 'pending' },
    ];

    const currentStages = jobStatus?.stages || defaultStages;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white">Image Processing Pipeline</h2>
                    <p className="text-zinc-400 text-sm mt-1">
                        Guardrail → Background Removal → Smart Placement → Generation → 4K Enhancement
                    </p>
                </div>
                {currentJobId && (
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={handleReset}
                        className="gap-2"
                    >
                        <RefreshCw className="w-4 h-4" />
                        New Job
                    </Button>
                )}
            </div>

            {/* Pipeline Stepper */}
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-2">
                    <PipelineStepper
                        stages={currentStages}
                        currentStage={jobStatus?.current_stage}
                        jobStatus={jobStatus?.status || 'READY'}
                    />
                </CardContent>
            </Card>

            {/* Metric Cards */}
            <MetricCards
                status={jobStatus || null}
                isProcessing={isPolling}
            />

            {/* Main Content Grid */}
            <div className="grid grid-cols-3 gap-6">
                {/* Left: Upload & Input */}
                <div className="space-y-4">
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardHeader>
                            <CardTitle className="text-lg">Upload Image</CardTitle>
                            <CardDescription>
                                Drag and drop or click to select
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div
                                {...getRootProps()}
                                className={`
                                    border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
                                    transition-colors duration-200
                                    ${isDragActive
                                        ? 'border-blue-500 bg-blue-500/10'
                                        : 'border-zinc-700 hover:border-zinc-600'
                                    }
                                    ${imagePreview ? 'p-2' : ''}
                                `}
                            >
                                <input {...getInputProps()} />
                                {imagePreview ? (
                                    <img
                                        src={imagePreview}
                                        alt="Preview"
                                        className="max-h-40 mx-auto rounded-lg"
                                    />
                                ) : (
                                    <div className="text-zinc-400">
                                        <Upload className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                        <p className="text-sm">Drop image here</p>
                                        <p className="text-xs text-zinc-600 mt-1">
                                            PNG, JPG, WEBP up to 10MB
                                        </p>
                                    </div>
                                )}
                            </div>

                            <div className="mt-4">
                                <Textarea
                                    placeholder="Describe the image (optional)..."
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    className="resize-none text-sm"
                                    rows={2}
                                />
                            </div>

                            <Button
                                className="w-full mt-4"
                                size="lg"
                                onClick={handleSubmit}
                                disabled={!imageBase64 || processMutation.isPending || isPolling}
                            >
                                {processMutation.isPending || isPolling ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="w-4 h-4 mr-2" />
                                        Start Pipeline
                                    </>
                                )}
                            </Button>
                        </CardContent>
                    </Card>

                    {/* Stage Details Panel */}
                    <StageDetailsPanel 
                        stages={currentStages}
                        currentStage={jobStatus?.current_stage}
                    />
                </div>

                {/* Middle: Comparison Tool */}
                <Card className="bg-zinc-900/50 border-zinc-800">
                    <CardHeader>
                        <CardTitle className="text-lg">Result Preview</CardTitle>
                        <CardDescription>
                            Compare original vs enhanced output
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ComparisonSlider
                            beforeUrl={imagePreview || undefined}
                            afterUrl={jobStatus?.storage_urls?.final || jobStatus?.storage_urls?.upscaled}
                        />

                        {/* Output thumbnails */}
                        {jobStatus?.storage_urls && (
                            <div className="grid grid-cols-4 gap-2 mt-4">
                                {jobStatus.storage_urls.original && (
                                    <div className="text-center">
                                        <img
                                            src={jobStatus.storage_urls.original}
                                            alt="Original"
                                            className="w-full aspect-square object-cover rounded border border-zinc-700"
                                        />
                                        <span className="text-[10px] text-zinc-500">Original</span>
                                    </div>
                                )}
                                {jobStatus.storage_urls.transparent && (
                                    <div className="text-center">
                                        <img
                                            src={jobStatus.storage_urls.transparent}
                                            alt="No BG"
                                            className="w-full aspect-square object-cover rounded border border-zinc-700"
                                        />
                                        <span className="text-[10px] text-zinc-500">No BG</span>
                                    </div>
                                )}
                                {jobStatus.storage_urls.generated && (
                                    <div className="text-center">
                                        <img
                                            src={jobStatus.storage_urls.generated}
                                            alt="Generated"
                                            className="w-full aspect-square object-cover rounded border border-zinc-700"
                                        />
                                        <span className="text-[10px] text-zinc-500">Generated</span>
                                    </div>
                                )}
                                {jobStatus.storage_urls.final && (
                                    <div className="text-center">
                                        <img
                                            src={jobStatus.storage_urls.final}
                                            alt="Final"
                                            className="w-full aspect-square object-cover rounded border border-zinc-700"
                                        />
                                        <span className="text-[10px] text-zinc-500">Final</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Right: Logs Terminal */}
                <div className="space-y-4">
                    <LogsTerminal 
                        jobId={currentJobId} 
                        isPolling={isPolling || !!currentJobId}
                        stages={currentStages}
                    />

                    {/* Error display */}
                    {jobStatus?.error && (
                        <Card className="bg-red-950/30 border-red-900">
                            <CardContent className="p-4">
                                <div className="flex items-start gap-3">
                                    <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                                    <div>
                                        <p className="text-red-400 font-medium">Pipeline Failed</p>
                                        <p className="text-red-300/70 text-sm mt-1">
                                            Stage: {jobStatus.error.stage} — {jobStatus.error.message}
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
}

export default PipelineDashboard;
