import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
    Upload,
    Shield,
    CheckCircle2,
    XCircle,
    Clock,
    Loader2,
    Activity,
    RefreshCw,
    AlertTriangle,
    Zap,
    BarChart3,
    TrendingUp,
    Database
} from 'lucide-react';
import { validationApi } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import type { ValidateResponse } from '@/lib/types';
import { VALIDATION_STAGES } from '@/lib/types';

// =============================================================================
// Validation Result Card
// =============================================================================

interface ValidationResultCardProps {
    result: ValidateResponse | null;
    isLoading: boolean;
}

function ValidationResultCard({ result, isLoading }: ValidationResultCardProps) {
    if (isLoading) {
        return (
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-6 flex flex-col items-center justify-center min-h-[200px]">
                    <Loader2 className="w-12 h-12 text-amber-400 animate-spin mb-4" />
                    <p className="text-zinc-400">Validating image...</p>
                    <p className="text-xs text-zinc-600 mt-2">Running L0-L4 checks</p>
                </CardContent>
            </Card>
        );
    }

    if (!result) {
        return (
            <Card className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-6 flex flex-col items-center justify-center min-h-[200px]">
                    <Shield className="w-12 h-12 text-zinc-600 mb-4" />
                    <p className="text-zinc-400">No validation result yet</p>
                    <p className="text-xs text-zinc-600 mt-2">Upload an image and click Validate</p>
                </CardContent>
            </Card>
        );
    }

    const isPassed = result.status === 'PASS';

    return (
        <Card className={`border ${isPassed ? 'bg-emerald-950/20 border-emerald-900/50' : 'bg-red-950/20 border-red-900/50'}`}>
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-lg flex items-center gap-2">
                        {isPassed ? (
                            <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                        ) : (
                            <XCircle className="w-6 h-6 text-red-400" />
                        )}
                        Validation Result
                    </CardTitle>
                    <span className={`px-3 py-1 rounded-full text-sm font-bold ${isPassed ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                        {result.status}
                    </span>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Failure Reason */}
                {result.failure_reason && (
                    <div className="p-3 rounded-lg bg-red-950/30 border border-red-900/30">
                        <div className="flex items-start gap-2">
                            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                            <div>
                                <p className="text-sm font-medium text-red-400">Blocking Reason</p>
                                <p className="text-xs text-red-300 mt-1">{result.failure_reason}</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Latency */}
                <div className="flex items-center justify-between p-2 rounded bg-zinc-800/50">
                    <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-blue-400" />
                        <span className="text-sm text-zinc-400">Processing Time</span>
                    </div>
                    <span className="font-mono text-white">{result.latency_ms}ms</span>
                </div>

                {/* Scores Grid */}
                {Object.keys(result.scores).length > 0 && (
                    <div className="space-y-2">
                        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">Validation Scores</h4>
                        <div className="grid grid-cols-3 gap-2">
                            {Object.entries(result.scores).map(([key, value]) => (
                                <ScoreCell key={key} label={key} value={value} />
                            ))}
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

// =============================================================================
// Score Cell Component
// =============================================================================

function ScoreCell({ label, value }: { label: string; value: number }) {
    const formatLabel = (key: string) => {
        return key.replace(/_/g, ' ').replace(/score|probability/gi, '').trim();
    };

    const formatValue = (key: string, val: number) => {
        if (key.includes('percentage') || key.includes('probability') || key.includes('score') && val <= 1) {
            return `${(val * 100).toFixed(1)}%`;
        }
        if (typeof val === 'number' && !Number.isInteger(val)) {
            return val.toFixed(2);
        }
        return String(val);
    };

    const getStatus = (key: string, val: number): 'good' | 'bad' | 'neutral' => {
        // Common patterns for bad scores
        if (key.includes('nsfw') || key.includes('foreign') || key.includes('blur')) {
            return val > 0.3 ? 'bad' : 'good';
        }
        if (key.includes('food') && val <= 1) {
            return val > 0.5 ? 'good' : val < 0.3 ? 'bad' : 'neutral';
        }
        if (key.includes('darkness') && val < 35) return 'bad';
        if (key.includes('glare') && val > 0.15) return 'bad';
        return 'neutral';
    };

    const status = getStatus(label, value);

    return (
        <div className={`p-2 rounded border text-center ${status === 'good' ? 'bg-emerald-950/20 border-emerald-900/30' :
            status === 'bad' ? 'bg-red-950/20 border-red-900/30' :
                'bg-zinc-800/30 border-zinc-700/30'
            }`}>
            <p className="text-[10px] text-zinc-500 uppercase truncate">{formatLabel(label)}</p>
            <p className={`text-sm font-mono font-bold ${status === 'good' ? 'text-emerald-400' :
                status === 'bad' ? 'text-red-400' :
                    'text-white'
                }`}>
                {formatValue(label, value)}
            </p>
        </div>
    );
}

// =============================================================================
// Validation History Panel
// =============================================================================

interface ValidationHistoryProps {
    onRefresh: () => void;
}

function ValidationHistory({ onRefresh }: ValidationHistoryProps) {
    const [page, setPage] = useState(1);
    const [statusFilter, setStatusFilter] = useState<string | undefined>();

    const { data, isLoading } = useQuery({
        queryKey: ['validationLogs', page, statusFilter],
        queryFn: () => validationApi.getLogs(page, 10, statusFilter),
        refetchInterval: 10000, // Refresh every 10 seconds
    });

    const logs = data?.data?.logs || [];
    const total = data?.data?.total || 0;

    return (
        <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader className="py-3 border-b border-zinc-800">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-sm flex items-center gap-2">
                        <Database className="w-4 h-4 text-zinc-400" />
                        Validation History
                    </CardTitle>
                    <div className="flex items-center gap-2">
                        <select
                            value={statusFilter || ''}
                            onChange={(e) => setStatusFilter(e.target.value || undefined)}
                            className="text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-300"
                        >
                            <option value="">All</option>
                            <option value="PASS">Pass Only</option>
                            <option value="BLOCK">Block Only</option>
                        </select>
                        <Button variant="ghost" size="sm" onClick={onRefresh} className="h-7 w-7 p-0">
                            <RefreshCw className="w-3 h-3" />
                        </Button>
                    </div>
                </div>
            </CardHeader>
            <CardContent className="p-0">
                <div className="max-h-64 overflow-y-auto">
                    {isLoading ? (
                        <div className="p-4 text-center text-zinc-500">
                            <Loader2 className="w-4 h-4 animate-spin mx-auto" />
                        </div>
                    ) : logs.length === 0 ? (
                        <div className="p-4 text-center text-zinc-600 text-sm">
                            No validation logs yet
                        </div>
                    ) : (
                        <div className="divide-y divide-zinc-800">
                            {logs.map((log) => (
                                <div key={log.id} className="px-4 py-2 flex items-center justify-between hover:bg-zinc-800/50">
                                    <div className="flex items-center gap-2">
                                        {log.status === 'PASS' ? (
                                            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                                        ) : (
                                            <XCircle className="w-3 h-3 text-red-400" />
                                        )}
                                        <span className={`text-xs font-medium ${log.status === 'PASS' ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {log.status}
                                        </span>
                                        {log.failure_level && (
                                            <span className="text-[10px] text-zinc-600 bg-zinc-800 px-1.5 py-0.5 rounded">
                                                {log.failure_level.toUpperCase()}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-3">
                                        {log.cache_hit && (
                                            <span className="text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">
                                                CACHED
                                            </span>
                                        )}
                                        <span className="text-[10px] font-mono text-zinc-500">
                                            {log.latency_ms}ms
                                        </span>
                                        <span className="text-[10px] text-zinc-600">
                                            {new Date(log.created_at).toLocaleTimeString()}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                {total > 10 && (
                    <div className="p-2 border-t border-zinc-800 flex items-center justify-between">
                        <span className="text-xs text-zinc-600">
                            Page {page} of {Math.ceil(total / 10)}
                        </span>
                        <div className="flex gap-1">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setPage(p => Math.max(1, p - 1))}
                                disabled={page === 1}
                                className="h-6 px-2 text-xs"
                            >
                                Prev
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setPage(p => p + 1)}
                                disabled={page >= Math.ceil(total / 10)}
                                className="h-6 px-2 text-xs"
                            >
                                Next
                            </Button>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

// =============================================================================
// Statistics Panel
// =============================================================================

function StatsPanel() {
    const { data, isLoading } = useQuery({
        queryKey: ['validationStats'],
        queryFn: () => validationApi.getStats(),
        refetchInterval: 30000, // Refresh every 30 seconds
    });

    const stats = data?.data;

    if (isLoading || !stats) {
        return null;
    }

    return (
        <div className="grid grid-cols-4 gap-4">
            <StatCard
                icon={<BarChart3 className="w-5 h-5 text-blue-400" />}
                label="Total Validations"
                value={stats.total_validations.toLocaleString()}
                bgColor="bg-blue-500/10"
            />
            <StatCard
                icon={<CheckCircle2 className="w-5 h-5 text-emerald-400" />}
                label="Pass Rate"
                value={`${(stats.pass_rate * 100).toFixed(1)}%`}
                bgColor="bg-emerald-500/10"
            />
            <StatCard
                icon={<Clock className="w-5 h-5 text-amber-400" />}
                label="Avg Latency"
                value={`${stats.average_latency_ms.toFixed(0)}ms`}
                bgColor="bg-amber-500/10"
            />
            <StatCard
                icon={<TrendingUp className="w-5 h-5 text-purple-400" />}
                label="Cache Hit Rate"
                value={`${(stats.cache_hit_rate * 100).toFixed(1)}%`}
                bgColor="bg-purple-500/10"
            />
        </div>
    );
}

function StatCard({
    icon,
    label,
    value,
    bgColor
}: {
    icon: React.ReactNode;
    label: string;
    value: string;
    bgColor: string;
}) {
    return (
        <Card className="bg-zinc-900/50 border-zinc-800">
            <CardContent className="p-4">
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${bgColor}`}>
                        {icon}
                    </div>
                    <div>
                        <p className="text-xs text-zinc-500 uppercase tracking-wide">{label}</p>
                        <p className="text-xl font-bold text-white">{value}</p>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

// =============================================================================
// Main Validation Dashboard Component
// =============================================================================

export function ValidationDashboard() {
    const [prompt, setPrompt] = useState('');
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [imageBase64, setImageBase64] = useState<string | null>(null);
    const [validationResult, setValidationResult] = useState<ValidateResponse | null>(null);
    const [imageSize, setImageSize] = useState<number | null>(null);

    // Max file size: 10MB
    const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
    const MAX_FILE_SIZE_MB = 10;

    const formatFileSize = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    // Validation mutation
    const validateMutation = useMutation({
        mutationFn: (data: { prompt: string; image_bytes: string }) =>
            validationApi.validate({
                prompt: data.prompt,
                image_bytes: data.image_bytes,
            }),
        onSuccess: (response) => {
            setValidationResult(response.data);
            if (response.data.status === 'PASS') {
                toast.success('Image validated successfully!');
            } else {
                toast.error(`Validation blocked: ${response.data.failure_reason || 'Unknown reason'}`);
            }
        },
        onError: (error: Error & { response?: { status?: number; data?: { detail?: string } } }) => {
            if (error.response?.status === 413) {
                toast.error(`Image too large. Maximum allowed: ${MAX_FILE_SIZE_MB}MB.`, { duration: 6000 });
            } else if (error.response?.data?.detail) {
                toast.error(error.response.data.detail, { duration: 6000 });
            } else {
                toast.error(`Validation failed: ${error.message}`);
            }
        }
    });

    // File drop handler
    const onDrop = useCallback((acceptedFiles: File[], fileRejections: Array<{ file: File; errors: readonly { code: string; message: string }[] }>) => {
        if (fileRejections.length > 0) {
            const rejection = fileRejections[0];
            for (const error of rejection.errors) {
                if (error.code === 'file-too-large') {
                    toast.error(`Image too large. Maximum allowed: ${MAX_FILE_SIZE_MB}MB.`);
                } else if (error.code === 'file-invalid-type') {
                    toast.error('Invalid file type. Please upload PNG, JPG, JPEG, or WEBP.');
                }
            }
            return;
        }

        const file = acceptedFiles[0];
        if (!file) return;

        setImageSize(file.size);
        setValidationResult(null);

        // Preview
        const reader = new FileReader();
        reader.onload = () => setImagePreview(reader.result as string);
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
        maxSize: MAX_FILE_SIZE_BYTES
    });

    const handleValidate = () => {
        if (!imageBase64) {
            toast.error('Please upload an image first');
            return;
        }

        validateMutation.mutate({
            prompt: prompt || 'Food image validation',
            image_bytes: imageBase64
        });
    };

    const handleReset = () => {
        setPrompt('');
        setImagePreview(null);
        setImageBase64(null);
        setValidationResult(null);
        setImageSize(null);
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white">Image Validation</h2>
                    <p className="text-zinc-400 text-sm mt-1">
                        L0 (Cache) → L1 (Text) → L2 (Physics) → L3 (YOLO) → L4 (CLIP)
                    </p>
                </div>
                {validationResult && (
                    <Button variant="outline" size="sm" onClick={handleReset} className="gap-2">
                        <RefreshCw className="w-4 h-4" />
                        New Validation
                    </Button>
                )}
            </div>

            {/* Statistics */}
            <StatsPanel />

            {/* Main Content Grid */}
            <div className="grid grid-cols-2 gap-6">
                {/* Left: Upload & Input */}
                <div className="space-y-4">
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardHeader>
                            <CardTitle className="text-lg">Upload Image</CardTitle>
                            <CardDescription>
                                Submit an image for guardrail validation
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div
                                {...getRootProps()}
                                className={`
                                    border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
                                    transition-colors duration-200
                                    ${isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 hover:border-zinc-600'}
                                    ${imagePreview ? 'p-2' : ''}
                                `}
                            >
                                <input {...getInputProps()} />
                                {imagePreview ? (
                                    <div className="space-y-2">
                                        <img
                                            src={imagePreview}
                                            alt="Preview"
                                            className="max-h-48 mx-auto rounded-lg"
                                        />
                                        {imageSize && (
                                            <div className="flex items-center justify-center gap-2 text-xs">
                                                <span className={`px-2 py-0.5 rounded ${imageSize > MAX_FILE_SIZE_BYTES * 0.8
                                                    ? 'bg-amber-500/20 text-amber-400'
                                                    : 'bg-emerald-500/20 text-emerald-400'
                                                    }`}>
                                                    {formatFileSize(imageSize)}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="text-zinc-400">
                                        <Upload className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                        <p className="text-sm">Drop image here</p>
                                        <p className="text-xs text-zinc-600 mt-1">
                                            PNG, JPG, WEBP up to {MAX_FILE_SIZE_MB}MB
                                        </p>
                                    </div>
                                )}
                            </div>

                            <div className="mt-4">
                                <Textarea
                                    placeholder="Describe the expected content (optional)..."
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    className="resize-none text-sm"
                                    rows={2}
                                />
                            </div>

                            <Button
                                className="w-full mt-4"
                                size="lg"
                                onClick={handleValidate}
                                disabled={!imageBase64 || validateMutation.isPending}
                            >
                                {validateMutation.isPending ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        Validating...
                                    </>
                                ) : (
                                    <>
                                        <Shield className="w-4 h-4 mr-2" />
                                        Validate Image
                                    </>
                                )}
                            </Button>
                        </CardContent>
                    </Card>

                    {/* Validation History */}
                    <ValidationHistory onRefresh={() => { }} />
                </div>

                {/* Right: Result */}
                <div className="space-y-4">
                    <ValidationResultCard
                        result={validationResult}
                        isLoading={validateMutation.isPending}
                    />

                    {/* Validation Stages Info */}
                    <Card className="bg-zinc-900/50 border-zinc-800">
                        <CardHeader className="py-3">
                            <CardTitle className="text-sm flex items-center gap-2">
                                <Activity className="w-4 h-4 text-zinc-400" />
                                Validation Pipeline
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2">
                            {VALIDATION_STAGES.map((stage) => (
                                <div
                                    key={stage.key}
                                    className="flex items-center gap-3 p-2 rounded-lg bg-zinc-800/30 border border-zinc-800"
                                >
                                    <div className="p-1.5 rounded-md bg-zinc-800">
                                        <Zap className="w-3 h-3 text-zinc-500" />
                                    </div>
                                    <div className="flex-1">
                                        <p className="text-xs font-medium text-zinc-300">{stage.label}</p>
                                        <p className="text-[10px] text-zinc-600">{stage.description}</p>
                                    </div>
                                </div>
                            ))}
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}

export default ValidationDashboard;
