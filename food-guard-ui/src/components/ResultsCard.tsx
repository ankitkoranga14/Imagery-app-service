import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from 'react-hot-toast';
import { GuardrailResult } from '@/lib/types';
import { cn } from '@/lib/utils';

interface ResultsCardProps {
    result: GuardrailResult;
    prompt?: string;
    imageFile?: File | null;
}

export function ResultsCard({ result, prompt, imageFile }: ResultsCardProps) {
    return (
        <Card>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle>Results</CardTitle>
                    <Badge variant={result.status === 'PASS' ? 'default' : 'destructive'}>
                        {result.status}
                    </Badge>
                </div>
                <div className="text-sm text-muted-foreground mt-2">
                    {prompt && (
                        <>
                            <p className="font-medium">Prompt:</p>
                            <p className="italic">"{prompt}"</p>
                        </>
                    )}
                    {imageFile && <p className="mt-1">Image: {imageFile.name}</p>}
                </div>
            </CardHeader>

            <CardContent className="space-y-6">
                {/* Processing Time */}
                <div className="flex justify-between items-center text-sm">
                    <span>Processing Time:</span>
                    <span className="font-mono">
                        {result.metadata.processing_time_ms}ms
                    </span>
                </div>

                {/* Scores Table */}
                <div className="grid gap-2">
                    {Object.entries(result.scores).map(([key, value]) => (
                        <div key={key} className="flex justify-between items-center py-2">
                            <span className="capitalize">{key.replace('_', ' ')}</span>
                            <div className="flex items-center gap-2">
                                <div className="w-24 bg-secondary h-2 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-primary transition-all"
                                        style={{ width: `${Math.min(value * 100, 100)}%` }}
                                    />
                                </div>
                                <span className="font-mono w-12 text-right">{value.toFixed(3)}</span>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Food Type */}
                {result.metadata.food_type && (
                    <div className="p-4 bg-emerald-50 dark:bg-emerald-950/30 rounded-lg border border-emerald-100 dark:border-emerald-900">
                        <span className="font-semibold text-emerald-700 dark:text-emerald-400">Detected:</span>{' '}
                        <span className="font-mono bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300 px-2 py-1 rounded">
                            {result.metadata.food_type}
                        </span>
                    </div>
                )}

                {/* Reasons if Blocked */}
                {result.reasons.length > 0 && (
                    <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/20">
                        <span className="font-semibold text-destructive">Reasons:</span>
                        <ul className="list-disc list-inside mt-2 text-sm text-destructive">
                            {result.reasons.map((reason, i) => (
                                <li key={i}>{reason}</li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Developer Details */}
                <div className="pt-4 border-t">
                    <p className="text-sm font-semibold mb-2">Developer Details</p>
                    <div className="grid grid-cols-2 gap-4 text-xs font-mono bg-muted/50 p-3 rounded-md">
                        <div>
                            <span className="text-muted-foreground">Total Latency:</span>
                            <span className="ml-2">{result.metadata.processing_time_ms}ms</span>
                        </div>
                        <div>
                            <span className="text-muted-foreground">Cache Hit:</span>
                            <span className={cn("ml-2", result.metadata.cache_hit ? "text-green-600" : "text-yellow-600")}>
                                {result.metadata.cache_hit ? 'YES' : 'NO'}
                            </span>
                        </div>
                        <div>
                            <span className="text-muted-foreground">Status Code:</span>
                            <span className="ml-2">200 OK</span>
                        </div>
                        <div>
                            <span className="text-muted-foreground">Timestamp:</span>
                            <span className="ml-2">{new Date().toLocaleTimeString()}</span>
                        </div>
                    </div>
                </div>

                {/* JSON Export */}
                <div className="flex gap-2 pt-4 border-t">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                            navigator.clipboard.writeText(JSON.stringify(result, null, 2));
                            toast.success('JSON copied!');
                        }}
                    >
                        Copy JSON
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                            const blob = new Blob([JSON.stringify(result, null, 2)], {
                                type: 'application/json',
                            });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `guardrail-${Date.now()}.json`;
                            a.click();
                        }}
                    >
                        Download JSON
                    </Button>
                </div>
            </CardContent>
        </Card>
    );
}
