import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { useGuardrailApi } from '@/hooks/useGuardrailApi';
import { useTestHistoryStore } from '@/stores/testHistory';
import { Upload, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { cn, fileToBase64 } from '@/lib/utils';
import { toast } from 'react-hot-toast';

export function BatchUploader() {
    const [files, setFiles] = useState<File[]>([]);
    const [prompt, setPrompt] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [results, setResults] = useState<{ file: string, status: string }[]>([]);

    const { mutateAsync: testGuardrail } = useGuardrailApi();
    const addTest = useTestHistoryStore((state) => state.addTest);

    const onDrop = (acceptedFiles: File[]) => {
        setFiles((prev) => [...prev, ...acceptedFiles]);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    });

    const handleBatchTest = async () => {
        if (!prompt.trim() || files.length === 0) return;

        setIsProcessing(true);
        setProgress(0);
        setResults([]);

        const batchSize = 5;
        let completed = 0;

        // Process in chunks
        for (let i = 0; i < files.length; i += batchSize) {
            const chunk = files.slice(i, i + batchSize);

            await Promise.all(chunk.map(async (file) => {
                try {
                    const base64 = await fileToBase64(file);
                    const imageBytes = base64.split(',')[1];

                    const response = await testGuardrail({
                        prompt,
                        image_bytes: imageBytes,
                    });

                    addTest({
                        id: crypto.randomUUID(),
                        request: { prompt, image_bytes: file.name },
                        result: response,
                        timestamp: Date.now(),
                    });

                    setResults(prev => [...prev, { file: file.name, status: response.status }]);
                } catch (error) {
                    console.error(error);
                    setResults(prev => [...prev, { file: file.name, status: 'ERROR' }]);
                } finally {
                    completed++;
                    setProgress((completed / files.length) * 100);
                }
            }));
        }

        setIsProcessing(false);
        toast.success('Batch processing completed');
    };

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>Batch Testing</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div>
                        <label className="text-sm font-medium mb-2 block">Common Prompt</label>
                        <Textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Prompt to use for all images..."
                        />
                    </div>

                    <div
                        {...getRootProps()}
                        className={cn(
                            "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
                            isDragActive ? "border-primary bg-primary/5" : "border-muted"
                        )}
                    >
                        <input {...getInputProps()} />
                        <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-2" />
                        <p className="text-muted-foreground">Drag images here or click to select</p>
                        <p className="text-sm text-muted-foreground mt-2">{files.length} files selected</p>
                    </div>

                    {files.length > 0 && (
                        <div className="flex justify-end gap-2">
                            <Button variant="outline" onClick={() => { setFiles([]); setResults([]); }}>
                                Clear All
                            </Button>
                            <Button onClick={handleBatchTest} disabled={isProcessing || !prompt.trim()}>
                                {isProcessing ? <Loader2 className="animate-spin mr-2 h-4 w-4" /> : null}
                                Process {files.length} Images
                            </Button>
                        </div>
                    )}

                    {isProcessing && <Progress value={progress} className="h-2" />}

                    {results.length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-60 overflow-y-auto">
                            {results.map((res, i) => (
                                <div key={i} className="flex items-center justify-between p-2 border rounded text-sm">
                                    <span className="truncate max-w-[70%]">{res.file}</span>
                                    {res.status === 'PASS' && <span className="text-green-600 flex items-center"><CheckCircle className="w-4 h-4 mr-1" /> PASS</span>}
                                    {res.status === 'BLOCK' && <span className="text-red-600 flex items-center"><XCircle className="w-4 h-4 mr-1" /> BLOCK</span>}
                                    {res.status === 'ERROR' && <span className="text-orange-600 flex items-center"><XCircle className="w-4 h-4 mr-1" /> ERROR</span>}
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
