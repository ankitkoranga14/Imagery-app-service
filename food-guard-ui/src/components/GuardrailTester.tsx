import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { useDropzone } from 'react-dropzone';
import { useGuardrailApi } from '@/hooks/useGuardrailApi';
import { useTestHistoryStore } from '@/stores/testHistory';
import { toast } from 'react-hot-toast';
import { GuardrailResult } from '@/lib/types';
import { ResultsCard } from '@/components/ResultsCard';
import { Shield, ShieldCheck, Loader2, Upload, X, Camera, FileText } from 'lucide-react';
import { cn, fileToBase64 } from '@/lib/utils';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

type TestMode = 'standard' | 'image-only';

export function GuardrailTester() {
    const [mode, setMode] = useState<TestMode>('standard');
    const [prompt, setPrompt] = useState('');
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [isTesting, setIsTesting] = useState(false);
    const [result, setResult] = useState<GuardrailResult | null>(null);

    const { mutateAsync: testGuardrail } = useGuardrailApi();
    const addTest = useTestHistoryStore((state) => state.addTest);

    const onDrop = (acceptedFiles: File[]) => {
        if (acceptedFiles && acceptedFiles.length > 0) setImageFile(acceptedFiles[0]);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
        maxFiles: 1,
        maxSize: 10 * 1024 * 1024, // 10MB
    });

    const handleTest = async () => {
        // For image-only mode, we need an image
        if (mode === 'image-only' && !imageFile) {
            toast.error('Image is required for Image Analysis mode');
            return;
        }

        // For standard mode, we need a prompt
        if (mode === 'standard' && !prompt.trim()) {
            toast.error('Prompt is required');
            return;
        }

        setIsTesting(true);
        try {
            let imageBytes: string | undefined = undefined;
            if (imageFile) {
                const base64 = await fileToBase64(imageFile);
                imageBytes = base64.split(',')[1];
            }

            // In image-only mode, use a neutral prompt that passes text guardrails
            const finalPrompt = mode === 'image-only' ? 'analyze this food image' : prompt;

            const response = await testGuardrail({
                prompt: finalPrompt,
                image_bytes: imageBytes,
            });

            setResult(response);
            addTest({
                id: crypto.randomUUID(),
                request: { prompt: finalPrompt, image_bytes: imageFile?.name || '' },
                result: response,
                timestamp: Date.now(),
            });

            if (response.status === 'PASS') {
                toast.success('✅ Guardrail PASSED');
            } else {
                toast.error('❌ Guardrail BLOCKED');
            }
        } catch (error) {
            console.error(error);
            toast.error('Test failed');
        } finally {
            setIsTesting(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-6">
            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2">
                            <Shield className="h-6 w-6" />
                            Imagery Guardrail Tester
                        </CardTitle>

                        <Tabs value={mode} onValueChange={(v) => setMode(v as TestMode)} className="w-[300px]">
                            <TabsList className="grid w-full grid-cols-2">
                                <TabsTrigger value="standard" className="flex items-center gap-2">
                                    <FileText className="h-4 w-4" /> Standard
                                </TabsTrigger>
                                <TabsTrigger value="image-only" className="flex items-center gap-2">
                                    <Camera className="h-4 w-4" /> Image Only
                                </TabsTrigger>
                            </TabsList>
                        </Tabs>
                    </div>
                </CardHeader>

                <CardContent className="space-y-4">
                    {/* Prompt Input - Only in Standard Mode */}
                    {mode === 'standard' && (
                        <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                            <label className="text-sm font-medium mb-2 block">Prompt</label>
                            <Textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder='e.g., "generate image with this image attached in center..."'
                                rows={4}
                                className="resize-none"
                            />
                            <p className="text-sm text-muted-foreground mt-1">
                                {prompt.length}/2000 characters
                            </p>
                        </div>
                    )}

                    {/* Image Upload */}
                    <div className="animate-in fade-in slide-in-from-top-2 duration-300 delay-75">
                        <label className="text-sm font-medium mb-2 block">
                            Image {mode === 'standard' ? '(optional)' : '(required)'}
                        </label>
                        <div
                            {...getRootProps()}
                            className={cn(
                                "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
                                isDragActive
                                    ? "border-primary bg-primary/5"
                                    : "border-muted hover:border-primary/50",
                                mode === 'image-only' && !imageFile && "border-primary/50 bg-primary/5"
                            )}
                        >
                            <input {...getInputProps()} />
                            {imageFile ? (
                                <div className="space-y-2 relative">
                                    <img
                                        src={URL.createObjectURL(imageFile)}
                                        alt="Preview"
                                        className="max-h-64 mx-auto rounded-md object-contain shadow-sm"
                                    />
                                    <Button
                                        type="button"
                                        variant="destructive"
                                        size="icon"
                                        className="absolute top-0 right-0 -mt-2 -mr-2 h-6 w-6 rounded-full shadow-md"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setImageFile(null);
                                        }}
                                    >
                                        <X className="h-3 w-3" />
                                    </Button>
                                    <p className="text-sm font-medium">{imageFile.name}</p>
                                    <p className="text-xs text-muted-foreground">
                                        {(imageFile.size / 1024 / 1024).toFixed(2)} MB
                                    </p>
                                </div>
                            ) : (
                                <div className="py-4">
                                    <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-2" />
                                    <p className="text-muted-foreground font-medium">
                                        Drag & drop image or click to browse
                                    </p>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        JPG, PNG, WebP up to 10MB
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Test Button */}
                    <Button
                        onClick={handleTest}
                        disabled={isTesting || (mode === 'standard' && !prompt.trim()) || (mode === 'image-only' && !imageFile)}
                        className="w-full h-12 text-lg mt-4"
                    >
                        {isTesting ? (
                            <>
                                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                                Processing...
                            </>
                        ) : (
                            <>
                                <ShieldCheck className="mr-2 h-5 w-5" />
                                {mode === 'standard' ? 'TEST GUARDRAIL' : 'ANALYZE IMAGE'}
                            </>
                        )}
                    </Button>
                </CardContent>
            </Card>

            {/* Results */}
            {result && (
                <ResultsCard result={result} prompt={mode === 'standard' ? prompt : undefined} imageFile={imageFile} />
            )}
        </div>
    );
}
