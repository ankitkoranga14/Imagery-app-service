import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from '@/components/theme-provider';
import { ModeToggle } from '@/components/mode-toggle';
import { PipelineDashboard } from '@/components/PipelineDashboard';
import { Workflow } from 'lucide-react';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 5000,
            retry: 1,
        },
    },
});

// Simplified Imagery Pipeline - Only Pipeline tab, no Guardrail/Batch/History/Analytics
function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
                <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 text-foreground transition-colors duration-300">
                    {/* Ambient background effects */}
                    <div className="fixed inset-0 -z-10 overflow-hidden">
                        <div className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-to-br from-indigo-500/5 via-transparent to-transparent rounded-full blur-3xl" />
                        <div className="absolute -bottom-1/2 -left-1/2 w-full h-full bg-gradient-to-tr from-emerald-500/5 via-transparent to-transparent rounded-full blur-3xl" />
                    </div>
                    
                    <header className="border-b border-zinc-800/50 sticky top-0 bg-zinc-950/80 backdrop-blur-xl supports-[backdrop-filter]:bg-zinc-950/60 z-50">
                        <div className="container mx-auto py-4 px-6 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="relative">
                                    <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-500 blur-lg opacity-50" />
                                    <Workflow className="relative h-8 w-8 text-white" />
                                </div>
                                <div>
                                    <h1 className="text-xl font-bold tracking-tight text-white">
                                        Imagery Pipeline
                                    </h1>
                                    <p className="text-xs text-zinc-500 font-medium tracking-wide">
                                        AI-POWERED IMAGE PROCESSING
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-4">
                                <div className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                                    <span className="text-xs font-medium text-emerald-400">v1.0.0</span>
                                </div>
                                <ModeToggle />
                            </div>
                        </div>
                    </header>

                    <main className="container mx-auto py-8 px-6">
                        <PipelineDashboard />
                    </main>
                    
                    <footer className="border-t border-zinc-800/50 mt-12 py-6">
                        <div className="container mx-auto px-6 flex items-center justify-between text-xs text-zinc-600">
                            <span>© 2024 Imagery Pipeline. All rights reserved.</span>
                            <div className="flex items-center gap-4">
                                <a href="/api/docs" className="hover:text-zinc-400 transition-colors">API Docs</a>
                            </div>
                        </div>
                    </footer>
                    
                    <Toaster 
                        position="bottom-right" 
                        toastOptions={{
                            style: {
                                background: '#18181b',
                                color: '#fff',
                                border: '1px solid #27272a',
                            },
                        }}
                    />
                </div>
            </ThemeProvider>
        </QueryClientProvider>
    );
}

export default App;
