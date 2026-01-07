import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from '@/components/theme-provider';
import { ModeToggle } from '@/components/mode-toggle';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { GuardrailTester } from '@/components/GuardrailTester';
import { HistoryTable } from '@/components/HistoryTable';
import { BatchUploader } from '@/components/BatchUploader';
import { AnalyticsDashboard } from '@/components/AnalyticsDashboard';
import { Shield } from 'lucide-react';

const queryClient = new QueryClient();

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
                <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
                    <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-50">
                        <div className="container mx-auto py-4 px-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Shield className="h-8 w-8 text-primary" />
                                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
                                    Imagery Guardrail
                                </h1>
                            </div>
                            <ModeToggle />
                        </div>
                    </header>

                    <main className="container mx-auto py-8 px-4">
                        <Tabs defaultValue="test" className="space-y-6">
                            <TabsList className="grid w-full grid-cols-4 lg:w-[400px]">
                                <TabsTrigger value="test">Test</TabsTrigger>
                                <TabsTrigger value="batch">Batch</TabsTrigger>
                                <TabsTrigger value="history">History</TabsTrigger>
                                <TabsTrigger value="analytics">Analytics</TabsTrigger>
                            </TabsList>

                            <TabsContent value="test" className="space-y-4 animate-in fade-in-50 duration-500 slide-in-from-bottom-5">
                                <GuardrailTester />
                            </TabsContent>

                            <TabsContent value="batch" className="space-y-4 animate-in fade-in-50 duration-500 slide-in-from-bottom-5">
                                <BatchUploader />
                            </TabsContent>

                            <TabsContent value="history" className="space-y-4 animate-in fade-in-50 duration-500 slide-in-from-bottom-5">
                                <HistoryTable />
                            </TabsContent>

                            <TabsContent value="analytics" className="space-y-4 animate-in fade-in-50 duration-500 slide-in-from-bottom-5">
                                <AnalyticsDashboard />
                            </TabsContent>
                        </Tabs>
                    </main>
                    <Toaster position="bottom-right" />
                </div>
            </ThemeProvider>
        </QueryClientProvider>
    );
}

export default App;
