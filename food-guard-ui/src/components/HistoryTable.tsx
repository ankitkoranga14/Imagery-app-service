import { useState } from 'react';
import { useTestHistoryStore } from '@/stores/testHistory';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Trash2, Search } from 'lucide-react';

export function HistoryTable() {
    const { history, clearHistory } = useTestHistoryStore();
    const [search, setSearch] = useState('');

    const filteredHistory = history.filter((test) =>
        test.request.prompt.toLowerCase().includes(search.toLowerCase()) ||
        test.result.metadata.food_type?.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between gap-4">
                <div className="relative flex-1 max-w-sm">
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                        placeholder="Search history..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="pl-8"
                    />
                </div>
                <Button variant="destructive" size="sm" onClick={clearHistory}>
                    <Trash2 className="mr-2 h-4 w-4" />
                    Clear History
                </Button>
            </div>

            <div className="rounded-md border">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Time</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead className="max-w-[300px]">Prompt</TableHead>
                            <TableHead>Food Type</TableHead>
                            <TableHead>Scores</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {filteredHistory.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={5} className="text-center h-24 text-muted-foreground">
                                    No history found
                                </TableCell>
                            </TableRow>
                        ) : (
                            filteredHistory.map((test) => (
                                <TableRow key={test.id}>
                                    <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                                        {new Date(test.timestamp).toLocaleString()}
                                    </TableCell>
                                    <TableCell>
                                        <Badge variant={test.result.status === 'PASS' ? 'default' : 'destructive'}>
                                            {test.result.status}
                                        </Badge>
                                    </TableCell>
                                    <TableCell className="max-w-[300px] truncate text-sm" title={test.request.prompt}>
                                        {test.request.prompt}
                                    </TableCell>
                                    <TableCell className="text-sm">
                                        {test.result.metadata.food_type || '-'}
                                    </TableCell>
                                    <TableCell>
                                        <div className="flex flex-col gap-1 text-xs">
                                            {Object.entries(test.result.scores).slice(0, 2).map(([k, v]) => (
                                                <div key={k} className="flex justify-between gap-2">
                                                    <span className="opacity-70">{k.split('_')[0]}:</span>
                                                    <span className="font-mono">{v.toFixed(2)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </TableCell>
                                </TableRow>
                            ))
                        )}
                    </TableBody>
                </Table>
            </div>
        </div>
    );
}
