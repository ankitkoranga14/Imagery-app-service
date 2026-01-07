import { useTestHistoryStore } from '@/stores/testHistory';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, BarChart, Bar, XAxis, YAxis } from 'recharts';

export function AnalyticsDashboard() {
    const history = useTestHistoryStore((state) => state.history);

    const passFailData = [
        { name: 'Pass', value: history.filter(h => h.result.status === 'PASS').length, color: '#22c55e' },
        { name: 'Block', value: history.filter(h => h.result.status === 'BLOCK').length, color: '#ef4444' },
    ];

    const foodTypeCounts = history.reduce((acc, curr) => {
        const type = curr.result.metadata.food_type || 'Unknown';
        acc[type] = (acc[type] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    const foodTypeData = Object.entries(foodTypeCounts)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 5);

    const avgProcessingTime = history.length
        ? history.reduce((acc, curr) => acc + curr.result.metadata.processing_time_ms, 0) / history.length
        : 0;

    return (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <Card>
                <CardHeader>
                    <CardTitle>Pass/Fail Ratio</CardTitle>
                </CardHeader>
                <CardContent className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={passFailData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {passFailData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Top Food Types</CardTitle>
                </CardHeader>
                <CardContent className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={foodTypeData} layout="vertical" margin={{ left: 20 }}>
                            <XAxis type="number" hide />
                            <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                            <Tooltip />
                            <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Stats</CardTitle>
                </CardHeader>
                <CardContent className="space-y-8">
                    <div>
                        <p className="text-sm font-medium text-muted-foreground">Total Tests</p>
                        <p className="text-4xl font-bold">{history.length}</p>
                    </div>
                    <div>
                        <p className="text-sm font-medium text-muted-foreground">Avg Processing Time</p>
                        <p className="text-4xl font-bold">{avgProcessingTime.toFixed(0)}ms</p>
                    </div>
                    <div>
                        <p className="text-sm font-medium text-muted-foreground">Pass Rate</p>
                        <p className="text-4xl font-bold">
                            {history.length ? ((passFailData[0].value / history.length) * 100).toFixed(1) : 0}%
                        </p>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
