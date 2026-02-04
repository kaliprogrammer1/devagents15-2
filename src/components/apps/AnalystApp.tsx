"use client";

import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, TrendingDown, Database, FileCode, Brain, Play, RefreshCcw, Filter, Download, Table, LineChart, PieChart, Layers } from 'lucide-react';

interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: number;
  size: string;
  lastUpdated: Date;
}

interface MetricCard {
  label: string;
  value: string;
  change: number;
  trend: 'up' | 'down' | 'neutral';
}

interface ChartData {
  labels: string[];
  values: number[];
}

export default function AnalystApp() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'notebook' | 'sql'>('dashboard');
  const [datasets] = useState<Dataset[]>([
    { id: '1', name: 'user_behavior.csv', rows: 125420, columns: 24, size: '45.2 MB', lastUpdated: new Date() },
    { id: '2', name: 'sales_metrics.parquet', rows: 892100, columns: 18, size: '128 MB', lastUpdated: new Date(Date.now() - 3600000) },
    { id: '3', name: 'model_predictions.csv', rows: 50000, columns: 12, size: '8.5 MB', lastUpdated: new Date(Date.now() - 7200000) },
  ]);

  const [metrics, setMetrics] = useState<MetricCard[]>([
    { label: 'Conversion Rate', value: '3.42%', change: 12.5, trend: 'up' },
    { label: 'Avg Session Duration', value: '4m 32s', change: -5.2, trend: 'down' },
    { label: 'Model Accuracy', value: '94.7%', change: 2.1, trend: 'up' },
    { label: 'Data Quality Score', value: '98.2%', change: 0, trend: 'neutral' },
  ]);

  const [chartData, setChartData] = useState<ChartData>({
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    values: [65, 72, 68, 85, 92, 78, 88],
  });

  const [sqlQuery, setSqlQuery] = useState(`SELECT 
  user_id,
  COUNT(*) as sessions,
  AVG(duration) as avg_duration
FROM user_sessions
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY user_id
ORDER BY sessions DESC
LIMIT 100;`);

  const [notebookCells] = useState([
    { type: 'code', content: 'import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split', output: '' },
    { type: 'code', content: 'df = pd.read_csv("user_behavior.csv")\ndf.head()', output: '   user_id  session_duration  page_views  ...\n0  1001     245               12          ...\n1  1002     189               8           ...' },
    { type: 'markdown', content: '## Feature Engineering\nCreating derived features for the model', output: '' },
    { type: 'code', content: 'df["engagement_score"] = df["page_views"] * df["session_duration"] / 1000\ndf["engagement_score"].describe()', output: 'count    125420.0\nmean         2.45\nstd          1.23\nmin          0.01\n25%          1.42\n50%          2.31\n75%          3.28\nmax         12.89' },
  ]);

  const [isQueryRunning, setIsQueryRunning] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(m => ({
        ...m,
        change: m.change + (Math.random() - 0.5) * 2,
      })));
      setChartData(prev => ({
        ...prev,
        values: prev.values.map(v => Math.max(40, Math.min(100, v + (Math.random() - 0.5) * 10))),
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const runQuery = () => {
    setIsQueryRunning(true);
    setTimeout(() => setIsQueryRunning(false), 2000);
  };

  return (
    <div className="h-full bg-slate-50 text-slate-800 flex flex-col font-mono text-xs">
      {/* Header */}
      <div className="p-3 border-b border-slate-200 bg-white flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <BarChart3 size={16} className="text-cyan-600" />
            <span className="font-bold text-slate-900 uppercase tracking-tight">Data Analysis Hub</span>
          </div>
          <div className="h-4 w-px bg-slate-200" />
          <div className="flex gap-1">
            {(['dashboard', 'notebook', 'sql'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase transition-colors ${
                  activeTab === tab
                    ? 'bg-cyan-500 text-white'
                    : 'hover:bg-slate-100 text-slate-600'
                }`}
              >
                {tab === 'sql' ? 'SQL Editor' : tab}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="p-1.5 hover:bg-slate-100 rounded transition-colors">
            <Filter size={14} className="text-slate-500" />
          </button>
          <button className="p-1.5 hover:bg-slate-100 rounded transition-colors">
            <Download size={14} className="text-slate-500" />
          </button>
          <button className="p-1.5 hover:bg-slate-100 rounded transition-colors">
            <RefreshCcw size={14} className="text-slate-500" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'dashboard' && (
          <div className="h-full flex">
            {/* Main Dashboard */}
            <div className="flex-1 p-4 overflow-y-auto space-y-4">
              {/* Metrics Row */}
              <div className="grid grid-cols-4 gap-3">
                {metrics.map((metric, i) => (
                  <div key={i} className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                    <div className="text-[10px] text-slate-500 uppercase font-bold mb-2">{metric.label}</div>
                    <div className="flex items-end justify-between">
                      <span className="text-2xl font-bold text-slate-900">{metric.value}</span>
                      <div className={`flex items-center gap-1 text-[10px] font-bold ${
                        metric.trend === 'up' ? 'text-emerald-600' :
                        metric.trend === 'down' ? 'text-rose-600' :
                        'text-slate-500'
                      }`}>
                        {metric.trend === 'up' && <TrendingUp size={12} />}
                        {metric.trend === 'down' && <TrendingDown size={12} />}
                        {metric.change > 0 ? '+' : ''}{metric.change.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-2 gap-4">
                {/* Line Chart */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <LineChart size={14} className="text-cyan-600" />
                      <span className="text-[11px] font-bold text-slate-700">Weekly Trend</span>
                    </div>
                    <div className="text-[10px] text-slate-400">Last 7 days</div>
                  </div>
                  <div className="h-40 flex items-end justify-between gap-2">
                    {chartData.values.map((val, i) => (
                      <div key={i} className="flex-1 flex flex-col items-center gap-1">
                        <div 
                          className="w-full bg-gradient-to-t from-cyan-500 to-cyan-400 rounded-t transition-all duration-500"
                          style={{ height: `${val}%` }}
                        />
                        <span className="text-[8px] text-slate-400">{chartData.labels[i]}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Pie Chart Placeholder */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <PieChart size={14} className="text-purple-600" />
                      <span className="text-[11px] font-bold text-slate-700">Distribution</span>
                    </div>
                  </div>
                  <div className="h-40 flex items-center justify-center">
                    <div className="relative w-32 h-32">
                      <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#e2e8f0" strokeWidth="20" />
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#06b6d4" strokeWidth="20" strokeDasharray="150 251" />
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#8b5cf6" strokeWidth="20" strokeDasharray="60 251" strokeDashoffset="-150" />
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#f59e0b" strokeWidth="20" strokeDasharray="41 251" strokeDashoffset="-210" />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-lg font-bold text-slate-700">100%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-center gap-4 mt-2 text-[9px]">
                    <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-cyan-500" />Training (60%)</div>
                    <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-purple-500" />Validation (24%)</div>
                    <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-amber-500" />Test (16%)</div>
                  </div>
                </div>
              </div>

              {/* Data Preview */}
              <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                <div className="p-3 border-b border-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Table size={14} className="text-purple-600" />
                    <span className="text-[11px] font-bold text-slate-700">Sample Data Preview</span>
                  </div>
                  <span className="text-[10px] text-slate-400">Showing 5 of 125,420 rows</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px]">
                    <thead className="bg-slate-50">
                      <tr>
                        <th className="px-3 py-2 text-left font-bold text-slate-500">user_id</th>
                        <th className="px-3 py-2 text-left font-bold text-slate-500">session_duration</th>
                        <th className="px-3 py-2 text-left font-bold text-slate-500">page_views</th>
                        <th className="px-3 py-2 text-left font-bold text-slate-500">engagement_score</th>
                        <th className="px-3 py-2 text-left font-bold text-slate-500">converted</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        [1001, 245, 12, 2.94, true],
                        [1002, 189, 8, 1.51, false],
                        [1003, 312, 15, 4.68, true],
                        [1004, 98, 4, 0.39, false],
                        [1005, 456, 22, 10.03, true],
                      ].map((row, i) => (
                        <tr key={i} className="border-t border-slate-100 hover:bg-slate-50">
                          {row.map((cell, j) => (
                            <td key={j} className="px-3 py-2 text-slate-700">
                              {typeof cell === 'boolean' ? (
                                <span className={`px-1.5 py-0.5 rounded text-[8px] font-bold ${cell ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'}`}>
                                  {cell ? 'Yes' : 'No'}
                                </span>
                              ) : cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Datasets Sidebar */}
            <div className="w-64 border-l border-slate-200 bg-white p-3 space-y-3">
              <div className="flex items-center gap-2">
                <Database size={14} className="text-slate-500" />
                <span className="text-[10px] font-bold text-slate-700 uppercase">Datasets</span>
              </div>
              {datasets.map(ds => (
                <div key={ds.id} className="p-3 bg-slate-50 rounded-lg border border-slate-200 hover:border-cyan-300 transition-colors cursor-pointer">
                  <div className="font-bold text-slate-800 text-[11px] truncate">{ds.name}</div>
                  <div className="flex gap-3 mt-2 text-[9px] text-slate-500">
                    <span>{ds.rows.toLocaleString()} rows</span>
                    <span>{ds.columns} cols</span>
                    <span>{ds.size}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'notebook' && (
          <div className="h-full p-4 overflow-y-auto space-y-3 bg-white">
            <div className="flex items-center gap-2 mb-4">
              <FileCode size={16} className="text-amber-600" />
              <span className="font-bold text-slate-700">analysis_notebook.ipynb</span>
              <span className="text-[10px] text-slate-400 ml-auto">Python 3.10 | pandas, numpy, sklearn</span>
            </div>
            {notebookCells.map((cell, i) => (
              <div key={i} className="border border-slate-200 rounded-lg overflow-hidden">
                <div className={`px-3 py-1.5 text-[9px] font-bold uppercase ${
                  cell.type === 'code' ? 'bg-slate-100 text-slate-600' : 'bg-blue-50 text-blue-600'
                }`}>
                  {cell.type === 'code' ? `In [${i + 1}]` : 'Markdown'}
                </div>
                <div className="p-3 bg-slate-50">
                  <pre className="text-[11px] text-slate-700 whitespace-pre-wrap font-mono">{cell.content}</pre>
                </div>
                {cell.output && (
                  <div className="p-3 bg-white border-t border-slate-200">
                    <pre className="text-[10px] text-emerald-700 whitespace-pre-wrap font-mono">{cell.output}</pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'sql' && (
          <div className="h-full flex flex-col p-4">
            <div className="flex items-center gap-2 mb-3">
              <Database size={16} className="text-purple-600" />
              <span className="font-bold text-slate-700 text-sm">SQL Query Editor</span>
              <button
                onClick={runQuery}
                disabled={isQueryRunning}
                className="ml-auto flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-[10px] font-bold transition-colors disabled:opacity-50"
              >
                <Play size={12} />
                {isQueryRunning ? 'Running...' : 'Run Query'}
              </button>
            </div>
            <div className="flex-1 flex gap-4">
              <div className="flex-1 flex flex-col">
                <textarea
                  value={sqlQuery}
                  onChange={(e) => setSqlQuery(e.target.value)}
                  className="flex-1 p-4 bg-slate-900 text-emerald-400 font-mono text-[11px] rounded-lg border border-slate-700 focus:outline-none focus:border-purple-500 resize-none"
                  spellCheck={false}
                />
              </div>
              <div className="w-48 space-y-2">
                <div className="text-[10px] font-bold text-slate-500 uppercase">Schema</div>
                {['user_sessions', 'users', 'events', 'conversions'].map(table => (
                  <div key={table} className="p-2 bg-white rounded border border-slate-200 text-[10px]">
                    <div className="flex items-center gap-1.5 text-purple-600 font-bold">
                      <Layers size={10} />
                      {table}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-2 px-4 border-t border-slate-200 bg-slate-50 text-[9px] text-slate-500 flex justify-between items-center">
        <span>ANALYST_MODE // DATA_EXPLORATION // STATISTICAL_ANALYSIS</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
          <span className="text-cyan-600">INSIGHTS ENGINE ACTIVE</span>
        </div>
      </div>
    </div>
  );
}
