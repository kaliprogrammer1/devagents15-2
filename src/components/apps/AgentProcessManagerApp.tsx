"use client";

import { useState, useEffect } from 'react';
import { Cpu, Activity, Clock, CheckCircle2, AlertCircle, Play, Square, RefreshCcw, Layers } from 'lucide-react';

interface Process {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'idle';
  cpu: number;
  memory: number;
  runtime: string;
  lastAction: string;
}

export default function AgentProcessManagerApp() {
  const [processes, setProcesses] = useState<Process[]>([]);
    const [systemStats, setSystemStats] = useState({
      cpuUsage: 0,
      memoryUsage: 0,
      diskUsage: 0,
      uptime: '00:00:00',
      activeTasks: 0,
      memDistribution: { workspace: 0, browser: 0, system: 0, free: 100 },
      pid: 0,
      port: 3000
    });
  const [loading, setLoading] = useState(true);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/system/stats');
      const data = await res.json();
      if (data.success) {
        setProcesses(data.processes);
        setSystemStats(data.stats);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full bg-[#0f172a] text-slate-300 flex flex-col font-mono text-xs">
      <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-cyan-400" />
            <span className="font-bold text-slate-100 uppercase tracking-tighter">Core Monitor</span>
          </div>
          <div className="h-4 w-[1px] bg-slate-700" />
          <div className="flex items-center gap-4 text-[10px]">
            <div className="flex items-center gap-1">
              <span className="text-slate-500">CPU</span>
              <span className="text-cyan-400 font-bold">{systemStats.cpuUsage}%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-slate-500">MEM</span>
              <span className="text-purple-400 font-bold">{systemStats.memoryUsage}%</span>
            </div>
          </div>
        </div>
        <button className="p-1.5 hover:bg-slate-800 rounded transition-colors">
          <RefreshCcw size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Uptime', value: systemStats.uptime, icon: Clock, color: 'text-blue-400' },
              { label: 'Active Processes', value: systemStats.activeTasks, icon: Activity, color: 'text-emerald-400' },
              { label: 'Disk Usage', value: `${systemStats.diskUsage}%`, icon: Layers, color: 'text-cyan-400' },
              { label: 'System Health', value: 'OPTIMAL', icon: CheckCircle2, color: 'text-emerald-500' },
            ].map((stat, i) => (

            <div key={i} className="bg-slate-900/40 border border-slate-800 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <stat.icon size={12} className={stat.color} />
                <span className="text-[10px] text-slate-500 uppercase font-bold">{stat.label}</span>
              </div>
              <div className="text-lg font-bold text-slate-100 tracking-tight">{stat.value}</div>
            </div>
          ))}
        </div>

        <div className="bg-slate-900/40 border border-slate-800 rounded-lg overflow-hidden">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-800/50 text-[10px] text-slate-500 uppercase font-bold">
                <th className="p-3">Process</th>
                <th className="p-3 text-center">Status</th>
                <th className="p-3 text-right">CPU</th>
                <th className="p-3 text-right">MEM</th>
                <th className="p-3">Runtime</th>
                <th className="p-3">Last Action</th>
                <th className="p-3"></th>
              </tr>
            </thead>
            <tbody>
              {processes.map((p) => (
                <tr key={p.id} className="border-t border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${p.status === 'running' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`} />
                      <span className="font-bold text-slate-200">{p.name}</span>
                    </div>
                  </td>
                  <td className="p-3 text-center">
                    <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${
                      p.status === 'running' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-slate-800 text-slate-500'
                    }`}>
                      {p.status}
                    </span>
                  </td>
                  <td className="p-3 text-right font-bold text-cyan-400">{p.cpu}%</td>
                  <td className="p-3 text-right font-bold text-purple-400">{p.memory}MB</td>
                  <td className="p-3 text-slate-400">{p.runtime}</td>
                  <td className="p-3 text-slate-500 truncate max-w-[150px] italic">"{p.lastAction}"</td>
                  <td className="p-3">
                    <div className="flex items-center gap-1">
                      <button className="p-1 hover:text-emerald-400 transition-colors"><Play size={12} /></button>
                      <button className="p-1 hover:text-rose-400 transition-colors"><Square size={12} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-900/40 border border-slate-800 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-4">
              <Layers size={14} className="text-cyan-400" />
              <span className="font-bold uppercase tracking-widest text-[10px]">Memory Distribution</span>
            </div>
            <div className="h-4 w-full bg-slate-800 rounded-full flex overflow-hidden">
              <div className="h-full bg-cyan-500 transition-all duration-500" style={{ width: `${systemStats.memDistribution.workspace}%` }} />
              <div className="h-full bg-purple-500 transition-all duration-500" style={{ width: `${systemStats.memDistribution.browser}%` }} />
              <div className="h-full bg-blue-500 transition-all duration-500" style={{ width: `${systemStats.memDistribution.system}%` }} />
              <div className="h-full bg-slate-700 transition-all duration-500" style={{ width: `${systemStats.memDistribution.free}%` }} />
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2 text-[9px]">
              <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-sm bg-cyan-500" /> Workspace (Node)</div>
              <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-sm bg-purple-500" /> Browser (Agent)</div>
              <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-sm bg-blue-500" /> Background (OS)</div>
              <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-sm bg-slate-700" /> Free Memory</div>
            </div>
          </div>
          <div className="bg-slate-900/40 border border-slate-800 p-4 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <Activity size={32} className="text-slate-700 mx-auto mb-2" />
              <div className="text-[10px] text-slate-500 font-bold uppercase">System Healthy</div>
              <div className="text-[9px] text-slate-600 mt-1 italic">All subsystems operating within parameters</div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-2 px-4 border-t border-slate-800 bg-slate-950 text-[9px] text-slate-600 flex justify-between items-center">
        <span>PID: {systemStats.pid} // PORT: {systemStats.port} // AGENT_MODE: AUTONOMOUS</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          <span>CONNECTED TO CORE</span>
        </div>
      </div>
    </div>
  );
}
