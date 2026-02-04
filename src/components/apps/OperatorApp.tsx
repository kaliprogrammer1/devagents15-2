"use client";

import { useState, useEffect } from 'react';
import { AlertTriangle, Shield, Activity, Cpu, HardDrive, Network, Clock, Zap, RefreshCcw, Bell, CheckCircle, XCircle, ArrowUp, ArrowDown } from 'lucide-react';

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  message: string;
  source: string;
  timestamp: Date;
  acknowledged: boolean;
}

interface ServiceStatus {
  name: string;
  status: 'operational' | 'degraded' | 'down';
  uptime: number;
  latency: number;
  requests: number;
}

export default function OperatorApp() {
  const [alerts, setAlerts] = useState<Alert[]>([
    { id: '1', type: 'warning', message: 'High memory usage detected on worker-03', source: 'worker-03', timestamp: new Date(), acknowledged: false },
    { id: '2', type: 'info', message: 'Deployment completed successfully', source: 'ci-pipeline', timestamp: new Date(Date.now() - 300000), acknowledged: true },
    { id: '3', type: 'critical', message: 'Database connection pool exhausted', source: 'db-primary', timestamp: new Date(Date.now() - 60000), acknowledged: false },
  ]);
  
  const [services, setServices] = useState<ServiceStatus[]>([
    { name: 'API Gateway', status: 'operational', uptime: 99.99, latency: 12, requests: 15420 },
    { name: 'Auth Service', status: 'operational', uptime: 99.95, latency: 8, requests: 8932 },
    { name: 'Worker Cluster', status: 'degraded', uptime: 98.50, latency: 45, requests: 3201 },
    { name: 'Database Primary', status: 'operational', uptime: 99.99, latency: 3, requests: 45230 },
    { name: 'Cache Layer', status: 'operational', uptime: 100, latency: 1, requests: 89102 },
    { name: 'CDN Edge', status: 'operational', uptime: 99.99, latency: 5, requests: 120450 },
  ]);

  const [metrics, setMetrics] = useState({
    cpu: 45,
    memory: 68,
    network: { in: 245, out: 189 },
    disk: 42,
    activeConnections: 1247,
    requestsPerSec: 3420,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        cpu: Math.max(10, Math.min(90, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(30, Math.min(95, prev.memory + (Math.random() - 0.5) * 5)),
        network: {
          in: Math.max(50, prev.network.in + (Math.random() - 0.5) * 50),
          out: Math.max(50, prev.network.out + (Math.random() - 0.5) * 50),
        },
        disk: prev.disk,
        activeConnections: Math.max(500, prev.activeConnections + Math.floor((Math.random() - 0.5) * 100)),
        requestsPerSec: Math.max(1000, prev.requestsPerSec + Math.floor((Math.random() - 0.5) * 500)),
      }));

      setServices(prev => prev.map(s => ({
        ...s,
        latency: Math.max(1, s.latency + (Math.random() - 0.5) * 5),
        requests: s.requests + Math.floor(Math.random() * 100),
      })));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const acknowledgeAlert = (id: string) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true } : a));
  };

  const criticalCount = alerts.filter(a => a.type === 'critical' && !a.acknowledged).length;

  return (
    <div className="h-full bg-[#0a0a12] text-slate-300 flex flex-col font-mono text-xs overflow-hidden">
      {/* Header - Urgent feel */}
      <div className={`p-3 border-b ${criticalCount > 0 ? 'bg-rose-900/30 border-rose-500/50' : 'bg-slate-900/50 border-slate-800'} flex items-center justify-between`}>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Shield size={16} className={criticalCount > 0 ? 'text-rose-400 animate-pulse' : 'text-purple-400'} />
            <span className="font-bold text-white uppercase tracking-tight">Operations Center</span>
          </div>
          {criticalCount > 0 && (
            <div className="flex items-center gap-2 px-3 py-1 bg-rose-500/20 border border-rose-500/40 rounded-lg animate-pulse">
              <AlertTriangle size={12} className="text-rose-400" />
              <span className="text-rose-400 font-bold text-[10px]">{criticalCount} CRITICAL ALERT{criticalCount > 1 ? 'S' : ''}</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-[10px]">
            <Clock size={12} className="text-slate-500" />
            <span className="text-slate-400">{new Date().toLocaleTimeString()}</span>
          </div>
          <button className="p-1.5 hover:bg-slate-800 rounded transition-colors">
            <RefreshCcw size={14} className="text-slate-500" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Main Dashboard */}
        <div className="flex-1 p-4 space-y-4 overflow-y-auto">
          {/* Metrics Row */}
          <div className="grid grid-cols-6 gap-3">
            {[
              { label: 'CPU', value: `${metrics.cpu.toFixed(0)}%`, icon: Cpu, color: metrics.cpu > 80 ? 'text-rose-400' : 'text-cyan-400' },
              { label: 'Memory', value: `${metrics.memory.toFixed(0)}%`, icon: HardDrive, color: metrics.memory > 85 ? 'text-rose-400' : 'text-purple-400' },
              { label: 'Network In', value: `${metrics.network.in.toFixed(0)} MB/s`, icon: ArrowDown, color: 'text-emerald-400' },
              { label: 'Network Out', value: `${metrics.network.out.toFixed(0)} MB/s`, icon: ArrowUp, color: 'text-blue-400' },
              { label: 'Connections', value: metrics.activeConnections.toLocaleString(), icon: Network, color: 'text-amber-400' },
              { label: 'Req/sec', value: metrics.requestsPerSec.toLocaleString(), icon: Zap, color: 'text-pink-400' },
            ].map((m, i) => (
              <div key={i} className="bg-slate-900/50 border border-slate-800 p-3 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <m.icon size={12} className={m.color} />
                  <span className="text-[9px] text-slate-500 uppercase font-bold">{m.label}</span>
                </div>
                <div className={`text-lg font-bold ${m.color}`}>{m.value}</div>
              </div>
            ))}
          </div>

          {/* Services Grid */}
          <div className="bg-slate-900/30 border border-slate-800 rounded-lg overflow-hidden">
            <div className="p-3 border-b border-slate-800 flex items-center gap-2">
              <Activity size={14} className="text-purple-400" />
              <span className="font-bold uppercase text-[10px] tracking-widest">Service Status</span>
            </div>
            <div className="p-3 grid grid-cols-3 gap-3">
              {services.map((service, i) => (
                <div key={i} className={`p-3 rounded-lg border ${
                  service.status === 'operational' ? 'bg-emerald-500/5 border-emerald-500/20' :
                  service.status === 'degraded' ? 'bg-amber-500/5 border-amber-500/20' :
                  'bg-rose-500/5 border-rose-500/20'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-white text-[11px]">{service.name}</span>
                    <div className={`w-2 h-2 rounded-full ${
                      service.status === 'operational' ? 'bg-emerald-500' :
                      service.status === 'degraded' ? 'bg-amber-500 animate-pulse' :
                      'bg-rose-500 animate-pulse'
                    }`} />
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-[9px]">
                    <div>
                      <div className="text-slate-500">Uptime</div>
                      <div className="text-slate-300 font-bold">{service.uptime}%</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Latency</div>
                      <div className="text-slate-300 font-bold">{service.latency.toFixed(0)}ms</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Requests</div>
                      <div className="text-slate-300 font-bold">{(service.requests / 1000).toFixed(1)}k</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: 'Scale Up', icon: ArrowUp, color: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20' },
              { label: 'Restart Services', icon: RefreshCcw, color: 'bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20' },
              { label: 'Enable Shield', icon: Shield, color: 'bg-purple-500/10 border-purple-500/30 text-purple-400 hover:bg-purple-500/20' },
              { label: 'Emergency Stop', icon: XCircle, color: 'bg-rose-500/10 border-rose-500/30 text-rose-400 hover:bg-rose-500/20' },
            ].map((action, i) => (
              <button key={i} className={`p-3 rounded-lg border ${action.color} transition-all flex items-center justify-center gap-2`}>
                <action.icon size={14} />
                <span className="font-bold text-[10px] uppercase">{action.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Alerts Sidebar */}
        <div className="w-80 border-l border-slate-800 bg-slate-900/30 flex flex-col">
          <div className="p-3 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell size={14} className="text-amber-400" />
              <span className="font-bold uppercase text-[10px] tracking-widest">Live Alerts</span>
            </div>
            <span className="text-[10px] text-slate-500">{alerts.filter(a => !a.acknowledged).length} unread</span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {alerts.map(alert => (
              <div 
                key={alert.id}
                className={`p-3 rounded-lg border ${
                  alert.type === 'critical' ? 'bg-rose-500/10 border-rose-500/30' :
                  alert.type === 'warning' ? 'bg-amber-500/10 border-amber-500/30' :
                  'bg-slate-800/50 border-slate-700'
                } ${alert.acknowledged ? 'opacity-50' : ''}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    {alert.type === 'critical' ? <AlertTriangle size={12} className="text-rose-400" /> :
                     alert.type === 'warning' ? <AlertTriangle size={12} className="text-amber-400" /> :
                     <CheckCircle size={12} className="text-blue-400" />}
                    <span className={`text-[9px] font-bold uppercase ${
                      alert.type === 'critical' ? 'text-rose-400' :
                      alert.type === 'warning' ? 'text-amber-400' :
                      'text-blue-400'
                    }`}>{alert.type}</span>
                  </div>
                  {!alert.acknowledged && (
                    <button 
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="text-[8px] px-2 py-0.5 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors"
                    >
                      ACK
                    </button>
                  )}
                </div>
                <p className="text-slate-300 text-[10px] mt-2 leading-relaxed">{alert.message}</p>
                <div className="flex items-center justify-between mt-2 text-[8px] text-slate-500">
                  <span>{alert.source}</span>
                  <span>{alert.timestamp.toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="p-2 px-4 border-t border-slate-800 bg-slate-950 text-[9px] text-slate-600 flex justify-between items-center">
        <span>OPERATOR_MODE // LIVE_MONITORING // RESPONSE_TIME: &lt;50ms</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
          <span className="text-purple-400">ALWAYS WATCHING</span>
        </div>
      </div>
    </div>
  );
}
