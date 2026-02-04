"use client";

import { useState, useEffect, useRef } from 'react';
import { Box, GitBranch, Layers, Database, Server, Cloud, ArrowRight, Zap, Shield, RefreshCcw, Plus, ZoomIn, ZoomOut, Move } from 'lucide-react';

interface ArchNode {
  id: string;
  type: 'service' | 'database' | 'api' | 'client' | 'queue' | 'cache';
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'idle';
  x: number;
  y: number;
  connections: string[];
  metrics?: { latency: number; throughput: number };
}

const defaultNodes: ArchNode[] = [
  { id: 'client', type: 'client', name: 'Web Client', status: 'healthy', x: 50, y: 150, connections: ['api-gateway'], metrics: { latency: 12, throughput: 1200 } },
  { id: 'api-gateway', type: 'api', name: 'API Gateway', status: 'healthy', x: 200, y: 150, connections: ['auth-service', 'core-service'], metrics: { latency: 8, throughput: 5000 } },
  { id: 'auth-service', type: 'service', name: 'Auth Service', status: 'healthy', x: 380, y: 80, connections: ['user-db', 'redis'], metrics: { latency: 15, throughput: 800 } },
  { id: 'core-service', type: 'service', name: 'Core Service', status: 'warning', x: 380, y: 220, connections: ['main-db', 'queue'], metrics: { latency: 45, throughput: 2200 } },
  { id: 'user-db', type: 'database', name: 'User DB', status: 'healthy', x: 550, y: 50, connections: [], metrics: { latency: 3, throughput: 10000 } },
  { id: 'main-db', type: 'database', name: 'Main DB', status: 'healthy', x: 550, y: 180, connections: [], metrics: { latency: 5, throughput: 8000 } },
  { id: 'redis', type: 'cache', name: 'Redis Cache', status: 'healthy', x: 550, y: 110, connections: [], metrics: { latency: 1, throughput: 50000 } },
  { id: 'queue', type: 'queue', name: 'Message Queue', status: 'healthy', x: 550, y: 280, connections: ['worker'], metrics: { latency: 2, throughput: 15000 } },
  { id: 'worker', type: 'service', name: 'Worker Service', status: 'idle', x: 700, y: 280, connections: [], metrics: { latency: 0, throughput: 0 } },
];

export default function ArchitectApp() {
  const [nodes, setNodes] = useState<ArchNode[]>(defaultNodes);
  const [selectedNode, setSelectedNode] = useState<ArchNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setNodes(prev => prev.map(node => ({
        ...node,
        metrics: node.metrics ? {
          latency: Math.max(1, node.metrics.latency + (Math.random() - 0.5) * 5),
          throughput: Math.max(0, node.metrics.throughput + (Math.random() - 0.5) * 200),
        } : undefined
      })));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const getNodeIcon = (type: ArchNode['type']) => {
    switch (type) {
      case 'service': return Server;
      case 'database': return Database;
      case 'api': return Zap;
      case 'client': return Box;
      case 'queue': return GitBranch;
      case 'cache': return Layers;
      default: return Box;
    }
  };

  const getStatusColor = (status: ArchNode['status']) => {
    switch (status) {
      case 'healthy': return 'bg-emerald-500';
      case 'warning': return 'bg-amber-500';
      case 'error': return 'bg-rose-500';
      case 'idle': return 'bg-slate-500';
    }
  };

  const getNodeStyle = (type: ArchNode['type']) => {
    switch (type) {
      case 'service': return 'bg-blue-500/10 border-blue-500/30 text-blue-400';
      case 'database': return 'bg-purple-500/10 border-purple-500/30 text-purple-400';
      case 'api': return 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400';
      case 'client': return 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400';
      case 'queue': return 'bg-amber-500/10 border-amber-500/30 text-amber-400';
      case 'cache': return 'bg-rose-500/10 border-rose-500/30 text-rose-400';
    }
  };

  return (
    <div className="h-full bg-slate-50 text-slate-800 flex flex-col font-mono text-xs">
      {/* Header */}
      <div className="p-3 border-b border-slate-200 bg-white flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Layers size={16} className="text-blue-500" />
            <span className="font-bold text-slate-900 uppercase tracking-tight">System Architecture</span>
          </div>
          <div className="h-4 w-px bg-slate-200" />
          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-slate-400">Nodes:</span>
            <span className="text-blue-600 font-bold">{nodes.length}</span>
            <span className="text-slate-400 ml-2">Connections:</span>
            <span className="text-purple-600 font-bold">{nodes.reduce((acc, n) => acc + n.connections.length, 0)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setZoom(z => Math.max(0.5, z - 0.1))} className="p-1.5 hover:bg-slate-100 rounded transition-colors">
            <ZoomOut size={14} className="text-slate-500" />
          </button>
          <span className="text-[10px] text-slate-500 w-12 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(2, z + 0.1))} className="p-1.5 hover:bg-slate-100 rounded transition-colors">
            <ZoomIn size={14} className="text-slate-500" />
          </button>
          <button className="p-1.5 hover:bg-slate-100 rounded transition-colors ml-2">
            <RefreshCcw size={14} className="text-slate-500" />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 flex">
        <div ref={canvasRef} className="flex-1 relative overflow-hidden bg-[linear-gradient(to_right,#f1f5f9_1px,transparent_1px),linear-gradient(to_bottom,#f1f5f9_1px,transparent_1px)] bg-[size:20px_20px]">
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)` }}>
            {nodes.map(node => 
              node.connections.map(targetId => {
                const target = nodes.find(n => n.id === targetId);
                if (!target) return null;
                return (
                  <g key={`${node.id}-${targetId}`}>
                    <line
                      x1={node.x + 60}
                      y1={node.y + 25}
                      x2={target.x}
                      y2={target.y + 25}
                      stroke="#94a3b8"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                      className="animate-pulse"
                    />
                    <circle cx={(node.x + 60 + target.x) / 2} cy={(node.y + target.y + 50) / 2} r="3" fill="#3b82f6" className="animate-ping" />
                  </g>
                );
              })
            )}
          </svg>

          <div style={{ transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`, transformOrigin: '0 0' }}>
            {nodes.map(node => {
              const Icon = getNodeIcon(node.type);
              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNode(node)}
                  className={`absolute cursor-pointer transition-all duration-200 hover:scale-105 ${selectedNode?.id === node.id ? 'ring-2 ring-blue-500 ring-offset-2' : ''}`}
                  style={{ left: node.x, top: node.y }}
                >
                  <div className={`relative px-3 py-2 rounded-lg border ${getNodeStyle(node.type)} backdrop-blur-sm min-w-[120px]`}>
                    <div className={`absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full ${getStatusColor(node.status)} border-2 border-white`} />
                    <div className="flex items-center gap-2">
                      <Icon size={14} />
                      <span className="font-bold text-[10px]">{node.name}</span>
                    </div>
                    {node.metrics && (
                      <div className="mt-1 flex gap-2 text-[8px] text-slate-500">
                        <span>{node.metrics.latency.toFixed(0)}ms</span>
                        <span>{(node.metrics.throughput / 1000).toFixed(1)}k/s</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Details Panel */}
        {selectedNode && (
          <div className="w-64 border-l border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-slate-900">{selectedNode.name}</h3>
              <div className={`w-2 h-2 rounded-full ${getStatusColor(selectedNode.status)}`} />
            </div>
            
            <div className="space-y-3">
              <div className="bg-slate-50 rounded-lg p-3">
                <div className="text-[10px] text-slate-500 uppercase font-bold mb-2">Type</div>
                <div className="text-sm font-bold text-slate-700 capitalize">{selectedNode.type}</div>
              </div>
              
              {selectedNode.metrics && (
                <>
                  <div className="bg-slate-50 rounded-lg p-3">
                    <div className="text-[10px] text-slate-500 uppercase font-bold mb-2">Latency</div>
                    <div className="text-xl font-bold text-blue-600">{selectedNode.metrics.latency.toFixed(1)}ms</div>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-3">
                    <div className="text-[10px] text-slate-500 uppercase font-bold mb-2">Throughput</div>
                    <div className="text-xl font-bold text-purple-600">{selectedNode.metrics.throughput.toFixed(0)}/s</div>
                  </div>
                </>
              )}
              
              <div className="bg-slate-50 rounded-lg p-3">
                <div className="text-[10px] text-slate-500 uppercase font-bold mb-2">Connections</div>
                <div className="space-y-1">
                  {selectedNode.connections.length > 0 ? (
                    selectedNode.connections.map(c => (
                      <div key={c} className="flex items-center gap-2 text-[10px]">
                        <ArrowRight size={10} className="text-slate-400" />
                        <span className="text-slate-700">{nodes.find(n => n.id === c)?.name || c}</span>
                      </div>
                    ))
                  ) : (
                    <span className="text-slate-400 text-[10px]">No outbound connections</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-2 px-4 border-t border-slate-200 bg-slate-50 text-[9px] text-slate-500 flex justify-between items-center">
        <span>ARCHITECTURE_VIEW // INTERACTIVE_MODE // REAL_TIME_METRICS</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          <span>ALL SYSTEMS OPERATIONAL</span>
        </div>
      </div>
    </div>
  );
}
