"use client";

import { useState, useEffect, useRef } from 'react';
import { Terminal, Shield, Network, Radio, AlertTriangle, Lock, Unlock, Eye, Wifi, Server, Globe, Zap, Skull, FileCode, ChevronRight } from 'lucide-react';

interface ScanResult {
  id: string;
  type: 'port' | 'vuln' | 'info';
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  target: string;
  port?: number;
  service?: string;
  message: string;
  timestamp: Date;
}

interface NetworkNode {
  id: string;
  ip: string;
  hostname: string;
  status: 'online' | 'offline' | 'scanning';
  openPorts: number[];
  os?: string;
}

export default function HackerApp() {
  const [activeTab, setActiveTab] = useState<'terminal' | 'scanner' | 'network' | 'traffic'>('terminal');
  const [terminalHistory, setTerminalHistory] = useState<string[]>([
    '╔════════════════════════════════════════════════════════════╗',
    '║  ███████╗███████╗ ██████╗    ████████╗███████╗██████╗ ███╗ ║',
    '║  ██╔════╝██╔════╝██╔════╝    ╚══██╔══╝██╔════╝██╔══██╗████╗║',
    '║  ███████╗█████╗  ██║            ██║   █████╗  ██████╔╝██╔██║',
    '║  ╚════██║██╔══╝  ██║            ██║   ██╔══╝  ██╔══██╗████║║',
    '║  ███████║███████╗╚██████╗       ██║   ███████╗██║  ██║╚███║║',
    '║  ╚══════╝╚══════╝ ╚═════╝       ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚══╝║',
    '╚════════════════════════════════════════════════════════════╝',
    '',
    '[*] Security Terminal v3.7.1 initialized',
    '[*] Loaded 847 exploit modules',
    '[*] Loaded 312 auxiliary modules',
    '[*] Type "help" for available commands',
    '',
  ]);
  const [currentInput, setCurrentInput] = useState('');
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [scanTarget, setScanTarget] = useState('192.168.1.0/24');
  const [networkNodes, setNetworkNodes] = useState<NetworkNode[]>([
    { id: '1', ip: '192.168.1.1', hostname: 'gateway.local', status: 'online', openPorts: [22, 80, 443], os: 'Linux' },
    { id: '2', ip: '192.168.1.10', hostname: 'web-server', status: 'online', openPorts: [80, 443, 8080, 3306], os: 'Ubuntu 22.04' },
    { id: '3', ip: '192.168.1.20', hostname: 'db-server', status: 'online', openPorts: [3306, 5432, 27017], os: 'Debian' },
    { id: '4', ip: '192.168.1.100', hostname: 'unknown', status: 'scanning', openPorts: [], os: undefined },
  ]);
  const [packetLog, setPacketLog] = useState<string[]>([]);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalHistory]);

  useEffect(() => {
    // Simulate packet capture
    const interval = setInterval(() => {
      if (activeTab === 'traffic') {
        const protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS', 'SSH'];
        const newPacket = `[${new Date().toLocaleTimeString()}] ${protocols[Math.floor(Math.random() * protocols.length)]} 192.168.1.${Math.floor(Math.random() * 255)}:${Math.floor(Math.random() * 65535)} -> 192.168.1.${Math.floor(Math.random() * 255)}:${[80, 443, 22, 3306, 53][Math.floor(Math.random() * 5)]} len=${Math.floor(Math.random() * 1500)}`;
        setPacketLog(prev => [...prev.slice(-50), newPacket]);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [activeTab]);

  const executeCommand = (cmd: string) => {
    const output: string[] = [`root@kali:~# ${cmd}`];
    const cmdLower = cmd.toLowerCase().trim();

    if (cmdLower === 'help') {
      output.push(...[
        '',
        'Available commands:',
        '  nmap <target>     - Scan target for open ports',
        '  nikto <target>    - Web vulnerability scanner',
        '  sqlmap <url>      - SQL injection scanner',
        '  hydra <target>    - Password brute force',
        '  msfconsole        - Start Metasploit',
        '  searchsploit <q>  - Search exploit database',
        '  clear             - Clear terminal',
        '',
      ]);
    } else if (cmdLower === 'clear') {
      setTerminalHistory([]);
      return;
    } else if (cmdLower.startsWith('nmap')) {
      output.push(...[
        '',
        'Starting Nmap 7.94 ( https://nmap.org )',
        `Scanning ${cmdLower.split(' ')[1] || '192.168.1.1'}...`,
        '',
        'PORT     STATE SERVICE     VERSION',
        '22/tcp   open  ssh         OpenSSH 8.9p1',
        '80/tcp   open  http        nginx 1.18.0',
        '443/tcp  open  ssl/http    nginx 1.18.0',
        '3306/tcp open  mysql       MySQL 8.0.32',
        '',
        'Nmap done: 1 IP address (1 host up) scanned in 2.34 seconds',
        '',
      ]);
    } else if (cmdLower === 'msfconsole') {
      output.push(...[
        '',
        '       =[ metasploit v6.3.25-dev                          ]',
        '+ -- --=[ 2345 exploits - 1220 auxiliary - 413 post       ]',
        '+ -- --=[ 1391 payloads - 46 encoders - 11 nops           ]',
        '+ -- --=[ 9 evasion                                       ]',
        '',
        'msf6 > ',
      ]);
    } else if (cmdLower.startsWith('searchsploit')) {
      const query = cmdLower.replace('searchsploit', '').trim() || 'apache';
      output.push(...[
        '',
        `Searching for: "${query}"`,
        '',
        '---------------------------------------------------------------------',
        ' Exploit Title                          | Path',
        '---------------------------------------------------------------------',
        ` Apache 2.4.49 - Path Traversal         | exploits/linux/remote/50383.py`,
        ` Apache mod_ssl < 2.8.7 - Off-By-One    | exploits/unix/remote/764.c`,
        ` Apache Tomcat - RCE (CVE-2020-9484)    | exploits/java/remote/48143.py`,
        '---------------------------------------------------------------------',
        '',
      ]);
    } else {
      output.push(`bash: ${cmd}: command not found`);
    }

    setTerminalHistory(prev => [...prev, ...output]);
  };

  const handleTerminalKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && currentInput.trim()) {
      executeCommand(currentInput);
      setCurrentInput('');
    }
  };

  const runScan = () => {
    setIsScanning(true);
    setScanResults([]);
    
    const mockResults: ScanResult[] = [
      { id: '1', type: 'port', severity: 'info', target: '192.168.1.10', port: 22, service: 'SSH', message: 'OpenSSH 8.9p1 detected', timestamp: new Date() },
      { id: '2', type: 'port', severity: 'info', target: '192.168.1.10', port: 80, service: 'HTTP', message: 'nginx 1.18.0 detected', timestamp: new Date() },
      { id: '3', type: 'vuln', severity: 'high', target: '192.168.1.10', port: 80, message: 'CVE-2021-23017: nginx DNS resolver vulnerability', timestamp: new Date() },
      { id: '4', type: 'vuln', severity: 'critical', target: '192.168.1.10', port: 3306, message: 'MySQL root account has weak password', timestamp: new Date() },
      { id: '5', type: 'vuln', severity: 'medium', target: '192.168.1.10', port: 443, message: 'TLS 1.0 enabled - consider disabling', timestamp: new Date() },
      { id: '6', type: 'info', severity: 'low', target: '192.168.1.10', message: 'X-Frame-Options header not set', timestamp: new Date() },
    ];

    let index = 0;
    const interval = setInterval(() => {
      if (index < mockResults.length) {
        setScanResults(prev => [...prev, mockResults[index]]);
        index++;
      } else {
        setIsScanning(false);
        clearInterval(interval);
      }
    }, 800);
  };

  const getSeverityColor = (severity: ScanResult['severity']) => {
    switch (severity) {
      case 'critical': return 'text-rose-500 bg-rose-500/10 border-rose-500/30';
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500/30';
      case 'medium': return 'text-amber-500 bg-amber-500/10 border-amber-500/30';
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/30';
      default: return 'text-slate-500 bg-slate-500/10 border-slate-500/30';
    }
  };

  return (
    <div className="h-full bg-[#0a0f0a] text-green-500 flex flex-col font-mono text-xs">
      {/* Header */}
      <div className="p-3 border-b border-green-900/50 bg-black/50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Skull size={16} className="text-green-400" />
            <span className="font-bold text-green-400 uppercase tracking-widest text-[11px]">Security Terminal</span>
          </div>
          <div className="h-4 w-px bg-green-900" />
          <div className="flex gap-1">
            {(['terminal', 'scanner', 'network', 'traffic'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-1.5 rounded text-[10px] font-bold uppercase transition-colors ${
                  activeTab === tab
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'hover:bg-green-900/30 text-green-600'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-green-600">
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            <span>STEALTH MODE</span>
          </div>
          <div className="flex items-center gap-1">
            <Lock size={10} />
            <span>VPN: ON</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'terminal' && (
          <div className="h-full flex flex-col p-2">
            <div 
              ref={terminalRef}
              className="flex-1 overflow-y-auto bg-black/50 rounded border border-green-900/30 p-3 font-mono text-[11px] leading-relaxed"
            >
              {terminalHistory.map((line, i) => (
                <div key={i} className={line.startsWith('[*]') ? 'text-green-400' : line.startsWith('[-]') ? 'text-red-400' : line.startsWith('[+]') ? 'text-cyan-400' : ''}>
                  {line || '\u00A0'}
                </div>
              ))}
            </div>
            <div className="mt-2 flex items-center gap-2 bg-black/50 rounded border border-green-900/30 px-3 py-2">
              <span className="text-green-400 font-bold">root@kali:~#</span>
              <input
                type="text"
                value={currentInput}
                onChange={(e) => setCurrentInput(e.target.value)}
                onKeyDown={handleTerminalKeyDown}
                className="flex-1 bg-transparent outline-none text-green-300 font-mono text-[11px]"
                placeholder="Enter command..."
                autoFocus
              />
            </div>
          </div>
        )}

        {activeTab === 'scanner' && (
          <div className="h-full flex flex-col p-4">
            <div className="flex items-center gap-3 mb-4">
              <Shield size={16} className="text-green-400" />
              <span className="font-bold text-green-400">Vulnerability Scanner</span>
              <div className="flex-1" />
              <input
                type="text"
                value={scanTarget}
                onChange={(e) => setScanTarget(e.target.value)}
                className="px-3 py-1.5 bg-black/50 border border-green-900/30 rounded text-[11px] text-green-300 w-48 focus:outline-none focus:border-green-500"
                placeholder="Target IP/Range"
              />
              <button
                onClick={runScan}
                disabled={isScanning}
                className="flex items-center gap-1.5 px-4 py-1.5 bg-green-500/20 border border-green-500/30 text-green-400 rounded text-[10px] font-bold hover:bg-green-500/30 transition-colors disabled:opacity-50"
              >
                <Zap size={12} />
                {isScanning ? 'Scanning...' : 'Start Scan'}
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2">
              {scanResults.length === 0 && !isScanning && (
                <div className="text-center text-green-700 py-12">
                  <Shield size={48} className="mx-auto mb-4 opacity-30" />
                  <p>Enter a target and click "Start Scan"</p>
                </div>
              )}
              {scanResults.map((result, i) => (
                <div 
                  key={result.id}
                  className={`p-3 rounded border ${getSeverityColor(result.severity)} animate-in slide-in-from-left`}
                  style={{ animationDelay: `${i * 100}ms` }}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      {result.type === 'vuln' ? <AlertTriangle size={12} /> : result.type === 'port' ? <Server size={12} /> : <Eye size={12} />}
                      <span className="font-bold uppercase text-[9px]">{result.severity}</span>
                      {result.port && <span className="text-[10px] opacity-70">:{result.port}</span>}
                    </div>
                    <span className="text-[9px] opacity-50">{result.target}</span>
                  </div>
                  <p className="text-[11px] opacity-90">{result.message}</p>
                </div>
              ))}
              {isScanning && (
                <div className="flex items-center gap-2 text-green-600 p-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-ping" />
                  <span className="animate-pulse">Scanning target...</span>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'network' && (
          <div className="h-full p-4">
            <div className="flex items-center gap-2 mb-4">
              <Network size={16} className="text-green-400" />
              <span className="font-bold text-green-400">Network Map</span>
              <span className="text-[10px] text-green-600 ml-auto">{networkNodes.length} hosts discovered</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {networkNodes.map(node => (
                <div 
                  key={node.id}
                  className={`p-4 rounded-lg border ${
                    node.status === 'online' ? 'bg-green-500/5 border-green-500/20' :
                    node.status === 'scanning' ? 'bg-amber-500/5 border-amber-500/20' :
                    'bg-slate-500/5 border-slate-500/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Server size={14} className={node.status === 'online' ? 'text-green-500' : node.status === 'scanning' ? 'text-amber-500' : 'text-slate-500'} />
                      <span className="font-bold text-green-300">{node.ip}</span>
                    </div>
                    <div className={`w-2 h-2 rounded-full ${
                      node.status === 'online' ? 'bg-green-500' :
                      node.status === 'scanning' ? 'bg-amber-500 animate-pulse' :
                      'bg-slate-500'
                    }`} />
                  </div>
                  <div className="text-[10px] text-green-600 mb-2">{node.hostname}</div>
                  {node.os && <div className="text-[9px] text-green-700 mb-2">OS: {node.os}</div>}
                  <div className="flex flex-wrap gap-1">
                    {node.openPorts.map(port => (
                      <span key={port} className="px-1.5 py-0.5 bg-green-900/30 rounded text-[8px] text-green-500">
                        {port}
                      </span>
                    ))}
                    {node.status === 'scanning' && (
                      <span className="px-1.5 py-0.5 bg-amber-900/30 rounded text-[8px] text-amber-500 animate-pulse">
                        scanning...
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'traffic' && (
          <div className="h-full flex flex-col p-4">
            <div className="flex items-center gap-2 mb-4">
              <Radio size={16} className="text-green-400" />
              <span className="font-bold text-green-400">Packet Capture</span>
              <div className="flex items-center gap-1 ml-auto text-[10px] text-green-600">
                <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                <span>CAPTURING</span>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto bg-black/50 rounded border border-green-900/30 p-3 font-mono text-[10px]">
              {packetLog.map((packet, i) => (
                <div key={i} className={`${
                  packet.includes('HTTP') ? 'text-cyan-400' :
                  packet.includes('SSH') ? 'text-amber-400' :
                  packet.includes('DNS') ? 'text-purple-400' :
                  'text-green-500'
                }`}>
                  {packet}
                </div>
              ))}
            </div>
            <div className="mt-3 flex gap-3">
              <div className="flex items-center gap-2 text-[9px]">
                <div className="w-2 h-2 rounded-full bg-cyan-500" />
                <span className="text-cyan-500">HTTP</span>
              </div>
              <div className="flex items-center gap-2 text-[9px]">
                <div className="w-2 h-2 rounded-full bg-amber-500" />
                <span className="text-amber-500">SSH</span>
              </div>
              <div className="flex items-center gap-2 text-[9px]">
                <div className="w-2 h-2 rounded-full bg-purple-500" />
                <span className="text-purple-500">DNS</span>
              </div>
              <div className="flex items-center gap-2 text-[9px]">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-green-500">Other</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-2 px-4 border-t border-green-900/50 bg-black/50 text-[9px] text-green-700 flex justify-between items-center">
        <span>HACKER_MODE // ROOT_ACCESS // ANONYMIZED</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          <span className="text-green-500">GHOST PROTOCOL ACTIVE</span>
        </div>
      </div>
    </div>
  );
}
