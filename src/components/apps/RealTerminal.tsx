"use client";

import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import { Terminal as XTerminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import { Wifi, Shield, Server, Terminal as TerminalIcon, Globe, Cpu, ChevronDown } from 'lucide-react';
import '@xterm/xterm/css/xterm.css';

export interface TerminalHandle {
  executeCommand: (command: string) => Promise<string>;
  getLastOutput: () => string;
  clear: () => void;
  focus: () => void;
  write: (text: string) => void;
}

interface RealTerminalProps {
  onOutput?: (output: string) => void;
  onCommandComplete?: (command: string, output: string) => void;
  disabled?: boolean;
  initialCommands?: string[];
}

const RealTerminal = forwardRef<TerminalHandle, RealTerminalProps>(({ 
  onOutput, 
  onCommandComplete,
  disabled = false,
  initialCommands = []
}, ref) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<XTerminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  
  const [currentLine, setCurrentLine] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [lastOutput, setLastOutput] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [cwd, setCwd] = useState('~');
  const [host, setHost] = useState('localhost');
  const [showHostMenu, setShowHostMenu] = useState(false);
  const [sessionId] = useState(`session-${Math.random().toString(36).substr(2, 9)}`);
  
  const pendingCommandRef = useRef<{
    resolve: (output: string) => void;
    command: string;
  } | null>(null);
  const outputBufferRef = useRef('');
  const cursorPosRef = useRef(0);

  const getPrompt = useCallback((path: string) => {
    return `\x1b[1;32muser@${host}\x1b[0m:\x1b[1;34m${path}\x1b[0m$ `;
  }, [host]);

  const executeShellCommand = useCallback(async (command: string): Promise<string> => {
    if (!command.trim()) return '';
    
    try {
      const response = await fetch('/api/terminal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          command: command.trim(),
          sessionId,
          host
        }),
      });
      
      if (!response.ok) {
        return `Error: Command failed with status ${response.status}`;
      }
      
      const data = await response.json();
      if (data.cwd) {
        const displayCwd = data.cwd.replace(process.cwd(), '~');
        setCwd(displayCwd);
      }
      return data.output || '';
    } catch (error) {
      return `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }, [sessionId, host]);

  const processCommand = useCallback(async (command: string) => {
    if (isExecuting) return;
    
    const term = xtermRef.current;
    if (!term) return;

    setIsExecuting(true);
    outputBufferRef.current = '';
    
    if (command.trim()) {
      setCommandHistory(prev => {
        const newHistory = [...prev.filter(c => c !== command), command];
        return newHistory.slice(-100);
      });
    }
    setHistoryIndex(-1);

    term.writeln('');

    const trimmedCommand = command.trim();
    
    if (trimmedCommand === 'clear') {
      term.clear();
      term.write(getPrompt(cwd));
      setIsExecuting(false);
      setCurrentLine('');
      cursorPosRef.current = 0;
      return;
    }

    if (trimmedCommand === 'help') {
      const helpText = [
        '\x1b[1;36mAvailable Commands:\x1b[0m',
        '  ls, dir     - List directory contents',
        '  cd <path>   - Change directory',
        '  pwd         - Print working directory',
        '  cat <file>  - Display file contents',
        '  echo <text> - Print text',
        '  mkdir <dir> - Create directory',
        '  touch <file>- Create empty file',
        '  rm <file>   - Remove file',
        '  node <file> - Run JavaScript',
        '  python <f>  - Run Python',
        '  npm <cmd>   - NPM commands',
        '  git <cmd>   - Git commands',
        '  clear       - Clear terminal',
        '  help        - Show this help',
      ].join('\r\n');
      
      term.writeln(helpText);
      term.write(getPrompt(cwd));
      setIsExecuting(false);
      setCurrentLine('');
      cursorPosRef.current = 0;
      return;
    }

    const output = await executeShellCommand(trimmedCommand);
    
    if (output) {
      const lines = output.split('\n');
      for (const line of lines) {
        term.writeln(line);
      }
      setLastOutput(output);
      outputBufferRef.current = output;
      onOutput?.(output);
      onCommandComplete?.(trimmedCommand, output);
    }

    if (pendingCommandRef.current && pendingCommandRef.current.command === trimmedCommand) {
      pendingCommandRef.current.resolve(output);
      pendingCommandRef.current = null;
    }

    term.write(getPrompt(cwd));
    setIsExecuting(false);
    setCurrentLine('');
    cursorPosRef.current = 0;
  }, [isExecuting, executeShellCommand, onOutput, onCommandComplete, cwd, getPrompt]);

  useImperativeHandle(ref, () => ({
    executeCommand: async (command: string): Promise<string> => {
      return new Promise((resolve) => {
        const term = xtermRef.current;
        if (!term) {
          resolve('');
          return;
        }

        pendingCommandRef.current = { resolve, command };
        
        term.write(command);
        setCurrentLine(command);
        cursorPosRef.current = command.length;
        
        setTimeout(() => {
          processCommand(command);
        }, 50);
      });
    },
    getLastOutput: () => lastOutput,
    clear: () => {
      xtermRef.current?.clear();
      xtermRef.current?.write(getPrompt(cwd));
    },
    focus: () => {
      xtermRef.current?.focus();
    },
    write: (text: string) => {
      xtermRef.current?.write(text);
    },
  }), [lastOutput, processCommand, cwd, getPrompt]);

  useEffect(() => {
    if (!terminalRef.current || xtermRef.current) return;

    const term = new XTerminal({
      cursorBlink: true,
      cursorStyle: 'block',
      fontSize: 13,
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
      theme: {
        background: '#0d1117',
        foreground: '#c9d1d9',
        cursor: '#58a6ff',
        cursorAccent: '#0d1117',
        selectionBackground: '#264f78',
        black: '#484f58',
        red: '#ff7b72',
        green: '#3fb950',
        yellow: '#d29922',
        blue: '#58a6ff',
        magenta: '#bc8cff',
        cyan: '#39c5cf',
        white: '#b1bac4',
        brightBlack: '#6e7681',
        brightRed: '#ffa198',
        brightGreen: '#56d364',
        brightYellow: '#e3b341',
        brightBlue: '#79c0ff',
        brightMagenta: '#d2a8ff',
        brightCyan: '#56d4dd',
        brightWhite: '#f0f6fc',
      },
      allowTransparency: true,
      scrollback: 5000,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();

    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);

    term.open(terminalRef.current);
    fitAddon.fit();

    xtermRef.current = term;
    fitAddonRef.current = fitAddon;

    term.writeln('\x1b[1;36m┌────────────────────────────────────────┐\x1b[0m');
    term.writeln('\x1b[1;36m│\x1b[0m  \x1b[1;32mDEV-OS TERMINAL\x1b[0m v2.1                  \x1b[1;36m│\x1b[0m');
    term.writeln('\x1b[1;36m│\x1b[0m  Connected to: \x1b[1;33m' + host + '\x1b[0m             \x1b[1;36m│\x1b[0m');
    term.writeln('\x1b[1;36m└────────────────────────────────────────┘\x1b[0m');
    term.writeln('');
    term.write(getPrompt(cwd));

    term.onKey(({ key, domEvent }) => {
      if (disabled || isExecuting) return;

      const printable = !domEvent.altKey && !domEvent.ctrlKey && !domEvent.metaKey;

      if (domEvent.keyCode === 13) {
        processCommand(currentLine);
      } else if (domEvent.keyCode === 8) {
        if (cursorPosRef.current > 0) {
          const newLine = currentLine.slice(0, cursorPosRef.current - 1) + currentLine.slice(cursorPosRef.current);
          setCurrentLine(newLine);
          cursorPosRef.current--;
          term.write('\b \b');
          if (cursorPosRef.current < newLine.length) {
            term.write(newLine.slice(cursorPosRef.current) + ' ');
            term.write('\x1b[' + (newLine.length - cursorPosRef.current + 1) + 'D');
          }
        }
      } else if (domEvent.keyCode === 38) {
        if (commandHistory.length > 0) {
          const newIndex = historyIndex < commandHistory.length - 1 ? historyIndex + 1 : historyIndex;
          if (newIndex !== historyIndex) {
            setHistoryIndex(newIndex);
            const historyCommand = commandHistory[commandHistory.length - 1 - newIndex];
            term.write('\x1b[2K\r' + getPrompt(cwd) + historyCommand);
            setCurrentLine(historyCommand);
            cursorPosRef.current = historyCommand.length;
          }
        }
      } else if (domEvent.keyCode === 40) {
        if (historyIndex > 0) {
          const newIndex = historyIndex - 1;
          setHistoryIndex(newIndex);
          const historyCommand = commandHistory[commandHistory.length - 1 - newIndex];
          term.write('\x1b[2K\r' + getPrompt(cwd) + historyCommand);
          setCurrentLine(historyCommand);
          cursorPosRef.current = historyCommand.length;
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          term.write('\x1b[2K\r' + getPrompt(cwd));
          setCurrentLine('');
          cursorPosRef.current = 0;
        }
      } else if (domEvent.keyCode === 37) {
        if (cursorPosRef.current > 0) {
          cursorPosRef.current--;
          term.write('\x1b[D');
        }
      } else if (domEvent.keyCode === 39) {
        if (cursorPosRef.current < currentLine.length) {
          cursorPosRef.current++;
          term.write('\x1b[C');
        }
      } else if (domEvent.ctrlKey && domEvent.key === 'c') {
        term.writeln('^C');
        term.write(getPrompt(cwd));
        setCurrentLine('');
        cursorPosRef.current = 0;
      } else if (domEvent.ctrlKey && domEvent.key === 'l') {
        term.clear();
        term.write(getPrompt(cwd) + currentLine);
      } else if (printable) {
        const newLine = currentLine.slice(0, cursorPosRef.current) + key + currentLine.slice(cursorPosRef.current);
        setCurrentLine(newLine);
        term.write(key);
        if (cursorPosRef.current < currentLine.length) {
          term.write(currentLine.slice(cursorPosRef.current));
          term.write('\x1b[' + (currentLine.length - cursorPosRef.current) + 'D');
        }
        cursorPosRef.current++;
      }
    });

    const resizeObserver = new ResizeObserver(() => {
      fitAddon.fit();
    });
    resizeObserver.observe(terminalRef.current);

    if (initialCommands.length > 0) {
      setTimeout(async () => {
        for (const cmd of initialCommands) {
          term.write(cmd);
          setCurrentLine(cmd);
          cursorPosRef.current = cmd.length;
          await processCommand(cmd);
          await new Promise(r => setTimeout(r, 500));
        }
      }, 500);
    }

    return () => {
      resizeObserver.disconnect();
      term.dispose();
      xtermRef.current = null;
      fitAddonRef.current = null;
    };
  }, [disabled, host, cwd, getPrompt, processCommand, initialCommands]);

  const changeHost = (newHost: string) => {
    setHost(newHost);
    setShowHostMenu(false);
    if (xtermRef.current) {
      xtermRef.current.writeln(`\r\n\x1b[1;33mSwitching to host: ${newHost}...\x1b[0m`);
      xtermRef.current.write(getPrompt(newHost === 'localhost' ? cwd : '/root'));
    }
  };

  return (
    <div className="w-full h-full bg-[#0d1117] flex flex-col">
      {/* Terminal Toolbar */}
      <div className="h-10 bg-[#161b22] border-b border-[#30363d] flex items-center justify-between px-3 shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-[#8b949e]">
            <TerminalIcon size={14} />
            <span className="text-xs font-medium">bash</span>
          </div>
          <div className="h-4 w-[1px] bg-[#30363d]" />
          <div className="relative">
            <button 
              onClick={() => setShowHostMenu(!showHostMenu)}
              className="flex items-center gap-2 px-2 py-1 rounded-md hover:bg-[#21262d] text-[#c9d1d9] transition-colors"
            >
              <Server size={14} className={host === 'localhost' ? 'text-emerald-500' : 'text-blue-500'} />
              <span className="text-[11px] font-bold">{host}</span>
              <ChevronDown size={12} className="text-[#8b949e]" />
            </button>
            
            {showHostMenu && (
              <div className="absolute top-full left-0 mt-1 w-48 bg-[#161b22] border border-[#30363d] rounded-lg shadow-xl z-50 overflow-hidden">
                <div className="p-2 text-[10px] font-bold text-[#8b949e] uppercase tracking-wider bg-[#0d1117]/50">Select Environment</div>
                <button 
                  onClick={() => changeHost('localhost')}
                  className="w-full flex items-center gap-3 px-3 py-2 text-xs text-[#c9d1d9] hover:bg-emerald-500/10 hover:text-emerald-400 transition-colors"
                >
                  <Cpu size={14} />
                  <span>Local Workspace</span>
                  {host === 'localhost' && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-500" />}
                </button>
                <button 
                  onClick={() => changeHost('docker-node-20')}
                  className="w-full flex items-center gap-3 px-3 py-2 text-xs text-[#c9d1d9] hover:bg-blue-500/10 hover:text-blue-400 transition-colors"
                >
                  <Globe size={14} />
                  <span>Docker (Node 20)</span>
                  {host === 'docker-node-20' && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-500" />}
                </button>
                <button 
                  onClick={() => changeHost('remote-ssh-prod')}
                  className="w-full flex items-center gap-3 px-3 py-2 text-xs text-[#c9d1d9] hover:bg-purple-500/10 hover:text-purple-400 transition-colors"
                >
                  <Shield size={14} />
                  <span>Production SSH</span>
                  {host === 'remote-ssh-prod' && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-purple-500" />}
                </button>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-medium text-[#8b949e]">Connected</span>
          </div>
          <Wifi size={14} className="text-[#8b949e]" />
        </div>
      </div>

      <div 
        ref={terminalRef} 
        className="flex-1 w-full"
        style={{ padding: '8px' }}
      />
    </div>
  );
});

RealTerminal.displayName = 'RealTerminal';

export default RealTerminal;
