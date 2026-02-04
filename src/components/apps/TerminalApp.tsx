"use client";

import { useState, useRef, useEffect } from 'react';
import { useWorkspaceStore } from '@/lib/workspaceStore';
import { Terminal, ChevronRight } from 'lucide-react';

const COMMANDS: Record<string, { description: string; handler: (args: string[], state: { cwd: string }) => { output: string[]; newCwd?: string } }> = {
  help: {
    description: 'Show available commands',
    handler: () => ({
      output: [
        'Available commands:',
        '  help              Show this help message',
        '  ls [path]         List directory contents',
        '  cd <path>         Change directory',
        '  pwd               Print working directory',
        '  cat <file>        Display file contents',
        '  mkdir <name>      Create directory',
        '  touch <name>      Create file',
        '  rm <file>         Remove file',
        '  echo <text>       Print text',
        '  clear             Clear terminal',
        '  npm <cmd>         Run npm commands',
        '  git <cmd>         Run git commands',
        '  node <file>       Run JavaScript file',
        '  whoami            Show current user',
        '  date              Show current date',
        '  uptime            Show system uptime',
        '  env               Show environment variables',
      ],
    }),
  },
  ls: {
    description: 'List directory contents',
    handler: (args, state) => {
      const items = [
        { name: 'src', type: 'dir' },
        { name: 'public', type: 'dir' },
        { name: 'node_modules', type: 'dir' },
        { name: 'package.json', type: 'file' },
        { name: 'tsconfig.json', type: 'file' },
        { name: 'README.md', type: 'file' },
        { name: 'next.config.js', type: 'file' },
        { name: '.gitignore', type: 'file' },
        { name: '.env.local', type: 'file' },
      ];
      
      if (args.includes('-la') || args.includes('-l')) {
        return {
          output: [
            'total 48',
            ...items.map(i => 
              `${i.type === 'dir' ? 'drwxr-xr-x' : '-rw-r--r--'}  1 devagent  staff   ${Math.floor(Math.random() * 10000).toString().padStart(5)}  Jan 21 09:00  ${i.type === 'dir' ? `\x1b[34m${i.name}\x1b[0m` : i.name}`
            ),
          ],
        };
      }
      
      return {
        output: [items.map(i => i.type === 'dir' ? `\x1b[34m${i.name}\x1b[0m` : i.name).join('  ')],
      };
    },
  },
  cd: {
    description: 'Change directory',
    handler: (args, state) => {
      const path = args[0] || '~';
      if (path === '..') {
        const parts = state.cwd.split('/').filter(Boolean);
        parts.pop();
        return { output: [], newCwd: parts.length ? '/' + parts.join('/') : '~' };
      }
      if (path === '~' || path === '/home/devagent') {
        return { output: [], newCwd: '~/devagent-project' };
      }
      if (path.startsWith('/')) {
        return { output: [], newCwd: path };
      }
      return { output: [], newCwd: `${state.cwd}/${path}` };
    },
  },
  pwd: {
    description: 'Print working directory',
    handler: (_, state) => ({
      output: [state.cwd.replace('~', '/home/devagent')],
    }),
  },
  cat: {
    description: 'Display file contents',
    handler: (args) => {
      if (!args[0]) return { output: ['cat: missing operand'] };
      
      const fileContents: Record<string, string[]> = {
        'package.json': [
          '{',
          '  "name": "devagent-project",',
          '  "version": "0.1.0",',
          '  "private": true,',
          '  "scripts": {',
          '    "dev": "next dev",',
          '    "build": "next build"',
          '  }',
          '}',
        ],
        'README.md': [
          '# DevAgent Project',
          '',
          'This project is being developed with AI assistance.',
        ],
        '.env.local': [
          'DATABASE_URL=postgresql://...',
          'NEXT_PUBLIC_API_URL=http://localhost:3000',
        ],
      };
      
      return { output: fileContents[args[0]] || [`cat: ${args[0]}: No such file or directory`] };
    },
  },
  mkdir: {
    description: 'Create directory',
    handler: (args) => {
      if (!args[0]) return { output: ['mkdir: missing operand'] };
      return { output: [`Created directory: ${args[0]}`] };
    },
  },
  touch: {
    description: 'Create file',
    handler: (args) => {
      if (!args[0]) return { output: ['touch: missing operand'] };
      return { output: [`Created file: ${args[0]}`] };
    },
  },
  rm: {
    description: 'Remove file',
    handler: (args) => {
      if (!args[0]) return { output: ['rm: missing operand'] };
      return { output: [`Removed: ${args[0]}`] };
    },
  },
  echo: {
    description: 'Print text',
    handler: (args) => ({ output: [args.join(' ')] }),
  },
  clear: {
    description: 'Clear terminal',
    handler: () => ({ output: ['__CLEAR__'] }),
  },
  npm: {
    description: 'Run npm commands',
    handler: (args) => {
      const cmd = args[0];
      if (cmd === 'install' || cmd === 'i') {
        return {
          output: [
            '',
            'added 245 packages, and audited 246 packages in 12s',
            '',
            '87 packages are looking for funding',
            '  run `npm fund` for details',
            '',
            'found 0 vulnerabilities',
          ],
        };
      }
      if (cmd === 'run') {
        const script = args[1];
        if (script === 'dev') {
          return {
            output: [
              '',
              '> devagent-project@0.1.0 dev',
              '> next dev',
              '',
              '  ▲ Next.js 14.0.0',
              '  - Local:        http://localhost:3000',
              '  - Environments: .env.local',
              '',
              ' ✓ Ready in 1.8s',
            ],
          };
        }
        if (script === 'build') {
          return {
            output: [
              '',
              '> devagent-project@0.1.0 build',
              '> next build',
              '',
              '  ▲ Next.js 14.0.0',
              '',
              '   Creating an optimized production build ...',
              ' ✓ Compiled successfully',
              ' ✓ Linting and checking validity of types',
              ' ✓ Collecting page data',
              ' ✓ Generating static pages (4/4)',
              ' ✓ Finalizing page optimization',
              '',
              'Route (app)                              Size     First Load JS',
              '┌ ○ /                                    5.2 kB        87.3 kB',
              '└ ○ /_not-found                          871 B         83 kB',
              '',
              '○  (Static)  prerendered as static content',
            ],
          };
        }
        if (script === 'lint') {
          return {
            output: [
              '',
              '> devagent-project@0.1.0 lint',
              '> next lint',
              '',
              '✔ No ESLint warnings or errors',
            ],
          };
        }
        if (script === 'test') {
          return {
            output: [
              '',
              '> devagent-project@0.1.0 test',
              '> jest',
              '',
              ' PASS  src/__tests__/utils.test.ts',
              '  ✓ formatDate returns correct format (3 ms)',
              '  ✓ cn combines classes correctly (1 ms)',
              '  ✓ sleep delays execution (101 ms)',
              '',
              'Test Suites: 1 passed, 1 total',
              'Tests:       3 passed, 3 total',
              'Time:        1.234 s',
            ],
          };
        }
      }
      return { output: [`npm ${args.join(' ')}`] };
    },
  },
  git: {
    description: 'Run git commands',
    handler: (args) => {
      const cmd = args[0];
      if (cmd === 'status') {
        return {
          output: [
            'On branch fix/navigation',
            'Changes not staged for commit:',
            '  (use "git add <file>..." to update what will be committed)',
            '',
            '        modified:   src/components/Nav.tsx',
            '',
            'no changes added to commit (use "git add" and/or "git commit -a")',
          ],
        };
      }
      if (cmd === 'branch') {
        return {
          output: [
            '  main',
            '  feature/auth',
            '* fix/navigation',
          ],
        };
      }
      if (cmd === 'log') {
        return {
          output: [
            'commit a1b2c3d (HEAD -> fix/navigation)',
            'Author: DevAgent <agent@devagent.ai>',
            'Date:   Tue Jan 21 09:00:00 2025',
            '',
            '    WIP: Fix mobile menu state',
            '',
            'commit e4f5g6h (main)',
            'Author: DevAgent <agent@devagent.ai>',
            'Date:   Mon Jan 20 14:30:00 2025',
            '',
            '    Add project structure',
          ],
        };
      }
      if (cmd === 'add') {
        return { output: [''] };
      }
      if (cmd === 'commit') {
        return {
          output: [
            '[fix/navigation 7h8i9j0] ' + (args.slice(2).join(' ') || 'Update'),
            ' 1 file changed, 15 insertions(+), 3 deletions(-)',
          ],
        };
      }
      if (cmd === 'push') {
        return {
          output: [
            'Enumerating objects: 5, done.',
            'Counting objects: 100% (5/5), done.',
            'Delta compression using up to 8 threads',
            'Compressing objects: 100% (3/3), done.',
            'Writing objects: 100% (3/3), 312 bytes | 312.00 KiB/s, done.',
            'Total 3 (delta 2), reused 0 (delta 0)',
            'To github.com:devagent/project.git',
            '   e4f5g6h..7h8i9j0  fix/navigation -> fix/navigation',
          ],
        };
      }
      if (cmd === 'pull') {
        return {
          output: [
            'Already up to date.',
          ],
        };
      }
      return { output: [`git ${args.join(' ')}`] };
    },
  },
  node: {
    description: 'Run JavaScript',
    handler: (args) => {
      if (args[0] === '-v' || args[0] === '--version') {
        return { output: ['v20.10.0'] };
      }
      if (args[0] === '-e') {
        try {
          const code = args.slice(1).join(' ');
          return { output: [`${eval(code)}`] };
        } catch (e) {
          return { output: [`Error: ${e}`] };
        }
      }
      return { output: ['Node.js v20.10.0'] };
    },
  },
  whoami: {
    description: 'Show current user',
    handler: () => ({ output: ['devagent'] }),
  },
  date: {
    description: 'Show current date',
    handler: () => ({ output: [new Date().toString()] }),
  },
  uptime: {
    description: 'Show system uptime',
    handler: () => ({ output: ['up 42 days, 3:14'] }),
  },
  env: {
    description: 'Show environment variables',
    handler: () => ({
      output: [
        'NODE_ENV=development',
        'PATH=/usr/local/bin:/usr/bin:/bin',
        'HOME=/home/devagent',
        'USER=devagent',
        'SHELL=/bin/bash',
        'LANG=en_US.UTF-8',
      ],
    }),
  },
};

export default function TerminalApp() {
  const { terminalHistory, terminalCwd, addTerminalLine, setTerminalCwd, clearTerminal } = useWorkspaceStore();
  const [input, setInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [terminalHistory]);

  const executeCommand = (cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) {
      addTerminalLine(`${terminalCwd}$ `);
      return;
    }

    setCommandHistory(prev => [...prev, trimmed]);
    setHistoryIndex(-1);
    addTerminalLine(`${terminalCwd}$ ${trimmed}`);

    const parts = trimmed.split(' ');
    const command = parts[0].toLowerCase();
    const args = parts.slice(1);

    const handler = COMMANDS[command];
    if (handler) {
      const result = handler.handler(args, { cwd: terminalCwd });
      
      if (result.output[0] === '__CLEAR__') {
        clearTerminal();
        addTerminalLine('Terminal cleared.');
        return;
      }
      
      result.output.forEach(line => addTerminalLine(line));
      if (result.newCwd) {
        setTerminalCwd(result.newCwd);
      }
    } else {
      addTerminalLine(`bash: ${command}: command not found`);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      executeCommand(input);
      setInput('');
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0) {
        const newIndex = historyIndex < commandHistory.length - 1 ? historyIndex + 1 : historyIndex;
        setHistoryIndex(newIndex);
        setInput(commandHistory[commandHistory.length - 1 - newIndex] || '');
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[commandHistory.length - 1 - newIndex] || '');
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setInput('');
      }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      const matches = Object.keys(COMMANDS).filter(c => c.startsWith(input));
      if (matches.length === 1) {
        setInput(matches[0] + ' ');
      } else if (matches.length > 1) {
        addTerminalLine(`${terminalCwd}$ ${input}`);
        addTerminalLine(matches.join('  '));
      }
    } else if (e.key === 'c' && e.ctrlKey) {
      addTerminalLine(`${terminalCwd}$ ${input}^C`);
      setInput('');
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault();
      clearTerminal();
    }
  };

  return (
    <div 
      className="h-full bg-[#0d1117] text-[#c9d1d9] font-mono text-sm flex flex-col"
      onClick={() => inputRef.current?.focus()}
    >
      <div className="h-8 bg-[#161b22] border-b border-[#30363d] flex items-center px-3 gap-2">
        <Terminal size={14} className="text-[#7ee787]" />
        <span className="text-xs">Terminal</span>
        <span className="text-xs text-[#8b949e]">— bash</span>
        <div className="flex-1" />
        <span className="text-xs text-[#8b949e]">{terminalCwd}</span>
      </div>
      
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto p-3 space-y-0.5"
      >
        {terminalHistory.map((line, i) => (
          <div key={i} className="whitespace-pre-wrap break-all">
            {line.includes('$') ? (
              <span>
                <span className="text-[#7ee787]">{line.split('$')[0]}$</span>
                <span className="text-white">{line.split('$').slice(1).join('$')}</span>
              </span>
            ) : line.startsWith('✓') || line.includes('passed') ? (
              <span className="text-[#7ee787]">{line}</span>
            ) : line.startsWith('✗') || line.includes('error') || line.includes('Error') ? (
              <span className="text-[#f85149]">{line}</span>
            ) : line.includes('warning') ? (
              <span className="text-[#d29922]">{line}</span>
            ) : (
              <span>{line}</span>
            )}
          </div>
        ))}
        
        <div className="flex items-center">
          <span className="text-[#7ee787]">{terminalCwd}$ </span>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent outline-none text-white caret-[#58a6ff]"
            autoFocus
            spellCheck={false}
          />
        </div>
      </div>
    </div>
  );
}
