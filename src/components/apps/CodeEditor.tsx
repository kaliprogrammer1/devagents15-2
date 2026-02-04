"use client";

import { useState, useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from 'react';
import Editor, { OnMount, OnChange } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import {
  Files, Search, GitBranch, Bug, Puzzle, Settings, User,
  ChevronRight, ChevronDown, File, Folder, FolderOpen,
  X, Circle, MoreHorizontal, SplitSquareHorizontal,
  Terminal as TerminalIcon, AlertCircle, Info, Bell,
  Play, RefreshCw, Save, Undo, Redo, Copy, Scissors,
  ClipboardPaste, ZoomIn, ZoomOut, Command, PanelBottom,
  PanelLeft, Search as SearchIcon, Replace, FileCode,
  FileJson, FileText, FileCog, Image, FileType, Loader2
} from 'lucide-react';

export interface CodeEditorHandle {
  openFile: (path: string, content?: string) => void;
  saveFile: () => Promise<void>;
  getContent: () => string;
  setContent: (content: string) => void;
  getCurrentFile: () => string | null;
  find: (text: string) => void;
  replace: (find: string, replaceWith: string) => void;
  undo: () => void;
  redo: () => void;
  formatDocument: () => void;
  goToLine: (line: number) => void;
  insertText: (text: string) => void;
}

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  isOpen?: boolean;
}

interface OpenTab {
  path: string;
  name: string;
  content: string;
  isDirty: boolean;
  language: string;
}

interface CodeEditorProps {
  onFileChange?: (path: string, content: string) => void;
  onSave?: (path: string, content: string) => Promise<void>;
  disabled?: boolean;
}

const FILE_ICONS: Record<string, { icon: typeof File; color: string }> = {
  'js': { icon: FileCode, color: '#f7df1e' },
  'jsx': { icon: FileCode, color: '#61dafb' },
  'ts': { icon: FileCode, color: '#3178c6' },
  'tsx': { icon: FileCode, color: '#3178c6' },
  'json': { icon: FileJson, color: '#cbcb41' },
  'md': { icon: FileText, color: '#519aba' },
  'txt': { icon: FileText, color: '#89e051' },
  'css': { icon: FileType, color: '#563d7c' },
  'scss': { icon: FileType, color: '#c6538c' },
  'html': { icon: FileCode, color: '#e34c26' },
  'svg': { icon: Image, color: '#ffb13b' },
  'png': { icon: Image, color: '#a074c4' },
  'jpg': { icon: Image, color: '#a074c4' },
  'config': { icon: FileCog, color: '#6d8086' },
  'env': { icon: FileCog, color: '#ecd53f' },
  'gitignore': { icon: GitBranch, color: '#f14e32' },
};

const getFileIcon = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const config = FILE_ICONS[ext] || { icon: File, color: '#8b949e' };
  return config;
};

const getLanguageFromFile = (filename: string): string => {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const langMap: Record<string, string> = {
    'js': 'javascript',
    'jsx': 'javascript',
    'ts': 'typescript',
    'tsx': 'typescript',
    'json': 'json',
    'md': 'markdown',
    'css': 'css',
    'scss': 'scss',
    'html': 'html',
    'py': 'python',
    'rs': 'rust',
    'go': 'go',
    'java': 'java',
    'c': 'c',
    'cpp': 'cpp',
    'h': 'c',
    'sh': 'shell',
    'bash': 'shell',
    'yml': 'yaml',
    'yaml': 'yaml',
    'xml': 'xml',
    'sql': 'sql',
    'graphql': 'graphql',
    'vue': 'vue',
    'svelte': 'svelte',
  };
  return langMap[ext] || 'plaintext';
};

const CodeEditor = forwardRef<CodeEditorHandle, CodeEditorProps>(({
  onFileChange,
  onSave,
  disabled = false,
}, ref) => {
  const [files, setFiles] = useState<FileNode[]>([]);
  const [openTabs, setOpenTabs] = useState<OpenTab[]>([]);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [showTerminal, setShowTerminal] = useState(true);
  const [activeActivity, setActiveActivity] = useState<'files' | 'search' | 'git' | 'debug' | 'extensions'>('files');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Array<{ file: string; line: number; content: string }>>([]);
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [showFindReplace, setShowFindReplace] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState<string[]>(['$ IDE Environment Ready', '$ Connected to local workspace']);
  const [terminalInput, setTerminalInput] = useState('');
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
  const [loading, setLoading] = useState(false);
  
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import('monaco-editor') | null>(null);

  const fetchFiles = useCallback(async (pathStr: string = ''): Promise<FileNode[]> => {
    try {
      const res = await fetch('/api/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'list', path: pathStr }),
      });
      const data = await res.json();
      if (data.files) {
        return data.files.map((f: any) => ({
          name: f.name,
          path: f.id,
          type: f.type,
          isOpen: false,
          children: f.type === 'folder' ? [] : undefined
        }));
      }
    } catch (err) {
      console.error('Fetch files error:', err);
    }
    return [];
  }, []);

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      const rootFiles = await fetchFiles();
      setFiles(rootFiles);
      setLoading(false);
    };
    init();
  }, [fetchFiles]);

  const handleEditorMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    monaco.editor.defineTheme('vscode-dark-custom', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'comment', foreground: '6A9955' },
        { token: 'keyword', foreground: '569CD6' },
        { token: 'string', foreground: 'CE9178' },
        { token: 'number', foreground: 'B5CEA8' },
        { token: 'type', foreground: '4EC9B0' },
        { token: 'function', foreground: 'DCDCAA' },
        { token: 'variable', foreground: '9CDCFE' },
        { token: 'constant', foreground: '4FC1FF' },
      ],
      colors: {
        'editor.background': '#1e1e1e',
        'editor.foreground': '#d4d4d4',
        'editor.lineHighlightBackground': '#2d2d30',
        'editor.selectionBackground': '#264f78',
        'editor.inactiveSelectionBackground': '#3a3d41',
        'editorLineNumber.foreground': '#858585',
        'editorLineNumber.activeForeground': '#c6c6c6',
        'editorCursor.foreground': '#aeafad',
        'editor.findMatchBackground': '#515c6a',
        'editor.findMatchHighlightBackground': '#ea5c0055',
        'editorBracketMatch.background': '#0d3a58',
        'editorBracketMatch.border': '#888888',
      },
    });

    monaco.editor.setTheme('vscode-dark-custom');

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      saveCurrentFile();
    });

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyF, () => {
      setShowFindReplace(true);
    });

    editor.onDidChangeCursorPosition((e) => {
      setCursorPosition({ line: e.position.lineNumber, column: e.position.column });
    });
  };

  const handleEditorChange: OnChange = (value) => {
    if (activeTab && value !== undefined) {
      setOpenTabs(prev => prev.map(tab =>
        tab.path === activeTab ? { ...tab, content: value, isDirty: true } : tab
      ));
      onFileChange?.(activeTab, value);
    }
  };

  const openFile = useCallback(async (filePath: string, content?: string) => {
    const existingTab = openTabs.find(t => t.path === filePath);
    if (existingTab) {
      setActiveTab(filePath);
      return;
    }

    setLoading(true);
    let fileContent = content;
    if (!fileContent) {
      try {
        const response = await fetch('/api/files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'read', path: filePath }),
        });
        const data = await response.json();
        fileContent = data.content || '';
      } catch (err) {
        fileContent = `// Error loading ${filePath}`;
      }
    }

    const filename = filePath.split('/').pop() || filePath;
    const newTab: OpenTab = {
      path: filePath,
      name: filename,
      content: fileContent,
      isDirty: false,
      language: getLanguageFromFile(filename),
    };

    setOpenTabs(prev => [...prev, newTab]);
    setActiveTab(filePath);
    setLoading(false);
  }, [openTabs]);

  const closeTab = (path: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    const tab = openTabs.find(t => t.path === path);
    if (tab?.isDirty) {
      if (!window.confirm('File has unsaved changes. Close anyway?')) {
        return;
      }
    }

    setOpenTabs(prev => prev.filter(t => t.path !== path));
    if (activeTab === path) {
      const remaining = openTabs.filter(t => t.path !== path);
      setActiveTab(remaining.length > 0 ? remaining[remaining.length - 1].path : null);
    }
  };

  const saveCurrentFile = async () => {
    if (!activeTab) return;
    const tab = openTabs.find(t => t.path === activeTab);
    if (!tab) return;

    try {
      await fetch('/api/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'write', path: activeTab, content: tab.content }),
      });
      
      setOpenTabs(prev => prev.map(t =>
        t.path === activeTab ? { ...t, isDirty: false } : t
      ));
      setTerminalOutput(prev => [...prev, `$ File saved successfully: ${activeTab}`]);
    } catch (error) {
      setTerminalOutput(prev => [...prev, `$ Error saving file: ${error}`]);
    }
  };

  const executeTerminalCommand = async () => {
    if (!terminalInput.trim()) return;
    setTerminalOutput(prev => [...prev, `$ ${terminalInput}`]);
    
    try {
      const response = await fetch('/api/terminal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: terminalInput }),
      });
      const data = await response.json();
      if (data.output) {
        setTerminalOutput(prev => [...prev, data.output]);
      }
    } catch (error) {
      setTerminalOutput(prev => [...prev, `Error: ${error}`]);
    }
    setTerminalInput('');
  };

  useImperativeHandle(ref, () => ({
    openFile,
    saveFile: saveCurrentFile,
    getContent: () => {
      const tab = openTabs.find(t => t.path === activeTab);
      return tab?.content || '';
    },
    setContent: (content: string) => {
      if (activeTab) {
        setOpenTabs(prev => prev.map(t =>
          t.path === activeTab ? { ...t, content, isDirty: true } : t
        ));
        editorRef.current?.setValue(content);
      }
    },
    getCurrentFile: () => activeTab,
    find: (text: string) => {
      setFindText(text);
      setShowFindReplace(true);
      if (editorRef.current && monacoRef.current) {
        const model = editorRef.current.getModel();
        if (model) {
          const matches = model.findMatches(text, true, false, false, null, true);
          if (matches.length > 0) {
            editorRef.current.setSelection(matches[0].range);
            editorRef.current.revealLineInCenter(matches[0].range.startLineNumber);
          }
        }
      }
    },
    replace: (find: string, replaceWith: string) => {
      if (editorRef.current) {
        const model = editorRef.current.getModel();
        if (model) {
          const content = model.getValue();
          model.setValue(content.replace(new RegExp(find, 'g'), replaceWith));
        }
      }
    },
    undo: () => editorRef.current?.trigger('keyboard', 'undo', null),
    redo: () => editorRef.current?.trigger('keyboard', 'redo', null),
    formatDocument: () => editorRef.current?.getAction('editor.action.formatDocument')?.run(),
    goToLine: (line: number) => {
      editorRef.current?.setPosition({ lineNumber: line, column: 1 });
      editorRef.current?.revealLineInCenter(line);
    },
    insertText: (text: string) => {
      editorRef.current?.trigger('keyboard', 'type', { text });
    },
  }), [activeTab, openTabs, openFile]);

  const toggleFolder = async (path: string) => {
    let newFiles = [...files];
    const updateNode = async (nodes: FileNode[]): Promise<FileNode[]> => {
      return Promise.all(nodes.map(async node => {
        if (node.path === path && node.type === 'folder') {
          const isOpen = !node.isOpen;
          let children = node.children;
          if (isOpen && (!children || children.length === 0)) {
            children = await fetchFiles(node.path);
          }
          return { ...node, isOpen, children };
        }
        if (node.children) {
          return { ...node, children: await updateNode(node.children) };
        }
        return node;
      }));
    };
    const updated = await updateNode(newFiles);
    setFiles(updated);
  };

  const renderFileTree = (nodes: FileNode[], depth = 0) => {
    return nodes.map(node => {
      const isActive = node.path === activeTab;
      const iconConfig = node.type === 'file' ? getFileIcon(node.name) : null;
      const Icon = iconConfig?.icon || File;

      if (node.type === 'folder') {
        return (
          <div key={node.path}>
            <div
              className={`flex items-center gap-1 px-2 py-1 cursor-pointer hover:bg-[#2a2d2e] text-[#cccccc] text-[13px] transition-colors`}
              style={{ paddingLeft: `${8 + depth * 12}px` }}
              onClick={() => toggleFolder(node.path)}
            >
              {node.isOpen ? (
                <ChevronDown size={14} className="text-[#cccccc] flex-shrink-0" />
              ) : (
                <ChevronRight size={14} className="text-[#cccccc] flex-shrink-0" />
              )}
              {node.isOpen ? (
                <FolderOpen size={14} className="text-[#dcb67a] flex-shrink-0" />
              ) : (
                <Folder size={14} className="text-[#dcb67a] flex-shrink-0" />
              )}
              <span className="truncate">{node.name}</span>
            </div>
            {node.isOpen && node.children && (
              <div>{renderFileTree(node.children, depth + 1)}</div>
            )}
          </div>
        );
      }

      return (
        <div
          key={node.path}
          className={`flex items-center gap-2 px-2 py-1 cursor-pointer text-[13px] transition-colors ${
            isActive ? 'bg-[#094771] text-white' : 'text-[#cccccc] hover:bg-[#2a2d2e]'
          }`}
          style={{ paddingLeft: `${20 + depth * 12}px` }}
          onClick={() => openFile(node.path)}
        >
          <Icon size={14} style={{ color: iconConfig?.color }} className="flex-shrink-0" />
          <span className="truncate">{node.name}</span>
        </div>
      );
    });
  };

  const activeTabContent = openTabs.find(t => t.path === activeTab);

  return (
    <div className="flex flex-col h-full bg-[#1e1e1e] text-[#cccccc] font-['Segoe_UI',system-ui,sans-serif] select-none">
      <div className="flex-1 flex overflow-hidden">
        {/* Activity Bar */}
        <div className="w-[48px] bg-[#333333] flex flex-col items-center py-2 gap-1 border-r border-[#1e1e1e]">
          {[
            { id: 'files', icon: Files, title: 'Explorer' },
            { id: 'search', icon: Search, title: 'Search' },
            { id: 'git', icon: GitBranch, title: 'Source Control' },
            { id: 'debug', icon: Bug, title: 'Run and Debug' },
            { id: 'extensions', icon: Puzzle, title: 'Extensions' },
          ].map(item => (
            <button
              key={item.id}
              onClick={() => {
                if (activeActivity === item.id && showSidebar) setShowSidebar(false);
                else { setActiveActivity(item.id as any); setShowSidebar(true); }
              }}
              className={`w-[48px] h-[48px] flex items-center justify-center relative transition-all ${
                activeActivity === item.id && showSidebar ? 'text-white border-l-2 border-white bg-[#252526]' : 'text-[#858585] hover:text-white'
              }`}
              title={item.title}
            >
              <item.icon size={22} />
            </button>
          ))}
          <div className="flex-1" />
          <button className="w-[48px] h-[48px] text-[#858585] hover:text-white flex items-center justify-center"><User size={20} /></button>
          <button className="w-[48px] h-[48px] text-[#858585] hover:text-white flex items-center justify-center"><Settings size={20} /></button>
        </div>

        {/* Sidebar */}
        {showSidebar && (
          <div className="bg-[#252526] border-r border-[#1e1e1e] flex flex-col overflow-hidden" style={{ width: 240 }}>
            <div className="h-[35px] flex items-center justify-between px-4 text-[11px] font-bold uppercase text-[#bbbbbb] tracking-wider">
              <span>{activeActivity.toUpperCase()}</span>
              <MoreHorizontal size={16} className="cursor-pointer hover:text-white" />
            </div>

            {activeActivity === 'files' && (
              <div className="flex-1 overflow-y-auto">
                <div className="py-1">
                  <div className="flex items-center gap-1 px-4 py-1 text-[11px] font-bold text-[#8b8b8b] uppercase tracking-tighter">
                    <ChevronDown size={12} />
                    <span>Dev-Workspace</span>
                  </div>
                  {loading && files.length === 0 ? (
                    <div className="flex items-center justify-center p-4"><Loader2 size={16} className="animate-spin text-[#8b8b8b]" /></div>
                  ) : renderFileTree(files)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Editor Area */}
        <div className="flex-1 flex flex-col min-w-0 bg-[#1e1e1e]">
          {/* Tabs */}
          <div className="h-[35px] bg-[#252526] flex items-center overflow-x-auto no-scrollbar border-b border-[#1e1e1e]">
            {openTabs.map(tab => {
              const iconConfig = getFileIcon(tab.name);
              const Icon = iconConfig.icon;
              return (
                <div
                  key={tab.path}
                  onClick={() => setActiveTab(tab.path)}
                  className={`h-[35px] flex items-center gap-2 px-3 cursor-pointer border-r border-[#1e1e1e] min-w-0 group transition-all ${
                    tab.path === activeTab ? 'bg-[#1e1e1e] border-t-2 border-t-[#007acc]' : 'bg-[#2d2d2d] hover:bg-[#2a2a2a] opacity-80'
                  }`}
                >
                  <Icon size={14} style={{ color: iconConfig.color }} className="flex-shrink-0" />
                  <span className={`text-[12px] truncate max-w-[120px] ${tab.path === activeTab ? 'text-white' : 'text-[#969696]'}`}>{tab.name}</span>
                  {tab.isDirty && <Circle size={6} className="fill-current text-white flex-shrink-0" />}
                  <button onClick={(e) => closeTab(tab.path, e)} className="ml-1 opacity-0 group-hover:opacity-100 hover:bg-[#404040] rounded p-0.5"><X size={14} /></button>
                </div>
              );
            })}
          </div>

          {/* Editor Container */}
          <div className="flex-1 relative overflow-hidden">
            {activeTabContent ? (
              <Editor
                height="100%"
                language={activeTabContent.language}
                value={activeTabContent.content}
                onChange={handleEditorChange}
                onMount={handleEditorMount}
                theme="vs-dark"
                options={{
                  fontSize: 13,
                  fontFamily: "'JetBrains Mono', monospace",
                  minimap: { enabled: true, scale: 0.75 },
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  tabSize: 2,
                  wordWrap: 'on',
                  lineNumbers: 'on',
                  renderWhitespace: 'selection',
                  bracketPairColorization: { enabled: true },
                }}
              />
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-[#3c3c3c]">
                <FileCode size={80} className="mb-4 opacity-10" />
                <h2 className="text-lg font-bold opacity-20">DEV-IDE v2.1</h2>
                <p className="text-xs opacity-20 mt-2">Open a file from the explorer</p>
              </div>
            )}
            
            {loading && (
              <div className="absolute inset-0 bg-black/20 backdrop-blur-[1px] flex items-center justify-center z-50">
                <Loader2 size={24} className="animate-spin text-[#007acc]" />
              </div>
            )}
          </div>

          {/* Terminal Panel */}
          {showTerminal && (
            <div className="h-40 bg-[#1e1e1e] border-t border-[#3c3c3c] flex flex-col shrink-0">
              <div className="h-8 bg-[#252526] flex items-center justify-between px-3 border-b border-[#1e1e1e]">
                <div className="flex gap-4 text-[11px] font-bold text-[#8b8b8b]">
                  <span className="text-white border-b border-white">TERMINAL</span>
                  <span className="hover:text-white cursor-pointer transition-colors">PROBLEMS</span>
                  <span className="hover:text-white cursor-pointer transition-colors">OUTPUT</span>
                </div>
                <div className="flex items-center gap-2">
                  <X size={14} className="cursor-pointer text-[#8b8b8b] hover:text-white" onClick={() => setShowTerminal(false)} />
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-2 font-mono text-[12px] scrollbar-thin scrollbar-thumb-[#3c3c3c]">
                {terminalOutput.map((line, i) => (
                  <div key={i} className={line.startsWith('$') ? 'text-emerald-400' : 'text-[#cccccc] mb-1'}>
                    {line}
                  </div>
                ))}
              </div>
              <div className="px-2 py-1 flex items-center border-t border-[#1e1e1e] bg-[#1e1e1e]">
                <span className="text-emerald-500 mr-2 text-xs">$</span>
                <input
                  type="text"
                  value={terminalInput}
                  onChange={(e) => setTerminalInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && executeTerminalCommand()}
                  className="flex-1 bg-transparent outline-none text-[12px] font-mono"
                  placeholder="Run command..."
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="h-[22px] bg-[#007acc] flex items-center justify-between px-2 text-[11px] text-white shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1"><GitBranch size={12} /><span>main*</span></div>
          <div className="flex items-center gap-1"><AlertCircle size={12} /><span>0</span><Info size={12} /><span>0</span></div>
        </div>
        <div className="flex items-center gap-3">
          <span>Ln {cursorPosition.line}, Col {cursorPosition.column}</span>
          <span>Spaces: 2</span>
          <span>UTF-8</span>
          <span>{activeTabContent?.language || 'Plain Text'}</span>
          <TerminalIcon size={12} className="cursor-pointer" onClick={() => setShowTerminal(!showTerminal)} />
        </div>
      </div>
    </div>
  );
});

CodeEditor.displayName = 'CodeEditor';

export default CodeEditor;
