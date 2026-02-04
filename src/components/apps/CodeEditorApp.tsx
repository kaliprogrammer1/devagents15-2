"use client";

import { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { useWorkspaceStore, findFileById, getFilePath, FileNode } from '@/lib/workspaceStore';
import { ChevronRight, ChevronDown, File, Folder, FolderOpen, X, Plus, Trash2, Search, RefreshCw, MoreVertical } from 'lucide-react';

const FILE_ICONS: Record<string, { color: string; icon?: string }> = {
  tsx: { color: '#3178c6' },
  ts: { color: '#3178c6' },
  jsx: { color: '#61dafb' },
  js: { color: '#f7df1e' },
  css: { color: '#264de4' },
  json: { color: '#cbcb41' },
  md: { color: '#ffffff' },
  html: { color: '#e34c26' },
  default: { color: '#8b8b8b' },
};

function FileTreeItem({ node, depth = 0 }: { node: FileNode; depth?: number }) {
  const { toggleFolder, openFile, activeFileId, deleteFile } = useWorkspaceStore();
  const [showMenu, setShowMenu] = useState(false);
  
  const isActive = node.id === activeFileId;
  const ext = node.name.split('.').pop() || 'default';
  const iconStyle = FILE_ICONS[ext] || FILE_ICONS.default;

  const handleClick = () => {
    if (node.type === 'folder') {
      toggleFolder(node.id);
    } else {
      openFile(node.id);
    }
  };

  return (
    <div>
      <div
        className={`flex items-center gap-1 px-2 py-0.5 cursor-pointer text-sm group relative ${
          isActive ? 'bg-[#37373d] text-white' : 'text-[#cccccc] hover:bg-[#2a2d2e]'
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={handleClick}
        onContextMenu={(e) => { e.preventDefault(); setShowMenu(true); }}
      >
        {node.type === 'folder' ? (
          <>
            {node.isOpen ? <ChevronDown size={14} className="text-[#8b8b8b]" /> : <ChevronRight size={14} className="text-[#8b8b8b]" />}
            {node.isOpen ? <FolderOpen size={14} className="text-[#dcb67a]" /> : <Folder size={14} className="text-[#dcb67a]" />}
          </>
        ) : (
          <>
            <span className="w-[14px]" />
            <File size={14} style={{ color: iconStyle.color }} />
          </>
        )}
        <span className="ml-1 truncate">{node.name}</span>
        {node.modified && <span className="ml-1 w-2 h-2 bg-white rounded-full" />}
        
        <button 
          className="ml-auto opacity-0 group-hover:opacity-100 p-0.5 hover:bg-white/10 rounded"
          onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu); }}
        >
          <MoreVertical size={12} />
        </button>
        
        {showMenu && (
          <div className="absolute right-0 top-full z-50 bg-[#252526] border border-[#454545] rounded shadow-lg py-1 min-w-32">
            <button 
              onClick={(e) => { e.stopPropagation(); deleteFile(node.id); setShowMenu(false); }}
              className="w-full text-left px-3 py-1 text-xs text-red-400 hover:bg-[#37373d] flex items-center gap-2"
            >
              <Trash2 size={12} /> Delete
            </button>
          </div>
        )}
      </div>
      
      {node.type === 'folder' && node.isOpen && node.children && (
        <div>
          {node.children.map(child => (
            <FileTreeItem key={child.id} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function CodeEditorApp() {
  const { 
    files, 
    openFiles, 
    activeFileId, 
    setActiveFile, 
    closeFile, 
    updateFileContent,
    createFile 
  } = useWorkspaceStore();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [showNewFile, setShowNewFile] = useState(false);
  const [newFileName, setNewFileName] = useState('');
  const editorRef = useRef<unknown>(null);

  const activeFile = activeFileId ? findFileById(files, activeFileId) : null;
  const filePath = activeFileId ? getFilePath(files, activeFileId) : null;

  const getLanguage = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      ts: 'typescript',
      tsx: 'typescript',
      js: 'javascript',
      jsx: 'javascript',
      json: 'json',
      css: 'css',
      scss: 'scss',
      html: 'html',
      md: 'markdown',
      py: 'python',
      rs: 'rust',
      go: 'go',
    };
    return langMap[ext || ''] || 'plaintext';
  };

  const handleEditorChange = (value: string | undefined) => {
    if (activeFileId && value !== undefined) {
      updateFileContent(activeFileId, value);
    }
  };

  const handleCreateFile = () => {
    if (newFileName.trim()) {
      createFile(null, newFileName.trim(), newFileName.includes('.') ? 'file' : 'folder');
      setNewFileName('');
      setShowNewFile(false);
    }
  };

  return (
    <div className="h-full flex bg-[#1e1e1e] text-white">
      <div className="w-56 bg-[#252526] border-r border-[#3c3c3c] flex flex-col flex-shrink-0">
        <div className="p-2 text-xs text-[#bbbbbb] uppercase font-semibold flex items-center justify-between">
          <span>Explorer</span>
          <div className="flex gap-1">
            <button 
              onClick={() => setShowNewFile(true)}
              className="p-1 hover:bg-[#37373d] rounded"
              title="New File"
            >
              <Plus size={14} />
            </button>
            <button className="p-1 hover:bg-[#37373d] rounded" title="Refresh">
              <RefreshCw size={14} />
            </button>
          </div>
        </div>
        
        {showNewFile && (
          <div className="px-2 pb-2">
            <input
              type="text"
              value={newFileName}
              onChange={(e) => setNewFileName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreateFile()}
              onBlur={() => setShowNewFile(false)}
              placeholder="filename.ts"
              className="w-full bg-[#3c3c3c] border border-[#007acc] rounded px-2 py-1 text-xs outline-none"
              autoFocus
            />
          </div>
        )}
        
        <div className="px-2 pb-2">
          <div className="flex items-center bg-[#3c3c3c] rounded px-2 py-1">
            <Search size={12} className="text-[#8b8b8b]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search files..."
              className="flex-1 bg-transparent text-xs outline-none ml-2 placeholder-[#8b8b8b]"
            />
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          {files.map(node => (
            <FileTreeItem key={node.id} node={node} />
          ))}
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="h-9 bg-[#252526] border-b border-[#3c3c3c] flex items-center overflow-x-auto">
          {openFiles.map(fileId => {
            const file = findFileById(files, fileId);
            if (!file) return null;
            const ext = file.name.split('.').pop() || 'default';
            const iconStyle = FILE_ICONS[ext] || FILE_ICONS.default;
            const isActive = fileId === activeFileId;
            
            return (
              <div
                key={fileId}
                className={`flex items-center gap-1.5 px-3 h-full border-r border-[#3c3c3c] cursor-pointer group ${
                  isActive ? 'bg-[#1e1e1e]' : 'bg-[#2d2d2d] hover:bg-[#2a2d2e]'
                }`}
                onClick={() => setActiveFile(fileId)}
              >
                <File size={14} style={{ color: iconStyle.color }} />
                <span className="text-sm text-[#cccccc] whitespace-nowrap">{file.name}</span>
                {file.modified && <span className="w-2 h-2 bg-white rounded-full" />}
                <button 
                  onClick={(e) => { e.stopPropagation(); closeFile(fileId); }}
                  className="ml-1 opacity-0 group-hover:opacity-100 hover:bg-white/10 rounded p-0.5"
                >
                  <X size={12} />
                </button>
              </div>
            );
          })}
        </div>

        {filePath && (
          <div className="h-6 bg-[#252526] border-b border-[#3c3c3c] flex items-center px-3 text-xs text-[#8b8b8b]">
            {filePath.join(' › ')}
          </div>
        )}

        <div className="flex-1">
          {activeFile && activeFile.content !== undefined ? (
            <Editor
              height="100%"
              language={getLanguage(activeFile.name)}
              value={activeFile.content}
              onChange={handleEditorChange}
              theme="vs-dark"
              options={{
                fontSize: 13,
                fontFamily: 'JetBrains Mono, Menlo, Monaco, Courier New, monospace',
                minimap: { enabled: true, scale: 1 },
                scrollBeyondLastLine: false,
                renderLineHighlight: 'all',
                cursorBlinking: 'smooth',
                smoothScrolling: true,
                padding: { top: 10 },
                automaticLayout: true,
                tabSize: 2,
                wordWrap: 'on',
              }}
              onMount={(editor) => {
                editorRef.current = editor;
              }}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-[#8b8b8b]">
              <div className="text-center">
                <div className="text-6xl mb-4">📝</div>
                <div className="text-lg">Select a file to edit</div>
                <div className="text-sm mt-2">or create a new one</div>
              </div>
            </div>
          )}
        </div>

        <div className="h-6 bg-[#007acc] flex items-center px-3 text-xs text-white justify-between">
          <div className="flex items-center gap-4">
            <span>DevAgent IDE</span>
            {activeFile && <span>{getLanguage(activeFile.name).toUpperCase()}</span>}
          </div>
          <div className="flex items-center gap-4">
            <span>UTF-8</span>
            <span>LF</span>
            <span>Spaces: 2</span>
          </div>
        </div>
      </div>
    </div>
  );
}
