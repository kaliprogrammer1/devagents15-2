"use client";

import { useState, useEffect, useCallback } from 'react';
import { Folder, File, ChevronRight, ChevronLeft, Home, Search, RefreshCw, HardDrive, Cpu, MoreVertical, Plus, Trash2, Edit3, FolderPlus, FilePlus, Loader2, Activity } from 'lucide-react';
import { useWorkspaceStore } from '@/lib/workspaceStore';

interface FileEntry {
  id: string;
  name: string;
  type: 'file' | 'folder';
}

export default function FileExplorerApp() {
  const { openFile } = useWorkspaceStore();
  const [currentPath, setCurrentPath] = useState<string[]>([]);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState<string | null>(null);

  const fetchFiles = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const pathStr = currentPath.join('/');
      const res = await fetch('/api/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'list', path: pathStr }),
      });
      const data = await res.json();
      if (data.files) {
        setFiles(data.files);
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to fetch files');
    } finally {
      setLoading(false);
    }
  }, [currentPath]);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const navigateTo = (folderName: string) => {
    setCurrentPath([...currentPath, folderName]);
  };

  const goBack = () => {
    if (currentPath.length === 0) return;
    setCurrentPath(currentPath.slice(0, -1));
  };

  const handleCreateFile = async () => {
    const name = prompt('Enter file name:');
    if (name) {
      try {
        const pathStr = currentPath.join('/');
        await fetch('/api/files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'create', path: pathStr, name, type: 'file' }),
        });
        fetchFiles();
      } catch (err) {
        alert('Failed to create file');
      }
    }
  };

  const handleCreateFolder = async () => {
    const name = prompt('Enter folder name:');
    if (name) {
      try {
        const pathStr = currentPath.join('/');
        await fetch('/api/files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'create', path: pathStr, name, type: 'folder' }),
        });
        fetchFiles();
      } catch (err) {
        alert('Failed to create folder');
      }
    }
  };

  const handleDelete = async (file: FileEntry) => {
    if (confirm(`Are you sure you want to delete ${file.name}?`)) {
      try {
        await fetch('/api/files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'delete', path: file.id }),
        });
        fetchFiles();
      } catch (err) {
        alert('Failed to delete');
      }
    }
  };

  const handleOpenFile = (file: FileEntry) => {
    if (file.type === 'folder') {
      navigateTo(file.name);
    } else {
      // For now, we use the id as the file path
      openFile(file.id);
    }
  };

  const filteredFiles = files.filter(f => 
    f.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-full bg-white flex flex-col font-sans select-none">
      {/* Toolbar */}
      <div className="p-3 border-b flex items-center gap-3 bg-slate-50/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-1">
          <button 
            onClick={goBack} 
            disabled={currentPath.length === 0}
            className="p-2 hover:bg-slate-200 disabled:opacity-30 rounded-lg text-slate-600 transition-colors"
          >
            <ChevronLeft size={18} />
          </button>
          <button 
            onClick={fetchFiles}
            className="p-2 hover:bg-slate-200 rounded-lg text-slate-600 transition-colors"
          >
            <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
        
        {/* Breadcrumbs */}
        <div className="flex-1 flex items-center bg-white border border-slate-200 rounded-xl px-3 py-1.5 gap-2 shadow-sm focus-within:ring-2 focus-within:ring-blue-500/10 transition-all overflow-hidden">
          <button 
            onClick={() => setCurrentPath([])}
            className="hover:text-blue-500 transition-colors"
          >
            <Home size={14} className="text-slate-400" />
          </button>
          {currentPath.map((segment, i) => (
            <div key={i} className="flex items-center gap-1 flex-shrink-0">
              <ChevronRight size={12} className="text-slate-300" />
              <button 
                onClick={() => setCurrentPath(currentPath.slice(0, i + 1))}
                className="text-xs text-slate-600 font-medium hover:text-blue-500 transition-colors max-w-[100px] truncate"
              >
                {segment}
              </button>
            </div>
          ))}
        </div>
  
        {/* Search */}
        <div className="w-48 relative hidden md:block">
          <input
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white border border-slate-200 rounded-xl px-4 py-1.5 pl-9 text-xs outline-none focus:ring-2 focus:ring-blue-500/10 transition-all"
          />
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={14} />
        </div>
  
        <div className="h-6 w-[1px] bg-slate-200 mx-1" />
  
        <div className="flex items-center gap-1">
          <button 
            onClick={handleCreateFile}
            className="p-2 hover:bg-blue-50 text-blue-600 rounded-lg transition-colors"
            title="New File"
          >
            <FilePlus size={18} />
          </button>
          <button 
            onClick={handleCreateFolder}
            className="p-2 hover:bg-amber-50 text-amber-600 rounded-lg transition-colors"
            title="New Folder"
          >
            <FolderPlus size={18} />
          </button>
        </div>
      </div>
  
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-44 bg-slate-50/50 border-r p-3 space-y-1 hidden sm:block">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 px-2">Locations</div>
          <button 
            onClick={() => setCurrentPath([])}
            className={`w-full flex items-center gap-2 px-3 py-2 text-xs rounded-lg transition-all font-medium ${currentPath.length === 0 ? 'bg-blue-500 text-white shadow-md shadow-blue-500/20' : 'text-slate-600 hover:bg-white hover:shadow-sm'}`}
          >
            <HardDrive size={14} className={currentPath.length === 0 ? 'text-white' : 'text-blue-500'} /> 
            Root
          </button>
          <button 
            onClick={() => navigateTo('src')}
            className={`w-full flex items-center gap-2 px-3 py-2 text-xs rounded-lg transition-all font-medium ${currentPath[0] === 'src' ? 'bg-blue-500 text-white shadow-md shadow-blue-500/20' : 'text-slate-600 hover:bg-white hover:shadow-sm'}`}
          >
            <Folder size={14} className={currentPath[0] === 'src' ? 'text-white' : 'text-amber-500'} /> 
            Source
          </button>
        </div>
  
        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="animate-spin text-blue-500" size={32} />
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-50 text-red-600 rounded-xl text-sm mb-4">
              {error}
            </div>
          )}

          {!loading && !error && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6">
              {filteredFiles.map((node) => (
                <div
                  key={node.id}
                  className="group flex flex-col items-center gap-2"
                  onDoubleClick={() => handleOpenFile(node)}
                >
                  <div className="relative w-16 h-16 flex items-center justify-center rounded-2xl group-hover:bg-blue-50 group-hover:scale-105 transition-all duration-200 cursor-pointer">
                    {node.type === 'folder' ? (
                      <Folder size={48} className="text-amber-400 fill-amber-100/50 group-hover:fill-amber-100" />
                    ) : (
                      <div className="relative">
                        <File size={42} className="text-slate-300 fill-slate-50" />
                        <div className="absolute inset-0 flex items-center justify-center opacity-40 group-hover:opacity-100 transition-opacity">
                          <span className="text-[8px] font-bold text-slate-500 uppercase">
                            {node.name.split('.').pop()?.slice(0, 3)}
                          </span>
                        </div>
                      </div>
                    )}
                    
                    {/* Item Actions */}
                    <div className="absolute -top-1 -right-1 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button 
                        onClick={(e) => { e.stopPropagation(); handleDelete(node); }}
                        className="p-1.5 bg-white border border-slate-200 rounded-full text-slate-400 hover:text-red-500 shadow-sm hover:shadow-md transition-all"
                      >
                        <Trash2 size={10} />
                      </button>
                    </div>
                  </div>
                  <span className="text-[11px] font-semibold text-slate-700 text-center break-all line-clamp-2 px-1 rounded hover:bg-slate-100 transition-colors">
                    {node.name}
                  </span>
                </div>
              ))}
              
              {/* Create Actions for empty space */}
              <button 
                onClick={handleCreateFile}
                className="flex flex-col items-center gap-2 p-4 rounded-2xl border-2 border-dashed border-slate-100 hover:border-blue-200 hover:bg-blue-50/30 transition-all opacity-40 hover:opacity-100"
              >
                <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 group-hover:text-blue-500">
                  <Plus size={20} />
                </div>
                <span className="text-[10px] font-bold text-slate-400">New Item</span>
              </button>
            </div>
          )}
          
          {!loading && !error && filteredFiles.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full py-20 text-slate-300">
              <Cpu size={48} className="mb-4 opacity-10" />
              <p className="text-sm font-medium">This folder is empty</p>
            </div>
          )}
        </div>
      </div>
  
      {/* Status Bar */}
      <div className="p-2 border-t bg-slate-50 flex items-center justify-between text-[10px] text-slate-500 px-4">
        <div className="flex items-center gap-3">
          <span>{filteredFiles.length} items</span>
          <div className="h-3 w-[1px] bg-slate-200" />
          <span>Available: 256 GB</span>
        </div>
        <div className="flex items-center gap-2">
          <Activity size={12} className="text-emerald-500" />
          <span className="font-bold tracking-tight">DEV-OS EXPLORER v2.0</span>
        </div>
      </div>
    </div>
  );
}
