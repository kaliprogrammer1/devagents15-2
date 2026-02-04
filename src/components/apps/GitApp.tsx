"use client";

import { useState } from 'react';
import { useWorkspaceStore } from '@/lib/workspaceStore';
import { GitBranch, GitCommit, GitMerge, GitPullRequest, Plus, Check, Circle, Clock, ChevronDown, RefreshCw, Upload, Download } from 'lucide-react';

export default function GitApp() {
  const { 
    gitBranches, 
    currentBranch, 
    stagedFiles, 
    files,
    switchBranch, 
    createBranch, 
    stageFile, 
    unstageFile, 
    commit 
  } = useWorkspaceStore();
  
  const [activeTab, setActiveTab] = useState<'changes' | 'branches' | 'history'>('changes');
  const [commitMessage, setCommitMessage] = useState('');
  const [showNewBranch, setShowNewBranch] = useState(false);
  const [newBranchName, setNewBranchName] = useState('');

  const currentBranchData = gitBranches.find(b => b.name === currentBranch);
  
  const modifiedFiles = [
    { id: 'nav-tsx', name: 'src/components/Nav.tsx', status: 'modified' as const },
    { id: 'page-tsx', name: 'src/app/page.tsx', status: 'modified' as const },
  ];

  const handleCommit = () => {
    if (commitMessage.trim() && stagedFiles.length > 0) {
      commit(commitMessage.trim());
      setCommitMessage('');
    }
  };

  const handleCreateBranch = () => {
    if (newBranchName.trim()) {
      createBranch(newBranchName.trim());
      setNewBranchName('');
      setShowNewBranch(false);
    }
  };

  const formatTimeAgo = (timestamp: number) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  return (
    <div className="h-full bg-[#0d1117] text-[#c9d1d9] flex flex-col">
      <div className="h-12 bg-[#161b22] border-b border-[#30363d] flex items-center px-4 gap-3">
        <GitBranch size={18} className="text-[#58a6ff]" />
        <div className="flex items-center gap-2 bg-[#21262d] rounded-md px-3 py-1.5">
          <span className="text-sm">{currentBranch}</span>
          <ChevronDown size={14} />
        </div>
        <div className="flex-1" />
        <button className="flex items-center gap-1.5 text-xs bg-[#238636] hover:bg-[#2ea043] px-3 py-1.5 rounded-md">
          <Upload size={14} />
          Push
        </button>
        <button className="flex items-center gap-1.5 text-xs bg-[#21262d] hover:bg-[#30363d] px-3 py-1.5 rounded-md">
          <Download size={14} />
          Pull
        </button>
        <button className="p-1.5 hover:bg-[#21262d] rounded-md">
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="flex border-b border-[#30363d]">
        {(['changes', 'branches', 'history'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm capitalize ${
              activeTab === tab
                ? 'text-white border-b-2 border-[#f78166]'
                : 'text-[#8b949e] hover:text-white'
            }`}
          >
            {tab}
            {tab === 'changes' && stagedFiles.length > 0 && (
              <span className="ml-2 bg-[#238636] text-white text-xs px-1.5 rounded-full">
                {stagedFiles.length}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {activeTab === 'changes' && (
          <div className="p-4 space-y-4">
            {stagedFiles.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-[#7ee787] uppercase font-semibold">Staged Changes</span>
                  <button 
                    onClick={() => stagedFiles.forEach(f => unstageFile(f))}
                    className="text-xs text-[#8b949e] hover:text-white"
                  >
                    Unstage All
                  </button>
                </div>
                <div className="space-y-1">
                  {stagedFiles.map(fileId => {
                    const file = modifiedFiles.find(f => f.id === fileId);
                    return (
                      <div 
                        key={fileId}
                        className="flex items-center gap-2 p-2 bg-[#161b22] rounded hover:bg-[#21262d] cursor-pointer"
                        onClick={() => unstageFile(fileId)}
                      >
                        <Check size={14} className="text-[#7ee787]" />
                        <span className="text-sm flex-1">{file?.name || fileId}</span>
                        <span className="text-xs text-[#7ee787]">M</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-[#8b949e] uppercase font-semibold">Changes</span>
                <button 
                  onClick={() => modifiedFiles.forEach(f => stageFile(f.id))}
                  className="text-xs text-[#8b949e] hover:text-white"
                >
                  Stage All
                </button>
              </div>
              <div className="space-y-1">
                {modifiedFiles.filter(f => !stagedFiles.includes(f.id)).map(file => (
                  <div 
                    key={file.id}
                    className="flex items-center gap-2 p-2 bg-[#161b22] rounded hover:bg-[#21262d] cursor-pointer"
                    onClick={() => stageFile(file.id)}
                  >
                    <Circle size={14} className="text-[#d29922]" />
                    <span className="text-sm flex-1">{file.name}</span>
                    <span className="text-xs text-[#d29922]">M</span>
                  </div>
                ))}
                {modifiedFiles.filter(f => !stagedFiles.includes(f.id)).length === 0 && (
                  <div className="text-center text-[#8b949e] py-4 text-sm">
                    No unstaged changes
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-[#30363d]">
              <textarea
                value={commitMessage}
                onChange={(e) => setCommitMessage(e.target.value)}
                placeholder="Commit message..."
                className="w-full h-20 bg-[#0d1117] border border-[#30363d] rounded-md p-2 text-sm resize-none outline-none focus:border-[#58a6ff]"
              />
              <button 
                onClick={handleCommit}
                disabled={!commitMessage.trim() || stagedFiles.length === 0}
                className="w-full bg-[#238636] hover:bg-[#2ea043] disabled:opacity-50 disabled:cursor-not-allowed py-2 rounded-md text-sm font-medium"
              >
                Commit {stagedFiles.length > 0 && `(${stagedFiles.length} files)`}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'branches' && (
          <div className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">Branches</span>
              <button 
                onClick={() => setShowNewBranch(true)}
                className="flex items-center gap-1 text-xs bg-[#238636] hover:bg-[#2ea043] px-2 py-1 rounded"
              >
                <Plus size={12} />
                New Branch
              </button>
            </div>

            {showNewBranch && (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newBranchName}
                  onChange={(e) => setNewBranchName(e.target.value)}
                  placeholder="feature/new-feature"
                  className="flex-1 bg-[#0d1117] border border-[#30363d] rounded px-2 py-1 text-sm outline-none focus:border-[#58a6ff]"
                  autoFocus
                />
                <button 
                  onClick={handleCreateBranch}
                  className="bg-[#238636] hover:bg-[#2ea043] px-3 py-1 rounded text-sm"
                >
                  Create
                </button>
                <button 
                  onClick={() => setShowNewBranch(false)}
                  className="bg-[#21262d] hover:bg-[#30363d] px-3 py-1 rounded text-sm"
                >
                  Cancel
                </button>
              </div>
            )}

            <div className="space-y-1">
              {gitBranches.map((branch) => (
                <div 
                  key={branch.name}
                  onClick={() => switchBranch(branch.name)}
                  className={`flex items-center gap-2 p-2 rounded cursor-pointer ${
                    branch.name === currentBranch
                      ? 'bg-[#388bfd26] border border-[#388bfd]'
                      : 'bg-[#161b22] hover:bg-[#21262d]'
                  }`}
                >
                  <GitBranch size={14} className={branch.name === currentBranch ? 'text-[#58a6ff]' : 'text-[#8b949e]'} />
                  <span className="text-sm flex-1">{branch.name}</span>
                  {branch.name === currentBranch && (
                    <span className="text-xs bg-[#238636] px-1.5 py-0.5 rounded">current</span>
                  )}
                  <span className="text-xs text-[#8b949e]">{branch.commits.length} commits</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="p-4">
            <div className="space-y-0">
              {currentBranchData?.commits.slice().reverse().map((commit, i) => (
                <div key={commit.id} className="flex gap-3">
                  <div className="flex flex-col items-center">
                    <div className="w-3 h-3 rounded-full bg-[#58a6ff] border-2 border-[#0d1117]" />
                    {i < (currentBranchData?.commits.length || 0) - 1 && (
                      <div className="w-0.5 flex-1 bg-[#30363d]" />
                    )}
                  </div>
                  <div className="flex-1 pb-4">
                    <div className="bg-[#161b22] rounded-md p-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="text-sm font-medium">{commit.message}</div>
                          <div className="text-xs text-[#8b949e] mt-1">
                            {commit.author} • {formatTimeAgo(commit.timestamp)}
                          </div>
                        </div>
                        <code className="text-xs text-[#8b949e] bg-[#21262d] px-1.5 py-0.5 rounded">
                          {commit.id.slice(0, 7)}
                        </code>
                      </div>
                      {commit.files.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {commit.files.map((file, j) => (
                            <span key={j} className="text-xs bg-[#21262d] px-1.5 py-0.5 rounded">
                              {file}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {(!currentBranchData?.commits || currentBranchData.commits.length === 0) && (
                <div className="text-center text-[#8b949e] py-8">
                  No commits yet
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
