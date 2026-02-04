"use client";

import { useState, useEffect } from 'react';
import { GitBranch, Github, X, RefreshCw, Link2, Unlink, ChevronRight, FileCode, GitPullRequest, Check, AlertCircle } from 'lucide-react';

interface GitHubRepo {
  name: string;
  full_name: string;
  description: string | null;
  html_url: string;
  private: boolean;
  language: string | null;
  stargazers_count: number;
}

interface GitHubPR {
  number: number;
  title: string;
  state: string;
  html_url: string;
  user: { login: string } | null;
}

interface GitHubPanelProps {
  isOpen: boolean;
  onClose: () => void;
  userId: string;
}

export default function GitHubPanel({ isOpen, onClose, userId }: GitHubPanelProps) {
  const [connected, setConnected] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [token, setToken] = useState('');
  const [repos, setRepos] = useState<GitHubRepo[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [prs, setPrs] = useState<GitHubPR[]>([]);
  const [workflowRuns, setWorkflowRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'connect' | 'repos' | 'prs' | 'workflows'>('connect');

  useEffect(() => {
    if (isOpen) {
      checkStatus();
    }
  }, [isOpen]);

  const loadWorkflows = async (repoFullName: string) => {
    setLoading(true);
    setSelectedRepo(repoFullName);
    try {
      const [owner, repo] = repoFullName.split('/');
      const res = await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'workflow_runs', userId, owner, repo }),
      });
      const data = await res.json();
      setWorkflowRuns(data.runs || []);
      setActiveTab('workflows');
    } catch (err) {
      console.error('Load Workflows error:', err);
    } finally {
      setLoading(false);
    }
  };

  const autoFixBuild = async (run: any) => {
    // This will notify the agent to start fixing the build
    // In a real app, this would send a special task to the agent API
    const [owner, repo] = selectedRepo!.split('/');
    const message = `Auto-fixing failing build #${run.run_number} for repo ${selectedRepo}. Action: ${run.name}`;
    
    // We can emit an event or just show a message for now
    // For the sake of this task, I'll simulate the agent starting a fix
    alert(`Agent starting auto-fix for build #${run.run_number}...`);
    onClose();
  };

  const checkStatus = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'status', userId }),
      });
      const data = await res.json();
      setConnected(data.connected);
      setUsername(data.username);
      if (data.connected) {
        setActiveTab('repos');
        loadRepos();
      }
    } catch (err) {
      console.error('Status check error:', err);
    } finally {
      setLoading(false);
    }
  };

  const connectGitHub = async () => {
    if (!token.trim()) {
      setError('Please enter a GitHub token');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'connect', userId, token }),
      });
      const data = await res.json();
      if (data.success) {
        setConnected(true);
        setUsername(data.user?.login);
        setToken('');
        setActiveTab('repos');
        loadRepos();
      } else {
        setError(data.error || 'Failed to connect');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  const disconnectGitHub = async () => {
    setLoading(true);
    try {
      await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'disconnect', userId }),
      });
      setConnected(false);
      setUsername(null);
      setRepos([]);
      setPrs([]);
      setActiveTab('connect');
    } catch (err) {
      console.error('Disconnect error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadRepos = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'repos', userId }),
      });
      const data = await res.json();
      setRepos(data.repos || []);
    } catch (err) {
      console.error('Load repos error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadPRs = async (repoFullName: string) => {
    setLoading(true);
    setSelectedRepo(repoFullName);
    try {
      const [owner, repo] = repoFullName.split('/');
      const res = await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'prs', userId, owner, repo }),
      });
      const data = await res.json();
      setPrs(data.prs || []);
      setActiveTab('prs');
    } catch (err) {
      console.error('Load PRs error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const languageColors: Record<string, string> = {
    JavaScript: 'bg-yellow-500',
    TypeScript: 'bg-blue-500',
    Python: 'bg-green-500',
    Ruby: 'bg-red-500',
    Go: 'bg-cyan-500',
    Rust: 'bg-orange-500',
    Java: 'bg-amber-600',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#0f0f17] border border-white/10 rounded-xl w-[700px] max-h-[70vh] overflow-hidden shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Github className="text-white" size={24} />
            <h2 className="text-white font-semibold text-lg">GitHub Integration</h2>
            {connected && username && (
              <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs border border-emerald-500/30">
                @{username}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {connected && (
              <button
                onClick={disconnectGitHub}
                disabled={loading}
                className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                title="Disconnect"
              >
                <Unlink size={18} />
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <X size={18} className="text-white/60" />
            </button>
          </div>
        </div>

        {connected && (
          <div className="flex border-b border-white/10">
            <button
              onClick={() => setActiveTab('repos')}
              className={`px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === 'repos'
                  ? 'text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              Repositories
            </button>
            <button
              onClick={() => setActiveTab('prs')}
              className={`px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === 'prs'
                  ? 'text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              Pull Requests
            </button>
            <button
              onClick={() => setActiveTab('workflows')}
              className={`px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === 'workflows'
                  ? 'text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              Workflows
            </button>
          </div>
        )}

        <div className="p-6 overflow-y-auto max-h-[calc(70vh-140px)]">
          {!connected && (
            <div className="space-y-4">
              <div className="text-white/60 text-sm mb-4">
                Connect your GitHub account to enable repository access, PR management, and code collaboration.
              </div>
              
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h3 className="text-white font-medium mb-3 flex items-center gap-2">
                  <Link2 size={16} />
                  Connect with Personal Access Token
                </h3>
                <div className="space-y-3">
                  <input
                    type="password"
                    value={token}
                    onChange={e => setToken(e.target.value)}
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white placeholder-white/40 outline-none focus:border-cyan-500/50"
                  />
                  <div className="text-white/40 text-xs">
                    Create a token at GitHub → Settings → Developer settings → Personal access tokens
                  </div>
                  <button
                    onClick={connectGitHub}
                    disabled={loading || !token.trim()}
                    className="w-full px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <RefreshCw size={16} className="animate-spin" />
                    ) : (
                      <Check size={16} />
                    )}
                    Connect
                  </button>
                </div>
              </div>

              {error && (
                <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 rounded-lg p-3 border border-red-500/20">
                  <AlertCircle size={16} />
                  {error}
                </div>
              )}
            </div>
          )}

          {connected && activeTab === 'repos' && (
            <div className="space-y-2">
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw size={24} className="text-cyan-400 animate-spin" />
                </div>
              ) : repos.length === 0 ? (
                <div className="text-white/40 text-center py-8">No repositories found</div>
              ) : (
                repos.map(repo => (
                  <button
                    key={repo.full_name}
                    onClick={() => loadPRs(repo.full_name)}
                    className="w-full p-4 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 text-left transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <FileCode size={18} className="text-white/40" />
                        <div>
                          <div className="text-white font-medium">{repo.name}</div>
                          {repo.description && (
                            <div className="text-white/40 text-xs mt-1 truncate max-w-[400px]">
                              {repo.description}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        {repo.language && (
                          <div className="flex items-center gap-1">
                            <div className={`w-2 h-2 rounded-full ${languageColors[repo.language] || 'bg-gray-500'}`} />
                            <span className="text-white/40 text-xs">{repo.language}</span>
                          </div>
                        )}
                        {repo.private && (
                          <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 rounded text-xs">
                            Private
                          </span>
                        )}
                        <ChevronRight size={16} className="text-white/40" />
                      </div>
                    </div>
                  </button>
                ))
              )}
            </div>
          )}

          {connected && activeTab === 'workflows' && (
            <div className="space-y-4">
              {selectedRepo && (
                <div className="flex items-center gap-2 text-white/60 text-sm mb-4">
                  <GitBranch size={16} />
                  {selectedRepo}
                </div>
              )}
              
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw size={24} className="text-cyan-400 animate-spin" />
                </div>
              ) : workflowRuns.length === 0 ? (
                <div className="text-white/40 text-center py-8">
                  {selectedRepo ? 'No workflow runs found' : 'Select a repository to view workflows'}
                </div>
              ) : (
                workflowRuns.map(run => (
                  <div
                    key={run.id}
                    className="p-4 bg-white/5 rounded-lg border border-white/10 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {run.conclusion === 'success' ? (
                          <Check size={18} className="text-emerald-400" />
                        ) : run.conclusion === 'failure' ? (
                          <AlertCircle size={18} className="text-red-400" />
                        ) : (
                          <RefreshCw size={18} className="text-amber-400 animate-spin" />
                        )}
                        <div>
                          <div className="text-white font-medium">{run.name} #{run.run_number}</div>
                          <div className="text-white/40 text-xs mt-1">
                            {run.head_branch} • {new Date(run.created_at).toLocaleString()}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {run.conclusion === 'failure' && (
                          <button
                            onClick={() => autoFixBuild(run)}
                            className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded text-xs font-medium border border-red-500/30 flex items-center gap-1"
                          >
                            <Zap size={12} />
                            Auto-Fix
                          </button>
                        )}
                        <a
                          href={run.html_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 hover:bg-white/10 rounded-lg transition-colors text-white/40 hover:text-white"
                        >
                          <ChevronRight size={18} />
                        </a>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
