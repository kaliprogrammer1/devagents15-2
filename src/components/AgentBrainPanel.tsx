"use client";

import { useState, useEffect } from 'react';
import { Brain, Zap, Database, Code, GitBranch, X, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

interface Skill {
  id: string;
  skill_name: string;
  skill_type: string;
  description: string;
  usage_count: number;
  success_rate: number;
  proficiency_level: number;
}

interface Memory {
  id: string;
  type: string;
  content: string;
}

interface AgentStats {
  totalSkills: number;
  totalMemories: number;
  topSkills: string[];
  recentLearnings: string[];
}

interface AgentBrainPanelProps {
  isOpen: boolean;
  onClose: () => void;
  userId: string;
}

export default function AgentBrainPanel({ isOpen, onClose, userId }: AgentBrainPanelProps) {
  const [stats, setStats] = useState<AgentStats | null>(null);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'skills' | 'memory' | 'code'>('overview');
  const [codeLanguage, setCodeLanguage] = useState('javascript');
  const [codeInput, setCodeInput] = useState('console.log("Hello World!");');
  const [codeOutput, setCodeOutput] = useState<string | null>(null);
  const [executing, setExecuting] = useState(false);
  const [expandedSkill, setExpandedSkill] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [statsRes, skillsRes, memoriesRes] = await Promise.all([
        fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'get_stats', userId }),
        }),
        fetch('/api/memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'get_all_skills' }),
        }),
        fetch('/api/memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'get_recent_memories', limit: 10 }),
        }),
      ]);

      if (statsRes.ok) setStats(await statsRes.json());
      if (skillsRes.ok) {
        const data = await skillsRes.json();
        setSkills(data.skills || []);
      }
      if (memoriesRes.ok) {
        const data = await memoriesRes.json();
        setMemories(data.memories || []);
      }
    } catch (error) {
      console.error('Error loading brain data:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeCode = async () => {
    setExecuting(true);
    setCodeOutput(null);
    try {
      const res = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'execute',
          userId,
          language: codeLanguage,
          code: codeInput,
        }),
      });
      const data = await res.json();
      setCodeOutput(data.success ? data.output : `Error: ${data.error}`);
    } catch (error) {
      setCodeOutput(`Error: ${error}`);
    } finally {
      setExecuting(false);
    }
  };

  if (!isOpen) return null;

  const categoryColors: Record<string, string> = {
    coding: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    research: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    communication: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    analysis: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    automation: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
    integration: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#0f0f17] border border-white/10 rounded-xl w-[900px] max-h-[80vh] overflow-hidden shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Brain className="text-cyan-400" size={24} />
            <h2 className="text-white font-semibold text-lg">Agent Brain</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadData}
              disabled={loading}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <RefreshCw size={18} className={`text-white/60 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <X size={18} className="text-white/60" />
            </button>
          </div>
        </div>

        <div className="flex border-b border-white/10">
          {(['overview', 'skills', 'memory', 'code'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? 'text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(80vh-140px)]">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center gap-2 text-emerald-400 mb-2">
                    <Zap size={18} />
                    <span className="text-sm">Skills</span>
                  </div>
                  <div className="text-2xl font-bold text-white">{stats?.totalSkills || skills.length}</div>
                </div>
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center gap-2 text-purple-400 mb-2">
                    <Database size={18} />
                    <span className="text-sm">Memories</span>
                  </div>
                  <div className="text-2xl font-bold text-white">{stats?.totalMemories || memories.length}</div>
                </div>
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center gap-2 text-cyan-400 mb-2">
                    <Code size={18} />
                    <span className="text-sm">Code Runs</span>
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {skills.find(s => s.skill_name === 'code_execution')?.usage_count || 0}
                  </div>
                </div>
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center gap-2 text-pink-400 mb-2">
                    <GitBranch size={18} />
                    <span className="text-sm">GitHub</span>
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {skills.find(s => s.skill_name === 'github_integration')?.usage_count || 0}
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-white/80 font-medium mb-3">Top Skills</h3>
                <div className="flex flex-wrap gap-2">
                  {(stats?.topSkills || skills.slice(0, 5).map(s => s.skill_name)).map((skill, i) => (
                    <span
                      key={i}
                      className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm border border-cyan-500/30"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-white/80 font-medium mb-3">Recent Learnings</h3>
                <div className="space-y-2">
                  {(stats?.recentLearnings || memories.slice(0, 3).map(m => m.content)).map((learning, i) => (
                    <div
                      key={i}
                      className="p-3 bg-white/5 rounded-lg border border-white/10 text-white/70 text-sm"
                    >
                      {learning}
                    </div>
                  ))}
                  {(!stats?.recentLearnings?.length && !memories.length) && (
                    <div className="text-white/40 text-sm">No recent learnings yet</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'skills' && (
            <div className="space-y-3">
              {skills.length === 0 ? (
                <div className="text-white/40 text-center py-8">No skills learned yet</div>
              ) : (
                skills.map(skill => (
                  <div
                    key={skill.id}
                    className="bg-white/5 rounded-lg border border-white/10 overflow-hidden"
                  >
                    <button
                      onClick={() => setExpandedSkill(expandedSkill === skill.id ? null : skill.id)}
                      className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className={`px-2 py-1 rounded text-xs border ${
                            categoryColors[skill.skill_type] || 'bg-white/10 text-white/60'
                          }`}
                        >
                          {skill.skill_type}
                        </span>
                        <span className="text-white font-medium">{skill.skill_name}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-white/40 text-sm">
                          Used {skill.usage_count || 0}x
                        </span>
                        <span className="text-emerald-400 text-sm">
                          {Math.round((skill.success_rate || 0) * 100)}% success
                        </span>
                        {expandedSkill === skill.id ? (
                          <ChevronUp size={18} className="text-white/40" />
                        ) : (
                          <ChevronDown size={18} className="text-white/40" />
                        )}
                      </div>
                    </button>
                    {expandedSkill === skill.id && (
                      <div className="px-4 pb-4 border-t border-white/10">
                        <p className="text-white/60 text-sm mt-3">{skill.description}</p>
                        <div className="mt-3 flex items-center gap-2">
                          <span className="text-white/40 text-xs">Proficiency:</span>
                          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-cyan-500"
                              style={{ width: `${Math.min((skill.proficiency_level || 1) * 20, 100)}%` }}
                            />
                          </div>
                          <span className="text-white/60 text-xs">Level {skill.proficiency_level || 1}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'memory' && (
            <div className="space-y-3">
              {memories.length === 0 ? (
                <div className="text-white/40 text-center py-8">No memories stored yet</div>
              ) : (
                memories.map(memory => (
                  <div
                    key={memory.id}
                    className="p-4 bg-white/5 rounded-lg border border-white/10"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs border border-purple-500/30">
                        {memory.type}
                      </span>
                    </div>
                    <p className="text-white/70 text-sm">{memory.content}</p>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'code' && (
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <select
                  value={codeLanguage}
                  onChange={e => setCodeLanguage(e.target.value)}
                  className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm outline-none focus:border-cyan-500/50"
                >
                  <option value="javascript">JavaScript</option>
                  <option value="python">Python</option>
                  <option value="typescript">TypeScript</option>
                  <option value="ruby">Ruby</option>
                  <option value="go">Go</option>
                  <option value="rust">Rust</option>
                  <option value="bash">Bash</option>
                </select>
                <button
                  onClick={executeCode}
                  disabled={executing}
                  className="px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {executing ? (
                    <RefreshCw size={16} className="animate-spin" />
                  ) : (
                    <Code size={16} />
                  )}
                  Run Code
                </button>
              </div>

              <textarea
                value={codeInput}
                onChange={e => setCodeInput(e.target.value)}
                className="w-full h-48 bg-[#1a1a24] border border-white/10 rounded-lg p-4 text-white font-mono text-sm outline-none focus:border-cyan-500/50 resize-none"
                placeholder="Enter your code here..."
              />

              {codeOutput !== null && (
                <div className="bg-[#1a1a24] border border-white/10 rounded-lg p-4">
                  <div className="text-white/40 text-xs mb-2">Output:</div>
                  <pre className="text-emerald-400 font-mono text-sm whitespace-pre-wrap">
                    {codeOutput || '(no output)'}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
