"use client";

import { useState, useEffect } from 'react';
import { Brain, Search, Database, Plus, Trash2, X, RefreshCw, Zap, Check, AlertCircle } from 'lucide-react';

interface Memory {
  id: string;
  type: string;
  content: string;
  importance?: number;
  similarity?: number;
}

interface EmbeddingStatus {
  source: 'openai' | 'huggingface' | 'local' | null;
  dimensions: number;
  working: boolean;
}

interface MemoryPanelProps {
  isOpen: boolean;
  onClose: () => void;
  userId: string;
}

export default function MemoryPanel({ isOpen, onClose, userId }: MemoryPanelProps) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Memory[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [loading, setLoading] = useState(false);
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus>({ source: null, dimensions: 0, working: false });
  
  // Add memory form
  const [showAddForm, setShowAddForm] = useState(false);
  const [newMemoryType, setNewMemoryType] = useState<'fact' | 'pattern' | 'solution' | 'error_fix' | 'optimization'>('fact');
  const [newMemoryContent, setNewMemoryContent] = useState('');
  const [addingMemory, setAddingMemory] = useState(false);

  // Similarity test
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [testingEmbeddings, setTestingEmbeddings] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadMemories();
      testEmbeddingPipeline();
    }
  }, [isOpen]);

  const testEmbeddingPipeline = async () => {
    try {
      const res = await fetch('/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'test' }),
      });
      const data = await res.json();
      setEmbeddingStatus({
        source: data.source || null,
        dimensions: data.dimensions || 0,
        working: data.success || false,
      });
    } catch (error) {
      setEmbeddingStatus({ source: null, dimensions: 0, working: false });
    }
  };

  const loadMemories = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'get_recent_memories', limit: 20 }),
      });
      const data = await res.json();
      setMemories(data.memories || []);
    } catch (error) {
      console.error('Error loading memories:', error);
    } finally {
      setLoading(false);
    }
  };

  const searchMemories = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    try {
      const res = await fetch('/api/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'search_universal_memory', 
          query: searchQuery,
          limit: 10 
        }),
      });
      const data = await res.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error('Error searching memories:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const addMemory = async () => {
    if (!newMemoryContent.trim()) return;
    
    setAddingMemory(true);
    try {
      const res = await fetch('/api/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'add_universal_memory',
          type: newMemoryType,
          content: newMemoryContent,
          importance: 0.7,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setNewMemoryContent('');
        setShowAddForm(false);
        await loadMemories();
      }
    } catch (error) {
      console.error('Error adding memory:', error);
    } finally {
      setAddingMemory(false);
    }
  };

  const testSimilarity = async () => {
    if (!text1.trim() || !text2.trim()) return;
    
    setTestingEmbeddings(true);
    setSimilarity(null);
    try {
      // Get embeddings for both texts
      const [res1, res2] = await Promise.all([
        fetch('/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'single', text: text1 }),
        }),
        fetch('/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'single', text: text2 }),
        }),
      ]);
      
      const [data1, data2] = await Promise.all([res1.json(), res2.json()]);
      
      if (data1.embedding && data2.embedding) {
        // Calculate similarity
        const simRes = await fetch('/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'similarity',
            embedding1: data1.embedding,
            embedding2: data2.embedding,
          }),
        });
        const simData = await simRes.json();
        setSimilarity(simData.similarity);
      }
    } catch (error) {
      console.error('Error testing similarity:', error);
    } finally {
      setTestingEmbeddings(false);
    }
  };

  if (!isOpen) return null;

  const memoryTypeColors: Record<string, string> = {
    fact: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    pattern: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    solution: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    error_fix: 'bg-red-500/20 text-red-400 border-red-500/30',
    optimization: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  };

  const sourceColors: Record<string, string> = {
    openai: 'text-emerald-400',
    huggingface: 'text-amber-400',
    local: 'text-slate-400',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#0f0f17] border border-white/10 rounded-xl w-[1000px] max-h-[85vh] overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Brain className="text-purple-400" size={24} />
            <h2 className="text-white font-semibold text-lg">Semantic Memory</h2>
            {embeddingStatus.working && (
              <span className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs ${sourceColors[embeddingStatus.source || 'local']} bg-white/5`}>
                <Check size={12} />
                {embeddingStatus.source === 'openai' && 'OpenAI'}
                {embeddingStatus.source === 'huggingface' && 'HuggingFace'}
                {embeddingStatus.source === 'local' && 'Local'}
                ({embeddingStatus.dimensions}d)
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadMemories}
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

        <div className="flex h-[calc(85vh-70px)]">
          {/* Left: Memories list */}
          <div className="w-1/2 border-r border-white/10 p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white/80 font-medium flex items-center gap-2">
                <Database size={16} />
                Stored Memories ({memories.length})
              </h3>
              <button
                onClick={() => setShowAddForm(!showAddForm)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg text-sm transition-colors"
              >
                <Plus size={14} />
                Add
              </button>
            </div>

            {/* Add memory form */}
            {showAddForm && (
              <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10">
                <select
                  value={newMemoryType}
                  onChange={(e) => setNewMemoryType(e.target.value as any)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm mb-3 outline-none focus:border-purple-500/50"
                >
                  <option value="fact">Fact</option>
                  <option value="pattern">Pattern</option>
                  <option value="solution">Solution</option>
                  <option value="error_fix">Error Fix</option>
                  <option value="optimization">Optimization</option>
                </select>
                <textarea
                  value={newMemoryContent}
                  onChange={(e) => setNewMemoryContent(e.target.value)}
                  placeholder="Enter memory content..."
                  className="w-full h-24 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm outline-none focus:border-purple-500/50 resize-none mb-3"
                />
                <button
                  onClick={addMemory}
                  disabled={addingMemory || !newMemoryContent.trim()}
                  className="w-full py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {addingMemory ? (
                    <RefreshCw size={14} className="animate-spin" />
                  ) : (
                    <Plus size={14} />
                  )}
                  Add Memory
                </button>
              </div>
            )}

            {/* Memory list */}
            <div className="space-y-2">
              {memories.length === 0 ? (
                <div className="text-white/40 text-center py-8">No memories stored yet</div>
              ) : (
                memories.map((memory) => (
                  <div
                    key={memory.id}
                    className="p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-0.5 rounded text-xs border ${memoryTypeColors[memory.type] || 'bg-white/10 text-white/60'}`}>
                        {memory.type}
                      </span>
                    </div>
                    <p className="text-white/70 text-sm line-clamp-2">{memory.content}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Right: Search & Test */}
          <div className="w-1/2 p-4 overflow-y-auto">
            {/* Semantic Search */}
            <div className="mb-6">
              <h3 className="text-white/80 font-medium flex items-center gap-2 mb-3">
                <Search size={16} />
                Semantic Search
              </h3>
              <div className="flex gap-2 mb-3">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && searchMemories()}
                  placeholder="Search memories semantically..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm outline-none focus:border-cyan-500/50"
                />
                <button
                  onClick={searchMemories}
                  disabled={isSearching || !searchQuery.trim()}
                  className="px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {isSearching ? (
                    <RefreshCw size={14} className="animate-spin" />
                  ) : (
                    <Search size={14} />
                  )}
                  Search
                </button>
              </div>

              {/* Search results */}
              {searchResults.length > 0 && (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {searchResults.map((result) => (
                    <div
                      key={result.id}
                      className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/20"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={`px-2 py-0.5 rounded text-xs border ${memoryTypeColors[result.type] || 'bg-white/10 text-white/60'}`}>
                          {result.type}
                        </span>
                        {result.similarity !== undefined && (
                          <span className="text-cyan-400 text-xs">
                            {(result.similarity * 100).toFixed(1)}% match
                          </span>
                        )}
                      </div>
                      <p className="text-white/70 text-sm">{result.content}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Embedding Similarity Test */}
            <div className="border-t border-white/10 pt-6">
              <h3 className="text-white/80 font-medium flex items-center gap-2 mb-3">
                <Zap size={16} />
                Test Embedding Similarity
              </h3>
              <p className="text-white/40 text-xs mb-3">
                Enter two texts to calculate their semantic similarity using embeddings.
              </p>
              <textarea
                value={text1}
                onChange={(e) => setText1(e.target.value)}
                placeholder="Text 1: e.g., 'The cat sat on the mat'"
                className="w-full h-20 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm outline-none focus:border-purple-500/50 resize-none mb-2"
              />
              <textarea
                value={text2}
                onChange={(e) => setText2(e.target.value)}
                placeholder="Text 2: e.g., 'A feline rested on the rug'"
                className="w-full h-20 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm outline-none focus:border-purple-500/50 resize-none mb-3"
              />
              <button
                onClick={testSimilarity}
                disabled={testingEmbeddings || !text1.trim() || !text2.trim()}
                className="w-full py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {testingEmbeddings ? (
                  <RefreshCw size={14} className="animate-spin" />
                ) : (
                  <Zap size={14} />
                )}
                Calculate Similarity
              </button>

              {similarity !== null && (
                <div className="mt-4 p-4 bg-white/5 rounded-lg border border-white/10">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-400 mb-1">
                      {(similarity * 100).toFixed(1)}%
                    </div>
                    <div className="text-white/40 text-xs">Semantic Similarity</div>
                    <div className="mt-2 w-full h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-500 ${
                          similarity > 0.8 ? 'bg-emerald-500' :
                          similarity > 0.5 ? 'bg-amber-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${Math.max(0, similarity) * 100}%` }}
                      />
                    </div>
                    <div className="mt-2 text-xs text-white/60">
                      {similarity > 0.8 && 'Highly similar - same topic/meaning'}
                      {similarity > 0.5 && similarity <= 0.8 && 'Moderately similar - related topics'}
                      {similarity <= 0.5 && 'Low similarity - different topics'}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
