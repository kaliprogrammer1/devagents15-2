import { supabaseAdmin, type AgentSkill, type UserMemory, type UniversalMemory } from './supabase';
import { generateEmbedding as generateOpenAIEmbedding, EMBEDDING_DIMENSIONS } from './embeddings';
import pako from 'pako';

export function compressText(text: string): string {
  const uint8 = new TextEncoder().encode(text);
  const compressed = pako.deflate(uint8);
  return Buffer.from(compressed).toString('base64');
}

export function decompressText(compressed: string): string {
  const uint8 = Buffer.from(compressed, 'base64');
  const decompressed = pako.inflate(uint8);
  return new TextDecoder().decode(decompressed);
}

export function summarizeForStorage(content: string, maxLength: number = 500): string {
  if (content.length <= maxLength) return content;
  
  const sentences = content.match(/[^.!?]+[.!?]+/g) || [content];
  let summary = '';
  
  for (const sentence of sentences) {
    if ((summary + sentence).length <= maxLength) {
      summary += sentence;
    } else {
      break;
    }
  }
  
  return summary.trim() || content.substring(0, maxLength) + '...';
}

export class AgentMemory {
  /**
   * Generate real semantic embedding using OpenAI text-embedding-3-small
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const result = await generateOpenAIEmbedding(text);
      return result.embedding;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      // Return zero vector as fallback (will have low similarity to everything)
      return new Array(EMBEDDING_DIMENSIONS).fill(0);
    }
  }

  async addUniversalMemory(
    type: 'fact' | 'pattern' | 'solution' | 'error_fix' | 'optimization',
    content: string,
    importance: number = 0.5
  ): Promise<string | null> {
    try {
      const compressed = compressText(content);
      const embedding = await this.generateEmbedding(content);
      
      const { data, error } = await supabaseAdmin
        .from('agent_universal_memory')
        .insert({
          memory_type: type,
          content_compressed: compressed,
          importance_score: importance,
          access_count: 0,
          embedding_1536: embedding,
        })
        .select('id')
        .single();
      
      if (error) throw error;
      return data?.id || null;
    } catch (error) {
      console.error('Error adding universal memory:', error);
      return null;
    }
  }
  
  async searchUniversalMemory(query: string, limit: number = 10): Promise<Array<{ id: string; type: string; content: string; importance: number; similarity?: number }>> {
    try {
      const embedding = await this.generateEmbedding(query);
      
      // Use new v2 RPC function with real semantic embeddings
      const { data, error } = await supabaseAdmin.rpc('match_universal_memories_v2', {
        query_embedding: embedding,
        match_threshold: 0.7, // Higher threshold for real embeddings
        match_count: limit,
      });

      if (error) throw error;
      if (!data) return [];
      
      const results = data.map((m: any) => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
        importance: m.importance_score,
        similarity: m.similarity,
      }));
      
      // Record access
      for (const result of results) {
        await supabaseAdmin
          .from('agent_universal_memory')
          .update({ 
            access_count: (data.find((d: any) => d.id === result.id)?.access_count || 0) + 1,
            last_accessed: new Date().toISOString()
          })
          .eq('id', result.id);
      }
      
      return results;
    } catch (error) {
      console.error('Error searching universal memory:', error);
      // Fallback to keyword search if RPC fails
      return this.fallbackSearchUniversal(query, limit);
    }
  }

  private async fallbackSearchUniversal(query: string, limit: number): Promise<any[]> {
    const { data } = await supabaseAdmin
      .from('agent_universal_memory')
      .select('*')
      .order('importance_score', { ascending: false })
      .limit(limit * 3);
    
    if (!data) return [];
    
    const queryLower = query.toLowerCase();
    return data
      .map(m => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
        importance: m.importance_score,
      }))
      .filter(r => r.content.toLowerCase().includes(queryLower))
      .slice(0, limit);
  }
  
  async getRecentMemories(limit: number = 20): Promise<Array<{ id: string; type: string; content: string }>> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_universal_memory')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(limit);
      
      if (!data) return [];
      
      return data.map(m => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
      }));
    } catch (error) {
      console.error('Error getting recent memories:', error);
      return [];
    }
  }
  
  async getMostAccessedMemories(limit: number = 10): Promise<Array<{ id: string; type: string; content: string; accessCount: number }>> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_universal_memory')
        .select('*')
        .order('access_count', { ascending: false })
        .limit(limit);
      
      if (!data) return [];
      
      return data.map(m => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
        accessCount: m.access_count,
      }));
    } catch (error) {
      console.error('Error getting most accessed memories:', error);
      return [];
    }
  }
}

export class UserMemoryManager {
  private userId: string;
  
  constructor(userId: string) {
    this.userId = userId;
  }

  /**
   * Generate real semantic embedding using OpenAI text-embedding-3-small
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const result = await generateOpenAIEmbedding(text);
      return result.embedding;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      return new Array(EMBEDDING_DIMENSIONS).fill(0);
    }
  }
  
  async addMemory(
    type: 'preference' | 'context' | 'conversation' | 'task_history' | 'feedback',
    content: string,
    context?: Record<string, unknown>,
    importance: number = 0.5
  ): Promise<string | null> {
    try {
      const compressed = compressText(content);
      const embedding = await this.generateEmbedding(content);
      
      const { data, error } = await supabaseAdmin
        .from('user_memory')
        .insert({
          user_id: this.userId,
          memory_type: type,
          content_compressed: compressed,
          context,
          importance_score: importance,
          embedding_1536: embedding,
        })
        .select('id')
        .single();
      
      if (error) throw error;
      return data?.id || null;
    } catch (error) {
      console.error('Error adding user memory:', error);
      return null;
    }
  }
  
  async getMemories(type?: string, limit: number = 20): Promise<Array<{ id: string; type: string; content: string; context?: Record<string, unknown> }>> {
    try {
      let query = supabaseAdmin
        .from('user_memory')
        .select('*')
        .eq('user_id', this.userId)
        .order('importance_score', { ascending: false })
        .limit(limit);
      
      if (type) {
        query = query.eq('memory_type', type);
      }
      
      const { data } = await query;
      if (!data) return [];
      
      return data.map(m => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
        context: m.context,
      }));
    } catch (error) {
      console.error('Error getting user memories:', error);
      return [];
    }
  }
  
  async searchMemories(query: string, limit: number = 10): Promise<Array<{ id: string; type: string; content: string; similarity?: number }>> {
    try {
      const embedding = await this.generateEmbedding(query);
      
      // Use new v2 RPC function with real semantic embeddings
      const { data, error } = await supabaseAdmin.rpc('match_user_memories_v2', {
        query_embedding: embedding,
        p_user_id: this.userId,
        match_threshold: 0.7,
        match_count: limit,
      });

      if (error) throw error;
      if (!data) return [];
      
      return data.map((m: any) => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
        similarity: m.similarity,
      }));
    } catch (error) {
      console.error('Error searching user memories:', error);
      return this.fallbackSearchUser(query, limit);
    }
  }

  private async fallbackSearchUser(query: string, limit: number): Promise<any[]> {
    const { data } = await supabaseAdmin
      .from('user_memory')
      .select('*')
      .eq('user_id', this.userId)
      .order('importance_score', { ascending: false })
      .limit(limit * 3);
    
    if (!data) return [];
    
    const queryLower = query.toLowerCase();
    return data
      .map(m => ({
        id: m.id,
        type: m.memory_type,
        content: decompressText(m.content_compressed),
      }))
      .filter(m => m.content.toLowerCase().includes(queryLower))
      .slice(0, limit);
  }
  
  async setPreference(key: string, value: unknown, learnedFrom?: string): Promise<void> {
    try {
      await supabaseAdmin
        .from('user_preferences')
        .upsert({
          user_id: this.userId,
          preference_key: key,
          preference_value: { value },
          learned_from: learnedFrom,
          confidence: 0.7,
          updated_at: new Date().toISOString(),
        }, { onConflict: 'user_id,preference_key' });
    } catch (error) {
      console.error('Error setting preference:', error);
    }
  }
  
  async getPreference(key: string): Promise<unknown | null> {
    try {
      const { data } = await supabaseAdmin
        .from('user_preferences')
        .select('preference_value')
        .eq('user_id', this.userId)
        .eq('preference_key', key)
        .single();
      
      return data?.preference_value?.value || null;
    } catch (error) {
      console.error('Error getting preference:', error);
      return null;
    }
  }
  
  async getAllPreferences(): Promise<Record<string, unknown>> {
    try {
      const { data } = await supabaseAdmin
        .from('user_preferences')
        .select('preference_key, preference_value')
        .eq('user_id', this.userId);
      
      if (!data) return {};
      
      const prefs: Record<string, unknown> = {};
      for (const p of data) {
        prefs[p.preference_key] = p.preference_value?.value;
      }
      return prefs;
    } catch (error) {
      console.error('Error getting all preferences:', error);
      return {};
    }
  }
  
  async deleteMemory(id: string): Promise<boolean> {
    try {
      const { error } = await supabaseAdmin
        .from('user_memory')
        .delete()
        .eq('id', id)
        .eq('user_id', this.userId);
      
      return !error;
    } catch (error) {
      console.error('Error deleting memory:', error);
      return false;
    }
  }
}

export class SkillManager {
  async learnSkill(
    name: string,
    category: 'coding' | 'research' | 'communication' | 'analysis' | 'automation' | 'integration',
    description: string,
    knowledge: Record<string, unknown>
  ): Promise<string | null> {
    try {
      const compressedKnowledge = {
        ...knowledge,
        _compressed: compressText(JSON.stringify(knowledge)),
      };
      
      const { data: existing } = await supabaseAdmin
        .from('agent_skills')
        .select('id, usage_count, compressed_knowledge')
        .eq('skill_name', name)
        .single();
      
      if (existing) {
        const existingKnowledge = existing.compressed_knowledge || {};
        const mergedKnowledge = { ...existingKnowledge, ...compressedKnowledge };
        
        await supabaseAdmin
          .from('agent_skills')
          .update({
            compressed_knowledge: mergedKnowledge,
            description,
            updated_at: new Date().toISOString(),
          })
          .eq('id', existing.id);
        
        return existing.id;
      }
      
      const { data, error } = await supabaseAdmin
        .from('agent_skills')
        .insert({
          skill_name: name,
          skill_type: category,
          skill_category: category,
          description,
          compressed_knowledge: compressedKnowledge,
          metadata: compressedKnowledge,
          usage_count: 0,
          times_used: 0,
          success_rate: 0,
          proficiency_level: 1,
        })
        .select('id')
        .single();
      
      if (error) throw error;
      return data?.id || null;
    } catch (error) {
      console.error('Error learning skill:', error);
      return null;
    }
  }
  
  async useSkill(name: string, success: boolean): Promise<void> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_skills')
        .select('id, usage_count, success_rate')
        .eq('skill_name', name)
        .single();
      
        if (data) {
          const newCount = (data.usage_count || 0) + 1;
          const currentSuccessRate = data.success_rate || 0;
          const newSuccessRate = ((currentSuccessRate * (newCount - 1)) + (success ? 1 : 0)) / newCount;
          
          await supabaseAdmin
            .from('agent_skills')
            .update({
              usage_count: newCount,
              times_used: newCount,
              success_rate: newSuccessRate,
              last_used: new Date().toISOString(),
              last_used_at: new Date().toISOString(),
            })
            .eq('id', data.id);
        }
    } catch (error) {
      console.error('Error recording skill use:', error);
    }
  }
  
  async getSkill(name: string): Promise<AgentSkill | null> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_skills')
        .select('*')
        .eq('skill_name', name)
        .single();
      
      return data as AgentSkill | null;
    } catch (error) {
      console.error('Error getting skill:', error);
      return null;
    }
  }
  
  async getSkillsByCategory(category: string): Promise<AgentSkill[]> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_skills')
        .select('*')
        .eq('skill_category', category)
        .order('usage_count', { ascending: false });
      
      return (data as AgentSkill[]) || [];
    } catch (error) {
      console.error('Error getting skills by category:', error);
      return [];
    }
  }
  
  async getMostUsedSkills(limit: number = 10): Promise<AgentSkill[]> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_skills')
        .select('*')
        .order('usage_count', { ascending: false })
        .limit(limit);
      
      return (data as AgentSkill[]) || [];
    } catch (error) {
      console.error('Error getting most used skills:', error);
      return [];
    }
  }
  
  async getAllSkills(): Promise<AgentSkill[]> {
    try {
      const { data } = await supabaseAdmin
        .from('agent_skills')
        .select('*')
        .order('skill_category', { ascending: true });
      
      return (data as AgentSkill[]) || [];
    } catch (error) {
      console.error('Error getting all skills:', error);
      return [];
    }
  }
  
  async searchSkills(query: string): Promise<AgentSkill[]> {
    try {
      // First get skills from Supabase
      const { data } = await supabaseAdmin
        .from('agent_skills')
        .select('*')
        .or(`skill_name.ilike.%${query}%,description.ilike.%${query}%`)
        .order('usage_count', { ascending: false })
        .limit(10);
      
      const supabaseSkills = (data as AgentSkill[]) || [];
      
      // Try to load skills from skills.md if possible (in a real server environment)
      // For this implementation, we'll assume the API handles the merge or we can fetch them
      return supabaseSkills;
    } catch (error) {
      console.error('Error searching skills:', error);
      return [];
    }
  }

  async loadSkillsFromMarkdown(markdown: string): Promise<void> {
    const sections = markdown.split('## ').slice(1);
    for (const section of sections) {
      const lines = section.split('\n');
      const name = lines[0].trim();
      const description = section.match(/- \*\*Trigger\*\*: (.*)/)?.[1] || `Markdown skill: ${name}`;
      const actions = [];
      const actionsStart = lines.findIndex(l => l.includes('- **Actions**:'));
      if (actionsStart !== -1) {
        for (let i = actionsStart + 1; i < lines.length; i++) {
          if (lines[i].startsWith('  - ')) actions.push(lines[i].replace('  - ', '').trim());
          else if (lines[i].trim() === '' || lines[i].startsWith('##')) break;
        }
      }
      
      await this.learnSkill(name, 'automation', description, { actions, source: 'markdown' });
    }
  }
}

export class KnowledgeGraphManager {
  async addNode(name: string, type: string, content: string, metadata: Record<string, unknown> = {}): Promise<string | null> {
    try {
      // Generate embedding for semantic search
      const embeddingResult = await generateOpenAIEmbedding(content);
      
      const { data, error } = await supabaseAdmin
        .from('knowledge_nodes')
        .insert({
          name,
          node_type: type,
          content_compressed: compressText(content),
          metadata,
          embedding_1536: embeddingResult.embedding,
        })
        .select('id')
        .single();
      
      if (error) throw error;
      return data?.id || null;
    } catch (error) {
      console.error('Error adding knowledge node:', error);
      return null;
    }
  }

  async addEdge(sourceId: string, targetId: string, type: string, weight: number = 1.0): Promise<string | null> {
    try {
      const { data, error } = await supabaseAdmin
        .from('knowledge_edges')
        .insert({
          source_id: sourceId,
          target_id: targetId,
          relation_type: type,
          weight,
        })
        .select('id')
        .single();
      
      if (error) throw error;
      return data?.id || null;
    } catch (error) {
      console.error('Error adding knowledge edge:', error);
      return null;
    }
  }

  async findNodeByName(name: string): Promise<any | null> {
    const { data } = await supabaseAdmin
      .from('knowledge_nodes')
      .select('*')
      .ilike('name', name)
      .single();
    return data;
  }

  async getRelatedNodes(nodeId: string): Promise<any[]> {
    const { data, error } = await supabaseAdmin
      .from('knowledge_edges')
      .select('relation_type, weight, knowledge_nodes!knowledge_edges_target_id_fkey(*)')
      .eq('source_id', nodeId);
    
    if (error) {
      console.error('Error getting related nodes:', error);
      return [];
    }
    return data.map(d => ({
      relation: d.relation_type,
      weight: d.weight,
      node: {
        ...d.knowledge_nodes,
        content: d.knowledge_nodes.content_compressed ? decompressText(d.knowledge_nodes.content_compressed) : ''
      }
    }));
  }

  async searchGraph(query: string): Promise<any[]> {
    try {
      // Use semantic search with real embeddings
      const embeddingResult = await generateOpenAIEmbedding(query);
      
      const { data, error } = await supabaseAdmin.rpc('match_knowledge_nodes', {
        query_embedding: embeddingResult.embedding,
        match_threshold: 0.7,
        match_count: 10,
      });
      
      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error searching knowledge graph:', error);
      // Fallback to text search
      const { data } = await supabaseAdmin
        .from('knowledge_nodes')
        .select('*')
        .or(`name.ilike.%${query}%,metadata->>'summary'.ilike.%${query}%`)
        .limit(10);
      return data || [];
    }
  }
}

export const knowledgeGraph = new KnowledgeGraphManager();
export const agentMemory = new AgentMemory();

export const skillManager = new SkillManager();

export function createUserMemory(userId: string): UserMemoryManager {
  return new UserMemoryManager(userId);
}
