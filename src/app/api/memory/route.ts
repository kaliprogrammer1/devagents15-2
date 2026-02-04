import { NextRequest, NextResponse } from 'next/server';
import { agentMemory, skillManager, createUserMemory } from '@/lib/agentMemory';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, userId, ...params } = body;
    
    switch (action) {
      case 'add_universal_memory': {
        const { type, content, importance } = params;
        if (!type || !content) {
          return NextResponse.json({ error: 'type and content are required' }, { status: 400 });
        }
        const id = await agentMemory.addUniversalMemory(type, content, importance);
        return NextResponse.json({ success: !!id, id });
      }
      
      case 'search_universal_memory': {
        const { query, limit = 10 } = params;
        if (!query) {
          return NextResponse.json({ error: 'query is required' }, { status: 400 });
        }
        const results = await agentMemory.searchUniversalMemory(query, limit);
        return NextResponse.json({ results });
      }
      
      case 'get_recent_memories': {
        const { limit = 20 } = params;
        const memories = await agentMemory.getRecentMemories(limit);
        return NextResponse.json({ memories });
      }
      
      case 'get_most_accessed_memories': {
        const { limit = 10 } = params;
        const memories = await agentMemory.getMostAccessedMemories(limit);
        return NextResponse.json({ memories });
      }
      
      case 'add_user_memory': {
        if (!userId) {
          return NextResponse.json({ error: 'userId is required' }, { status: 400 });
        }
        const { type, content, context, importance } = params;
        if (!type || !content) {
          return NextResponse.json({ error: 'type and content are required' }, { status: 400 });
        }
        const userMemory = createUserMemory(userId);
        const id = await userMemory.addMemory(type, content, context, importance);
        return NextResponse.json({ success: !!id, id });
      }
      
      case 'get_user_memories': {
        if (!userId) {
          return NextResponse.json({ error: 'userId is required' }, { status: 400 });
        }
        const { type, limit = 20 } = params;
        const userMemory = createUserMemory(userId);
        const memories = await userMemory.getMemories(type, limit);
        return NextResponse.json({ memories });
      }
      
      case 'search_user_memories': {
        if (!userId) {
          return NextResponse.json({ error: 'userId is required' }, { status: 400 });
        }
        const { query, limit = 10 } = params;
        if (!query) {
          return NextResponse.json({ error: 'query is required' }, { status: 400 });
        }
        const userMemory = createUserMemory(userId);
        const results = await userMemory.searchMemories(query, limit);
        return NextResponse.json({ results });
      }
      
      case 'set_preference': {
        if (!userId) {
          return NextResponse.json({ error: 'userId is required' }, { status: 400 });
        }
        const { key, value, learnedFrom } = params;
        if (!key) {
          return NextResponse.json({ error: 'key is required' }, { status: 400 });
        }
        const userMemory = createUserMemory(userId);
        await userMemory.setPreference(key, value, learnedFrom);
        return NextResponse.json({ success: true });
      }
      
      case 'get_preference': {
        if (!userId) {
          return NextResponse.json({ error: 'userId is required' }, { status: 400 });
        }
        const { key } = params;
        if (!key) {
          return NextResponse.json({ error: 'key is required' }, { status: 400 });
        }
        const userMemory = createUserMemory(userId);
        const value = await userMemory.getPreference(key);
        return NextResponse.json({ value });
      }
      
      case 'get_all_preferences': {
        if (!userId) {
          return NextResponse.json({ error: 'userId is required' }, { status: 400 });
        }
        const userMemory = createUserMemory(userId);
        const preferences = await userMemory.getAllPreferences();
        return NextResponse.json({ preferences });
      }
      
      case 'learn_skill': {
        const { name, category, description, knowledge } = params;
        if (!name || !category || !description) {
          return NextResponse.json({ error: 'name, category, and description are required' }, { status: 400 });
        }
        const id = await skillManager.learnSkill(name, category, description, knowledge || {});
        return NextResponse.json({ success: !!id, id });
      }
      
      case 'use_skill': {
        const { name, success } = params;
        if (!name) {
          return NextResponse.json({ error: 'name is required' }, { status: 400 });
        }
        await skillManager.useSkill(name, success ?? true);
        return NextResponse.json({ success: true });
      }
      
      case 'get_skill': {
        const { name } = params;
        if (!name) {
          return NextResponse.json({ error: 'name is required' }, { status: 400 });
        }
        const skill = await skillManager.getSkill(name);
        return NextResponse.json({ skill });
      }
      
      case 'get_skills_by_category': {
        const { category } = params;
        if (!category) {
          return NextResponse.json({ error: 'category is required' }, { status: 400 });
        }
        const skills = await skillManager.getSkillsByCategory(category);
        return NextResponse.json({ skills });
      }
      
      case 'get_most_used_skills': {
        const { limit = 10 } = params;
        const skills = await skillManager.getMostUsedSkills(limit);
        return NextResponse.json({ skills });
      }
      
      case 'get_all_skills': {
        const skills = await skillManager.getAllSkills();
        return NextResponse.json({ skills });
      }
      
      case 'search_skills': {
        const { query } = params;
        if (!query) {
          return NextResponse.json({ error: 'query is required' }, { status: 400 });
        }
        const skills = await skillManager.searchSkills(query);
        return NextResponse.json({ skills });
      }
      
      default:
        return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Memory API error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
