# Real Semantic Embeddings Implementation Plan

## Requirements
Replace the current fake hash-based embedding system with real semantic embeddings using OpenAI's text-embedding-3-small model and Supabase pgvector for proper vector similarity search.

---

## Current Problem Analysis

### The Fake Embedding Code (src/lib/agentMemory.ts:34-53)
```typescript
private async generateEmbedding(text: string): Promise<number[]> {
  const dims = 384;
  const embedding = new Array(dims).fill(0);
  const words = text.toLowerCase().split(/\W+/);
  
  for (let i = 0; i < words.length; i++) {
    let hash = 0;
    for (let j = 0; j < words[i].length; j++) {
      hash = (hash << 5) - hash + words[i].charCodeAt(j);
      hash |= 0;
    }
    const idx = Math.abs(hash) % dims;
    embedding[idx] += 1;
  }
  
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0)) || 1;
  return embedding.map(val => val / magnitude);
}
```

### Why This Is Broken
1. **No Semantic Understanding**: "I love programming" and "Coding is my passion" produce completely different vectors
2. **Hash Collisions**: Different words can map to the same dimension, destroying information
3. **384 Dimensions**: Arbitrary choice, not optimized for any embedding space
4. **No Context**: Word order is ignored, losing all syntactic meaning

### Impact on Agent
- Memory search returns irrelevant results
- Similar tasks are not recognized as related
- Knowledge graph relationships are meaningless
- Learning from experience is effectively broken

---

## Implementation Architecture

### High-Level Flow
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Agent Brain    │────▶│  Embeddings API  │────▶│  OpenAI API     │
│  agentMemory.ts │     │  /api/embeddings │     │  text-embed-3   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  Supabase       │◀────│  Vector Storage  │
│  pgvector RPC   │     │  1536 dimensions │
└─────────────────┘     └──────────────────┘
```

### Key Design Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding Model | text-embedding-3-small | Cost-effective ($0.02/1M tokens), 1536 dims, good performance |
| Vector Dimensions | 1536 (full) | Maximum accuracy; can reduce to 512 later if cost is concern |
| Index Type | HNSW | Better recall and QPS than IVFFlat, works on empty tables |
| Similarity Metric | Cosine | Standard for text embeddings, normalized vectors |
| Batch Size | 100 texts | Balance between API efficiency and memory |
| Cache Strategy | None initially | Add Redis/memory cache in Phase 2 if needed |

---

## Database Schema Changes

### New SQL Migrations

#### 1. Enable pgvector Extension
```sql
-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

#### 2. Modify agent_universal_memory Table
```sql
-- Add new embedding column (1536 dimensions for text-embedding-3-small)
ALTER TABLE agent_universal_memory 
ADD COLUMN IF NOT EXISTS embedding_1536 VECTOR(1536);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_universal_memory_embedding_hnsw 
ON agent_universal_memory 
USING hnsw (embedding_1536 vector_cosine_ops);

-- Drop old embedding column after migration (optional, keep for rollback)
-- ALTER TABLE agent_universal_memory DROP COLUMN IF EXISTS embedding;
```

#### 3. Modify user_memory Table
```sql
-- Add new embedding column
ALTER TABLE user_memory 
ADD COLUMN IF NOT EXISTS embedding_1536 VECTOR(1536);

-- Create HNSW index
CREATE INDEX IF NOT EXISTS idx_user_memory_embedding_hnsw 
ON user_memory 
USING hnsw (embedding_1536 vector_cosine_ops);
```

#### 4. Modify knowledge_nodes Table
```sql
-- Add new embedding column
ALTER TABLE knowledge_nodes 
ADD COLUMN IF NOT EXISTS embedding_1536 VECTOR(1536);

-- Create HNSW index
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding_hnsw 
ON knowledge_nodes 
USING hnsw (embedding_1536 vector_cosine_ops);
```

#### 5. Create New RPC Functions
```sql
-- Universal Memory Similarity Search
CREATE OR REPLACE FUNCTION match_universal_memories_v2(
  query_embedding VECTOR(1536),
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  memory_type TEXT,
  content_compressed TEXT,
  importance_score FLOAT,
  similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    m.id,
    m.memory_type,
    m.content_compressed,
    m.importance_score,
    1 - (m.embedding_1536 <=> query_embedding) AS similarity
  FROM agent_universal_memory m
  WHERE m.embedding_1536 IS NOT NULL
    AND 1 - (m.embedding_1536 <=> query_embedding) > match_threshold
  ORDER BY m.embedding_1536 <=> query_embedding
  LIMIT match_count;
$$;

-- User Memory Similarity Search
CREATE OR REPLACE FUNCTION match_user_memories_v2(
  query_embedding VECTOR(1536),
  p_user_id UUID,
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  memory_type TEXT,
  content_compressed TEXT,
  context JSONB,
  importance_score FLOAT,
  similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    m.id,
    m.memory_type,
    m.content_compressed,
    m.context,
    m.importance_score,
    1 - (m.embedding_1536 <=> query_embedding) AS similarity
  FROM user_memory m
  WHERE m.user_id = p_user_id
    AND m.embedding_1536 IS NOT NULL
    AND 1 - (m.embedding_1536 <=> query_embedding) > match_threshold
  ORDER BY m.embedding_1536 <=> query_embedding
  LIMIT match_count;
$$;

-- Knowledge Graph Node Similarity Search
CREATE OR REPLACE FUNCTION match_knowledge_nodes(
  query_embedding VECTOR(1536),
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  name TEXT,
  node_type TEXT,
  content_compressed TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    n.id,
    n.name,
    n.node_type,
    n.content_compressed,
    n.metadata,
    1 - (n.embedding_1536 <=> query_embedding) AS similarity
  FROM knowledge_nodes n
  WHERE n.embedding_1536 IS NOT NULL
    AND 1 - (n.embedding_1536 <=> query_embedding) > match_threshold
  ORDER BY n.embedding_1536 <=> query_embedding
  LIMIT match_count;
$$;
```

---

## Code Changes

### 1. New File: src/lib/embeddings.ts
```typescript
/**
 * Real Semantic Embeddings Service
 * Uses OpenAI text-embedding-3-small (1536 dimensions)
 */

import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

export const EMBEDDING_MODEL = 'text-embedding-3-small';
export const EMBEDDING_DIMENSIONS = 1536;

export interface EmbeddingResult {
  embedding: number[];
  tokens: number;
}

export interface BatchEmbeddingResult {
  embeddings: number[][];
  totalTokens: number;
}

/**
 * Generate embedding for a single text
 */
export async function generateEmbedding(text: string): Promise<EmbeddingResult> {
  if (!text || text.trim().length === 0) {
    throw new Error('Text cannot be empty');
  }

  // Truncate to max token limit (8191 tokens ≈ 32000 chars as rough estimate)
  const truncatedText = text.slice(0, 30000);

  const response = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: truncatedText,
    encoding_format: 'float',
  });

  return {
    embedding: response.data[0].embedding,
    tokens: response.usage.total_tokens,
  };
}

/**
 * Generate embeddings for multiple texts (batch)
 * More efficient for bulk operations
 */
export async function generateBatchEmbeddings(
  texts: string[],
  batchSize: number = 100
): Promise<BatchEmbeddingResult> {
  if (!texts || texts.length === 0) {
    return { embeddings: [], totalTokens: 0 };
  }

  const allEmbeddings: number[][] = [];
  let totalTokens = 0;

  // Process in batches to avoid API limits
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const truncatedBatch = batch.map(t => t.slice(0, 30000));

    const response = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: truncatedBatch,
      encoding_format: 'float',
    });

    // Embeddings come back in same order as input
    for (const item of response.data) {
      allEmbeddings.push(item.embedding);
    }
    totalTokens += response.usage.total_tokens;
  }

  return { embeddings: allEmbeddings, totalTokens };
}

/**
 * Calculate cosine similarity between two embeddings
 * Returns value between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite)
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Embeddings must have same dimensions');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}
```

### 2. New File: src/app/api/embeddings/route.ts
```typescript
/**
 * Embeddings API Route
 * Server-side endpoint for generating embeddings (keeps API key secure)
 */

import { NextRequest, NextResponse } from 'next/server';
import { generateEmbedding, generateBatchEmbeddings } from '@/lib/embeddings';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, text, texts } = body;

    switch (action) {
      case 'single': {
        if (!text || typeof text !== 'string') {
          return NextResponse.json(
            { error: 'Text is required for single embedding' },
            { status: 400 }
          );
        }
        const result = await generateEmbedding(text);
        return NextResponse.json(result);
      }

      case 'batch': {
        if (!texts || !Array.isArray(texts)) {
          return NextResponse.json(
            { error: 'Texts array is required for batch embedding' },
            { status: 400 }
          );
        }
        const result = await generateBatchEmbeddings(texts);
        return NextResponse.json(result);
      }

      default:
        return NextResponse.json(
          { error: 'Unknown action. Use "single" or "batch"' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Embeddings API error:', error);
    
    if (error instanceof Error) {
      // Handle OpenAI specific errors
      if (error.message.includes('API key')) {
        return NextResponse.json(
          { error: 'OpenAI API key not configured' },
          { status: 500 }
        );
      }
      if (error.message.includes('rate limit')) {
        return NextResponse.json(
          { error: 'Rate limit exceeded, please retry' },
          { status: 429 }
        );
      }
    }

    return NextResponse.json(
      { error: 'Failed to generate embedding' },
      { status: 500 }
    );
  }
}
```

### 3. Modified: src/lib/agentMemory.ts

#### Changes to AgentMemory class:
```typescript
// REPLACE the fake generateEmbedding method with:

import { generateEmbedding, generateBatchEmbeddings, EMBEDDING_DIMENSIONS } from './embeddings';

export class AgentMemory {
  /**
   * Generate real semantic embedding using OpenAI
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const result = await generateEmbedding(text);
      return result.embedding;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      // Return zero vector as fallback (will have low similarity to everything)
      return new Array(EMBEDDING_DIMENSIONS).fill(0);
    }
  }

  // ... rest of class unchanged, but update column names from 'embedding' to 'embedding_1536'
}
```

#### Update addUniversalMemory method:
```typescript
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
        embedding_1536: embedding,  // Changed from 'embedding'
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
```

#### Update searchUniversalMemory method:
```typescript
async searchUniversalMemory(
  query: string, 
  limit: number = 10
): Promise<Array<{ id: string; type: string; content: string; importance: number; similarity?: number }>> {
  try {
    const embedding = await this.generateEmbedding(query);
    
    // Use new v2 RPC function
    const { data, error } = await supabaseAdmin.rpc('match_universal_memories_v2', {
      query_embedding: embedding,
      match_threshold: 0.7,  // Higher threshold for real embeddings
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
    
    // Update access counts
    for (const result of results) {
      await supabaseAdmin
        .from('agent_universal_memory')
        .update({ 
          access_count: supabaseAdmin.raw('access_count + 1'),
          last_accessed: new Date().toISOString()
        })
        .eq('id', result.id);
    }
    
    return results;
  } catch (error) {
    console.error('Error searching universal memory:', error);
    return this.fallbackSearchUniversal(query, limit);
  }
}
```

#### Update UserMemoryManager class similarly:
```typescript
// In UserMemoryManager class

private async generateEmbedding(text: string): Promise<number[]> {
  try {
    const result = await generateEmbedding(text);
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
        embedding_1536: embedding,  // Changed from 'embedding'
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

async searchMemories(
  query: string, 
  limit: number = 10
): Promise<Array<{ id: string; type: string; content: string; similarity?: number }>> {
  try {
    const embedding = await this.generateEmbedding(query);
    
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
```

#### Update KnowledgeGraphManager class:
```typescript
// In KnowledgeGraphManager class

async addNode(
  name: string, 
  type: string, 
  content: string, 
  metadata: Record<string, unknown> = {}
): Promise<string | null> {
  try {
    const embedding = await generateEmbedding(content);
    
    const { data, error } = await supabaseAdmin
      .from('knowledge_nodes')
      .insert({
        name,
        node_type: type,
        content_compressed: compressText(content),
        metadata,
        embedding_1536: embedding.embedding,  // Add embedding
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

async searchGraph(query: string): Promise<any[]> {
  try {
    const embedding = await generateEmbedding(query);
    
    const { data, error } = await supabaseAdmin.rpc('match_knowledge_nodes', {
      query_embedding: embedding.embedding,
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
      .or(`name.ilike.%${query}%`)
      .limit(10);
    return data || [];
  }
}
```

---

## Environment Variables

### Required New Variables
```env
# .env.local
OPENAI_API_KEY=sk-proj-...your-openai-api-key...
```

### Verification
- OpenAI API key must have access to embeddings endpoint
- Recommended: Set usage limits in OpenAI dashboard to prevent runaway costs

---

## Migration Strategy

### Phase 1: Database Setup
1. Run SQL migrations to add new columns and RPC functions
2. Keep old `embedding` column for rollback capability

### Phase 2: Code Deployment
1. Deploy new `embeddings.ts` library
2. Deploy `/api/embeddings` route
3. Update `agentMemory.ts` to use new embedding system

### Phase 3: Data Migration
1. Create migration script to re-embed existing memories
2. Run in batches to avoid API rate limits
3. Estimate: ~100 memories = $0.002 (very cheap)

### Phase 4: Cleanup (After Validation)
1. Monitor similarity search quality for 1 week
2. Drop old `embedding` column
3. Remove fallback hash-based code

---

## Testing Plan

### Unit Tests
```typescript
// tests/embeddings.test.ts
import { generateEmbedding, cosineSimilarity } from '@/lib/embeddings';

describe('Semantic Embeddings', () => {
  it('should generate 1536-dimension embeddings', async () => {
    const result = await generateEmbedding('Hello world');
    expect(result.embedding.length).toBe(1536);
  });

  it('should find similar texts have high similarity', async () => {
    const e1 = await generateEmbedding('I love programming');
    const e2 = await generateEmbedding('Coding is my passion');
    const e3 = await generateEmbedding('The weather is nice today');
    
    const sim12 = cosineSimilarity(e1.embedding, e2.embedding);
    const sim13 = cosineSimilarity(e1.embedding, e3.embedding);
    
    expect(sim12).toBeGreaterThan(0.8);  // Similar texts
    expect(sim13).toBeLessThan(0.5);     // Unrelated texts
  });
});
```

### Integration Tests
1. Add memory with new embedding
2. Search for semantically similar query
3. Verify results include the added memory
4. Verify similarity scores are reasonable (>0.7 for relevant)

---

## Cost Analysis

### OpenAI Pricing (text-embedding-3-small)
- $0.02 per 1 million tokens
- Average memory: ~100 tokens
- 10,000 memories = $0.02

### Expected Usage
| Operation | Frequency | Est. Tokens/Day | Daily Cost |
|-----------|-----------|-----------------|------------|
| Memory Add | 50/day | 5,000 | $0.0001 |
| Memory Search | 200/day | 20,000 | $0.0004 |
| Knowledge Graph | 100/day | 10,000 | $0.0002 |
| **Total** | | 35,000 | **$0.0007/day** |

Annual cost estimate: ~$0.26/year (negligible)

---

## Rollback Plan

1. Keep old `embedding` (384-dim) column for 30 days
2. Feature flag to switch between old/new embedding system
3. If issues: revert code, use fallback keyword search
4. Database rollback: just ignore `embedding_1536` column

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/lib/embeddings.ts` | CREATE | OpenAI embeddings service |
| `src/app/api/embeddings/route.ts` | CREATE | Server-side embeddings API |
| `src/lib/agentMemory.ts` | MODIFY | Replace fake embeddings with real ones |
| `src/lib/supabase.ts` | MODIFY | Add embedding_1536 to type definitions |
| `.env.local` | MODIFY | Add OPENAI_API_KEY |
| `package.json` | MODIFY | Add `openai` dependency |

---

## Success Criteria

1. **Semantic Search Works**: "I love coding" finds "Programming is fun"
2. **Performance**: Embedding generation < 500ms
3. **Accuracy**: Similarity scores > 0.8 for related content
4. **Cost**: < $1/month for typical usage
5. **Reliability**: < 1% error rate on embedding generation

---

## Critical Files for Implementation
- `src/lib/agentMemory.ts` - Core file with fake embedding implementation to replace
- `src/lib/supabase.ts` - Type definitions need embedding_1536 field
- `src/lib/agentBrain.ts` - Uses memory search, will benefit from real embeddings
- `src/app/api/agent/route.ts` - API endpoint that triggers memory operations
- `package.json` - Need to add openai dependency
