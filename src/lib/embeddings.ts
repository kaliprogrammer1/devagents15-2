/**
 * Real Semantic Embeddings Service
 * Primary: OpenAI text-embedding-3-small (1536 dimensions)
 * Fallback: Hugging Face Inference API (free, no API key required)
 */

import OpenAI from 'openai';

// Initialize OpenAI client only if API key exists
const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

export const EMBEDDING_MODEL = 'text-embedding-3-small';
export const EMBEDDING_DIMENSIONS = 384; // Using 384 for HuggingFace all-MiniLM-L6-v2
export const OPENAI_EMBEDDING_DIMENSIONS = 1536;

// Hugging Face free inference API endpoint
const HF_INFERENCE_URL = 'https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2';

export interface EmbeddingResult {
  embedding: number[];
  tokens: number;
  source: 'openai' | 'huggingface' | 'local';
}

export interface BatchEmbeddingResult {
  embeddings: number[][];
  totalTokens: number;
  source: 'openai' | 'huggingface' | 'local';
}

/**
 * Generate embedding using Hugging Face free inference API
 * Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
 */
async function generateHuggingFaceEmbedding(text: string): Promise<number[]> {
  const response = await fetch(HF_INFERENCE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      // HF_TOKEN is optional for public models
      ...(process.env.HF_TOKEN && { Authorization: `Bearer ${process.env.HF_TOKEN}` }),
    },
    body: JSON.stringify({
      inputs: text,
      options: { wait_for_model: true },
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HuggingFace API error: ${response.status} - ${error}`);
  }

  const result = await response.json();
  
  // The API returns the embedding directly as an array
  if (Array.isArray(result) && typeof result[0] === 'number') {
    return result;
  }
  
  // Sometimes it returns nested array
  if (Array.isArray(result) && Array.isArray(result[0])) {
    return result[0];
  }
  
  throw new Error('Unexpected HuggingFace response format');
}

/**
 * Simple local embedding fallback using TF-IDF-like approach
 * Used when both OpenAI and HuggingFace are unavailable
 */
function generateLocalEmbedding(text: string, dimensions: number = 384): number[] {
  const words = text.toLowerCase().split(/\s+/);
  const embedding = new Array(dimensions).fill(0);
  
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    // Simple hash-based embedding
    for (let j = 0; j < word.length; j++) {
      const charCode = word.charCodeAt(j);
      const idx = (charCode * (j + 1) * (i + 1)) % dimensions;
      embedding[idx] += 1 / (words.length * Math.sqrt(word.length));
    }
  }
  
  // Normalize to unit vector
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  if (magnitude > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= magnitude;
    }
  }
  
  return embedding;
}

/**
 * Generate embedding for a single text
 * Falls back through: OpenAI -> HuggingFace -> Local
 */
export async function generateEmbedding(text: string): Promise<EmbeddingResult> {
  if (!text || text.trim().length === 0) {
    throw new Error('Text cannot be empty');
  }

  // Truncate to reasonable limit
  const truncatedText = text.slice(0, 8000);

  // Try OpenAI first if available
  if (openai && process.env.OPENAI_API_KEY) {
    try {
      const response = await openai.embeddings.create({
        model: EMBEDDING_MODEL,
        input: truncatedText,
        encoding_format: 'float',
      });

      return {
        embedding: response.data[0].embedding,
        tokens: response.usage.total_tokens,
        source: 'openai',
      };
    } catch (error) {
      console.warn('OpenAI embedding failed, falling back to HuggingFace:', error);
    }
  }

  // Try HuggingFace free inference
  try {
    const embedding = await generateHuggingFaceEmbedding(truncatedText);
    return {
      embedding,
      tokens: truncatedText.split(/\s+/).length, // Approximate token count
      source: 'huggingface',
    };
  } catch (error) {
    console.warn('HuggingFace embedding failed, using local fallback:', error);
  }

  // Final fallback to local embedding
  const embedding = generateLocalEmbedding(truncatedText);
  return {
    embedding,
    tokens: truncatedText.split(/\s+/).length,
    source: 'local',
  };
}

/**
 * Generate embeddings for multiple texts (batch)
 * Falls back through: OpenAI -> HuggingFace -> Local
 */
export async function generateBatchEmbeddings(
  texts: string[],
  batchSize: number = 100
): Promise<BatchEmbeddingResult> {
  if (!texts || texts.length === 0) {
    return { embeddings: [], totalTokens: 0, source: 'local' };
  }

  const allEmbeddings: number[][] = [];
  let totalTokens = 0;
  let source: 'openai' | 'huggingface' | 'local' = 'local';

  // Try OpenAI batch processing first
  if (openai && process.env.OPENAI_API_KEY) {
    try {
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        const truncatedBatch = batch.map(t => t.slice(0, 8000));

        const response = await openai.embeddings.create({
          model: EMBEDDING_MODEL,
          input: truncatedBatch,
          encoding_format: 'float',
        });

        for (const item of response.data) {
          allEmbeddings.push(item.embedding);
        }
        totalTokens += response.usage.total_tokens;
      }
      return { embeddings: allEmbeddings, totalTokens, source: 'openai' };
    } catch (error) {
      console.warn('OpenAI batch embedding failed, falling back:', error);
      allEmbeddings.length = 0;
      totalTokens = 0;
    }
  }

  // Fallback: process individually through the cascade
  for (const text of texts) {
    try {
      const result = await generateEmbedding(text);
      allEmbeddings.push(result.embedding);
      totalTokens += result.tokens;
      source = result.source;
    } catch (error) {
      // Use local embedding as last resort
      allEmbeddings.push(generateLocalEmbedding(text.slice(0, 8000)));
      totalTokens += text.split(/\s+/).length;
      source = 'local';
    }
  }

  return { embeddings: allEmbeddings, totalTokens, source };
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
