/**
 * Embeddings API Route
 * Server-side endpoint for generating embeddings (keeps API key secure)
 * Supports OpenAI, HuggingFace (free), and local fallback
 */

import { NextRequest, NextResponse } from 'next/server';
import { generateEmbedding, generateBatchEmbeddings, cosineSimilarity, EMBEDDING_DIMENSIONS } from '@/lib/embeddings';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, text, texts, embedding1, embedding2 } = body;

    switch (action) {
      case 'single': {
        if (!text || typeof text !== 'string') {
          return NextResponse.json(
            { error: 'Text is required for single embedding' },
            { status: 400 }
          );
        }
        const result = await generateEmbedding(text);
        return NextResponse.json({
          ...result,
          dimensions: result.embedding.length,
        });
      }

      case 'batch': {
        if (!texts || !Array.isArray(texts)) {
          return NextResponse.json(
            { error: 'Texts array is required for batch embedding' },
            { status: 400 }
          );
        }
        const result = await generateBatchEmbeddings(texts);
        return NextResponse.json({
          ...result,
          dimensions: result.embeddings[0]?.length || EMBEDDING_DIMENSIONS,
        });
      }

      case 'similarity': {
        if (!embedding1 || !embedding2 || !Array.isArray(embedding1) || !Array.isArray(embedding2)) {
          return NextResponse.json(
            { error: 'Two embeddings arrays required for similarity' },
            { status: 400 }
          );
        }
        const similarity = cosineSimilarity(embedding1, embedding2);
        return NextResponse.json({ similarity });
      }

      case 'test': {
        // Test the embedding pipeline
        const testText = text || 'This is a test sentence for embedding generation.';
        const result = await generateEmbedding(testText);
        return NextResponse.json({
          success: true,
          source: result.source,
          dimensions: result.embedding.length,
          tokens: result.tokens,
          sample: result.embedding.slice(0, 5), // First 5 values as sample
        });
      }

      default:
        return NextResponse.json(
          { error: 'Unknown action. Use "single", "batch", "similarity", or "test"' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Embeddings API error:', error);
    
    if (error instanceof Error) {
      if (error.message.includes('API key')) {
        return NextResponse.json(
          { error: 'OpenAI API key not configured, using fallback' },
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
