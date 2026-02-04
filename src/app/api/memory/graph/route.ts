import { NextResponse } from 'next/server';
import { knowledgeGraph } from '@/lib/agentMemory';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('query');
  const nodeId = searchParams.get('nodeId');

  try {
    if (nodeId) {
      const related = await knowledgeGraph.getRelatedNodes(nodeId);
      return NextResponse.json({ related });
    }

    if (query) {
      const nodes = await knowledgeGraph.searchGraph(query);
      return NextResponse.json({ nodes });
    }

    return NextResponse.json({ error: 'Missing query or nodeId' }, { status: 400 });
  } catch (error) {
    console.error('Graph API error:', error);
    return NextResponse.json({ error: 'Failed to fetch graph data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const { action, name, type, content, metadata, sourceId, targetId, relationType } = await request.json();

    if (action === 'addNode') {
      const id = await knowledgeGraph.addNode(name, type, content, metadata);
      return NextResponse.json({ id });
    }

    if (action === 'addEdge') {
      const id = await knowledgeGraph.addEdge(sourceId, targetId, relationType);
      return NextResponse.json({ id });
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
  } catch (error) {
    console.error('Graph API error:', error);
    return NextResponse.json({ error: 'Failed to update graph' }, { status: 500 });
  }
}
