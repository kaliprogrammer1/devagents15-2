/**
 * Code Analysis API
 * Provides deep code understanding through AST parsing and semantic analysis
 */

import { NextRequest, NextResponse } from 'next/server';
import { codeAnalyzer } from '@/lib/codeAnalysis';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, files, functionName, pattern, type, filePath } = body;

    switch (action) {
      case 'analyze': {
        // Analyze code files
        if (!files || !Array.isArray(files)) {
          return NextResponse.json(
            { error: 'Files array is required' },
            { status: 400 }
          );
        }

        codeAnalyzer.clear();
        codeAnalyzer.addFiles(files);

        const results = [];
        for (const file of files) {
          const result = codeAnalyzer.analyzeFile(file.path);
          results.push({
            filePath: file.path,
            ...result,
          });
        }

        // Build dependency graph
        const graph = codeAnalyzer.buildDependencyGraph();
        const circularDeps = codeAnalyzer.findCircularDependencies();
        const summary = codeAnalyzer.getCodebaseSummary();

        return NextResponse.json({
          results,
          dependencyGraph: {
            nodes: Array.from(graph.nodes.keys()),
            edges: graph.edges.map(e => ({ from: e.from, to: e.to, type: e.type })),
            circularDependencies: circularDeps,
          },
          summary,
        });
      }

      case 'findCallers': {
        // Find all callers of a function
        if (!functionName) {
          return NextResponse.json(
            { error: 'functionName is required' },
            { status: 400 }
          );
        }

        const callers = codeAnalyzer.findCallers(functionName, filePath);
        return NextResponse.json({ callers });
      }

      case 'findCallees': {
        // Find all functions called by a function
        if (!functionName) {
          return NextResponse.json(
            { error: 'functionName is required' },
            { status: 400 }
          );
        }

        const callees = codeAnalyzer.findCallees(functionName);
        return NextResponse.json({ callees });
      }

      case 'searchEntities': {
        // Search for entities by pattern
        if (!pattern) {
          return NextResponse.json(
            { error: 'pattern is required' },
            { status: 400 }
          );
        }

        const entities = codeAnalyzer.searchEntities(pattern, type);
        return NextResponse.json({ entities });
      }

      case 'getEntity': {
        // Get a specific entity by name
        if (!functionName) {
          return NextResponse.json(
            { error: 'functionName is required' },
            { status: 400 }
          );
        }

        const entity = codeAnalyzer.getEntity(functionName);
        return NextResponse.json({ entity });
      }

      case 'getDependents': {
        // Get files that depend on a specific file
        if (!filePath) {
          return NextResponse.json(
            { error: 'filePath is required' },
            { status: 400 }
          );
        }

        const dependents = codeAnalyzer.getDependents(filePath);
        return NextResponse.json({ dependents });
      }

      case 'getDependencies': {
        // Get files that a specific file depends on
        if (!filePath) {
          return NextResponse.json(
            { error: 'filePath is required' },
            { status: 400 }
          );
        }

        const dependencies = codeAnalyzer.getDependencies(filePath);
        return NextResponse.json({ dependencies });
      }

      case 'findCircularDependencies': {
        const circularDeps = codeAnalyzer.findCircularDependencies();
        return NextResponse.json({ circularDependencies: circularDeps });
      }

      case 'getSummary': {
        const summary = codeAnalyzer.getCodebaseSummary();
        return NextResponse.json({ summary });
      }

      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Code analysis error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    service: 'Code Analysis API',
    version: '1.0.0',
    actions: [
      'analyze - Analyze code files and extract entities, complexity, and issues',
      'findCallers - Find all callers of a specific function',
      'findCallees - Find all functions called by a specific function',
      'searchEntities - Search for entities by pattern',
      'getEntity - Get a specific entity by name',
      'getDependents - Get files that depend on a specific file',
      'getDependencies - Get files that a specific file depends on',
      'findCircularDependencies - Detect circular dependencies in the codebase',
      'getSummary - Get a summary of the analyzed codebase',
    ],
    capabilities: [
      'AST parsing with ts-morph',
      'Function/class/variable extraction',
      'Dependency graph building',
      'Cyclomatic complexity calculation',
      'Code issue detection',
      'Circular dependency detection',
    ],
  });
}
