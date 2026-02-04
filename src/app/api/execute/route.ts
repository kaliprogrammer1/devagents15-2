import { NextRequest, NextResponse } from 'next/server';
import { CodeExecutor, TestRunner } from '@/lib/codeExecution';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, userId = 'default', ...params } = body;
    
    switch (action) {
      case 'execute': {
        const { language, code, stdin, args, timeout } = params;
        if (!language || !code) {
          return NextResponse.json({ error: 'language and code are required' }, { status: 400 });
        }
        const executor = new CodeExecutor(userId);
        const result = await executor.execute(language, code, { stdin, args, timeout });
        return NextResponse.json(result);
      }
      
      case 'run_tests': {
        const { language, code, testCode, framework } = params;
        if (!language || !code || !testCode) {
          return NextResponse.json({ error: 'language, code, and testCode are required' }, { status: 400 });
        }
        const runner = new TestRunner(userId);
        const result = await runner.runTests(language, code, testCode, framework);
        return NextResponse.json(result);
      }
      
      case 'runtimes': {
        const executor = new CodeExecutor(userId);
        const runtimes = await executor.getRuntimes();
        return NextResponse.json({ runtimes });
      }
      
      case 'languages': {
        const executor = new CodeExecutor(userId);
        const languages = await executor.getAvailableLanguages();
        return NextResponse.json({ languages });
      }
      
      case 'history': {
        const { limit = 20 } = params;
        const executor = new CodeExecutor(userId);
        const history = await executor.getExecutionHistory(limit);
        return NextResponse.json({ history });
      }
      
      case 'test_history': {
        const { limit = 20 } = params;
        const runner = new TestRunner(userId);
        const history = await runner.getTestHistory(limit);
        return NextResponse.json({ history });
      }
      
      default:
        return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Code execution API error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
