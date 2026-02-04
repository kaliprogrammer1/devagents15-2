import { supabaseAdmin, type CodeExecution, type TestResult } from './supabase';

const PISTON_API = 'https://emkc.org/api/v2/piston';

export interface Runtime {
  language: string;
  version: string;
  aliases: string[];
}

export interface ExecuteBody {
  language: string;
  version: string;
  files: Array<{ name?: string; content: string }>;
  stdin?: string;
  args?: string[];
  compile_timeout?: number;
  run_timeout?: number;
  compile_memory_limit?: number;
  run_memory_limit?: number;
}

export interface StageResult {
  stdout: string;
  stderr: string;
  output: string;
  code: number | null;
  signal: string | null;
  message?: string;
  cpu_time?: number;
  wall_time?: number;
  memory?: number;
}

export interface ExecuteResponse {
  language: string;
  version: string;
  run: StageResult;
  compile?: StageResult;
}

const LANGUAGE_ALIASES: Record<string, string> = {
  'js': 'javascript',
  'node': 'javascript',
  'nodejs': 'javascript',
  'py': 'python',
  'python3': 'python',
  'ts': 'typescript',
  'rb': 'ruby',
  'rs': 'rust',
  'go': 'go',
  'java': 'java',
  'c': 'c',
  'cpp': 'c++',
  'c++': 'c++',
  'csharp': 'csharp',
  'cs': 'csharp',
  'php': 'php',
  'swift': 'swift',
  'kotlin': 'kotlin',
  'r': 'r',
  'bash': 'bash',
  'sh': 'bash',
  'sql': 'sqlite3',
};

export class CodeExecutor {
  private userId: string;
  private runtimes: Runtime[] = [];
  
  constructor(userId: string) {
    this.userId = userId;
  }
  
  async getRuntimes(): Promise<Runtime[]> {
    if (this.runtimes.length > 0) return this.runtimes;
    
    try {
      const res = await fetch(`${PISTON_API}/runtimes`);
      if (!res.ok) throw new Error('Failed to fetch runtimes');
      this.runtimes = await res.json();
      return this.runtimes;
    } catch (error) {
      console.error('Error fetching runtimes:', error);
      return [];
    }
  }
  
  async getAvailableLanguages(): Promise<string[]> {
    const runtimes = await this.getRuntimes();
    return [...new Set(runtimes.map(r => r.language))].sort();
  }
  
  async getLanguageVersion(language: string): Promise<string | null> {
    const normalizedLang = LANGUAGE_ALIASES[language.toLowerCase()] || language.toLowerCase();
    const runtimes = await this.getRuntimes();
    const runtime = runtimes.find(r => 
      r.language === normalizedLang || 
      r.aliases.includes(normalizedLang) ||
      r.aliases.includes(language.toLowerCase())
    );
    return runtime?.version || null;
  }
  
  async execute(
    language: string,
    code: string,
    options: {
      stdin?: string;
      args?: string[];
      timeout?: number;
      memoryLimit?: number;
      fileName?: string;
    } = {}
  ): Promise<{ success: boolean; output: string; error?: string; executionTime?: number; execution?: CodeExecution }> {
    const startTime = Date.now();
    const normalizedLang = LANGUAGE_ALIASES[language.toLowerCase()] || language.toLowerCase();
    
    try {
      const version = await this.getLanguageVersion(normalizedLang);
      if (!version) {
        return { success: false, output: '', error: `Unsupported language: ${language}` };
      }
      
      const body: ExecuteBody = {
        language: normalizedLang,
        version: '*',
        files: [{ 
          name: options.fileName || this.getDefaultFileName(normalizedLang),
          content: code 
        }],
        stdin: options.stdin,
        args: options.args,
        run_timeout: options.timeout || 10000,
        run_memory_limit: options.memoryLimit || -1,
      };
      
      const res = await fetch(`${PISTON_API}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Execution failed: ${errorText}`);
      }
      
      const result: ExecuteResponse = await res.json();
      const executionTime = Date.now() - startTime;
      
      const success = result.run.code === 0 && !result.run.stderr;
      const output = result.run.output || result.run.stdout;
      const error = result.run.stderr || result.compile?.stderr;
      
      const execution: CodeExecution = {
        user_id: this.userId,
        language: normalizedLang,
        code,
        output: output.substring(0, 10000),
        exit_code: result.run.code || 0,
        execution_time_ms: executionTime,
        success,
      };
      
      await this.saveExecution(execution);
      
      return {
        success,
        output: output.substring(0, 10000),
        error: error ? error.substring(0, 5000) : undefined,
        executionTime,
        execution,
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      const execution: CodeExecution = {
        user_id: this.userId,
        language: normalizedLang,
        code,
        output: '',
        exit_code: 1,
        execution_time_ms: executionTime,
        success: false,
      };
      
      await this.saveExecution(execution);
      
      return { success: false, output: '', error: errorMessage, executionTime, execution };
    }
  }
  
  private getDefaultFileName(language: string): string {
    const extensions: Record<string, string> = {
      'javascript': 'main.js',
      'typescript': 'main.ts',
      'python': 'main.py',
      'ruby': 'main.rb',
      'rust': 'main.rs',
      'go': 'main.go',
      'java': 'Main.java',
      'c': 'main.c',
      'c++': 'main.cpp',
      'csharp': 'Main.cs',
      'php': 'main.php',
      'swift': 'main.swift',
      'kotlin': 'Main.kt',
      'r': 'main.r',
      'bash': 'main.sh',
      'sqlite3': 'main.sql',
    };
    return extensions[language] || 'main.txt';
  }
  
  private async saveExecution(execution: CodeExecution): Promise<string | null> {
    try {
      const { data, error } = await supabaseAdmin
        .from('code_executions')
        .insert(execution)
        .select('id')
        .single();
      
      if (error) {
        console.error('Error saving execution:', error);
        return null;
      }
      return data?.id || null;
    } catch (error) {
      console.error('Error saving execution:', error);
      return null;
    }
  }
  
  async getExecutionHistory(limit: number = 20): Promise<CodeExecution[]> {
    const { data } = await supabaseAdmin
      .from('code_executions')
      .select('*')
      .eq('user_id', this.userId)
      .order('created_at', { ascending: false })
      .limit(limit);
    
    return data || [];
  }
}

export class TestRunner {
  private executor: CodeExecutor;
  private userId: string;
  
  constructor(userId: string) {
    this.userId = userId;
    this.executor = new CodeExecutor(userId);
  }
  
  async runTests(
    language: string,
    code: string,
    testCode: string,
    framework?: string
  ): Promise<TestResult> {
    const combinedCode = this.combineCodeWithTests(language, code, testCode, framework);
    const result = await this.executor.execute(language, combinedCode, { timeout: 30000 });
    
    const parsed = this.parseTestOutput(result.output, framework || this.detectFramework(language));
    
    const testResult: TestResult = {
      user_id: this.userId,
      execution_id: result.execution?.id,
      test_framework: framework || this.detectFramework(language),
      total_tests: parsed.total,
      passed: parsed.passed,
      failed: parsed.failed,
      skipped: parsed.skipped,
      coverage_percent: parsed.coverage,
      output: result.output,
    };
    
    await this.saveTestResult(testResult);
    return testResult;
  }
  
  private combineCodeWithTests(language: string, code: string, testCode: string, framework?: string): string {
    switch (language.toLowerCase()) {
      case 'javascript':
      case 'typescript':
        return `${code}\n\n// Tests\n${testCode}\n\n// Run tests\nif (typeof describe === 'undefined') {\n  const tests = [];\n  global.describe = (name, fn) => { console.log('Suite:', name); fn(); };\n  global.it = global.test = (name, fn) => { try { fn(); console.log('✓', name); tests.push({pass:true}); } catch(e) { console.log('✗', name, e.message); tests.push({pass:false}); } };\n  global.expect = (v) => ({ toBe: (e) => { if(v!==e) throw new Error(\`Expected \${e} but got \${v}\`); }, toEqual: (e) => { if(JSON.stringify(v)!==JSON.stringify(e)) throw new Error('Not equal'); }, toBeTruthy: () => { if(!v) throw new Error('Not truthy'); } });\n}\n`;
      case 'python':
        return `${code}\n\n# Tests\nimport unittest\nimport sys\n\n${testCode}\n\nif __name__ == '__main__':\n    unittest.main(verbosity=2)\n`;
      default:
        return `${code}\n\n${testCode}`;
    }
  }
  
  private detectFramework(language: string): string {
    const frameworks: Record<string, string> = {
      'javascript': 'jest',
      'typescript': 'jest',
      'python': 'unittest',
      'ruby': 'rspec',
      'go': 'testing',
      'rust': 'cargo-test',
      'java': 'junit',
    };
    return frameworks[language.toLowerCase()] || 'custom';
  }
  
  private parseTestOutput(output: string, framework: string): { total: number; passed: number; failed: number; skipped: number; coverage?: number } {
    const lines = output.split('\n');
    let passed = 0, failed = 0, skipped = 0;
    
    for (const line of lines) {
      if (line.includes('✓') || line.includes('PASS') || line.includes('ok') || line.includes('passed')) {
        passed++;
      }
      if (line.includes('✗') || line.includes('FAIL') || line.includes('failed') || line.includes('error')) {
        failed++;
      }
      if (line.includes('skip') || line.includes('pending')) {
        skipped++;
      }
    }
    
    const coverageMatch = output.match(/(\d+(?:\.\d+)?)\s*%\s*(?:coverage|covered)/i);
    const coverage = coverageMatch ? parseFloat(coverageMatch[1]) : undefined;
    
    return { total: passed + failed + skipped, passed, failed, skipped, coverage };
  }
  
  private async saveTestResult(result: TestResult): Promise<void> {
    try {
      await supabaseAdmin.from('test_results').insert(result);
    } catch (error) {
      console.error('Error saving test result:', error);
    }
  }
  
  async getTestHistory(limit: number = 20): Promise<TestResult[]> {
    const { data } = await supabaseAdmin
      .from('test_results')
      .select('*')
      .eq('user_id', this.userId)
      .order('created_at', { ascending: false })
      .limit(limit);
    
    return data || [];
  }
}

export async function executeCode(userId: string, language: string, code: string): Promise<{ success: boolean; output: string; error?: string }> {
  const executor = new CodeExecutor(userId);
  return executor.execute(language, code);
}

export async function runTests(userId: string, language: string, code: string, testCode: string): Promise<TestResult> {
  const runner = new TestRunner(userId);
  return runner.runTests(language, code, testCode);
}
