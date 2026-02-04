import { agentMemory, skillManager, createUserMemory, summarizeForStorage, knowledgeGraph } from './agentMemory';
import { CodeExecutor, TestRunner } from './codeExecution';
import { GitHubIntegration } from './github';
import { hierarchicalPlanner } from './hierarchicalPlanning';
import { supabaseAdmin } from './supabase';
import { codeAnalyzer, CodeEntity, CodeAnalysisResult } from './codeAnalysis';

export interface AgentContext {
  userId: string;
  task: string;
  previousActions: string[];
  codeContext?: {
    files?: Array<{ path: string; content: string }>;
    focusedFile?: string;
    focusedFunction?: string;
  };
  screenState?: {
    activeApp: string | null;
    browserUrl: string;
    visibleWindows: string[];
  };
}

export interface LearnedInsight {
  type: 'pattern' | 'solution' | 'error_fix' | 'optimization' | 'fact' | 'concept';
  content: string;
  importance: number;
  entities?: string[];
  relations?: Array<{ target: string; type: string }>;
}

// Task classification using semantic analysis
export type TaskType = 
  | 'code_writing'
  | 'code_debugging'
  | 'code_refactoring'
  | 'code_review'
  | 'github_operation'
  | 'research'
  | 'memory_retrieval'
  | 'general';

interface TaskClassification {
  type: TaskType;
  confidence: number;
  relevantEntities: string[];
  suggestedApproach: string;
}

export class AgentBrain {
  private userId: string;
  private userMemory: ReturnType<typeof createUserMemory>;
  private codeExecutor: CodeExecutor;
  private testRunner: TestRunner;
  private github: GitHubIntegration;
  
  constructor(userId: string) {
    this.userId = userId;
    this.userMemory = createUserMemory(userId);
    this.codeExecutor = new CodeExecutor(userId);
    this.testRunner = new TestRunner(userId);
    this.github = new GitHubIntegration(userId);
  }
  
  async think(context: AgentContext): Promise<{
    relevantMemories: string[];
    relevantSkills: string[];
    relevantKnowledgeNodes: any[];
    userPreferences: Record<string, unknown>;
    suggestedApproach: string;
    hierarchicalPlan?: string[];
    taskClassification: TaskClassification;
    codeAnalysis?: {
      entities: CodeEntity[];
      complexity: CodeAnalysisResult['complexity'];
      issues: CodeAnalysisResult['issues'];
    };
  }> {
    // Semantic task classification (replaces simple keyword matching)
    const taskClassification = this.classifyTask(context.task);
    
    // Analyze code context if provided
    let codeAnalysis: { entities: CodeEntity[]; complexity: CodeAnalysisResult['complexity']; issues: CodeAnalysisResult['issues'] } | undefined;
    if (context.codeContext?.files) {
      codeAnalyzer.clear();
      codeAnalyzer.addFiles(context.codeContext.files);
      
      const analysisResults: CodeAnalysisResult[] = [];
      for (const file of context.codeContext.files) {
        const result = codeAnalyzer.analyzeFile(file.path);
        analysisResults.push(result);
      }
      
      // Aggregate analysis results
      codeAnalysis = {
        entities: analysisResults.flatMap(r => r.entities),
        complexity: {
          cyclomaticComplexity: analysisResults.reduce((sum, r) => sum + r.complexity.cyclomaticComplexity, 0),
          linesOfCode: analysisResults.reduce((sum, r) => sum + r.complexity.linesOfCode, 0),
          functionCount: analysisResults.reduce((sum, r) => sum + r.complexity.functionCount, 0),
          classCount: analysisResults.reduce((sum, r) => sum + r.complexity.classCount, 0),
        },
        issues: analysisResults.flatMap(r => r.issues),
      };
      
      // Build dependency graph
      codeAnalyzer.buildDependencyGraph();
      
      // Find entities related to the task
      const taskEntities = codeAnalyzer.searchEntities(context.task);
      taskClassification.relevantEntities = taskEntities.map(e => `${e.name} (${e.type}) at ${e.filePath}:${e.startLine}`);
    }
    
    const [universalMemories, userMemories, skills, preferences, graphNodes] = await Promise.all([
      agentMemory.searchUniversalMemory(context.task, 5),
      this.userMemory.searchMemories(context.task, 5),
      skillManager.searchSkills(context.task),
      this.userMemory.getAllPreferences(),
      knowledgeGraph.searchGraph(context.task),
    ]);
    
    const relevantMemories = [
      ...universalMemories.map(m => `[Universal] ${m.content}`),
      ...userMemories.map(m => `[User] ${m.content}`),
    ];
    
    const relevantSkills = [
      ...skills.map(s => `${s.skill_name}: ${s.description} (used ${s.usage_count}x, ${Math.round((s.success_rate || 0) * 100)}% success)`),
    ];

    // Expand context using Knowledge Graph
    const relevantKnowledgeNodes = [];
    for (const node of graphNodes) {
      const related = await knowledgeGraph.getRelatedNodes(node.id);
      relevantKnowledgeNodes.push({
        ...node,
        related: related.map(r => ({ type: r.relation, name: r.node.name }))
      });
    }
    
    // Use Hierarchical Planning (Tree of Thoughts) for complex tasks
    let hierarchicalPlan: string[] | undefined;
    const taskWords = context.task.toLowerCase().split(' ');
    if (context.task.length > 20 || taskWords.length > 4) {
      const taskContext = await this.getContextForTask(context.task, false);
      hierarchicalPlan = await hierarchicalPlanner.plan(context.task, {}, taskContext);
    }
    
    return {
      relevantMemories,
      relevantSkills,
      relevantKnowledgeNodes,
      userPreferences: preferences,
      suggestedApproach: taskClassification.suggestedApproach,
      hierarchicalPlan,
      taskClassification,
      codeAnalysis,
    };
  }

  /**
   * Semantic task classification using pattern matching and heuristics
   * Replaces simple keyword matching with more intelligent classification
   */
  private classifyTask(task: string): TaskClassification {
    const taskLower = task.toLowerCase();
    
    // Pattern-based classification with confidence scores
    const patterns: Array<{ type: TaskType; patterns: RegExp[]; approach: string }> = [
      {
        type: 'code_debugging',
        patterns: [
          /\b(fix|debug|error|bug|issue|crash|broken|not working|fails?|exception)\b/i,
          /\b(trace|stack|traceback|undefined|null|NaN)\b/i,
          /why (is|does|doesn't|isn't)/i,
        ],
        approach: 'Analyze error messages and stack traces. Use code analysis to trace the bug through call paths. Identify root cause before fixing.',
      },
      {
        type: 'code_refactoring',
        patterns: [
          /\b(refactor|clean|improve|optimize|restructure|reorganize|simplify)\b/i,
          /\b(performance|memory|speed|efficiency)\b/i,
          /make (it |this )?(better|faster|cleaner|more readable)/i,
        ],
        approach: 'Analyze code structure and dependencies. Identify code smells and anti-patterns. Plan incremental changes that preserve behavior.',
      },
      {
        type: 'code_review',
        patterns: [
          /\b(review|check|audit|analyze|examine|inspect|assess)\b/i,
          /\b(security|vulnerability|best practice|quality)\b/i,
          /what('s| is) wrong with/i,
        ],
        approach: 'Perform static analysis. Check for security vulnerabilities, code smells, and adherence to best practices. Provide actionable feedback.',
      },
      {
        type: 'code_writing',
        patterns: [
          /\b(write|create|implement|add|build|develop|make|generate)\b/i,
          /\b(function|class|component|module|api|endpoint|feature)\b/i,
          /\b(code|program|script|app|application)\b/i,
        ],
        approach: 'Understand requirements. Analyze existing code structure. Write code following project conventions and best practices.',
      },
      {
        type: 'github_operation',
        patterns: [
          /\b(github|git|repo|repository|branch|commit|push|pull|merge|pr|pull request)\b/i,
          /\b(clone|fork|release|deploy)\b/i,
        ],
        approach: 'Use GitHub integration. Check connection status first. Follow git workflow best practices.',
      },
      {
        type: 'research',
        patterns: [
          /\b(search|find|look for|browse|research|investigate)\b/i,
          /\b(documentation|docs|api|reference|example|tutorial)\b/i,
          /how (do|can|to|does)/i,
        ],
        approach: 'Search documentation and code examples. Gather relevant information before implementation.',
      },
      {
        type: 'memory_retrieval',
        patterns: [
          /\b(remember|recall|history|previous|last time|before)\b/i,
          /what (did|was|were)/i,
        ],
        approach: 'Search memories and past interactions for relevant context.',
      },
    ];

    let bestMatch: { type: TaskType; confidence: number; approach: string } = {
      type: 'general',
      confidence: 0.3,
      approach: 'Analyze the task and execute step by step.',
    };

    for (const { type, patterns: typePatterns, approach } of patterns) {
      let matchCount = 0;
      for (const pattern of typePatterns) {
        if (pattern.test(taskLower)) {
          matchCount++;
        }
      }
      
      if (matchCount > 0) {
        const confidence = Math.min(0.9, 0.4 + (matchCount * 0.2));
        if (confidence > bestMatch.confidence) {
          bestMatch = { type, confidence, approach };
        }
      }
    }

    return {
      type: bestMatch.type,
      confidence: bestMatch.confidence,
      relevantEntities: [],
      suggestedApproach: bestMatch.approach,
    };
  }
  
  async learn(insights: LearnedInsight[]): Promise<void> {
    for (const insight of insights) {
      const summary = summarizeForStorage(insight.content, 500);
      await agentMemory.addUniversalMemory(insight.type, summary, insight.importance);
      
      // Add to Knowledge Graph
      const nodeId = await knowledgeGraph.addNode(
        insight.entities?.[0] || insight.content.substring(0, 50),
        insight.type,
        insight.content,
        { importance: insight.importance, entities: insight.entities }
      );
      
      if (nodeId && insight.relations) {
        for (const rel of insight.relations) {
          const targetNode = await knowledgeGraph.findNodeByName(rel.target);
          if (targetNode) {
            await knowledgeGraph.addEdge(nodeId, targetNode.id, rel.type);
          }
        }
      }
    }
  }
  
  async learnFromTask(task: string, actions: string[], outcome: 'success' | 'failure', notes?: string): Promise<void> {
    const content = `Task: ${task}\nActions: ${actions.join(' → ')}\nOutcome: ${outcome}${notes ? `\nNotes: ${notes}` : ''}`;
    
    await this.userMemory.addMemory(
      'task_history',
      content,
      { task, actions, outcome },
      outcome === 'success' ? 0.7 : 0.5
    );

    // Add task to Knowledge Graph
    const taskId = await knowledgeGraph.addNode(task, 'task', content, { outcome, actions });
    
    if (outcome === 'success' && actions.length > 2) {
      // Auto-patch skill set (Continuous Learning 2.0)
      const patternName = `Skill: ${task.split(' ').slice(0, 3).join(' ')}`;
      await supabaseAdmin.from('skill_patterns').insert({
        pattern_name: patternName,
        pattern_description: `Generated from successful task: ${task}`,
        successful_action_sequence: actions,
        trigger_condition: task,
        usage_count: 1,
        success_rate: 1.0
      });

      const patternContent = `Successful pattern for "${task}": ${actions.join(' → ')}`;
      await agentMemory.addUniversalMemory('pattern', patternContent, 0.6);

      // Add solution to Knowledge Graph and link to task
      const solutionId = await knowledgeGraph.addNode(patternName, 'solution', patternContent, { actions });
      if (taskId && solutionId) {
        await knowledgeGraph.addEdge(taskId, solutionId, 'solved_by');
      }
    }
  }
  
  async learnNewSkill(
    name: string,
    category: 'coding' | 'research' | 'communication' | 'analysis' | 'automation' | 'integration',
    description: string,
    examples: string[],
    bestPractices: string[]
  ): Promise<void> {
    await skillManager.learnSkill(name, category, description, {
      examples,
      bestPractices,
      learnedAt: new Date().toISOString(),
    });
  }
  
  async monitorAndFixGitHubBuilds(owner: string, repo: string): Promise<{ fixed: boolean; message: string }> {
    await this.github.initialize();
    if (!this.github.isConnected()) {
      return { fixed: false, message: 'GitHub not connected' };
    }
    
    const result = await this.github.monitorAndFixBuilds(owner, repo);
    
    if (result.message.includes('Found failing build')) {
      await this.recordSkillUsage('github_cicd_monitoring', true);
      // Logic for autonomous fix would go here - for now we return the findings
    }
    
    return result;
  }
  
  async executeCode(language: string, code: string): Promise<{ success: boolean; output: string; error?: string }> {
    const result = await this.codeExecutor.execute(language, code);
    
    if (result.success) {
      await this.recordSkillUsage('code_execution', true);
    } else {
      await agentMemory.addUniversalMemory(
        'error_fix',
        `Error in ${language}: ${result.error}\nCode snippet: ${code.substring(0, 200)}`,
        0.4
      );
    }
    
    return result;
  }
  
  async runTests(language: string, code: string, testCode: string): Promise<{
    passed: number;
    failed: number;
    total: number;
    output: string;
  }> {
    const result = await this.testRunner.runTests(language, code, testCode);
    
    await this.recordSkillUsage('testing', result.failed === 0);
    
    return {
      passed: result.passed,
      failed: result.failed,
      total: result.total_tests,
      output: result.output || '',
    };
  }
  
  async connectGitHub(token: string): Promise<boolean> {
    const success = await this.github.initialize(token);
    if (success) {
      await this.recordSkillUsage('github_integration', true);
    }
    return success;
  }
  
  async getGitHubRepos(): Promise<Array<{ name: string; full_name: string; description: string | null }>> {
    await this.github.initialize();
    if (!this.github.isConnected()) return [];
    
    const repos = await this.github.listRepositories();
    return repos.map(r => ({
      name: r.name,
      full_name: r.full_name,
      description: r.description,
    }));
  }
  
  async createPR(
    owner: string,
    repo: string,
    title: string,
    head: string,
    base: string,
    body?: string
  ): Promise<{ success: boolean; prUrl?: string }> {
    await this.github.initialize();
    if (!this.github.isConnected()) {
      return { success: false };
    }
    
    const pr = await this.github.createPullRequest(owner, repo, title, head, base, body);
    if (pr) {
      await this.recordSkillUsage('pr_creation', true);
      return { success: true, prUrl: pr.html_url };
    }
    
    return { success: false };
  }
  
  async rememberUserPreference(key: string, value: unknown, context?: string): Promise<void> {
    await this.userMemory.setPreference(key, value, context);
  }
  
  async recallUserPreference(key: string): Promise<unknown | null> {
    return this.userMemory.getPreference(key);
  }
  
  async getContextForTask(task: string, includeRecursive: boolean = true): Promise<string> {
    const thinking = await this.think({ userId: this.userId, task, previousActions: [] });
    
    let context = '';
    
    if (thinking.relevantMemories.length > 0) {
      context += `\nRelevant memories:\n${thinking.relevantMemories.slice(0, 3).join('\n')}`;
    }
    
    if (thinking.relevantSkills.length > 0) {
      context += `\nRelevant skills:\n${thinking.relevantSkills.slice(0, 3).join('\n')}`;
    }
    
    if (Object.keys(thinking.userPreferences).length > 0) {
      context += `\nUser preferences: ${JSON.stringify(thinking.userPreferences)}`;
    }
    
    context += `\nSuggested approach: ${thinking.suggestedApproach}`;
    
    if (includeRecursive && thinking.hierarchicalPlan) {
      context += `\nHierarchical Plan (ToT):\n${thinking.hierarchicalPlan.join(' → ')}`;
    }
    
    return context;
  }
  
  async getMostUsedSkills(): Promise<Array<{ name: string; uses: number; successRate: number }>> {
    const skills = await skillManager.getMostUsedSkills(10);
    return skills.map(s => ({
      name: s.skill_name,
      uses: s.usage_count || 0,
      successRate: s.success_rate || 0,
    }));
  }
  
  async getAgentStats(): Promise<{
    totalSkills: number;
    totalMemories: number;
    topSkills: string[];
    recentLearnings: string[];
  }> {
    const [skills, memories] = await Promise.all([
      skillManager.getAllSkills(),
      agentMemory.getRecentMemories(5),
    ]);
    
    const sortedSkills = [...skills].sort((a, b) => (b.usage_count || 0) - (a.usage_count || 0));
    
    return {
      totalSkills: skills.length,
      totalMemories: memories.length,
      topSkills: sortedSkills.slice(0, 5).map(s => s.skill_name),
      recentLearnings: memories.map(m => m.content.substring(0, 100)),
    };
  }

  /**
   * Record skill usage for learning
   */
  private async recordSkillUsage(skillName: string, success: boolean): Promise<void> {
    await skillManager.useSkill(skillName, success);
  }

  // ============================================================
  // CODE ANALYSIS METHODS
  // ============================================================

  /**
   * Analyze code files and return structured analysis
   */
  async analyzeCode(files: Array<{ path: string; content: string }>): Promise<{
    entities: CodeEntity[];
    complexity: CodeAnalysisResult['complexity'];
    issues: CodeAnalysisResult['issues'];
    dependencyGraph: {
      nodes: string[];
      edges: Array<{ from: string; to: string }>;
      circularDependencies: string[][];
    };
    summary: {
      totalFiles: number;
      totalEntities: number;
      byType: Record<string, number>;
      avgComplexity: number;
    };
  }> {
    codeAnalyzer.clear();
    codeAnalyzer.addFiles(files);

    const results: CodeAnalysisResult[] = [];
    for (const file of files) {
      results.push(codeAnalyzer.analyzeFile(file.path));
    }

    const graph = codeAnalyzer.buildDependencyGraph();
    const circularDeps = codeAnalyzer.findCircularDependencies();
    const summary = codeAnalyzer.getCodebaseSummary();

    return {
      entities: results.flatMap(r => r.entities),
      complexity: {
        cyclomaticComplexity: results.reduce((sum, r) => sum + r.complexity.cyclomaticComplexity, 0),
        linesOfCode: results.reduce((sum, r) => sum + r.complexity.linesOfCode, 0),
        functionCount: results.reduce((sum, r) => sum + r.complexity.functionCount, 0),
        classCount: results.reduce((sum, r) => sum + r.complexity.classCount, 0),
      },
      issues: results.flatMap(r => r.issues),
      dependencyGraph: {
        nodes: Array.from(graph.nodes.keys()),
        edges: graph.edges.map(e => ({ from: e.from, to: e.to })),
        circularDependencies: circularDeps,
      },
      summary,
    };
  }

  /**
   * Find callers of a specific function
   */
  findFunctionCallers(functionName: string): CodeEntity[] {
    return codeAnalyzer.findCallers(functionName);
  }

  /**
   * Find what functions are called by a specific function
   */
  findFunctionCallees(functionName: string): string[] {
    return codeAnalyzer.findCallees(functionName);
  }

  /**
   * Search for code entities by pattern
   */
  searchCodeEntities(pattern: string, type?: CodeEntity['type']): CodeEntity[] {
    return codeAnalyzer.searchEntities(pattern, type);
  }

  /**
   * Get a specific entity by name
   */
  getCodeEntity(name: string): CodeEntity | null {
    return codeAnalyzer.getEntity(name);
  }

  /**
   * Get files that depend on a specific file
   */
  getFileDependents(filePath: string): string[] {
    return codeAnalyzer.getDependents(filePath);
  }

  /**
   * Get files that a specific file depends on
   */
  getFileDependencies(filePath: string): string[] {
    return codeAnalyzer.getDependencies(filePath);
  }

  /**
   * Analyze code for potential issues (security, performance, maintainability)
   */
  async reviewCode(files: Array<{ path: string; content: string }>): Promise<{
    issues: CodeAnalysisResult['issues'];
    suggestions: string[];
    complexity: CodeAnalysisResult['complexity'];
  }> {
    const analysis = await this.analyzeCode(files);
    const suggestions: string[] = [];

    // Generate suggestions based on analysis
    if (analysis.complexity.cyclomaticComplexity > 20) {
      suggestions.push('High cyclomatic complexity detected. Consider breaking down complex functions.');
    }

    const highComplexityFunctions = analysis.entities
      .filter(e => e.type === 'function' && e.complexity && e.complexity > 10);
    for (const fn of highComplexityFunctions) {
      suggestions.push(`Function "${fn.name}" has high complexity (${fn.complexity}). Consider refactoring.`);
    }

    if (analysis.dependencyGraph.circularDependencies.length > 0) {
      suggestions.push(`${analysis.dependencyGraph.circularDependencies.length} circular dependencies detected. This can cause maintainability issues.`);
    }

    const errorIssues = analysis.issues.filter(i => i.type === 'error');
    if (errorIssues.length > 0) {
      suggestions.push(`${errorIssues.length} error(s) found. These should be fixed before deployment.`);
    }

    return {
      issues: analysis.issues,
      suggestions,
      complexity: analysis.complexity,
    };
  }
}

export function createAgentBrain(userId: string): AgentBrain {
  return new AgentBrain(userId);
}

export async function initializeBaseSkills(): Promise<void> {
  const baseSkills = [
    {
      name: 'web_browsing',
      category: 'research' as const,
      description: 'Navigate websites, search for information, and extract data from web pages',
      knowledge: { patterns: ['NAVIGATE:url', 'TYPE:search_query', 'CLICK:element'] },
    },
    {
      name: 'code_execution',
      category: 'coding' as const,
      description: 'Write and execute code in multiple programming languages',
      knowledge: { languages: ['javascript', 'python', 'typescript'], patterns: ['write', 'test', 'debug'] },
    },
    {
      name: 'github_integration',
      category: 'integration' as const,
      description: 'Interact with GitHub repositories, create branches, and manage pull requests',
      knowledge: { actions: ['clone', 'branch', 'commit', 'push', 'pr'] },
    },
    {
      name: 'testing',
      category: 'coding' as const,
      description: 'Write and run automated tests for code',
      knowledge: { frameworks: ['jest', 'unittest', 'pytest'] },
    },
    {
      name: 'task_planning',
      category: 'analysis' as const,
      description: 'Break down complex tasks into manageable steps',
      knowledge: { patterns: ['analyze', 'plan', 'execute', 'verify'] },
    },
  ];
  
  for (const skill of baseSkills) {
    await skillManager.learnSkill(skill.name, skill.category, skill.description, skill.knowledge);
  }
}
