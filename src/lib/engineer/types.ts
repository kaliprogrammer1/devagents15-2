export interface EngineerSession {
  id: string;
  startedAt: Date;
  endedAt?: Date;
  totalTasksCompleted: number;
  totalCodeWritten: number;
  languagesUsed: string[];
  frameworksUsed: string[];
  performanceScore: number;
  learnings: Learning[];
}

export interface Learning {
  topic: string;
  insight: string;
  confidence: number;
  timestamp: Date;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  techStack: string[];
  fileStructure: FileNode[];
  createdAt: Date;
  updatedAt: Date;
  status: 'active' | 'completed' | 'archived';
  gitRepo?: string;
  branch: string;
}

export interface FileNode {
  path: string;
  name: string;
  type: 'file' | 'directory';
  content?: string;
  language?: string;
  size: number;
  children?: FileNode[];
  metadata?: Record<string, unknown>;
}

export interface Task {
  id: string;
  sessionId: string;
  projectId?: string;
  description: string;
  status: 'pending' | 'analyzing' | 'planning' | 'executing' | 'testing' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  reasoningChain: ReasoningStep[];
  actionsTaken: Action[];
  filesModified: string[];
  testsRun: TestResult[];
  result?: string;
  error?: string;
  durationMs?: number;
  createdAt: Date;
  completedAt?: Date;
}

export interface ReasoningStep {
  stepNumber: number;
  type: 'observation' | 'hypothesis' | 'analysis' | 'decision' | 'reflection' | 'self_correction';
  thought: string;
  evidence: string[];
  conclusion?: string;
  confidence: number;
  timestamp: Date;
}

export interface Action {
  id: string;
  type: ActionType;
  target?: string;
  content?: string;
  result?: string;
  error?: string;
  durationMs: number;
  timestamp: Date;
}

export type ActionType = 
  | 'READ_FILE'
  | 'WRITE_FILE'
  | 'CREATE_FILE'
  | 'DELETE_FILE'
  | 'SEARCH_CODE'
  | 'RUN_COMMAND'
  | 'RUN_TESTS'
  | 'INSTALL_PACKAGE'
  | 'GIT_COMMIT'
  | 'GIT_PUSH'
  | 'GIT_PULL'
  | 'GIT_BRANCH'
  | 'BROWSE_WEB'
  | 'SEARCH_WEB'
  | 'READ_DOCS'
  | 'ANALYZE_CODE'
  | 'GENERATE_CODE'
  | 'REFACTOR_CODE'
  | 'DEBUG_CODE'
  | 'ASK_USER'
  | 'THINK'
  | 'PLAN'
  | 'COMPLETE';

export interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  error?: string;
  output?: string;
}

export interface Knowledge {
  id: string;
  category: string;
  topic: string;
  content: string;
  confidence: number;
  timesUsed: number;
  lastUsed: Date;
  source?: string;
  metadata?: Record<string, unknown>;
}

export interface EngineerState {
  session: EngineerSession | null;
  currentProject: Project | null;
  currentTask: Task | null;
  openFiles: FileNode[];
  terminalHistory: TerminalEntry[];
  browserState: BrowserState;
  isWorking: boolean;
  workPhase: WorkPhase;
  thoughts: Thought[];
  plan: PlanStep[];
}

export type WorkPhase = 
  | 'idle'
  | 'understanding'
  | 'researching'
  | 'planning'
  | 'implementing'
  | 'testing'
  | 'debugging'
  | 'reviewing'
  | 'documenting'
  | 'deploying';

export interface Thought {
  id: string;
  type: 'observation' | 'analysis' | 'hypothesis' | 'decision' | 'reflection' | 'question' | 'insight';
  content: string;
  confidence: number;
  timestamp: Date;
  relatedFiles?: string[];
  relatedCode?: string;
}

export interface PlanStep {
  id: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped' | 'failed';
  substeps?: PlanStep[];
  estimatedDuration?: number;
  actualDuration?: number;
  dependencies?: string[];
}

export interface TerminalEntry {
  id: string;
  command: string;
  output: string;
  exitCode: number;
  timestamp: Date;
  cwd: string;
}

export interface BrowserState {
  url: string;
  title: string;
  isLoading: boolean;
  history: string[];
  tabs: BrowserTab[];
}

export interface BrowserTab {
  id: string;
  url: string;
  title: string;
  isActive: boolean;
}

export interface CodeAnalysis {
  file: string;
  language: string;
  lines: number;
  functions: FunctionInfo[];
  imports: string[];
  exports: string[];
  issues: CodeIssue[];
  complexity: number;
  suggestions: string[];
}

export interface FunctionInfo {
  name: string;
  line: number;
  params: string[];
  returnType?: string;
  complexity: number;
  description?: string;
}

export interface CodeIssue {
  type: 'error' | 'warning' | 'info' | 'style';
  message: string;
  line: number;
  column?: number;
  rule?: string;
  fix?: string;
}

export interface EngineerCapabilities {
  languages: string[];
  frameworks: string[];
  tools: string[];
  specializations: string[];
  currentSkillLevel: Record<string, number>;
}

export const ENGINEER_CAPABILITIES: EngineerCapabilities = {
  languages: [
    'TypeScript', 'JavaScript', 'Python', 'Rust', 'Go', 'Java', 'C++', 
    'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'SQL', 'HTML', 'CSS', 'Shell'
  ],
  frameworks: [
    'React', 'Next.js', 'Vue', 'Angular', 'Svelte', 'Node.js', 'Express',
    'FastAPI', 'Django', 'Flask', 'Spring', 'Rails', 'Laravel', 'TailwindCSS'
  ],
  tools: [
    'Git', 'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure', 'PostgreSQL',
    'MongoDB', 'Redis', 'GraphQL', 'REST', 'WebSocket', 'Jest', 'Pytest'
  ],
  specializations: [
    'Full-Stack Development', 'API Design', 'Database Design', 'DevOps',
    'Performance Optimization', 'Security', 'Testing', 'Code Review'
  ],
  currentSkillLevel: {}
};
