export interface ComputerState {
  activeApp: 'terminal' | 'browser' | 'editor' | 'files' | 'todo' | 'chat' | 'process' | null;
  browserUrl: string;
  browserTitle: string;
  terminalCwd: string;
  terminalLastOutput: string;
  terminalLastCommand: string;
  editorActiveFile: string | null;
  editorContent: string;
  filesCurrentDir: string;
  visibleElements: string[];
}

export interface TaskContext {
  task: string;
  subtasks: string[];
  currentSubtaskIndex: number;
  attempts: number;
  maxAttempts: number;
  startTime: number;
  lastActionTime: number;
  actionHistory: ActionHistoryItem[];
  computerState: ComputerState;
  errors: string[];
  learnings: string[];
}

export interface ActionHistoryItem {
  action: string;
  target?: string;
  result: 'success' | 'failure' | 'partial';
  output?: string;
  timestamp: number;
  duration: number;
}

export interface ThinkingResult {
  phase: 'observe' | 'reflect' | 'plan' | 'act' | 'verify';
  thought: string;
  confidence: number;
  action?: ComputerAction;
  plan?: string[];
  shouldRetry?: boolean;
  isComplete?: boolean;
}

export interface ComputerAction {
  type: 'TERMINAL' | 'BROWSER' | 'EDITOR' | 'FILES' | 'SWITCH_APP' | 'WAIT' | 'DONE';
  subAction: string;
  target?: string;
  content?: string;
  waitMs?: number;
}

export const COMPUTER_SKILLS = {
  terminal: {
    name: 'Terminal Operations',
    actions: {
      RUN_COMMAND: {
        description: 'Execute a shell command',
        format: 'TERMINAL:RUN_COMMAND:<command>',
        examples: ['ls -la', 'npm install', 'git status', 'cat package.json'],
      },
      CHANGE_DIR: {
        description: 'Change current directory',
        format: 'TERMINAL:CHANGE_DIR:<path>',
        examples: ['cd src', 'cd ..', 'cd /home/user/project'],
      },
      CREATE_FILE: {
        description: 'Create a new file with content',
        format: 'TERMINAL:CREATE_FILE:<filename>:<content>',
        examples: ['echo "content" > file.txt', 'touch newfile.js'],
      },
      READ_FILE: {
        description: 'Read file contents',
        format: 'TERMINAL:READ_FILE:<filename>',
        examples: ['cat file.txt', 'head -20 large.log'],
      },
      SEARCH: {
        description: 'Search for text in files',
        format: 'TERMINAL:SEARCH:<pattern>',
        examples: ['grep -r "pattern" .', 'find . -name "*.js"'],
      },
    },
  },
  browser: {
    name: 'Browser Operations',
    actions: {
      NAVIGATE: {
        description: 'Navigate to a URL',
        format: 'BROWSER:NAVIGATE:<url>',
        examples: ['https://google.com', 'https://github.com'],
      },
      SEARCH: {
        description: 'Search on Google',
        format: 'BROWSER:SEARCH:<query>',
        examples: ['React tutorials', 'Node.js documentation'],
      },
      SCROLL: {
        description: 'Scroll the page',
        format: 'BROWSER:SCROLL:<direction>',
        examples: ['down', 'up'],
      },
      BACK: {
        description: 'Go back in browser history',
        format: 'BROWSER:BACK',
        examples: [],
      },
      REFRESH: {
        description: 'Refresh the current page',
        format: 'BROWSER:REFRESH',
        examples: [],
      },
    },
  },
  editor: {
    name: 'Editor Operations',
    actions: {
      OPEN_FILE: {
        description: 'Open a file in editor',
        format: 'EDITOR:OPEN_FILE:<filepath>',
        examples: ['src/app.js', 'package.json'],
      },
      WRITE: {
        description: 'Write content at cursor',
        format: 'EDITOR:WRITE:<content>',
        examples: ['function hello() {}', 'const x = 5;'],
      },
      SAVE: {
        description: 'Save the current file',
        format: 'EDITOR:SAVE',
        examples: [],
      },
      FIND: {
        description: 'Find text in file',
        format: 'EDITOR:FIND:<text>',
        examples: ['TODO', 'function'],
      },
      REPLACE: {
        description: 'Find and replace text',
        format: 'EDITOR:REPLACE:<find>:<replace>',
        examples: ['oldName:newName'],
      },
    },
  },
  system: {
    name: 'System Operations',
    actions: {
      SWITCH_APP: {
        description: 'Switch to different application',
        format: 'SWITCH_APP:<app>',
        examples: ['terminal', 'browser', 'editor', 'files'],
      },
      WAIT: {
        description: 'Wait for operation to complete',
        format: 'WAIT:<milliseconds>',
        examples: ['1000', '2000'],
      },
      DONE: {
        description: 'Mark task as complete',
        format: 'DONE',
        examples: [],
      },
    },
  },
};

export function parseAction(actionString: string): ComputerAction {
  const parts = actionString.split(':');
  const type = parts[0].toUpperCase();
  
  switch (type) {
    case 'TERMINAL':
      return {
        type: 'TERMINAL',
        subAction: parts[1] || 'RUN_COMMAND',
        content: parts.slice(2).join(':'),
      };
    case 'BROWSER':
      return {
        type: 'BROWSER',
        subAction: parts[1] || 'NAVIGATE',
        target: parts[2],
        content: parts.slice(2).join(':'),
      };
    case 'EDITOR':
      return {
        type: 'EDITOR',
        subAction: parts[1] || 'OPEN_FILE',
        target: parts[2],
        content: parts.slice(3).join(':'),
      };
    case 'FILES':
      return {
        type: 'FILES',
        subAction: parts[1] || 'LIST',
        target: parts[2],
      };
    case 'SWITCH_APP':
      return {
        type: 'SWITCH_APP',
        subAction: 'SWITCH',
        target: parts[1],
      };
    case 'WAIT':
      return {
        type: 'WAIT',
        subAction: 'WAIT',
        waitMs: parseInt(parts[1]) || 1000,
      };
    case 'DONE':
      return {
        type: 'DONE',
        subAction: 'COMPLETE',
      };
    default:
      return {
        type: 'TERMINAL',
        subAction: 'RUN_COMMAND',
        content: actionString,
      };
  }
}

export function formatAction(action: ComputerAction): string {
  switch (action.type) {
    case 'TERMINAL':
      return `TERMINAL:${action.subAction}:${action.content || ''}`;
    case 'BROWSER':
      return `BROWSER:${action.subAction}:${action.target || action.content || ''}`;
    case 'EDITOR':
      return `EDITOR:${action.subAction}:${action.target || ''}${action.content ? ':' + action.content : ''}`;
    case 'FILES':
      return `FILES:${action.subAction}:${action.target || ''}`;
    case 'SWITCH_APP':
      return `SWITCH_APP:${action.target}`;
    case 'WAIT':
      return `WAIT:${action.waitMs}`;
    case 'DONE':
      return 'DONE';
    default:
      return 'DONE';
  }
}

export function createInitialContext(task: string): TaskContext {
  return {
    task,
    subtasks: [],
    currentSubtaskIndex: 0,
    attempts: 0,
    maxAttempts: 20,
    startTime: Date.now(),
    lastActionTime: Date.now(),
    actionHistory: [],
    computerState: {
      activeApp: null,
      browserUrl: '',
      browserTitle: '',
      terminalCwd: '~',
      terminalLastOutput: '',
      terminalLastCommand: '',
      editorActiveFile: null,
      editorContent: '',
      filesCurrentDir: '~',
      visibleElements: [],
    },
    errors: [],
    learnings: [],
  };
}

export function shouldContinue(context: TaskContext): boolean {
  if (context.attempts >= context.maxAttempts) return false;
  
  const elapsed = Date.now() - context.startTime;
  if (elapsed > 5 * 60 * 1000) return false;
  
  const recentActions = context.actionHistory.slice(-5);
  const repeatCount = recentActions.filter(a => a.action === recentActions[0]?.action).length;
  if (repeatCount >= 4) return false;
  
  return true;
}

export function analyzeFailure(context: TaskContext): { reason: string; suggestion: string } {
  const lastActions = context.actionHistory.slice(-3);
  
  if (lastActions.every(a => a.result === 'failure')) {
    return {
      reason: 'Multiple consecutive failures',
      suggestion: 'Try a different approach or verify prerequisites',
    };
  }
  
  if (context.errors.some(e => e.includes('not found'))) {
    return {
      reason: 'Resource not found',
      suggestion: 'Check if the file/path exists or search for alternatives',
    };
  }
  
  if (context.errors.some(e => e.includes('permission'))) {
    return {
      reason: 'Permission denied',
      suggestion: 'Check file permissions or try a different location',
    };
  }
  
  return {
    reason: 'Unknown failure',
    suggestion: 'Review the task and try breaking it into smaller steps',
  };
}

export function selectBestApp(task: string, context: TaskContext): 'terminal' | 'browser' | 'editor' {
  const taskLower = task.toLowerCase();
  
  if (taskLower.includes('search') || taskLower.includes('browse') || 
      taskLower.includes('website') || taskLower.includes('google') ||
      taskLower.includes('look up') || taskLower.includes('find online')) {
    return 'browser';
  }
  
  if (taskLower.includes('edit') || taskLower.includes('write code') ||
      taskLower.includes('modify') || taskLower.includes('create file') ||
      taskLower.includes('update file')) {
    return 'editor';
  }
  
  if (taskLower.includes('run') || taskLower.includes('execute') ||
      taskLower.includes('install') || taskLower.includes('npm') ||
      taskLower.includes('git') || taskLower.includes('command') ||
      taskLower.includes('terminal') || taskLower.includes('shell')) {
    return 'terminal';
  }
  
  return 'terminal';
}

export function getAvailableActions(state: ComputerState): string[] {
  const actions: string[] = [];
  
  switch (state.activeApp) {
    case 'terminal':
      actions.push(
        'TERMINAL:RUN_COMMAND:<command>',
        'TERMINAL:CHANGE_DIR:<path>',
        'TERMINAL:READ_FILE:<file>',
        'TERMINAL:SEARCH:<pattern>',
      );
      break;
    case 'browser':
      actions.push(
        'BROWSER:NAVIGATE:<url>',
        'BROWSER:SEARCH:<query>',
        'BROWSER:SCROLL:down',
        'BROWSER:SCROLL:up',
        'BROWSER:BACK',
        'BROWSER:REFRESH',
      );
      break;
    case 'editor':
      actions.push(
        'EDITOR:OPEN_FILE:<path>',
        'EDITOR:WRITE:<content>',
        'EDITOR:SAVE',
        'EDITOR:FIND:<text>',
        'EDITOR:REPLACE:<old>:<new>',
      );
      break;
  }
  
  actions.push('SWITCH_APP:terminal', 'SWITCH_APP:browser', 'SWITCH_APP:editor', 'DONE');
  
  return actions;
}
