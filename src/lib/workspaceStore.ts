import { create } from 'zustand';

export interface FileNode {
  id: string;
  name: string;
  type: 'file' | 'folder';
  content?: string;
  language?: string;
  children?: FileNode[];
  isOpen?: boolean;
  modified?: boolean;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  status: 'backlog' | 'todo' | 'in_progress' | 'review' | 'done';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  assignee: string;
  storyPoints: number;
  createdAt: number;
  updatedAt: number;
  labels: string[];
}

export interface GitCommit {
  id: string;
  message: string;
  author: string;
  timestamp: number;
  files: string[];
}

export interface GitBranch {
  name: string;
  isActive: boolean;
  commits: GitCommit[];
}

export interface ChatMessage {
  id: string;
  channel: string;
  sender: string;
  content: string;
  timestamp: number;
  isAgent: boolean;
  reactions?: string[];
}

export interface CalendarEvent {
  id: string;
  title: string;
  start: number;
  end: number;
  type: 'meeting' | 'focus' | 'break' | 'standup' | 'review';
  description?: string;
}

export interface ScreenState {
  activeApp: string | null;
  browserUrl: string;
  browserTitle: string;
  terminalLastOutput: string;
  editorActiveFile: string | null;
  editorContent: string;
  visibleWindows: string[];
}

export interface WorkspaceState {
  files: FileNode[];
  openFiles: string[];
  activeFileId: string | null;
  
  tasks: Task[];
  activeTaskId: string | null;
  
  gitBranches: GitBranch[];
  currentBranch: string;
  stagedFiles: string[];
  
  chatMessages: ChatMessage[];
  activeChannel: string;
  channels: string[];
  
  calendarEvents: CalendarEvent[];
  currentTime: number;
  
  terminalHistory: string[];
  terminalCwd: string;

  screenState: ScreenState;

  // Skills
  skills: Skill[];
  isRecordingSkill: boolean;
  recordedActions: string[];
  
  setFiles: (files: FileNode[]) => void;
  updateFileContent: (fileId: string, content: string) => void;
  openFile: (fileId: string) => void;
  closeFile: (fileId: string) => void;
  setActiveFile: (fileId: string | null) => void;
  createFile: (parentId: string | null, name: string, type: 'file' | 'folder') => void;
  deleteFile: (fileId: string) => void;
  toggleFolder: (folderId: string) => void;
  
  addTask: (task: Omit<Task, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  moveTask: (taskId: string, newStatus: Task['status']) => void;
  setActiveTask: (taskId: string | null) => void;
  
  createBranch: (name: string) => void;
  switchBranch: (name: string) => void;
  stageFile: (fileId: string) => void;
  unstageFile: (fileId: string) => void;
  commit: (message: string) => void;
  
  sendMessage: (channel: string, content: string, isAgent: boolean) => void;
  setActiveChannel: (channel: string) => void;
  
  addCalendarEvent: (event: Omit<CalendarEvent, 'id'>) => void;
  removeCalendarEvent: (eventId: string) => void;
  setCurrentTime: (time: number) => void;
  
  addTerminalLine: (line: string) => void;
  setTerminalCwd: (cwd: string) => void;
  clearTerminal: () => void;

  updateScreenState: (state: Partial<ScreenState>) => void;
  getScreenState: () => ScreenState;

  // Skill actions
  startRecordingSkill: () => void;
  stopRecordingSkill: (title: string, trigger: string) => Promise<void>;
  addRecordedAction: (action: string) => void;
  setSkills: (skills: Skill[]) => void;
}

export interface Skill {
  id: string;
  title: string;
  trigger: string;
  actions: string[];
  createdAt: number;
}

const EMPTY_FILES: FileNode[] = [];

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  files: EMPTY_FILES,
  openFiles: [],
  activeFileId: null,
  
  tasks: [],
  activeTaskId: null,
  
  gitBranches: [
    {
      name: 'main',
      isActive: true,
      commits: [],
    },
  ],
  currentBranch: 'main',
  stagedFiles: [],
  
  chatMessages: [],
  activeChannel: 'general',
  channels: ['general'],
  
  calendarEvents: [],
  currentTime: Date.now(),
  
  terminalHistory: [],
  terminalCwd: '~',

  screenState: {
    activeApp: null,
    browserUrl: '',
    browserTitle: '',
    terminalLastOutput: '',
    editorActiveFile: null,
    editorContent: '',
    visibleWindows: [],
  },

  skills: [],
  isRecordingSkill: false,
  recordedActions: [],
  
  setFiles: (files) => set({ files }),
  
  updateFileContent: (fileId, content) => {
    const updateNode = (nodes: FileNode[]): FileNode[] => {
      return nodes.map(node => {
        if (node.id === fileId) {
          return { ...node, content, modified: true };
        }
        if (node.children) {
          return { ...node, children: updateNode(node.children) };
        }
        return node;
      });
    };
    set({ files: updateNode(get().files) });
    
    const file = findFileById(get().files, fileId);
    if (file) {
      set(state => ({
        screenState: {
          ...state.screenState,
          editorActiveFile: file.name,
          editorContent: content.slice(0, 500),
        }
      }));
    }
  },
  
  openFile: (fileId) => {
    const { openFiles } = get();
    if (!openFiles.includes(fileId)) {
      set({ openFiles: [...openFiles, fileId], activeFileId: fileId });
    } else {
      set({ activeFileId: fileId });
    }
  },
  
  closeFile: (fileId) => {
    const { openFiles, activeFileId } = get();
    const newOpenFiles = openFiles.filter(id => id !== fileId);
    let newActiveFileId = activeFileId;
    if (activeFileId === fileId) {
      const index = openFiles.indexOf(fileId);
      newActiveFileId = newOpenFiles[Math.min(index, newOpenFiles.length - 1)] || null;
    }
    set({ openFiles: newOpenFiles, activeFileId: newActiveFileId });
  },
  
  setActiveFile: (fileId) => set({ activeFileId: fileId }),
  
  createFile: (parentId, name, type) => {
    const newFile: FileNode = {
      id: `file-${Date.now()}`,
      name,
      type,
      content: type === 'file' ? '' : undefined,
      children: type === 'folder' ? [] : undefined,
      isOpen: type === 'folder' ? true : undefined,
    };
    
    const addToParent = (nodes: FileNode[]): FileNode[] => {
      if (!parentId) {
        return [...nodes, newFile];
      }
      return nodes.map(node => {
        if (node.id === parentId && node.type === 'folder') {
          return { ...node, children: [...(node.children || []), newFile] };
        }
        if (node.children) {
          return { ...node, children: addToParent(node.children) };
        }
        return node;
      });
    };
    
    set({ files: addToParent(get().files) });
  },
  
  deleteFile: (fileId) => {
    const removeNode = (nodes: FileNode[]): FileNode[] => {
      return nodes
        .filter(node => node.id !== fileId)
        .map(node => {
          if (node.children) {
            return { ...node, children: removeNode(node.children) };
          }
          return node;
        });
    };
    
    const { openFiles, activeFileId } = get();
    set({
      files: removeNode(get().files),
      openFiles: openFiles.filter(id => id !== fileId),
      activeFileId: activeFileId === fileId ? null : activeFileId,
    });
  },
  
  toggleFolder: (folderId) => {
    const toggle = (nodes: FileNode[]): FileNode[] => {
      return nodes.map(node => {
        if (node.id === folderId) {
          return { ...node, isOpen: !node.isOpen };
        }
        if (node.children) {
          return { ...node, children: toggle(node.children) };
        }
        return node;
      });
    };
    set({ files: toggle(get().files) });
  },
  
  addTask: (task) => {
    const newTask: Task = {
      ...task,
      id: `task-${Date.now()}`,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    set({ tasks: [...get().tasks, newTask] });
  },
  
  updateTask: (taskId, updates) => {
    set({
      tasks: get().tasks.map(task =>
        task.id === taskId ? { ...task, ...updates, updatedAt: Date.now() } : task
      ),
    });
  },
  
  moveTask: (taskId, newStatus) => {
    set({
      tasks: get().tasks.map(task =>
        task.id === taskId ? { ...task, status: newStatus, updatedAt: Date.now() } : task
      ),
    });
  },
  
  setActiveTask: (taskId) => set({ activeTaskId: taskId }),
  
  createBranch: (name) => {
    const { gitBranches, currentBranch } = get();
    const currentBranchData = gitBranches.find(b => b.name === currentBranch);
    const newBranch: GitBranch = {
      name,
      isActive: true,
      commits: currentBranchData?.commits ? [...currentBranchData.commits] : [],
    };
    set({
      gitBranches: gitBranches.map(b => ({ ...b, isActive: false })).concat(newBranch),
      currentBranch: name,
    });
  },
  
  switchBranch: (name) => {
    set({
      gitBranches: get().gitBranches.map(b => ({ ...b, isActive: b.name === name })),
      currentBranch: name,
    });
  },
  
  stageFile: (fileId) => {
    const { stagedFiles } = get();
    if (!stagedFiles.includes(fileId)) {
      set({ stagedFiles: [...stagedFiles, fileId] });
    }
  },
  
  unstageFile: (fileId) => {
    set({ stagedFiles: get().stagedFiles.filter(id => id !== fileId) });
  },
  
  commit: (message) => {
    const { gitBranches, currentBranch, stagedFiles } = get();
    const newCommit: GitCommit = {
      id: `commit-${Date.now()}`,
      message,
      author: 'DevAgent',
      timestamp: Date.now(),
      files: stagedFiles,
    };
    
    set({
      gitBranches: gitBranches.map(b =>
        b.name === currentBranch
          ? { ...b, commits: [...b.commits, newCommit] }
          : b
      ),
      stagedFiles: [],
    });
  },
  
  sendMessage: (channel, content, isAgent) => {
    const newMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      channel,
      sender: isAgent ? 'DevAgent' : 'You',
      content,
      timestamp: Date.now(),
      isAgent,
    };
    set({ chatMessages: [...get().chatMessages, newMessage] });
  },
  
  setActiveChannel: (channel) => set({ activeChannel: channel }),
  
  addCalendarEvent: (event) => {
    const newEvent: CalendarEvent = {
      ...event,
      id: `evt-${Date.now()}`,
    };
    set({ calendarEvents: [...get().calendarEvents, newEvent] });
  },
  
  removeCalendarEvent: (eventId) => {
    set({ calendarEvents: get().calendarEvents.filter(e => e.id !== eventId) });
  },
  
  setCurrentTime: (time) => set({ currentTime: time }),
  
  addTerminalLine: (line) => {
    set({ 
      terminalHistory: [...get().terminalHistory, line],
      screenState: {
        ...get().screenState,
        terminalLastOutput: line,
      }
    });
  },
  
  setTerminalCwd: (cwd) => set({ terminalCwd: cwd }),
  
  clearTerminal: () => set({ terminalHistory: [] }),

  updateScreenState: (state) => {
    set(prev => ({
      screenState: { ...prev.screenState, ...state }
    }));
  },

  getScreenState: () => get().screenState,

  startRecordingSkill: () => {
    set({ isRecordingSkill: true, recordedActions: [] });
  },

  stopRecordingSkill: async (title, trigger) => {
    const actions = get().recordedActions;
    if (actions.length === 0) {
      set({ isRecordingSkill: false });
      return;
    }

    const newSkill: Skill = {
      id: `skill-${Date.now()}`,
      title,
      trigger,
      actions,
      createdAt: Date.now(),
    };

    try {
      const response = await fetch('/api/skills', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'save', skill: newSkill }),
      });

      if (response.ok) {
        set(state => ({
          skills: [...state.skills, newSkill],
          isRecordingSkill: false,
          recordedActions: [],
        }));
      }
    } catch (error) {
      console.error('Error saving skill:', error);
      set({ isRecordingSkill: false });
    }
  },

  addRecordedAction: (action) => {
    if (get().isRecordingSkill) {
      set(state => ({ recordedActions: [...state.recordedActions, action] }));
    }
  },

  setSkills: (skills) => set({ skills }),
}));

export function findFileById(files: FileNode[], id: string): FileNode | null {
  for (const file of files) {
    if (file.id === id) return file;
    if (file.children) {
      const found = findFileById(file.children, id);
      if (found) return found;
    }
  }
  return null;
}

export function getFilePath(files: FileNode[], id: string, path: string[] = []): string[] | null {
  for (const file of files) {
    if (file.id === id) return [...path, file.name];
    if (file.children) {
      const found = getFilePath(file.children, id, [...path, file.name]);
      if (found) return found;
    }
  }
  return null;
}
