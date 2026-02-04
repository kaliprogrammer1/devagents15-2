import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Settings {
  agentName: string;
  language: string;
  timeFormat: '12h' | '24h';
  soundEnabled: boolean;
  notificationsEnabled: boolean;
  
  theme: string;
  fontFamily: string;
  fontSize: number;
  animationsEnabled: boolean;
  
  agentPersonality: string;
  agentSpeed: 'slow' | 'normal' | 'fast';
  autoStartEnabled: boolean;
  breakRemindersEnabled: boolean;
  verboseLogging: boolean;
  
  workspaceName: string;
  defaultBranch: string;
  tabSize: number;
  autoSaveEnabled: boolean;
  formatOnSave: boolean;
  showMinimap: boolean;
  
  githubToken: string;
  vercelToken: string;
  openaiKey: string;
  slackWebhook: string;
}

interface SettingsState {
  settings: Settings;
  updateSettings: (newSettings: Partial<Settings>) => void;
  resetSettings: () => void;
  getSetting: <K extends keyof Settings>(key: K) => Settings[K];
}

const DEFAULT_SETTINGS: Settings = {
  agentName: 'DevAgent',
  language: 'en',
  timeFormat: '12h',
  soundEnabled: true,
  notificationsEnabled: true,
  
  theme: 'dark',
  fontFamily: 'jetbrains',
  fontSize: 13,
  animationsEnabled: true,
  
  agentPersonality: 'friendly',
  agentSpeed: 'normal',
  autoStartEnabled: false,
  breakRemindersEnabled: true,
  verboseLogging: false,
  
  workspaceName: 'devagent-project',
  defaultBranch: 'main',
  tabSize: 2,
  autoSaveEnabled: true,
  formatOnSave: true,
  showMinimap: true,
  
  githubToken: '',
  vercelToken: '',
  openaiKey: '',
  slackWebhook: '',
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: DEFAULT_SETTINGS,
      
      updateSettings: (newSettings) => {
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        }));
      },
      
      resetSettings: () => {
        set({ settings: DEFAULT_SETTINGS });
      },
      
      getSetting: (key) => {
        return get().settings[key];
      },
    }),
    {
      name: 'devagent-settings',
    }
  )
);

export const THEME_COLORS: Record<string, { bg: string; bgSecondary: string; accent: string; text: string; textMuted: string; border: string }> = {
  dark: {
    bg: '#0d1117',
    bgSecondary: '#161b22',
    accent: '#58a6ff',
    text: '#c9d1d9',
    textMuted: '#8b949e',
    border: '#30363d',
  },
  light: {
    bg: '#ffffff',
    bgSecondary: '#f6f8fa',
    accent: '#0969da',
    text: '#24292f',
    textMuted: '#57606a',
    border: '#d0d7de',
  },
  system: {
    bg: '#1e1e2e',
    bgSecondary: '#313244',
    accent: '#89b4fa',
    text: '#cdd6f4',
    textMuted: '#a6adc8',
    border: '#45475a',
  },
  midnight: {
    bg: '#0a0a0f',
    bgSecondary: '#12121a',
    accent: '#a855f7',
    text: '#e2e8f0',
    textMuted: '#94a3b8',
    border: '#1e1e2e',
  },
  forest: {
    bg: '#0d1912',
    bgSecondary: '#132a1c',
    accent: '#22c55e',
    text: '#dcfce7',
    textMuted: '#86efac',
    border: '#166534',
  },
};

export const FONT_FAMILIES: Record<string, string> = {
  jetbrains: "'JetBrains Mono', monospace",
  fira: "'Fira Code', monospace",
  cascadia: "'Cascadia Code', monospace",
  source: "'Source Code Pro', monospace",
  inter: "'Inter', sans-serif",
};

export const getSpeedMultiplier = (speed: 'slow' | 'normal' | 'fast'): number => {
  switch (speed) {
    case 'slow': return 1.5;
    case 'normal': return 1;
    case 'fast': return 0.6;
  }
};
