// Persona-specific skill storage and database management
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { PersonaType, PersonaConfig, PERSONA_CONFIGS } from './personaApps';

// Skill entry for each persona
export interface PersonaSkillEntry {
  id: string;
  name: string;
  description: string;
  proficiency: number; // 0-100
  lastUsed: Date;
  usageCount: number;
  category: string;
  relatedSkills: string[];
  learnedFrom?: string;
}

// Memory entry for agent recall
export interface PersonaMemoryEntry {
  id: string;
  type: 'task' | 'insight' | 'pattern' | 'error' | 'success';
  content: string;
  timestamp: Date;
  importance: number; // 0-1
  tags: string[];
  context?: Record<string, unknown>;
}

// Project/workspace data per persona
export interface PersonaProjectData {
  id: string;
  name: string;
  type: string;
  data: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

// Complete persona database
export interface PersonaDatabase {
  personaId: PersonaType;
  skills: PersonaSkillEntry[];
  memories: PersonaMemoryEntry[];
  projects: PersonaProjectData[];
  settings: Record<string, unknown>;
  statistics: {
    tasksCompleted: number;
    totalWorkTime: number;
    skillsLearned: number;
    lastActiveAt: Date;
  };
}

// Store for all persona databases
interface PersonaSkillStore {
  databases: Record<PersonaType, PersonaDatabase>;
  currentPersona: PersonaType;
  
  // Persona selection
  setCurrentPersona: (personaId: PersonaType) => void;
  getCurrentDatabase: () => PersonaDatabase;
  
  // Skill management
  addSkill: (personaId: PersonaType, skill: Omit<PersonaSkillEntry, 'id' | 'usageCount' | 'lastUsed'>) => void;
  updateSkillProficiency: (personaId: PersonaType, skillId: string, proficiency: number) => void;
  useSkill: (personaId: PersonaType, skillId: string) => void;
  getSkillsByCategory: (personaId: PersonaType, category: string) => PersonaSkillEntry[];
  
  // Memory management
  addMemory: (personaId: PersonaType, memory: Omit<PersonaMemoryEntry, 'id' | 'timestamp'>) => void;
  getRecentMemories: (personaId: PersonaType, limit: number) => PersonaMemoryEntry[];
  searchMemories: (personaId: PersonaType, query: string) => PersonaMemoryEntry[];
  
  // Project management
  addProject: (personaId: PersonaType, project: Omit<PersonaProjectData, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateProject: (personaId: PersonaType, projectId: string, data: Partial<PersonaProjectData>) => void;
  getProjects: (personaId: PersonaType) => PersonaProjectData[];
  
  // Statistics
  incrementTasksCompleted: (personaId: PersonaType) => void;
  addWorkTime: (personaId: PersonaType, minutes: number) => void;
  
  // Import/Export
  exportPersonaData: (personaId: PersonaType) => string;
  importPersonaData: (personaId: PersonaType, jsonData: string) => void;
  
  // Reset
  resetPersona: (personaId: PersonaType) => void;
}

// Default skills for each persona
const getDefaultSkills = (personaId: PersonaType): PersonaSkillEntry[] => {
  const config = PERSONA_CONFIGS[personaId];
  const skills: PersonaSkillEntry[] = [];
  
  config.skills.primarySkills.forEach((skill, i) => {
    skills.push({
      id: `${personaId}-skill-${i}`,
      name: skill,
      description: `Core ${config.name} skill`,
      proficiency: 75 + Math.random() * 20,
      lastUsed: new Date(),
      usageCount: Math.floor(Math.random() * 50),
      category: 'primary',
      relatedSkills: [],
    });
  });
  
  config.skills.tools.forEach((tool, i) => {
    skills.push({
      id: `${personaId}-tool-${i}`,
      name: tool,
      description: `Proficient in ${tool}`,
      proficiency: 60 + Math.random() * 30,
      lastUsed: new Date(),
      usageCount: Math.floor(Math.random() * 30),
      category: 'tools',
      relatedSkills: [],
    });
  });
  
  return skills;
};

// Initialize empty database for a persona
const createEmptyDatabase = (personaId: PersonaType): PersonaDatabase => ({
  personaId,
  skills: getDefaultSkills(personaId),
  memories: [],
  projects: [],
  settings: {},
  statistics: {
    tasksCompleted: 0,
    totalWorkTime: 0,
    skillsLearned: getDefaultSkills(personaId).length,
    lastActiveAt: new Date(),
  },
});

// Create all persona databases
const createAllDatabases = (): Record<PersonaType, PersonaDatabase> => ({
  architect: createEmptyDatabase('architect'),
  operator: createEmptyDatabase('operator'),
  creator: createEmptyDatabase('creator'),
  analyst: createEmptyDatabase('analyst'),
  hacker: createEmptyDatabase('hacker'),
});

export const usePersonaSkillStore = create<PersonaSkillStore>()(
  persist(
    (set, get) => ({
      databases: createAllDatabases(),
      currentPersona: 'architect',
      
      setCurrentPersona: (personaId) => {
        set({ currentPersona: personaId });
        // Update last active time
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              statistics: {
                ...state.databases[personaId].statistics,
                lastActiveAt: new Date(),
              },
            },
          },
        }));
      },
      
      getCurrentDatabase: () => {
        const state = get();
        return state.databases[state.currentPersona];
      },
      
      addSkill: (personaId, skill) => {
        const newSkill: PersonaSkillEntry = {
          ...skill,
          id: `${personaId}-skill-${Date.now()}`,
          usageCount: 0,
          lastUsed: new Date(),
        };
        
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              skills: [...state.databases[personaId].skills, newSkill],
              statistics: {
                ...state.databases[personaId].statistics,
                skillsLearned: state.databases[personaId].statistics.skillsLearned + 1,
              },
            },
          },
        }));
      },
      
      updateSkillProficiency: (personaId, skillId, proficiency) => {
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              skills: state.databases[personaId].skills.map(s =>
                s.id === skillId ? { ...s, proficiency: Math.min(100, Math.max(0, proficiency)) } : s
              ),
            },
          },
        }));
      },
      
      useSkill: (personaId, skillId) => {
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              skills: state.databases[personaId].skills.map(s =>
                s.id === skillId ? {
                  ...s,
                  usageCount: s.usageCount + 1,
                  lastUsed: new Date(),
                  proficiency: Math.min(100, s.proficiency + 0.5),
                } : s
              ),
            },
          },
        }));
      },
      
      getSkillsByCategory: (personaId, category) => {
        return get().databases[personaId].skills.filter(s => s.category === category);
      },
      
      addMemory: (personaId, memory) => {
        const newMemory: PersonaMemoryEntry = {
          ...memory,
          id: `${personaId}-mem-${Date.now()}`,
          timestamp: new Date(),
        };
        
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              memories: [...state.databases[personaId].memories, newMemory].slice(-500), // Keep last 500 memories
            },
          },
        }));
      },
      
      getRecentMemories: (personaId, limit) => {
        return get().databases[personaId].memories
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
          .slice(0, limit);
      },
      
      searchMemories: (personaId, query) => {
        const queryLower = query.toLowerCase();
        return get().databases[personaId].memories.filter(m =>
          m.content.toLowerCase().includes(queryLower) ||
          m.tags.some(t => t.toLowerCase().includes(queryLower))
        );
      },
      
      addProject: (personaId, project) => {
        const newProject: PersonaProjectData = {
          ...project,
          id: `${personaId}-proj-${Date.now()}`,
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              projects: [...state.databases[personaId].projects, newProject],
            },
          },
        }));
      },
      
      updateProject: (personaId, projectId, data) => {
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              projects: state.databases[personaId].projects.map(p =>
                p.id === projectId ? { ...p, ...data, updatedAt: new Date() } : p
              ),
            },
          },
        }));
      },
      
      getProjects: (personaId) => {
        return get().databases[personaId].projects;
      },
      
      incrementTasksCompleted: (personaId) => {
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              statistics: {
                ...state.databases[personaId].statistics,
                tasksCompleted: state.databases[personaId].statistics.tasksCompleted + 1,
              },
            },
          },
        }));
      },
      
      addWorkTime: (personaId, minutes) => {
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: {
              ...state.databases[personaId],
              statistics: {
                ...state.databases[personaId].statistics,
                totalWorkTime: state.databases[personaId].statistics.totalWorkTime + minutes,
              },
            },
          },
        }));
      },
      
      exportPersonaData: (personaId) => {
        return JSON.stringify(get().databases[personaId], null, 2);
      },
      
      importPersonaData: (personaId, jsonData) => {
        try {
          const data = JSON.parse(jsonData) as PersonaDatabase;
          if (data.personaId !== personaId) {
            throw new Error('Persona ID mismatch');
          }
          set(state => ({
            databases: {
              ...state.databases,
              [personaId]: data,
            },
          }));
        } catch (e) {
          console.error('Failed to import persona data:', e);
        }
      },
      
      resetPersona: (personaId) => {
        set(state => ({
          databases: {
            ...state.databases,
            [personaId]: createEmptyDatabase(personaId),
          },
        }));
      },
    }),
    {
      name: 'persona-skill-storage',
      partialize: (state) => ({
        databases: state.databases,
        currentPersona: state.currentPersona,
      }),
    }
  )
);

// Helper hook to get current persona's data
export const useCurrentPersonaData = () => {
  const store = usePersonaSkillStore();
  return {
    database: store.databases[store.currentPersona],
    persona: store.currentPersona,
    config: PERSONA_CONFIGS[store.currentPersona],
  };
};

// Helper to get persona theme colors
export const getPersonaTheme = (personaId: PersonaType) => {
  const themes: Record<PersonaType, { primary: string; secondary: string; accent: string; bg: string; bgGradient: string }> = {
    architect: {
      primary: '#3b82f6',
      secondary: '#64748b',
      accent: '#06b6d4',
      bg: '#f8fafc',
      bgGradient: 'from-blue-500/10 via-cyan-500/10 to-slate-500/10',
    },
    operator: {
      primary: '#a855f7',
      secondary: '#22d3ee',
      accent: '#f472b6',
      bg: '#0f172a',
      bgGradient: 'from-purple-500/20 via-cyan-500/10 to-pink-500/10',
    },
    creator: {
      primary: '#f97316',
      secondary: '#84cc16',
      accent: '#ec4899',
      bg: '#fdf6e3',
      bgGradient: 'from-orange-500/10 via-amber-500/10 to-rose-500/10',
    },
    analyst: {
      primary: '#0891b2',
      secondary: '#8b5cf6',
      accent: '#10b981',
      bg: '#f8fafc',
      bgGradient: 'from-cyan-500/10 via-purple-500/10 to-emerald-500/10',
    },
    hacker: {
      primary: '#22c55e',
      secondary: '#14532d',
      accent: '#4ade80',
      bg: '#020617',
      bgGradient: 'from-green-500/10 via-emerald-900/20 to-black',
    },
  };
  
  return themes[personaId];
};
