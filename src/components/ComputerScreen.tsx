"use client";

import { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Minus, Maximize2, Terminal, Globe, FileText, Folder, ChevronRight, Eye, Camera, CheckSquare, MessageSquare, Activity, Monitor, Search, Lock, ChevronLeft, Wifi, Battery, Clock as ClockIcon, Box, Shield, Palette, BarChart3, Skull, Layers, Rocket, Database, Brain } from 'lucide-react';
import { ComputerAction, parseAction, ComputerState } from '@/lib/computerSkills';
import { ScreenState, useWorkspaceStore } from '@/lib/workspaceStore';
import { PersonaType, PERSONA_CONFIGS } from '@/lib/personaApps';
import { getPersonaTheme } from '@/lib/personaSkillStore';

const RealTerminal = dynamic(() => import('./apps/RealTerminal'), { ssr: false });
const RealBrowser = dynamic(() => import('./apps/RealBrowser'), { ssr: false });
const CodeEditor = dynamic(() => import('./apps/CodeEditor'), { ssr: false });
const TodoApp = dynamic(() => import('./apps/TodoApp'), { ssr: false });
const TeamChatApp = dynamic(() => import('./apps/TeamChatApp'), { ssr: false });
const FileExplorerApp = dynamic(() => import('./apps/FileExplorerApp'), { ssr: false });
const AgentProcessManagerApp = dynamic(() => import('./apps/AgentProcessManagerApp'), { ssr: false });

// Persona-specific apps
const ArchitectApp = dynamic(() => import('./apps/ArchitectApp'), { ssr: false });
const OperatorApp = dynamic(() => import('./apps/OperatorApp'), { ssr: false });
const CreatorApp = dynamic(() => import('./apps/CreatorApp'), { ssr: false });
const AnalystApp = dynamic(() => import('./apps/AnalystApp'), { ssr: false });
const HackerApp = dynamic(() => import('./apps/HackerApp'), { ssr: false });

interface ComputerScreenProps {
  isActive: boolean;
  task: string | null;
  onClose: () => void;
  onActionComplete?: (action: string) => void;
  agentActions?: Array<{ type: string; target?: string; content?: string }>;
  isUserMode?: boolean;
  onTypingChange?: (isTyping: boolean) => void;
  onScrollingChange?: (isScrolling: boolean) => void;
  onScreenStateChange?: (state: ScreenState) => void;
  personaId?: PersonaType;
}

type AppType = 'terminal' | 'browser' | 'editor' | 'files' | 'todo' | 'chat' | 'process' | 'persona';

interface WindowState {
  id: string;
  type: AppType;
  title: string;
  isMaximized: boolean;
  isMinimized: boolean;
  zIndex: number;
  position: { x: number; y: number };
  size: { width: number | string; height: number | string };
}

interface TerminalHandle {
  executeCommand: (cmd: string) => Promise<string>;
}

interface BrowserHandle {
  navigate: (url: string) => void;
  getUrl: () => string;
  getTitle: () => string;
  search: (query: string) => void;
  scroll: (direction: 'up' | 'down', amount?: number) => void;
}

interface EditorHandle {
  openFile: (path: string, content?: string) => void;
  saveFile: () => Promise<void>;
  getContent: () => string;
  getCurrentFile: () => string | null;
  insertText: (text: string) => void;
  find: (text: string) => void;
}

interface ScreenCapture {
  timestamp: number;
  activeApp: AppType | null;
  visibleWindows: AppType[];
  browserState: { url: string; title: string } | null;
  terminalState: { lastCommand: string; lastOutput: string; cwd: string } | null;
  editorState: { activeFile: string | null; content: string } | null;
}

export default function ComputerScreen({
  isActive,
  task,
  onClose,
  onActionComplete,
  isUserMode = false,
  onScreenStateChange,
  personaId = 'architect',
}: ComputerScreenProps) {
  const [activeApp, setActiveApp] = useState<AppType | null>(null);
  const [windows, setWindows] = useState<WindowState[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentThought, setCurrentThought] = useState<string>('');
  const [currentPlan, setCurrentPlan] = useState<string[]>([]);
  const [screenCapture, setScreenCapture] = useState<ScreenCapture | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [snapPreview, setSnapPreview] = useState<{ x: number, y: number, w: string, h: string } | null>(null);
  
  const theme = getPersonaTheme(personaId);
  const personaConfig = PERSONA_CONFIGS[personaId];
  
  const containerRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<TerminalHandle | null>(null);
  const browserRef = useRef<BrowserHandle | null>(null);
  const editorRef = useRef<EditorHandle | null>(null);
  const isExecutingRef = useRef(false);
  const previousActionsRef = useRef<string[]>([]);
  const screenStateRef = useRef<ScreenCapture | null>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const computerStateRef = useRef<ComputerState>({
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
  });

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const captureScreenState = useCallback((): ScreenCapture => {
    const capture: ScreenCapture = {
      timestamp: Date.now(),
      activeApp,
      visibleWindows: windows.filter(w => !w.isMinimized).map(w => w.type),
      browserState: null,
      terminalState: null,
      editorState: null,
    };

    if (browserRef.current && windows.some(w => w.type === 'browser' && !w.isMinimized)) {
      capture.browserState = {
        url: browserRef.current.getUrl?.() || computerStateRef.current.browserUrl,
        title: browserRef.current.getTitle?.() || computerStateRef.current.browserTitle,
      };
    }

    if (terminalRef.current && windows.some(w => w.type === 'terminal' && !w.isMinimized)) {
      capture.terminalState = {
        lastCommand: computerStateRef.current.terminalLastCommand,
        lastOutput: computerStateRef.current.terminalLastOutput.slice(-500),
        cwd: computerStateRef.current.terminalCwd,
      };
    }

    if (editorRef.current && windows.some(w => w.type === 'editor' && !w.isMinimized)) {
      capture.editorState = {
        activeFile: editorRef.current.getCurrentFile?.() || null,
        content: editorRef.current.getContent?.()?.slice(0, 1000) || '',
      };
    }

    return capture;
  }, [activeApp, windows]);

  useEffect(() => {
    if (isActive && !isUserMode) {
      captureIntervalRef.current = setInterval(() => {
        const capture = captureScreenState();
        screenStateRef.current = capture;
        setScreenCapture(capture);
        
        if (onScreenStateChange) {
          onScreenStateChange({
            activeApp: capture.activeApp,
            browserUrl: capture.browserState?.url || '',
            browserTitle: capture.browserState?.title || '',
            terminalLastOutput: capture.terminalState?.lastOutput || '',
            editorActiveFile: capture.editorState?.activeFile || null,
            editorContent: capture.editorState?.content || '',
            visibleWindows: capture.visibleWindows,
          });
        }
      }, 200);
    }

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    };
  }, [isActive, isUserMode, captureScreenState, onScreenStateChange]);

  useEffect(() => {
    if (task && !isUserMode && !isExecutingRef.current) {
      initializeAgent(task);
    }
  }, [task, isUserMode]);

  const initializeAgent = async (taskStr: string) => {
    isExecutingRef.current = true;
    setIsProcessing(true);
    previousActionsRef.current = [];

    try {
      const taskLower = taskStr.toLowerCase();
      let initialApp: AppType = 'terminal';
      if (taskLower.includes('search') || taskLower.includes('browse') || 
          taskLower.includes('website') || taskLower.includes('google') ||
          taskLower.includes('youtube') || taskLower.includes('look up') ||
          taskLower.includes('open') && taskLower.includes('http')) {
        initialApp = 'browser';
      } else if (taskLower.includes('code') || taskLower.includes('edit') || 
                 taskLower.includes('file') || taskLower.includes('write')) {
        initialApp = 'editor';
      } else if (taskLower.includes('todo') || taskLower.includes('task')) {
        initialApp = 'todo';
      } else if (taskLower.includes('chat') || taskLower.includes('team') || taskLower.includes('message')) {
        initialApp = 'chat';
      }
      
      openApp(initialApp);
      await new Promise(r => setTimeout(r, 1000));

      await runAgentLoop(taskStr);
    } catch (error) {
      console.error('Agent initialization error:', error);
    } finally {
      isExecutingRef.current = false;
      setIsProcessing(false);
    }
  };

  const runAgentLoop = async (taskStr: string) => {
    let attempts = 0;
    const maxAttempts = 25;
    let consecutiveFailures = 0;

    while (attempts < maxAttempts && consecutiveFailures < 3) {
      attempts++;
      
      await new Promise(r => setTimeout(r, 300));
      const currentScreen = captureScreenState();
      screenStateRef.current = currentScreen;
      setScreenCapture(currentScreen);

      try {
        const thinkRes = await fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'computer_step',
            task: taskStr,
            screenState: {
              activeApp: currentScreen.activeApp,
              visibleWindows: currentScreen.visibleWindows,
              browserUrl: currentScreen.browserState?.url || '',
              browserTitle: currentScreen.browserState?.title || '',
              terminalLastCommand: currentScreen.terminalState?.lastCommand || '',
              terminalLastOutput: currentScreen.terminalState?.lastOutput || '',
              terminalCwd: currentScreen.terminalState?.cwd || '~',
              editorActiveFile: currentScreen.editorState?.activeFile || null,
              editorContent: currentScreen.editorState?.content || '',
            },
            previousActions: previousActionsRef.current.slice(-5),
            attempt: attempts,
          }),
        });

        if (!thinkRes.ok) {
          consecutiveFailures++;
          continue;
        }

        const result = await thinkRes.json();
        consecutiveFailures = 0;
        
        setCurrentThought(result.thought || '');
        if (result.plan && result.plan.length > 0) {
          setCurrentPlan(result.plan);
        }

        if (result.done || result.action === 'DONE') {
          onActionComplete?.('Task completed');
          setCurrentThought('Task completed successfully!');
          break;
        }

        const actionStr = result.action || 'DONE';
        
        if (previousActionsRef.current.slice(-3).every(a => a === actionStr)) {
          setCurrentThought('Detected action loop, trying different approach...');
          await new Promise(r => setTimeout(r, 1000));
          continue;
        }
        
        previousActionsRef.current.push(actionStr);
        const action = parseAction(actionStr);
        await executeAction(action);
        
        onActionComplete?.(actionStr);
        await new Promise(r => setTimeout(r, 800));

      } catch (error) {
        console.error('Agent loop error:', error);
        consecutiveFailures++;
        await new Promise(r => setTimeout(r, 500));
      }
    }
  };

  const executeAction = async (action: ComputerAction) => {
    const currentScreen = screenStateRef.current;
    
    try {
      switch (action.type) {
        case 'TERMINAL': {
          if (!currentScreen?.visibleWindows.includes('terminal')) openApp('terminal');
          else if (activeApp !== 'terminal') focusWindowByType('terminal');
          
          if (action.content) {
            const output = await terminalRef.current?.executeCommand(action.content) || '';
            computerStateRef.current.terminalLastCommand = action.content;
            computerStateRef.current.terminalLastOutput = output;
          }
          break;
        }
        case 'BROWSER': {
          if (!currentScreen?.visibleWindows.includes('browser')) openApp('browser');
          else if (activeApp !== 'browser') focusWindowByType('browser');
          
          if (action.subAction === 'NAVIGATE' && action.content) {
            browserRef.current?.navigate(action.content);
          } else if (action.subAction === 'SEARCH' && action.content) {
            browserRef.current?.search(action.content);
          }
          break;
        }
        case 'EDITOR': {
          if (!currentScreen?.visibleWindows.includes('editor')) openApp('editor');
          else if (activeApp !== 'editor') focusWindowByType('editor');
          
          if (action.subAction === 'OPEN_FILE' && action.target) {
            editorRef.current?.openFile(action.target);
          } else if (action.subAction === 'WRITE' && action.content) {
            editorRef.current?.insertText(action.content);
          } else if (action.subAction === 'SAVE') {
            await editorRef.current?.saveFile();
          }
          break;
        }
        case 'SWITCH_APP': {
          const app = action.target as AppType;
          if (app) openApp(app);
          break;
        }
        case 'WAIT': {
          await new Promise(r => setTimeout(r, action.waitMs || 1000));
          break;
        }
      }
    } catch (error) {
      console.error('Action execution error:', error);
    }
  };

  const openApp = (type: AppType) => {
    const existingWindow = windows.find(w => w.type === type);
    
    if (existingWindow) {
      focusWindow(existingWindow.id);
    } else {
      const titles: Record<AppType, string> = {
        terminal: personaConfig.terminalPrompt.split('@')[0] || 'Terminal',
        browser: 'Browser',
        editor: 'Code Editor',
        files: 'Finder',
        todo: 'Reminders',
        chat: 'Messages',
        process: 'Activity Monitor',
        persona: personaConfig.name,
      };
      
      const newZ = windows.length > 0 ? Math.max(...windows.map(w => w.zIndex)) + 1 : 1;
      
      // Cascade positions
      const offset = (windows.length % 5) * 40;
      
      const newWindow: WindowState = {
        id: `${type}-${Date.now()}`,
        type,
        title: titles[type],
        isMaximized: false,
        isMinimized: false,
        zIndex: newZ,
        position: { x: 100 + offset, y: 100 + offset },
        size: { width: 900, height: 600 }
      };
      
      setWindows(prev => [...prev, newWindow]);
      setActiveApp(type);
      computerStateRef.current.activeApp = type;
    }
  };

  const focusWindow = (id: string) => {
    setWindows(prev => {
      const maxZ = Math.max(...prev.map(w => w.zIndex), 0);
      return prev.map(w => {
        if (w.id === id) {
          setActiveApp(w.type);
          computerStateRef.current.activeApp = w.type;
          return { ...w, isMinimized: false, zIndex: maxZ + 1 };
        }
        return w;
      });
    });
  };

  const focusWindowByType = (type: AppType) => {
    const win = windows.find(w => w.type === type);
    if (win) focusWindow(win.id);
  };

  const toggleMaximize = (id: string) => {
    setWindows(prev => prev.map(w => 
      w.id === id ? { ...w, isMaximized: !w.isMaximized } : w
    ));
  };

  const minimizeWindow = (id: string) => {
    setWindows(prev => prev.map(w => 
      w.id === id ? { ...w, isMinimized: true } : w
    ));
    setActiveApp(null);
  };

  const closeWindow = (id: string) => {
    const closingWindow = windows.find(w => w.id === id);
    setWindows(prev => prev.filter(w => w.id !== id));
    
    if (closingWindow && activeApp === closingWindow.type) {
      const remaining = windows.filter(w => w.id !== id && !w.isMinimized);
      if (remaining.length > 0) {
        focusWindow(remaining[remaining.length - 1].id);
      } else {
        setActiveApp(null);
      }
    }
  };

  const handleDrag = (id: string, info: any) => {
    if (!containerRef.current) return;
    const { x, y } = info.point;
    const { width, height } = containerRef.current.getBoundingClientRect();
    
    const threshold = 50;
    if (y < threshold + 40) { // Top snap (maximize)
      setSnapPreview({ x: 0, y: 0, w: '100%', h: '100%' });
    } else if (x < threshold) { // Left snap
      setSnapPreview({ x: 0, y: 0, w: '50%', h: '100%' });
    } else if (x > width - threshold) { // Right snap
      setSnapPreview({ x: width / 2, y: 0, w: '50%', h: '100%' });
    } else {
      setSnapPreview(null);
    }
  };

  const handleDragEnd = (id: string, info: any) => {
    if (!containerRef.current) return;
    const { x, y } = info.point;
    const { width, height } = containerRef.current.getBoundingClientRect();
    const threshold = 50;

    setWindows(prev => prev.map(w => {
      if (w.id === id) {
        if (y < threshold + 40) {
          return { ...w, isMaximized: true, position: { x: 0, y: 0 } };
        } else if (x < threshold) {
          return { ...w, isMaximized: false, position: { x: 0, y: 0 }, size: { width: '50%', height: '100%' } };
        } else if (x > width - threshold) {
          return { ...w, isMaximized: false, position: { x: width / 2, y: 0 }, size: { width: '50%', height: '100%' } };
        }
        return { ...w, position: { x: info.offset.x + w.position.x, y: info.offset.y + w.position.y } };
      }
      return w;
    }));
    setSnapPreview(null);
  };

  const handleResize = (id: string, info: any) => {
    setWindows(prev => prev.map(w => {
      if (w.id === id) {
        const newWidth = Math.max(400, (w.size.width as number) + info.delta.x);
        const newHeight = Math.max(300, (w.size.height as number) + info.delta.y);
        return { ...w, size: { width: newWidth, height: newHeight } };
      }
      return w;
    }));
  };

  if (!isActive) return null;

  // Get persona-specific icon
  const getPersonaIcon = () => {
    switch (personaId) {
      case 'architect': return Box;
      case 'operator': return Shield;
      case 'creator': return Palette;
      case 'analyst': return BarChart3;
      case 'hacker': return Skull;
      default: return Brain;
    }
  };

  const PersonaIcon = getPersonaIcon();

  // Persona-specific dock apps
  const dockApps = [
    { type: 'persona' as AppType, icon: PersonaIcon, label: personaConfig.name, color: `text-[${theme.primary}]`, isPrimary: true },
    { type: 'files' as AppType, icon: Folder, label: 'Finder', color: 'text-blue-400' },
    { type: 'browser' as AppType, icon: Globe, label: 'Browser', color: 'text-sky-500' },
    { type: 'editor' as AppType, icon: FileText, label: 'Editor', color: 'text-purple-400' },
    { type: 'terminal' as AppType, icon: Terminal, label: 'Terminal', color: 'text-slate-300' },
    { type: 'chat' as AppType, icon: MessageSquare, label: 'Messages', color: 'text-emerald-500' },
    { type: 'todo' as AppType, icon: CheckSquare, label: 'Reminders', color: 'text-rose-500' },
    { type: 'process' as AppType, icon: Activity, label: 'Monitor', color: 'text-amber-500' },
  ];

  // Render persona-specific app content
  const renderPersonaApp = () => {
    switch (personaId) {
      case 'architect': return <ArchitectApp />;
      case 'operator': return <OperatorApp />;
      case 'creator': return <CreatorApp />;
      case 'analyst': return <AnalystApp />;
      case 'hacker': return <HackerApp />;
      default: return <ArchitectApp />;
    }
  };

  // Get wallpaper gradient based on persona
  const getWallpaperGradient = () => {
    switch (personaId) {
      case 'architect': return 'from-blue-500/20 via-cyan-500/20 to-slate-500/20';
      case 'operator': return 'from-purple-500/30 via-pink-500/20 to-cyan-500/20';
      case 'creator': return 'from-orange-500/20 via-amber-500/20 to-rose-500/20';
      case 'analyst': return 'from-cyan-500/20 via-blue-500/20 to-purple-500/20';
      case 'hacker': return 'from-green-500/30 via-emerald-900/30 to-black';
      default: return 'from-indigo-500/20 via-purple-500/20 to-pink-500/20';
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-6 animate-in fade-in zoom-in duration-300">
      <div className="relative w-full h-full max-w-[1440px] max-h-[900px] bg-[#f8fafc]/10 backdrop-blur-2xl rounded-[32px] overflow-hidden shadow-[0_0_100px_rgba(0,0,0,0.5)] border border-white/20 flex flex-col">
        
        {/* Apple Style Top Bar */}
        <div className="h-8 bg-white/10 backdrop-blur-md flex items-center justify-between px-4 z-[1000] border-b border-white/10 select-none">
          <div className="flex items-center gap-4">
            <div className="flex gap-2">
              <button onClick={onClose} className="w-3 h-3 rounded-full bg-rose-500 hover:bg-rose-600 transition-colors shadow-sm" />
              <button className="w-3 h-3 rounded-full bg-amber-500 hover:bg-amber-600 transition-colors shadow-sm" />
              <button className="w-3 h-3 rounded-full bg-emerald-500 hover:bg-emerald-600 transition-colors shadow-sm" />
            </div>
            <div className="flex items-center gap-4 ml-2">
              <span className="text-[11px] font-bold text-white/90">Finder</span>
              <span className="text-[11px] font-medium text-white/60">File</span>
              <span className="text-[11px] font-medium text-white/60">Edit</span>
              <span className="text-[11px] font-medium text-white/60">View</span>
              <span className="text-[11px] font-medium text-white/60">Go</span>
              <span className="text-[11px] font-medium text-white/60">Window</span>
              <span className="text-[11px] font-medium text-white/60">Help</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <Wifi size={14} className="text-white/80" />
              <Battery size={14} className="text-white/80" />
              <Search size={14} className="text-white/80" />
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[11px] font-bold text-white/90">
                {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </div>
        </div>

        <div ref={containerRef} className="flex-1 relative overflow-hidden p-4">
          {/* Snap Preview Overlay */}
          <AnimatePresence>
            {snapPreview && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute z-[40] bg-blue-500/20 border-2 border-blue-500/40 rounded-xl transition-all duration-200 pointer-events-none"
                style={{
                  left: snapPreview.x,
                  top: snapPreview.y,
                  width: snapPreview.w,
                  height: snapPreview.h
                }}
              />
            )}
          </AnimatePresence>

          <AnimatePresence>
            {windows.map(window => !window.isMinimized && (
              <motion.div
                key={window.id}
                drag={!window.isMaximized}
                dragMomentum={false}
                onDragStart={() => focusWindow(window.id)}
                onDrag={(e, info) => handleDrag(window.id, info)}
                onDragEnd={(e, info) => handleDragEnd(window.id, info)}
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ 
                  opacity: 1, 
                  scale: 1, 
                  y: 0,
                  zIndex: window.zIndex,
                  width: window.isMaximized ? 'calc(100% - 32px)' : window.size.width,
                  height: window.isMaximized ? 'calc(100% - 32px)' : window.size.height,
                  x: window.isMaximized ? 0 : window.position.x,
                  y: window.isMaximized ? 0 : window.position.y,
                }}
                exit={{ opacity: 0, scale: 0.9 }}
                className={`absolute rounded-2xl overflow-hidden shadow-2xl border border-white/10 flex flex-col bg-white ${activeApp === window.type ? 'ring-1 ring-blue-500/30' : ''}`}
                onMouseDown={() => focusWindow(window.id)}
              >
                {/* Window Title Bar */}
                <div className="h-11 bg-slate-50/80 backdrop-blur-md flex items-center justify-between px-4 border-b border-slate-200 cursor-default select-none group/title">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5 opacity-0 group-hover/title:opacity-100 transition-opacity">
                      <button 
                        onClick={(e) => { e.stopPropagation(); closeWindow(window.id); }} 
                        className="w-3 h-3 rounded-full bg-rose-500 flex items-center justify-center group/btn"
                      >
                        <X size={8} className="text-rose-900 opacity-0 group-hover/btn:opacity-100" />
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); minimizeWindow(window.id); }}
                        className="w-3 h-3 rounded-full bg-amber-500 flex items-center justify-center group/btn"
                      >
                        <Minus size={8} className="text-amber-900 opacity-0 group-hover/btn:opacity-100" />
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); toggleMaximize(window.id); }}
                        className="w-3 h-3 rounded-full bg-emerald-500 flex items-center justify-center group/btn"
                      >
                        <Maximize2 size={8} className="text-emerald-900 opacity-0 group-hover/btn:opacity-100" />
                      </button>
                    </div>
                    <span className="text-xs font-bold text-slate-800 ml-2">{window.title}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1 bg-slate-200/50 rounded-lg px-2 py-1">
                      <ChevronLeft size={14} className="text-slate-400" />
                      <ChevronRight size={14} className="text-slate-400" />
                    </div>
                  </div>
                </div>
                
                  {/* Window Content */}
                    <div className="flex-1 overflow-hidden relative">
                      <div className={`h-full ${activeApp !== window.type && !isUserMode ? 'pointer-events-none grayscale-[0.3] opacity-90' : ''}`}>
                        {window.type === 'terminal' && <RealTerminal ref={terminalRef as any} disabled={!isUserMode && isProcessing} />}
                        {window.type === 'browser' && <RealBrowser ref={browserRef as any} disabled={!isUserMode && isProcessing} />}
                        {window.type === 'editor' && <CodeEditor ref={editorRef as any} disabled={!isUserMode && isProcessing} />}
                        {window.type === 'todo' && <TodoApp />}
                        {window.type === 'chat' && <TeamChatApp />}
                        {window.type === 'files' && <FileExplorerApp />}
                        {window.type === 'process' && <AgentProcessManagerApp />}
                        {window.type === 'persona' && renderPersonaApp()}
                      </div>
                    </div>

                  {/* Resize Handle */}
                  {!window.isMaximized && (
                    <motion.div
                      drag
                      dragMomentum={false}
                      dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
                      onDrag={(e, info) => handleResize(window.id, info)}
                      className="absolute bottom-0 right-0 w-6 h-6 cursor-nwse-resize z-[50] flex items-center justify-center"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-slate-300 mr-1 mb-1" />
                    </motion.div>
                  )}
                </motion.div>
            ))}
          </AnimatePresence>

            {/* Desktop Wallpaper Feel */}
            <div className={`absolute inset-0 z-[-1] bg-gradient-to-br ${getWallpaperGradient()}`} />
        </div>

        {/* Floating Thinking Panel */}
        {currentThought && !isUserMode && (
          <div className="absolute top-12 left-1/2 -translate-x-1/2 w-full max-w-xl z-[2000] animate-in slide-in-from-top duration-500">
            <div className="bg-white/10 backdrop-blur-2xl border border-white/20 rounded-2xl p-4 shadow-2xl flex items-start gap-4 mx-4">
              <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center flex-shrink-0 border border-purple-500/30">
                <Monitor size={20} className="text-purple-400" />
              </div>
              <div className="flex-1">
                <div className="text-[10px] font-bold text-purple-400 uppercase tracking-widest mb-1">Reasoning System</div>
                <div className="text-sm text-white/90 font-medium leading-relaxed">{currentThought}</div>
                {currentPlan.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {currentPlan.slice(0, 3).map((step, i) => (
                      <div key={i} className="px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-[10px] text-white/50 font-medium">
                        {step}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

          {/* Modern Dock */}
            <div className="h-24 flex items-center justify-center px-8 z-[2000]">
              <div className="bg-white/10 backdrop-blur-3xl border border-white/20 rounded-[28px] p-2 flex items-end gap-1 px-3 shadow-2xl relative group">
                {dockApps.map((app, index) => {
                  const Icon = app.icon;
                  const win = windows.find(w => w.type === app.type);
                  const isOpen = !!win;
                  const isMinimized = win?.isMinimized;
                  const isActiveApp = activeApp === app.type;
                  const isPrimary = 'isPrimary' in app && app.isPrimary;
                  
                  // Get persona-specific colors
                  const getPersonaColors = () => {
                    switch (personaId) {
                      case 'architect': return { icon: 'text-blue-400', bg: 'bg-blue-500/30', ring: 'ring-blue-400/50', glow: 'shadow-blue-500/50' };
                      case 'operator': return { icon: 'text-purple-400', bg: 'bg-purple-500/30', ring: 'ring-purple-400/50', glow: 'shadow-purple-500/50' };
                      case 'creator': return { icon: 'text-orange-400', bg: 'bg-orange-500/30', ring: 'ring-orange-400/50', glow: 'shadow-orange-500/50' };
                      case 'analyst': return { icon: 'text-cyan-400', bg: 'bg-cyan-500/30', ring: 'ring-cyan-400/50', glow: 'shadow-cyan-500/50' };
                      case 'hacker': return { icon: 'text-green-400', bg: 'bg-green-500/30', ring: 'ring-green-400/50', glow: 'shadow-green-500/50' };
                      default: return { icon: 'text-blue-400', bg: 'bg-blue-500/30', ring: 'ring-blue-400/50', glow: 'shadow-blue-500/50' };
                    }
                  };
                  const personaColors = getPersonaColors();
                  
                  return (
                    <div key={app.type} className="flex items-end">
                      {index === 1 && <div className="w-px h-10 bg-white/20 mx-1" />}
                      <button
                        onClick={() => openApp(app.type)}
                        className={`group relative p-3 rounded-2xl transition-all duration-300 hover:scale-125 hover:-translate-y-4 ${
                          isActiveApp ? 'bg-white/20' : 'hover:bg-white/10'
                        } ${isPrimary ? `ring-2 ${personaColors.ring} ${personaColors.bg} shadow-lg ${personaColors.glow} -translate-y-2` : ''}`}
                      >
                        <Icon size={isPrimary ? 32 : 28} className={`${
                          isPrimary ? personaColors.icon : app.color
                        } transition-transform group-hover:scale-110`} />
                        {/* Running indicator dot */}
                        {isOpen && (
                          <div className={`absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full ${
                            isPrimary ? personaColors.icon.replace('text-', 'bg-') : isActiveApp ? 'bg-white' : 'bg-white/40'
                          } ${isActiveApp ? 'scale-150' : ''}`} />
                        )}
                        {/* Persona badge */}
                        {isPrimary && (
                          <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${personaColors.bg} border border-white/30 animate-pulse`} />
                        )}
                        <div className={`absolute -top-10 left-1/2 -translate-x-1/2 ${
                          isPrimary ? `${personaColors.bg} border ${personaColors.ring}` : 'bg-white/10'
                        } backdrop-blur-md text-white text-[9px] font-bold py-1 px-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap`}>
                          {app.label}
                        </div>
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
      </div>
    </div>
  );
}
