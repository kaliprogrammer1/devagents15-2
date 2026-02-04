"use client";

import { useState, useCallback, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Send, Brain, Laptop, User as UserIcon, Play, Pause, Zap, Github, Mic, MicOff, Camera, ScreenShare, ChevronLeft, ChevronRight, Database } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import AgentBrainPanel from '@/components/AgentBrainPanel';
import GitHubPanel from '@/components/GitHubPanel';
import MemoryPanel from '@/components/MemoryPanel';
import RoomSelector from '@/components/ui/RoomSelector';
import { ScreenState } from '@/lib/workspaceStore';
import { ROOMS } from '@/lib/rooms';

const Room3D = dynamic(() => import('@/components/Room3D'), { ssr: false });
const ComputerScreen = dynamic(() => import('@/components/ComputerScreen'), { ssr: false });

interface AgentThought {
  type: 'observation' | 'reflection' | 'plan' | 'action';
  content: string;
  timestamp: number;
}

interface ComputerAction {
  type: 'OPEN' | 'TYPE' | 'CLICK' | 'SCROLL' | 'NAVIGATE' | 'CLOSE' | 'PRESS' | 'WAIT' | 'DONE';
  target?: string;
  content?: string;
}

interface AgentState {
  position: { x: number; y: number; z: number };
  rotation: number;
  currentAction: string;
  currentTask: string | null;
  thoughts: AgentThought[];
  isProcessing: boolean;
  isTyping: boolean;
  isScrolling: boolean;
  mood: string;
}

export default function Home() {
  const [agentState, setAgentState] = useState<AgentState>({
    position: { x: 0, y: 0, z: 0.5 },
    rotation: 0,
    currentAction: 'IDLE',
    currentTask: null,
    thoughts: [],
    isProcessing: false,
    isTyping: false,
    isScrolling: false,
    mood: 'curious',
  });

  const [currentRoomId, setCurrentRoomId] = useState(ROOMS[0].id);
  const [showRoomSelector, setShowRoomSelector] = useState(true);
  const roomConfig = ROOMS.find(r => r.id === currentRoomId) || ROOMS[0];

  const [taskInput, setTaskInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [showComputerScreen, setShowComputerScreen] = useState(false);
  const [computerTask, setComputerTask] = useState<string | null>(null);
  const [computerActions, setComputerActions] = useState<ComputerAction[]>([]);
  const [isUserComputerMode, setIsUserComputerMode] = useState(false);
  const [focusOnAgent, setFocusOnAgent] = useState(false);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [previousActions, setPreviousActions] = useState<string[]>([]);
    const [showBrainPanel, setShowBrainPanel] = useState(false);
    const [showGitHubPanel, setShowGitHubPanel] = useState(false);
    const [showMemoryPanel, setShowMemoryPanel] = useState(false);
    const [ciStatus, setCiStatus] = useState<{ failing: boolean; count: number }>({ failing: false, count: 0 });
  const [userId] = useState('default-user');
  const [screenState, setScreenState] = useState<ScreenState>({
    activeApp: null,
    browserUrl: '',
    browserTitle: '',
    terminalLastOutput: '',
    editorActiveFile: null,
    editorContent: '',
    visibleWindows: [],
  });

  // Thoughts panel visibility states
  const [showThoughts, setShowThoughts] = useState(false);
  const thoughtsTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastThoughtTimeRef = useRef<number>(0);

  // Input drawer state
  const [showInputDrawer, setShowInputDrawer] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const recognitionRef = useRef<any>(null);

  const thoughtsEndRef = useRef<HTMLDivElement>(null);
  const isExecutingRef = useRef(false);
  const autonomousIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const agentPositionRef = useRef(agentState.position);
  const screenStateRef = useRef(screenState);

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'speechRecognition' in window)) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).speechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event: any) => {
        const current = event.resultIndex;
        const transcriptText = event.results[current][0].transcript;
        setTranscript(transcriptText);
        
        if (event.results[current].isFinal) {
          setTaskInput(transcriptText);
          processTask(transcriptText);
          setIsListening(false);
        }
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    if (showGitHubPanel) {
      const checkCI = async () => {
        try {
          const res = await fetch('/api/github', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'status', userId }),
          });
          const status = await res.json();
          if (status.connected) {
            const reposRes = await fetch('/api/github', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ action: 'repos', userId }),
            });
            const reposData = await reposRes.json();
            if (reposData.repos && reposData.repos.length > 0) {
              const repo = reposData.repos[0];
              const [owner, name] = repo.full_name.split('/');
              const runsRes = await fetch('/api/github', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'workflow_runs', userId, owner, repo: name }),
              });
              const runsData = await runsRes.json();
              const failing = runsData.runs?.filter((r: any) => r.status === 'completed' && r.conclusion === 'failure') || [];
              setCiStatus({ failing: failing.length > 0, count: failing.length });
            }
          }
        } catch (err) {
          console.error('CI check error:', err);
        }
      };

      checkCI();
      const interval = setInterval(checkCI, 60000);
      return () => clearInterval(interval);
    }
  }, [showGitHubPanel, userId]);

  useEffect(() => {
    thoughtsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [agentState.thoughts]);

  // Auto-collapse thoughts panel after 2 seconds of inactivity
  useEffect(() => {
    if (agentState.thoughts.length > 0) {
      const latestThought = agentState.thoughts[agentState.thoughts.length - 1];
      
      // Show thoughts when new thought arrives
      if (latestThought.timestamp > lastThoughtTimeRef.current) {
        lastThoughtTimeRef.current = latestThought.timestamp;
        setShowThoughts(true);
        
        // Clear existing timeout
        if (thoughtsTimeoutRef.current) {
          clearTimeout(thoughtsTimeoutRef.current);
        }
        
        // Set new timeout to hide after 2 seconds
        thoughtsTimeoutRef.current = setTimeout(() => {
          setShowThoughts(false);
        }, 2000);
      }
    }
    
    return () => {
      if (thoughtsTimeoutRef.current) {
        clearTimeout(thoughtsTimeoutRef.current);
      }
    };
  }, [agentState.thoughts]);

  const addThought = useCallback((type: AgentThought['type'], content: string) => {
    setAgentState(prev => ({
      ...prev,
      thoughts: [...prev.thoughts.slice(-30), { type, content, timestamp: Date.now() }],
    }));
  }, []);

  const executeAction = useCallback(async (actionStr: string) => {
    setAgentState(prev => ({ ...prev, currentAction: actionStr }));
    setPreviousActions(prev => [...prev.slice(-10), actionStr]);

    const actionDuration = actionStr.startsWith('WALK_TO') ? 2000 : 
                           actionStr === 'SIT_COMPUTER' ? 1200 :
                           actionStr === 'USE_COMPUTER' ? 800 : 
                           actionStr === 'LOOK_AROUND' ? 1500 :
                           actionStr === 'STAND_UP' ? 1000 : 
                           actionStr.startsWith('STUDY_FILE:') ? 3000 :
                           actionStr === 'ANALYZE_PROJECT' ? 4000 : 1000;
    
    if (actionStr.startsWith('STUDY_FILE:')) {
      const filePath = actionStr.split(':')[1];
      addThought('action', `Studying file: ${filePath}`);
      try {
        const res = await fetch('/api/files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'read', path: filePath }),
        });
        const data = await res.json();
        if (data.content) {
          await fetch('/api/agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'learn',
              userId,
              insights: [{
                type: 'concept',
                content: `Analyzed file ${filePath}: ${data.content.substring(0, 1000)}`,
                importance: 0.6,
                entities: [filePath],
                relations: [{ target: 'project', type: 'part_of' }]
              }]
            }),
          });
          addThought('reflection', `I've learned a lot from ${filePath}!`);
        }
      } catch (err) {
        console.error('Study error:', err);
      }
    }

    if (actionStr === 'ANALYZE_PROJECT') {
      addThought('action', 'Analyzing project architecture...');
      try {
        const res = await fetch('/api/files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'list', path: '.' }),
        });
        const data = await res.json();
        if (data.files) {
          await fetch('/api/agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'learn',
              userId,
              insights: [{
                type: 'pattern',
                content: `Project Structure: ${JSON.stringify(data.files)}`,
                importance: 0.8,
                entities: ['project_root'],
              }]
            }),
          });
          addThought('reflection', 'Project architecture mapped to my knowledge graph.');
        }
      } catch (err) {
        console.error('Analyze error:', err);
      }
    }

    await new Promise(resolve => setTimeout(resolve, actionDuration));
    return true;
  }, [addThought, userId]);

  const processComputerWork = useCallback(async (task: string) => {
    addThought('action', 'Opening computer...');
    
    setComputerTask(task);
    setIsUserComputerMode(false);
    setShowComputerScreen(true);

    await new Promise(resolve => setTimeout(resolve, 500));

    let isDone = false;
    let iterations = 0;
    const maxIterations = 12;
    const computerActionsList: string[] = [];
    const parsedActions: ComputerAction[] = [];

    while (!isDone && iterations < maxIterations) {
      iterations++;
      
      const currentScreenState = screenStateRef.current;
      
      const actionResponse = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'computer_step',
          task,
          screenState: currentScreenState,
          previousActions: computerActionsList,
        }),
      });

      if (!actionResponse.ok) {
        addThought('action', 'Error getting next action');
        break;
      }

      const result = await actionResponse.json();
      
      if (result.thought) {
        addThought('reflection', result.thought);
      }

      const actionStr = result.action || 'DONE';
      computerActionsList.push(actionStr);
      
      const [type, ...rest] = actionStr.split(':');
      const newAction: ComputerAction = {
        type: type as ComputerAction['type'],
        target: rest[0],
        content: rest.join(':') || rest[0],
      };
      
      parsedActions.push(newAction);
      setComputerActions([...parsedActions]);

      addThought('action', `Computer: ${actionStr}`);
      
      await new Promise(resolve => setTimeout(resolve, 2000));

      isDone = result.done === true || actionStr === 'DONE';
    }

    addThought('action', 'Finished using computer');
  }, [addThought]);

  const processTask = useCallback(async (task: string) => {
    if (isExecutingRef.current) return;

    isExecutingRef.current = true;
    setError(null);
    setFocusOnAgent(true);
    const taskActions: string[] = [];
    let currentAction = 'IDLE';
    
    setAgentState(prev => ({ 
      ...prev, 
      isProcessing: true, 
      currentTask: task, 
      currentAction: 'IDLE'
    }));

    try {
      addThought('observation', `Task: "${task}"`);

        const planResponse = await fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'reflective',
            task,
            currentState: { ...agentState, position: agentPositionRef.current },
            roomConfig: {
              id: roomConfig.id,
              name: roomConfig.name,
              environment: {
                geometry: roomConfig.environment.geometry,
                layout: roomConfig.environment.layout,
              },
            },
          }),
        });

      if (planResponse.ok) {
        const plan = await planResponse.json();
        if (plan.thought) addThought('reflection', plan.thought);
        if (plan.steps?.length > 0) addThought('plan', `Plan: ${plan.steps.join(' → ')}`);
      }

      let isDone = false;
      let iterations = 0;
      const maxIterations = 6;

      while (!isDone && iterations < maxIterations) {
        iterations++;
        
        const currentState = {
          ...agentState,
          position: agentPositionRef.current,
          currentAction: currentAction,
        };
        
          const actionResponse = await fetch('/api/agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'immediate',
              task,
              currentState,
              previousActions: taskActions,
              roomConfig: {
                id: roomConfig.id,
                name: roomConfig.name,
                environment: {
                  geometry: roomConfig.environment.geometry,
                  layout: roomConfig.environment.layout,
                },
              },
            }),
          });

        if (!actionResponse.ok) throw new Error(`Action failed: ${actionResponse.status}`);

        const result = await actionResponse.json();
        
        if (result.thought) addThought('reflection', result.thought);

        const actionStr = result.action || 'IDLE';
        taskActions.push(actionStr);
        currentAction = actionStr;
        
        addThought('action', `Doing: ${actionStr}`);
        await executeAction(actionStr);

        if (actionStr === 'USE_COMPUTER') {
          await processComputerWork(task);
          isDone = true;
        } else {
          isDone = result.done === true;
        }

        if (actionStr === 'IDLE' && iterations > 1) isDone = true;
      }

      addThought('action', '✓ Task completed!');

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      addThought('action', `Error: ${errorMessage}`);
    } finally {
      setAgentState(prev => ({ 
        ...prev, 
        isProcessing: false, 
        currentTask: null 
      }));
      isExecutingRef.current = false;
      setTimeout(() => setFocusOnAgent(false), 2000);
    }
    }, [agentState, addThought, executeAction, processComputerWork, roomConfig]);

  const runAutonomousAction = useCallback(async () => {
    if (isExecutingRef.current || !autonomousMode) return;

    isExecutingRef.current = true;
    setAgentState(prev => ({ ...prev, isProcessing: true }));

    try {
        const response = await fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'autonomous',
            currentState: agentState,
            previousActions,
            mood: agentState.mood,
            roomConfig: {
              id: roomConfig.id,
              name: roomConfig.name,
              environment: {
                geometry: roomConfig.environment.geometry,
                layout: roomConfig.environment.layout,
              },
            },
            unrestricted: true, // Signal for unrestricted mode
          }),
        });

      if (response.ok) {
        const result = await response.json();
        
        addThought('reflection', result.thought);
        setAgentState(prev => ({ ...prev, mood: result.nextMood || 'curious' }));

        const actionStr = result.action || 'LOOK_AROUND';
        addThought('action', `Doing: ${actionStr}`);
        await executeAction(actionStr);

        if (actionStr === 'USE_COMPUTER') {
          await processComputerWork('browse the internet and explore');
        }
      }
    } catch (err) {
      console.error('Autonomous action error:', err);
    } finally {
      setAgentState(prev => ({ ...prev, isProcessing: false }));
      isExecutingRef.current = false;
    }
    }, [agentState, autonomousMode, previousActions, addThought, executeAction, processComputerWork, roomConfig]);

  useEffect(() => {
    if (autonomousMode && !isExecutingRef.current) {
      runAutonomousAction();
      
      autonomousIntervalRef.current = setInterval(() => {
        if (!isExecutingRef.current && autonomousMode) {
          runAutonomousAction();
        }
      }, 8000);
    }

    return () => {
      if (autonomousIntervalRef.current) {
        clearInterval(autonomousIntervalRef.current);
        autonomousIntervalRef.current = null;
      }
    };
  }, [autonomousMode, runAutonomousAction]);

  const toggleAutonomousMode = () => {
    if (autonomousMode) {
      setAutonomousMode(false);
      if (autonomousIntervalRef.current) {
        clearInterval(autonomousIntervalRef.current);
        autonomousIntervalRef.current = null;
      }
      addThought('observation', 'Autonomous mode disabled.');
    } else {
      setAutonomousMode(true);
      addThought('observation', 'Autonomous mode enabled.');
    }
  };

  const handleSubmitTask = (e: React.FormEvent) => {
    e.preventDefault();
    if (taskInput.trim() && !agentState.isProcessing) {
      if (autonomousMode) {
        setAutonomousMode(false);
        if (autonomousIntervalRef.current) {
          clearInterval(autonomousIntervalRef.current);
          autonomousIntervalRef.current = null;
        }
      }
      processTask(taskInput.trim());
      setTaskInput('');
    }
  };

  const handleVisualInput = async (type: 'screen' | 'camera') => {
    addThought('observation', `User is showing ${type}...`);
    setAgentState(prev => ({ ...prev, isProcessing: true }));
    
    const analysisTask = type === 'screen' 
      ? `Analyze my current screen: ${JSON.stringify(screenState)}`
      : 'Look at what I am showing you via camera';
    
    await processTask(analysisTask);
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      setTranscript('');
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const openUserComputer = () => {
    if (agentState.currentAction !== 'USE_COMPUTER') {
      setIsUserComputerMode(true);
      setComputerTask(null);
      setComputerActions([]);
      setShowComputerScreen(true);
    }
  };

  const handleRoomChange = (roomId: string) => {
    setCurrentRoomId(roomId);
    const room = ROOMS.find(r => r.id === roomId);
    if (room) {
      addThought('observation', `Entering workspace: ${room.name}.`);
      if (agentState.currentAction === 'USE_COMPUTER') {
        executeAction('STAND_UP');
      }
    }
  };

  const handleAgentPositionChange = useCallback((pos: { x: number; y: number; z: number }) => {
    agentPositionRef.current = pos;
    setAgentState(prev => ({ ...prev, position: pos }));
  }, []);

  const handleScreenStateChange = useCallback((state: ScreenState) => {
    screenStateRef.current = state;
    setScreenState(state);
  }, []);

  const isAgentUsingComputer = agentState.currentAction === 'USE_COMPUTER' || agentState.currentAction === 'SIT_COMPUTER';

  const moodEmoji: Record<string, string> = {
    curious: '🤔',
    relaxed: '😌',
    focused: '🎯',
    bored: '😑',
    energetic: '⚡',
  };

  return (
    <div className="h-screen w-screen bg-[#f8fafc] text-slate-900 overflow-hidden flex flex-col font-['JetBrains_Mono',monospace]">
        <div className="flex-1 flex overflow-hidden">
          <RoomSelector 
            currentRoomId={currentRoomId}
            onRoomChange={handleRoomChange}
            isOpen={showRoomSelector}
            onToggle={() => setShowRoomSelector(!showRoomSelector)}
          />
          <div className="flex-1 relative w-full">
          <Room3D
            agentState={agentState}
            onAgentPositionChange={handleAgentPositionChange}
            wireframe={false}
            showAgentView={false}
            focusOnAgent={focusOnAgent}
            roomConfig={roomConfig}
          />
          
          {/* Top left controls - simplified */}
          <div className="absolute top-4 left-4 flex gap-2">
            <button
              onClick={openUserComputer}
              disabled={isAgentUsingComputer}
              className={`p-2 rounded-lg backdrop-blur-md border transition-all ${
                isAgentUsingComputer 
                  ? 'bg-red-500/20 border-red-500/30 text-red-600 cursor-not-allowed' 
                  : 'bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60'
              }`}
              title={isAgentUsingComputer ? 'Agent is using computer' : 'Use Computer'}
            >
              <Laptop size={20} />
            </button>
            <button
              onClick={toggleAutonomousMode}
              className={`p-2 rounded-lg backdrop-blur-md border transition-all flex items-center gap-1 ${
                autonomousMode 
                  ? 'bg-amber-500/30 border-amber-500/50 text-amber-700 animate-pulse' 
                  : 'bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60'
              }`}
              title={autonomousMode ? 'Stop Autonomous Mode' : 'Start Autonomous Mode'}
            >
              {autonomousMode ? <Pause size={20} /> : <Play size={20} />}
              <Zap size={14} />
            </button>
            <button
                onClick={() => setShowBrainPanel(true)}
                className="p-2 rounded-lg backdrop-blur-md border bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60 transition-all"
                title="Agent Brain"
              >
                <Brain size={20} />
              </button>
              <button
                onClick={() => setShowMemoryPanel(true)}
                className="p-2 rounded-lg backdrop-blur-md border bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60 transition-all"
                title="Semantic Memory"
              >
                <Database size={20} className="text-purple-500" />
              </button>
              <button
                onClick={() => setShowGitHubPanel(true)}
              className="p-2 rounded-lg backdrop-blur-md border bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60 transition-all relative"
              title="GitHub"
            >
              <Github size={20} />
              {ciStatus.failing && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse border-2 border-white" />
              )}
            </button>
            <button
              onClick={() => setShowSkillRecorder(true)}
              className="p-2 rounded-lg backdrop-blur-md border bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60 transition-all"
              title="Record New Skill"
            >
              <Zap size={20} className="text-purple-500" />
            </button>
          </div>

          {/* Top right - autonomous mode indicator */}
          <div className="absolute top-4 right-4 flex items-center gap-2">
            {autonomousMode && (
              <div className="px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/30 text-amber-700 text-xs flex items-center gap-2">
                <span className="animate-pulse text-amber-500">●</span>
                Autonomous Mode {moodEmoji[agentState.mood] || '🤔'}
              </div>
            )}
          </div>

          {/* Agent status panel */}
          <div className="absolute bottom-4 left-4 bg-white/60 border border-slate-200 backdrop-blur-md rounded-lg p-3 text-xs shadow-sm">
            <div className="flex items-center gap-2 mb-1">
              <UserIcon size={14} className="text-cyan-600" />
              <span className="text-slate-500">Agent Status</span>
              {autonomousMode && <span className="text-amber-600 text-[10px] font-bold">AUTO</span>}
            </div>
            <div className="text-slate-700 font-medium">
              Position: ({agentState.position.x.toFixed(1)}, {agentState.position.z.toFixed(1)})
            </div>
            <div className="text-slate-700 font-medium">Action: {agentState.currentAction}</div>
            <div className="text-slate-500">Mood: {moodEmoji[agentState.mood]} {agentState.mood}</div>
            {agentState.currentTask && (
              <div className="text-cyan-600 mt-1 truncate max-w-48 font-medium">Task: {agentState.currentTask}</div>
            )}
            {agentState.isProcessing && (
              <div className="text-amber-600 mt-1 flex items-center gap-1 font-medium">
                <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                Processing...
              </div>
            )}
          </div>

          {error && (
            <div className="absolute top-16 left-4 bg-red-500/20 border border-red-500/50 text-red-300 px-3 py-2 rounded-lg text-sm max-w-sm">
              {error}
            </div>
          )}

          {/* Left sliding drawer for input controls */}
          <div className="absolute left-0 top-1/2 -translate-y-1/2 z-30">
            <AnimatePresence>
              {showInputDrawer && (
                <motion.div
                  initial={{ x: -100, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -100, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="bg-white/10 backdrop-blur-2xl border border-white/10 rounded-r-2xl p-3 shadow-2xl flex flex-col gap-3"
                >
                  <button
                    onClick={toggleListening}
                    className={`p-3 rounded-xl transition-all ${
                      isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-white/10 text-white/70 hover:text-white hover:bg-white/20'
                    }`}
                    title={isListening ? 'Stop Listening' : 'Voice Command'}
                  >
                    {isListening ? <Mic size={22} /> : <MicOff size={22} />}
                  </button>

                  <button
                    onClick={() => handleVisualInput('screen')}
                    className="p-3 bg-white/10 text-white/70 hover:text-white hover:bg-white/20 transition-all rounded-xl"
                    title="Show Screen"
                  >
                    <ScreenShare size={22} />
                  </button>

                  <button
                    onClick={() => handleVisualInput('camera')}
                    className="p-3 bg-white/10 text-white/70 hover:text-white hover:bg-white/20 transition-all rounded-xl"
                    title="Show via Camera"
                  >
                    <Camera size={22} />
                  </button>

                  {isListening && (
                    <div className="px-2 py-1 text-xs text-white/80 max-w-[120px] truncate">
                      {transcript || 'Listening...'}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Drawer toggle */}
            <button
              onClick={() => setShowInputDrawer(!showInputDrawer)}
              className={`absolute top-1/2 -translate-y-1/2 ${showInputDrawer ? 'left-full' : 'left-0'} bg-white/10 backdrop-blur-xl border border-white/10 border-l-0 py-4 px-1.5 rounded-r-xl text-white/40 hover:text-white/70 transition-all`}
            >
              {showInputDrawer ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
            </button>
          </div>
        </div>
      </div>

        {/* Floating Glassmorphism Agent Thoughts Panel - Auto-collapsing */}
        <AnimatePresence>
          {showThoughts && agentState.thoughts.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.3 }}
              className="absolute bottom-28 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 z-20 pointer-events-none"
            >
              <div className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-2xl pointer-events-auto max-h-64 overflow-hidden">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center">
                    <Brain size={14} className="text-cyan-400" />
                  </div>
                  <span className="text-white/60 text-xs font-medium uppercase tracking-wider">Agent Thoughts</span>
                  {agentState.isProcessing && (
                    <div className="ml-auto flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                      <span className="text-cyan-400/60 text-[10px]">Processing</span>
                    </div>
                  )}
                </div>
                <div className="space-y-2 overflow-y-auto max-h-44 pr-1 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                  {agentState.thoughts.slice(-5).map((thought, i) => (
                    <div
                      key={i}
                      className={`p-2.5 rounded-xl text-sm backdrop-blur-sm ${
                        thought.type === 'observation'
                          ? 'bg-blue-500/10 border-l-2 border-blue-400'
                          : thought.type === 'reflection'
                          ? 'bg-purple-500/10 border-l-2 border-purple-400'
                          : thought.type === 'plan'
                          ? 'bg-amber-500/10 border-l-2 border-amber-400'
                          : 'bg-emerald-500/10 border-l-2 border-emerald-400'
                      }`}
                    >
                      <div className="text-white/40 text-[10px] mb-0.5 uppercase tracking-wider">
                        {thought.type}
                      </div>
                      <div className="text-white/90 text-xs leading-relaxed">{thought.content}</div>
                    </div>
                  ))}
                  <div ref={thoughtsEndRef} />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="border-t border-white/10 bg-[#12121a]/80 backdrop-blur-xl p-4">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmitTask} className="flex gap-3">
              <input
                type="text"
                value={taskInput}
                onChange={e => setTaskInput(e.target.value)}
                placeholder={autonomousMode ? "Give a task to interrupt autonomous mode..." : "Give the agent a task... (e.g., 'Search for React tutorials', 'Open the browser')"}
                className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-white/40 outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 transition-all"
                disabled={agentState.isProcessing && !autonomousMode}
              />
              <button
                type="submit"
                disabled={(agentState.isProcessing && !autonomousMode) || !taskInput.trim()}
                className="bg-cyan-500/20 hover:bg-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed text-cyan-300 px-6 py-3 rounded-lg flex items-center gap-2 transition-all"
              >
                {agentState.isProcessing && !autonomousMode ? (
                  <div className="w-5 h-5 border-2 border-cyan-300/30 border-t-cyan-300 rounded-full animate-spin" />
                ) : (
                  <Send size={20} />
                )}
                <span>Send</span>
              </button>
            </form>
          </div>
        </div>

        <ComputerScreen
          isActive={showComputerScreen}
          task={computerTask}
          agentActions={computerActions}
          isUserMode={isUserComputerMode}
          personaId={currentRoomId as 'architect' | 'operator' | 'creator' | 'analyst' | 'hacker'}
          onClose={() => {
            setShowComputerScreen(false);
            setComputerTask(null);
            setComputerActions([]);
            setIsUserComputerMode(false);
          }}
          onActionComplete={(action) => {
            addThought('action', action);
          }}
          onTypingChange={(isTyping) => {
            setAgentState(prev => ({ ...prev, isTyping }));
          }}
          onScrollingChange={(isScrolling) => {
            setAgentState(prev => ({ ...prev, isScrolling }));
          }}
          onScreenStateChange={handleScreenStateChange}
          />

      <AgentBrainPanel
        isOpen={showBrainPanel}
        onClose={() => setShowBrainPanel(false)}
        userId={userId}
      />

      <GitHubPanel
          isOpen={showGitHubPanel}
          onClose={() => setShowGitHubPanel(false)}
          userId={userId}
        />

        <MemoryPanel
          isOpen={showMemoryPanel}
          onClose={() => setShowMemoryPanel(false)}
          userId={userId}
        />
      </div>
    );
  }
