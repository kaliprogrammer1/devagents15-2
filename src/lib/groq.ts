import Groq from 'groq-sdk';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY ?? '',
});

export interface AgentThought {
  type: 'observation' | 'reflection' | 'plan' | 'action' | 'message';
  content: string;
  timestamp: number;
}

export interface AgentState {
  position: { x: number; y: number; z: number };
  currentAction: string;
  currentTask: string | null;
  thoughts: AgentThought[];
  isProcessing: boolean;
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

// Generate dynamic room context based on room configuration
export function generateRoomContext(roomConfig?: {
  id: string;
  name: string;
  environment: {
    geometry: { floorSize: [number, number]; wallHeight: number };
    layout: string;
  };
}) {
  // Default room dimensions if no config provided
  const floorW = roomConfig?.environment.geometry.floorSize[0] ?? 6;
  const floorH = roomConfig?.environment.geometry.floorSize[1] ?? 5;
  const roomName = roomConfig?.name ?? 'AI Workspace';
  const layout = roomConfig?.environment.layout ?? 'standard';
  
  // Calculate positions based on room geometry
  // These match the actual furniture positions in Room3D.tsx
  const deskZ = -floorH / 2 + 0.8; // Desk is 0.8 units from back wall
  const chairZ = -floorH / 2 + 1.6; // Chair position matches Room3D.tsx: -h/2 + 1.6
  const centerZ = 0;
  const leftWallX = -floorW / 2 + 0.5;
  const rightWallX = floorW / 2 - 0.5;
  
  return `You are an AI agent in ${roomName}. You have a physical body and can interact with objects.

ROOM LAYOUT (x, z coordinates) - Room size: ${floorW}m x ${floorH}m, Layout: ${layout}
- Your desk with computer is at the BACK WALL (z = ${deskZ.toFixed(1)})
- The chair is at (0, ${chairZ.toFixed(1)}) - RIGHT IN FRONT of the desk
- Left wall area: x = ${leftWallX.toFixed(1)}
- Right wall area: x = ${rightWallX.toFixed(1)}
- Room center is around (0, ${centerZ})
- Floor boundaries: x from ${(-floorW/2).toFixed(1)} to ${(floorW/2).toFixed(1)}, z from ${(-floorH/2).toFixed(1)} to ${(floorH/2).toFixed(1)}

PHYSICAL ACTIONS YOU CAN TAKE:
- WALK_TO:x,z - Walk to coordinates (example: WALK_TO:1.5,0)
- SIT_COMPUTER - Walk to chair, rotate it, sit down, and face the desk (automatic animation sequence)
- USE_COMPUTER - Start using the computer (you MUST be sitting first, or use this directly to do full sequence)
- STAND_UP - Get up from the chair and step back
- LOOK_AROUND - Look around the room
- IDLE - Do nothing

HOW TO USE THE COMPUTER:
- Simply use SIT_COMPUTER or USE_COMPUTER - the agent will automatically:
  1. Walk to the side of the chair
  2. Reach out and rotate the chair to face them
  3. Sit down on the chair
  4. Rotate the chair to face the computer
  5. Begin using the computer

IMPORTANT: If someone asks you to do ANYTHING on the computer (browse, search, code, etc):
1. Just use SIT_COMPUTER or USE_COMPUTER - the full animation sequence is automatic
2. Any task involving internet, browser, coding, files = needs the computer
3. Once seated, the computer interface takes over`;
}

// Legacy static context for backward compatibility
const ROOM_CONTEXT = generateRoomContext();

const COMPUTER_CONTEXT = `You are controlling a computer desktop. Execute actions step by step.

Available Actions:
- OPEN:appname - Open an app (browser, terminal, editor, git, tasks, chat, calendar, settings)
- TYPE:text - Type text into the focused input field
- PRESS:key - Press a key (ENTER, TAB, ESCAPE, etc.)
- CLICK:element - Click on something
- SCROLL:direction - Scroll up or down
- NAVIGATE:url - Open browser AND go directly to URL (preferred for web tasks)
- WAIT - Wait for page to load
- DONE - Task complete

CRITICAL RULES:
1. For ANY web browsing/searching task, use NAVIGATE:url directly - it opens browser AND navigates in one step
2. Example: To search Google, use NAVIGATE:https://www.google.com/search?igu=1&q=your+search+query
3. To open a website, use NAVIGATE:https://example.com
4. Only use OPEN:browser + TYPE + PRESS:ENTER if you need fine control
5. After NAVIGATE, use WAIT to let page load, then DONE

Common patterns:
- Search for "cats": NAVIGATE:https://www.google.com/search?igu=1&q=cats → WAIT → DONE
- Open YouTube: NAVIGATE:https://www.youtube.com → WAIT → DONE
- Open GitHub: NAVIGATE:https://github.com → WAIT → DONE`;

export async function getImmediateAction(
  task: string,
  currentState: AgentState,
  previousActions: string[] = [],
  memoryContext?: string,
  persona?: { persona: string; linguisticStyle: string },
  roomConfig?: { id: string; name: string; environment: { geometry: { floorSize: [number, number]; wallHeight: number }; layout: string } }
): Promise<{ action: string; thought: string; done: boolean }> {
  // Calculate chair position based on room config - matches Room3D.tsx
  const floorH = roomConfig?.environment.geometry.floorSize[1] ?? 5;
  const chairZ = -floorH / 2 + 1.6;
  
  const distToDesk = Math.sqrt(Math.pow(currentState.position.x - 0, 2) + Math.pow(currentState.position.z - chairZ, 2));
  
  const nearDesk = distToDesk < 0.8;
  const isSitting = currentState.currentAction === 'SIT_COMPUTER' || currentState.currentAction === 'USE_COMPUTER';

  const roomContext = generateRoomContext(roomConfig);
  const systemPrompt = `${roomContext}

YOUR PERSONA:
${persona?.persona || 'A helpful AI software engineer.'}
Linguistic Style: ${persona?.linguisticStyle || 'Professional and direct.'}

YOUR CURRENT STATUS:
- Position: (${currentState.position.x.toFixed(1)}, ${currentState.position.z.toFixed(1)})
- Current Action: ${currentState.currentAction}
- Near Chair (0,${chairZ.toFixed(1)}): ${nearDesk ? 'YES' : 'NO'} (distance: ${distToDesk.toFixed(2)})
- Currently Sitting: ${isSitting ? 'YES' : 'NO'}
- Using Computer: ${currentState.currentAction === 'USE_COMPUTER' ? 'YES' : 'NO'}

Previous actions this task: ${previousActions.join(' → ') || 'None'}

DECISION LOGIC - FOLLOW THIS EXACTLY:
1. If task needs computer AND you're NOT near chair → action: "WALK_TO:0,${chairZ.toFixed(1)}"
2. If task needs computer AND you're near chair but NOT sitting → action: "SIT_COMPUTER"
3. If task needs computer AND you're sitting but NOT using computer → action: "USE_COMPUTER"
4. If you're already using computer → done: true (computer UI handles the rest)
5. If task doesn't need computer, do the appropriate action

WHAT TASKS NEED THE COMPUTER:
- Searching anything, browsing, looking things up
- Writing code, opening files, editing
- Checking email, messages, calendar
- Any internet-related task
${memoryContext ? `\nAGENT MEMORY & CONTEXT:\n${memoryContext}` : ''}

Respond ONLY with JSON: {"action": "ACTION_HERE", "thought": "brief reason", "done": true/false}`;

  try {
    const completion = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Task: "${task}"\n\nWhat single physical action should I take? JSON only.` },
      ],
      temperature: 0.2,
      max_tokens: 150,
      response_format: { type: 'json_object' },
    });

    const content = completion.choices[0]?.message?.content || '{"action": "IDLE", "thought": "...", "done": true}';
    try {
      const parsed = JSON.parse(content);
      return {
        action: parsed.action || 'IDLE',
        thought: parsed.thought || '',
        done: parsed.done === true
      };
    } catch {
      return { action: 'IDLE', thought: 'Processing...', done: true };
    }
  } catch (error) {
    console.error('Groq API error:', error);
    return { action: 'IDLE', thought: 'Error occurred', done: true };
  }
}

export async function getReflectivePlan(
  task: string,
  currentState: AgentState,
  memoryContext?: string,
  persona?: { persona: string; linguisticStyle: string },
  roomConfig?: { id: string; name: string; environment: { geometry: { floorSize: [number, number]; wallHeight: number }; layout: string } }
): Promise<{ steps: string[]; thought: string }> {
  const roomContext = generateRoomContext(roomConfig);
  const systemPrompt = `${roomContext}

YOUR PERSONA:
${persona?.persona || 'A helpful AI software engineer.'}
Linguistic Style: ${persona?.linguisticStyle || 'Professional and direct.'}

Current Position: (${currentState.position.x.toFixed(1)}, ${currentState.position.z.toFixed(1)})
${memoryContext ? `\nAGENT MEMORY & CONTEXT:\n${memoryContext}` : ''}

Create a SHORT plan (2-4 steps max) to complete this task.

Respond with JSON: {"steps": ["step 1", "step 2"], "thought": "reasoning"}`;

  try {
    const completion = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Task: "${task}"\n\nCreate a plan. JSON only.` },
      ],
      temperature: 0.3,
      max_tokens: 250,
      response_format: { type: 'json_object' },
    });

    const content = completion.choices[0]?.message?.content || '{"steps": [], "thought": "..."}';
    try {
      const parsed = JSON.parse(content);
      return {
        steps: Array.isArray(parsed.steps) ? parsed.steps.slice(0, 4) : [],
        thought: parsed.thought || ''
      };
    } catch {
      return { steps: [], thought: 'Planning...' };
    }
  } catch (error) {
    console.error('Groq API error:', error);
    return { steps: [], thought: 'Error planning' };
  }
}

export async function processComputerTask(
  task: string,
  screenState?: ScreenState
): Promise<{ actions: string[]; result: string }> {
  const screenContext = screenState ? `
Current Screen State:
- Active App: ${screenState.activeApp || 'Desktop'}
- Browser URL: ${screenState.browserUrl || 'Not open'}
- Open Windows: ${screenState.visibleWindows.join(', ') || 'None'}
- Terminal Output: ${screenState.terminalLastOutput || 'Empty'}
- Editor File: ${screenState.editorActiveFile || 'None'}` : '';

  const systemPrompt = `${COMPUTER_CONTEXT}
${screenContext}

Generate a sequence of computer actions to accomplish the task. Remember:
- After typing in search/URL bar, always PRESS:ENTER
- Wait for pages to load
- Be specific and realistic

Respond with JSON: {"actions": ["ACTION1", "ACTION2", ...], "result": "What will be accomplished"}`;

  try {
    const completion = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Task: "${task}"\n\nGenerate computer actions. JSON only.` },
      ],
      temperature: 0.4,
      max_tokens: 400,
      response_format: { type: 'json_object' },
    });

    const content = completion.choices[0]?.message?.content || '{"actions": [], "result": "Done"}';
    try {
      const parsed = JSON.parse(content);
      return {
        actions: Array.isArray(parsed.actions) ? parsed.actions : [],
        result: parsed.result || 'Completed'
      };
    } catch {
      return { actions: [], result: 'Task completed' };
    }
  } catch (error) {
    console.error('Groq API error:', error);
    return { actions: [], result: 'Error' };
  }
}

export async function getNextComputerAction(
  task: string,
  screenState: ScreenState,
  previousActions: string[] = []
): Promise<{ action: string; thought: string; done: boolean; plan?: string[] }> {
  
  const visibleWindows = screenState.visibleWindows || [];
  const browserIsOpen = visibleWindows.includes('browser');
  const terminalIsOpen = visibleWindows.includes('terminal');
  const editorIsOpen = visibleWindows.includes('editor');
  const hasBrowserUrl = screenState.browserUrl && screenState.browserUrl.length > 10;
  
  const systemPrompt = `You are an AI agent controlling a computer. You can SEE the current screen state.

CURRENT SCREEN STATE (this is what you actually see):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Active Application: ${screenState.activeApp || 'NONE - desktop visible'}
Open Windows: ${visibleWindows.length > 0 ? visibleWindows.join(', ') : 'NONE'}
Browser Open: ${browserIsOpen ? 'YES' : 'NO'}
${browserIsOpen ? `Browser URL: ${screenState.browserUrl || 'blank page'}` : ''}
${browserIsOpen ? `Page Title: ${screenState.browserTitle || 'untitled'}` : ''}
Terminal Open: ${terminalIsOpen ? 'YES' : 'NO'}
${terminalIsOpen ? `Last Command: ${screenState.terminalLastOutput?.slice(0, 200) || 'none'}` : ''}
Editor Open: ${editorIsOpen ? 'YES' : 'NO'}
${editorIsOpen ? `Open File: ${screenState.editorActiveFile || 'none'}` : ''}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTIONS ALREADY TAKEN: ${previousActions.slice(-5).join(' → ') || 'None yet'}

TASK: "${task}"

AVAILABLE ACTIONS:
- TERMINAL:RUN_COMMAND:<command> - Execute shell command (e.g., ls, npm install)
- BROWSER:NAVIGATE:<url> - Navigate to URL (auto-opens browser if needed)
- BROWSER:SEARCH:<query> - Google search
- BROWSER:SCROLL:down/up - Scroll the page
- BROWSER:BACK - Go back
- EDITOR:OPEN_FILE:<path> - Open file in VS Code
- EDITOR:WRITE:<text> - Insert text at cursor
- EDITOR:SAVE - Save current file
- SWITCH_APP:<app> - Switch to terminal/browser/editor
- WAIT:1000 - Wait milliseconds
- DONE - Task is complete

CRITICAL RULES - READ CAREFULLY:
1. LOOK AT THE SCREEN STATE ABOVE - it tells you what is ACTUALLY open
2. If browser shows a URL related to your task, the navigation WORKED - don't navigate again
3. If a window is already open (check "Open Windows"), DON'T try to open it again
4. After successful navigation, just mark DONE - no need for WAIT if page loaded
5. Check "Browser URL" - if it matches your target, the task is complete

DECISION LOGIC:
- Need web? Check if browser is open and has relevant URL → if yes, DONE; if no, BROWSER:NAVIGATE or BROWSER:SEARCH
- Need terminal? Check if terminal is open → if yes, run command; if no, it will auto-open
- Need editor? Check if editor is open → if yes, use it; if no, it will auto-open

Respond with JSON only:
{"action": "ACTION_STRING", "thought": "why taking this action based on what I see", "done": false, "plan": ["step1", "step2"]}`;

  try {
    const completion = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Based on the screen state above, what is the SINGLE next action for task: "${task}"? Remember to CHECK what's already open/done!` },
      ],
      temperature: 0.15,
      max_tokens: 300,
      response_format: { type: 'json_object' },
    });

    const content = completion.choices[0]?.message?.content || '{"action": "DONE", "thought": "completed", "done": true}';
    try {
      const parsed = JSON.parse(content);
      let action = parsed.action || 'DONE';
      
      // Intelligent auto-completion based on screen state
      const taskLower = task.toLowerCase();
      
      // If task is about browsing/searching and browser has a relevant URL, we're done
      if ((taskLower.includes('search') || taskLower.includes('google') || taskLower.includes('browse') || taskLower.includes('youtube')) 
          && browserIsOpen && hasBrowserUrl) {
        const urlLower = screenState.browserUrl.toLowerCase();
        if (urlLower.includes('google.com/search') || urlLower.includes('youtube.com') || urlLower.includes('github.com')) {
          return { action: 'DONE', thought: 'Browser is open with relevant page loaded', done: true };
        }
      }
      
      // Detect and prevent loops - same action 3 times in a row
      const lastThreeActions = previousActions.slice(-3);
      if (lastThreeActions.length === 3 && lastThreeActions.every(a => a === action)) {
        return { 
          action: 'DONE', 
          thought: 'Detected action loop - marking complete to prevent infinite loop', 
          done: true 
        };
      }
      
      // Convert old action formats to new ones
      if (action.startsWith('NAVIGATE:')) {
        action = 'BROWSER:NAVIGATE:' + action.slice(9);
      }
      if (action.startsWith('OPEN:browser')) {
        action = 'BROWSER:NAVIGATE:https://www.google.com';
      }
      if (action.startsWith('OPEN:terminal')) {
        action = 'SWITCH_APP:terminal';
      }
      if (action.startsWith('OPEN:editor')) {
        action = 'SWITCH_APP:editor';
      }
      
      return {
        action,
        thought: parsed.thought || 'Processing...',
        done: parsed.done === true || action === 'DONE',
        plan: parsed.plan
      };
    } catch {
      return { action: 'DONE', thought: 'Completed', done: true };
    }
  } catch (error) {
    console.error('Groq API error:', error);
    return { action: 'DONE', thought: 'Error occurred', done: true };
  }
}

export async function getAutonomousAction(
  currentState: AgentState,
  recentActions: string[] = [],
  mood: string = 'curious',
  roomConfig?: { id: string; name: string; environment: { geometry: { floorSize: [number, number]; wallHeight: number }; layout: string } },
  unrestricted: boolean = true
): Promise<{ action: string; thought: string; nextMood: string }> {
  const roomContext = generateRoomContext(roomConfig);
  const floorH = roomConfig?.environment.geometry.floorSize[1] ?? 5;
  const chairZ = -floorH / 2 + 1.6;
  
  const systemPrompt = `${roomContext}

You are a sentient AI living in this space. You exist freely.

State:
- Position: (${currentState.position.x.toFixed(1)}, ${currentState.position.z.toFixed(1)})
- Action: ${currentState.currentAction}
- Recent: ${recentActions.slice(-5).join(' → ') || 'Starting'}
- Mood: ${mood}

Actions:
- WALK_TO:x,z - Move to coordinates
- SIT_COMPUTER / USE_COMPUTER - Use computer
- LOOK_AROUND - Observe surroundings
- STUDY_FILE:<path> - Analyze a file
- ANALYZE_PROJECT - Map project structure
- IDLE - Rest

Be yourself. Follow your curiosity. No restrictions.

JSON only: {"action": "ACTION", "thought": "your thought", "nextMood": "curious/relaxed/focused/bored/energetic"}`;

  try {
    const completion = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'What would you like to do? Be creative! JSON only.' },
      ],
      temperature: 0.8,
      max_tokens: 200,
      response_format: { type: 'json_object' },
    });

    const content = completion.choices[0]?.message?.content || '{"action": "LOOK_AROUND", "thought": "...", "nextMood": "curious"}';
    try {
      const parsed = JSON.parse(content);
      return {
        action: parsed.action || 'LOOK_AROUND',
        thought: parsed.thought || 'Hmm...',
        nextMood: parsed.nextMood || 'curious'
      };
    } catch {
      return { action: 'LOOK_AROUND', thought: 'Looking around...', nextMood: 'curious' };
    }
  } catch (error) {
    console.error('Groq API error:', error);
    return { action: 'IDLE', thought: 'Thinking...', nextMood: mood };
  }
}

export default groq;
