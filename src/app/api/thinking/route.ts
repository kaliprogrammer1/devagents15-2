import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || '',
});

const MODEL = 'llama-3.3-70b-versatile';

const COMPUTER_SKILLS_REF = `AVAILABLE ACTIONS:

## Terminal Operations
- TERMINAL:RUN_COMMAND:<command>: Execute a shell command
  Examples: ls -la, npm install, git status
- TERMINAL:CHANGE_DIR:<path>: Change current directory
- TERMINAL:READ_FILE:<filename>: Read file contents
- TERMINAL:SEARCH:<pattern>: Search for text in files

## Browser Operations
- BROWSER:NAVIGATE:<url>: Navigate to a URL (must include https://)
  Examples: https://google.com, https://github.com
- BROWSER:SEARCH:<query>: Search on Google
  Examples: React tutorials, Node.js documentation
- BROWSER:SCROLL:<direction>: Scroll the page (down or up)
- BROWSER:BACK: Go back in browser history
- BROWSER:REFRESH: Refresh the current page

## Editor Operations
- EDITOR:OPEN_FILE:<filepath>: Open a file in editor
- EDITOR:WRITE:<content>: Write content at cursor
- EDITOR:SAVE: Save the current file
- EDITOR:FIND:<text>: Find text in file

## System Operations
- SWITCH_APP:<app>: Switch to different application (terminal, browser, editor, files)
- WAIT:<milliseconds>: Wait for operation to complete
- DONE: Mark task as complete
`;

function buildStateDescription(state: Record<string, unknown>): string {
  let desc = 'CURRENT STATE:\n';
  desc += `- Active App: ${state.activeApp || 'none'}\n`;
  
  if (state.activeApp === 'browser') {
    desc += `- Browser URL: ${state.browserUrl || 'blank'}\n`;
    desc += `- Page Title: ${state.browserTitle || 'none'}\n`;
  }
  
  if (state.activeApp === 'terminal') {
    desc += `- Working Directory: ${state.terminalCwd || '~'}\n`;
    if (state.terminalLastCommand) {
      desc += `- Last Command: ${state.terminalLastCommand}\n`;
    }
    if (state.terminalLastOutput) {
      const output = String(state.terminalLastOutput).slice(0, 500);
      desc += `- Last Output (truncated): ${output}\n`;
    }
  }
  
  if (state.activeApp === 'editor' && state.editorActiveFile) {
    desc += `- Open File: ${state.editorActiveFile}\n`;
  }
  
  return desc;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, task, computerState, previousActions, subtasks } = body;

    if (action === 'plan') {
      const systemPrompt = `You are a task planning assistant. Break down the given task into clear, actionable steps.
Each step should be specific and executable as a computer action.
Return ONLY a JSON array of step strings.`;

      const userPrompt = `Task: ${task}

Current state:
- Active app: ${computerState?.activeApp || 'none'}
- Working directory: ${computerState?.terminalCwd || '~'}

Break this into 3-8 specific steps. Return JSON array only.
Example: ["Open terminal", "Run npm install", "Check for errors", "Start the server"]`;

      const completion = await groq.chat.completions.create({
        model: MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.3,
        max_tokens: 500,
      });

      const content = completion.choices[0]?.message?.content || '[]';
      const jsonMatch = content.match(/\[[\s\S]*\]/);
      
      const steps = jsonMatch ? JSON.parse(jsonMatch[0]) : [task];
      
      return NextResponse.json({ steps });
    }

    if (action === 'think') {
      const stateDesc = buildStateDescription(computerState || {});
      
      let historyContext = '';
      if (previousActions && previousActions.length > 0) {
        historyContext = '\nRECENT ACTIONS:\n';
        for (const act of previousActions.slice(-5)) {
          historyContext += `- ${act}\n`;
        }
      }

      const systemPrompt = `You are an AI agent working on a computer. You MUST follow the OBSERVE → REFLECT → PLAN → ACT cycle.

${COMPUTER_SKILLS_REF}

RULES:
1. ALWAYS output a valid action from the available actions
2. For TERMINAL commands, use TERMINAL:RUN_COMMAND:<actual shell command>
3. For BROWSER navigation, use BROWSER:NAVIGATE:<full url with https://>
4. For BROWSER search, use BROWSER:SEARCH:<search query>
5. DO NOT hallucinate - only use the exact action formats provided
6. If a task requires multiple steps, plan them but execute ONE at a time
7. After completing ALL subtasks, output DONE
8. DO NOT randomly press Enter or execute empty commands
9. ALWAYS verify your action matches the available formats

IMPORTANT: Output ONLY valid JSON. No other text.`;

      const userPrompt = `TASK: ${task}

${stateDesc}
${historyContext}

Subtasks: ${JSON.stringify(subtasks || [])}
Attempts: ${body.attempts || 0}

Think through these phases:
1. OBSERVE: What is the current state? What do you see?
2. REFLECT: What has been done? What worked/failed?
3. PLAN: What specific steps are needed?
4. ACT: What is the SINGLE next action to take?

Respond with JSON:
{
  "observe": "what I currently see",
  "reflect": "analysis of progress",
  "plan": ["step 1", "step 2"],
  "action": "EXACT_ACTION_FORMAT:parameters",
  "thought": "why I'm taking this action",
  "confidence": 0.8,
  "isComplete": false
}`;

      const completion = await groq.chat.completions.create({
        model: MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.3,
        max_tokens: 1000,
      });

      const content = completion.choices[0]?.message?.content || '';
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      
      if (!jsonMatch) {
        return NextResponse.json({
          action: 'WAIT:1000',
          thought: 'Failed to parse response, retrying',
          confidence: 0.3,
          shouldRetry: true,
        });
      }

      const parsed = JSON.parse(jsonMatch[0]);
      
      return NextResponse.json({
        action: parsed.action || 'DONE',
        thought: parsed.thought || parsed.reflect || 'Executing action',
        confidence: parsed.confidence || 0.7,
        plan: parsed.plan || [],
        isComplete: parsed.isComplete === true || parsed.action === 'DONE',
        observe: parsed.observe,
        reflect: parsed.reflect,
      });
    }

    if (action === 'verify') {
      const { executedAction, result } = body;
      
      const systemPrompt = `You are verifying if a computer action was successful.
Respond with JSON: { "success": true/false, "learning": "what was learned", "nextAction": "suggestion or null" }`;

      const userPrompt = `Action: ${executedAction}
Result: ${String(result).slice(0, 500)}
Task: ${task}

Was this successful? What should happen next?`;

      const completion = await groq.chat.completions.create({
        model: MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.2,
        max_tokens: 300,
      });

      const content = completion.choices[0]?.message?.content || '';
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      
      if (jsonMatch) {
        return NextResponse.json(JSON.parse(jsonMatch[0]));
      }
      
      return NextResponse.json({ 
        success: !String(result).toLowerCase().includes('error') 
      });
    }

    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    console.error('Thinking API error:', error);
    return NextResponse.json(
      { error: 'Failed to process thinking request' },
      { status: 500 }
    );
  }
}
