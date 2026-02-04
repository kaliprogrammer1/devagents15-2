import { NextResponse } from 'next/server';
import { getImmediateAction, getReflectivePlan, processComputerTask, getAutonomousAction, getNextComputerAction, ScreenState } from '@/lib/groq';
import { createAgentBrain, initializeBaseSkills } from '@/lib/agentBrain';
import { agentMemory, skillManager } from '@/lib/agentMemory';

let skillsInitialized = false;

async function ensureSkillsInitialized() {
  if (!skillsInitialized) {
    await initializeBaseSkills();
    skillsInitialized = true;
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, task, currentState, previousActions, mood, screenState, userId = 'default', roomConfig } = body;

    await ensureSkillsInitialized();

    const brain = createAgentBrain(userId);

    switch (action) {
      case 'immediate': {
        const context = await brain.getContextForTask(task);
        const result = await getImmediateAction(task, currentState, previousActions || [], context, undefined, roomConfig);
        return NextResponse.json(result);
      }

      case 'reflective': {
        const context = await brain.getContextForTask(task);
        const result = await getReflectivePlan(task, currentState, context, undefined, roomConfig);
        return NextResponse.json(result);
      }

      case 'computer': {
        const result = await processComputerTask(task, screenState as ScreenState | undefined);
        return NextResponse.json(result);
      }

      case 'computer_step': {
        const result = await getNextComputerAction(task, screenState as ScreenState, previousActions || []);
        return NextResponse.json(result);
      }

      case 'autonomous': {
        const { unrestricted } = body;
        const result = await getAutonomousAction(currentState, previousActions || [], mood || 'curious', roomConfig, unrestricted ?? true);
        return NextResponse.json(result);
      }

      case 'think': {
        const thinking = await brain.think({
          userId,
          task,
          previousActions: previousActions || [],
          screenState,
        });
        return NextResponse.json(thinking);
      }

      case 'learn': {
        const { insights } = body;
        if (insights && Array.isArray(insights)) {
          await brain.learn(insights);
        }
        return NextResponse.json({ success: true });
      }

      case 'learn_from_task': {
        const { outcome, notes, actions: taskActions } = body;
        await brain.learnFromTask(task, taskActions || previousActions || [], outcome, notes);
        return NextResponse.json({ success: true });
      }

      case 'execute_code': {
        const { language, code } = body;
        const result = await brain.executeCode(language, code);
        return NextResponse.json(result);
      }

      case 'run_tests': {
        const { language, code, testCode } = body;
        const result = await brain.runTests(language, code, testCode);
        return NextResponse.json(result);
      }

      case 'get_stats': {
        const stats = await brain.getAgentStats();
        return NextResponse.json(stats);
      }

      case 'get_skills': {
        const skills = await skillManager.getAllSkills();
        return NextResponse.json({ skills });
      }

      case 'get_memories': {
        const memories = await agentMemory.getRecentMemories(20);
        return NextResponse.json({ memories });
      }

      default:
        return NextResponse.json({ error: 'Unknown action type' }, { status: 400 });
    }
  } catch (error) {
    console.error('Agent API error:', error);
    return NextResponse.json(
      { error: 'Failed to process request', action: 'IDLE', thought: 'Error', done: true },
      { status: 500 }
    );
  }
}
