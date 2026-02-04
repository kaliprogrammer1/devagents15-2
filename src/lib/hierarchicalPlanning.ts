import { getImmediateAction, getReflectivePlan } from './groq';
import { agentMemory } from './agentMemory';

interface ThoughtNode {
  id: string;
  thought: string;
  score: number;
  children: ThoughtNode[];
  parent?: string;
  depth: number;
}

export class HierarchicalPlanner {
  private maxDepth = 3;
  private branchFactor = 3;

  async plan(task: string, currentState: any, context: string): Promise<string[]> {
    console.log(`Starting Tree of Thoughts for task: ${task}`);
    
    // Level 0: Root
    const root: ThoughtNode = {
      id: 'root',
      thought: task,
      score: 1.0,
      children: [],
      depth: 0
    };

    // BFS/DFS to build the tree
    let currentLevelNodes = [root];
    
    for (let d = 1; d <= this.maxDepth; d++) {
      const nextLevelNodes: ThoughtNode[] = [];
      
      for (const node of currentLevelNodes) {
        const branches = await this.generateBranches(node, task, currentState, context);
        node.children = branches;
        nextLevelNodes.push(...branches);
      }
      
      // Keep only top nodes to prevent explosion
      currentLevelNodes = nextLevelNodes
        .sort((a, b) => b.score - a.score)
        .slice(0, this.branchFactor);
      
      if (currentLevelNodes.length === 0) break;
    }

    // Find the best leaf node and trace back
    const bestPath = this.getBestPath(root);
    return bestPath;
  }

  private async generateBranches(node: ThoughtNode, task: string, currentState: any, context: string): Promise<ThoughtNode[]> {
    // Simulate branching by asking the LLM to provide alternative strategies
    const prompt = `Task: ${task}
Current Node Thought: ${node.thought}
Context: ${context}
State: ${JSON.stringify(currentState)}

Generate ${this.branchFactor} distinct possible next steps or strategies to achieve the task. 
For each strategy, provide a brief thought and a feasibility score (0.0 to 1.0).
Format: Thought | Score`;

    // In a real implementation, we'd use a specific ToT prompt. 
    // Here we'll use the existing Groq lib but mock the multi-branching for demo purposes
    // or call it multiple times with different temperatures.
    
    const branches: ThoughtNode[] = [];
    
    // For now, let's use the reflective plan to get steps and treat them as a "branch"
    const plan = await getReflectivePlan(task, currentState, context);
    const steps = plan.steps || [];
    
    for (let i = 0; i < Math.min(steps.length, this.branchFactor); i++) {
      branches.push({
        id: `${node.id}-${i}`,
        thought: steps[i],
        score: Math.random() * 0.5 + 0.5, // Mock score
        children: [],
        parent: node.id,
        depth: node.depth + 1
      });
    }

    return branches;
  }

  private getBestPath(root: ThoughtNode): string[] {
    let path: string[] = [];
    let current = root;
    
    while (current.children.length > 0) {
      // Pick the child with the highest score
      const bestChild = current.children.reduce((prev, curr) => (curr.score > prev.score ? curr : prev));
      path.push(bestChild.thought);
      current = bestChild;
    }
    
    return path;
  }
}

export const hierarchicalPlanner = new HierarchicalPlanner();
