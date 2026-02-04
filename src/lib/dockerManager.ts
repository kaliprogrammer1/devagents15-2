import { exec, spawn, ChildProcess } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs/promises';

const execAsync = promisify(exec);

export interface DockerWorkspace {
  id: string;
  containerId: string;
  containerName: string;
  image: string;
  mountPath: string;       // Host path mounted into container
  workspaceDir: string;    // Path inside container
  status: 'running' | 'stopped' | 'creating' | 'error';
  ports: Record<number, number>;  // container port -> host port
  env: Record<string, string>;
  createdAt: number;
  projectType?: 'node' | 'python' | 'go' | 'rust' | 'generic';
}

export interface DockerExecResult {
  output: string;
  exitCode: number;
  stderr?: string;
}

export interface DockerConfig {
  defaultImage: string;
  memoryLimit: string;
  cpuLimit: string;
  networkMode: string;
  autoRemove: boolean;
}

// Pre-built development images for different project types
const PROJECT_IMAGES: Record<string, string> = {
  node: 'node:20-alpine',
  python: 'python:3.12-slim',
  go: 'golang:1.22-alpine',
  rust: 'rust:1.75-slim',
  generic: 'alpine:3.19',
};

// Default packages to install per project type
const PROJECT_SETUP: Record<string, string[]> = {
  node: ['npm install'],
  python: ['pip install -r requirements.txt 2>/dev/null || true'],
  go: ['go mod download 2>/dev/null || true'],
  rust: ['cargo fetch 2>/dev/null || true'],
  generic: [],
};

const DEFAULT_CONFIG: DockerConfig = {
  defaultImage: 'node:20-alpine',
  memoryLimit: '2g',
  cpuLimit: '2',
  networkMode: 'bridge',
  autoRemove: false,
};

export class DockerManager {
  private workspaces: Map<string, DockerWorkspace> = new Map();
  private activeWorkspaceId: string | null = null;
  private config: DockerConfig;
  private dockerAvailable: boolean = false;

  constructor(config?: Partial<DockerConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async initialize(): Promise<void> {
    // Check if Docker is available
    try {
      await execAsync('docker info');
      this.dockerAvailable = true;
    } catch {
      this.dockerAvailable = false;
      console.warn('Docker is not available. Docker features will be disabled.');
    }

    // Load existing workspaces from persisted state
    await this.loadWorkspaces();
    
    // Clean up any orphaned containers
    await this.cleanupOrphanedContainers();
  }

  isAvailable(): boolean {
    return this.dockerAvailable;
  }

  private async loadWorkspaces(): Promise<void> {
    try {
      const configPath = path.join(process.cwd(), '.workspaces', 'docker-workspaces.json');
      const data = await fs.readFile(configPath, 'utf-8');
      const workspaces: DockerWorkspace[] = JSON.parse(data);
      
      for (const ws of workspaces) {
        // Verify container still exists
        if (await this.containerExists(ws.containerId)) {
          const status = await this.getContainerStatus(ws.containerId);
          ws.status = status;
          this.workspaces.set(ws.id, ws);
        }
      }
    } catch {
      // No config file yet
    }
  }

  private async saveWorkspaces(): Promise<void> {
    const configDir = path.join(process.cwd(), '.workspaces');
    await fs.mkdir(configDir, { recursive: true });
    const configPath = path.join(configDir, 'docker-workspaces.json');
    const workspaces = Array.from(this.workspaces.values());
    await fs.writeFile(configPath, JSON.stringify(workspaces, null, 2));
  }

  private async containerExists(containerId: string): Promise<boolean> {
    if (!this.dockerAvailable) return false;
    try {
      await execAsync(`docker inspect ${containerId}`);
      return true;
    } catch {
      return false;
    }
  }

  private async getContainerStatus(containerId: string): Promise<'running' | 'stopped' | 'error'> {
    try {
      const { stdout } = await execAsync(
        `docker inspect --format='{{.State.Status}}' ${containerId}`
      );
      const status = stdout.trim();
      if (status === 'running') return 'running';
      if (status === 'exited' || status === 'stopped') return 'stopped';
      return 'error';
    } catch {
      return 'error';
    }
  }

  private async cleanupOrphanedContainers(): Promise<void> {
    if (!this.dockerAvailable) return;
    
    try {
      // Find containers with our label that aren't in our workspace list
      const { stdout } = await execAsync(
        'docker ps -a --filter "label=ai-engineer-workspace" --format "{{.ID}}"'
      );
      const containerIds = stdout.trim().split('\n').filter(Boolean);
      const knownIds = new Set(Array.from(this.workspaces.values()).map(w => w.containerId));
      
      for (const id of containerIds) {
        if (!knownIds.has(id)) {
          try {
            await execAsync(`docker rm -f ${id}`);
            console.log(`Cleaned up orphaned container: ${id}`);
          } catch (e) {
            console.warn(`Failed to clean up container ${id}:`, e);
          }
        }
      }
    } catch {
      // Ignore errors during cleanup
    }
  }

  // Detect project type from directory contents
  async detectProjectType(projectPath: string): Promise<'node' | 'python' | 'go' | 'rust' | 'generic'> {
    const checks: Array<{ file: string; type: 'node' | 'python' | 'go' | 'rust' }> = [
      { file: 'package.json', type: 'node' },
      { file: 'requirements.txt', type: 'python' },
      { file: 'pyproject.toml', type: 'python' },
      { file: 'setup.py', type: 'python' },
      { file: 'go.mod', type: 'go' },
      { file: 'Cargo.toml', type: 'rust' },
    ];

    for (const { file, type } of checks) {
      try {
        await fs.access(path.join(projectPath, file));
        return type;
      } catch {
        continue;
      }
    }

    return 'generic';
  }

  // Create a sandboxed Docker environment for a project
  async createWorkspace(
    hostPath: string,
    options?: {
      name?: string;
      image?: string;
      ports?: Record<number, number>;
      env?: Record<string, string>;
      installDeps?: boolean;
    }
  ): Promise<DockerWorkspace> {
    if (!this.dockerAvailable) {
      throw new Error('Docker is not available');
    }

    const resolvedPath = path.resolve(hostPath);
    
    // Verify path exists
    try {
      const stat = await fs.stat(resolvedPath);
      if (!stat.isDirectory()) {
        throw new Error(`Path is not a directory: ${resolvedPath}`);
      }
    } catch (e: any) {
      if (e.code === 'ENOENT') {
        throw new Error(`Directory does not exist: ${resolvedPath}`);
      }
      throw e;
    }

    const projectType = await this.detectProjectType(resolvedPath);
    const image = options?.image || PROJECT_IMAGES[projectType];
    const containerName = `ai-engineer-${options?.name || path.basename(resolvedPath)}-${Date.now()}`;
    const workspaceDir = '/workspace';

    // Build port mappings
    const ports = options?.ports || {};
    const portArgs = Object.entries(ports)
      .map(([container, host]) => `-p ${host}:${container}`)
      .join(' ');

    // Build environment variables
    const envVars = { ...options?.env, PROJECT_TYPE: projectType };
    const envArgs = Object.entries(envVars)
      .map(([k, v]) => `-e ${k}="${v}"`)
      .join(' ');

    // Create container with workspace mounted
    const createCmd = [
      'docker run -d',
      `--name ${containerName}`,
      `--label ai-engineer-workspace=true`,
      `-v "${resolvedPath}:${workspaceDir}"`,
      `-w ${workspaceDir}`,
      `--memory=${this.config.memoryLimit}`,
      `--cpus=${this.config.cpuLimit}`,
      portArgs,
      envArgs,
      image,
      'tail -f /dev/null', // Keep container running
    ].filter(Boolean).join(' ');

    const { stdout } = await execAsync(createCmd);
    const containerId = stdout.trim();

    const id = `docker-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const workspace: DockerWorkspace = {
      id,
      containerId,
      containerName,
      image,
      mountPath: resolvedPath,
      workspaceDir,
      status: 'running',
      ports,
      env: envVars,
      createdAt: Date.now(),
      projectType,
    };

    // Install dependencies if requested
    if (options?.installDeps !== false) {
      const setupCommands = PROJECT_SETUP[projectType] || [];
      for (const cmd of setupCommands) {
        try {
          await this.execInContainer(containerId, cmd, workspaceDir);
        } catch {
          // Continue even if setup fails
        }
      }
    }

    this.workspaces.set(id, workspace);
    await this.saveWorkspaces();
    return workspace;
  }

  // Execute a command inside a Docker container
  async execInContainer(
    containerId: string,
    command: string,
    workDir?: string
  ): Promise<DockerExecResult> {
    if (!this.dockerAvailable) {
      throw new Error('Docker is not available');
    }

    const workDirArg = workDir ? `-w "${workDir}"` : '';
    const execCmd = `docker exec ${workDirArg} ${containerId} sh -c "${command.replace(/"/g, '\\"')}"`;

    try {
      const { stdout, stderr } = await execAsync(execCmd, {
        timeout: 60000,
        maxBuffer: 1024 * 1024 * 10,
      });
      return {
        output: stdout + (stderr ? `\n${stderr}` : ''),
        exitCode: 0,
        stderr,
      };
    } catch (error: any) {
      return {
        output: error.stdout || error.message || 'Command failed',
        exitCode: error.code || 1,
        stderr: error.stderr,
      };
    }
  }

  // Execute with real-time streaming output
  execInContainerStream(
    containerId: string,
    command: string,
    onData: (data: string) => void,
    workDir?: string
  ): Promise<number> {
    return new Promise((resolve, reject) => {
      if (!this.dockerAvailable) {
        reject(new Error('Docker is not available'));
        return;
      }

      const workDirArg = workDir ? `-w ${workDir}` : '';
      const child = spawn('docker', [
        'exec',
        ...(workDir ? ['-w', workDir] : []),
        containerId,
        'sh', '-c', command,
      ]);

      child.stdout.on('data', (data) => onData(data.toString()));
      child.stderr.on('data', (data) => onData(data.toString()));
      
      child.on('close', (code) => resolve(code || 0));
      child.on('error', reject);
    });
  }

  // Copy files to/from container
  async copyToContainer(containerId: string, hostPath: string, containerPath: string): Promise<void> {
    if (!this.dockerAvailable) {
      throw new Error('Docker is not available');
    }
    await execAsync(`docker cp "${hostPath}" ${containerId}:${containerPath}`);
  }

  async copyFromContainer(containerId: string, containerPath: string, hostPath: string): Promise<void> {
    if (!this.dockerAvailable) {
      throw new Error('Docker is not available');
    }
    await execAsync(`docker cp ${containerId}:${containerPath} "${hostPath}"`);
  }

  // Read file from container
  async readFile(containerId: string, filePath: string): Promise<string> {
    const result = await this.execInContainer(containerId, `cat "${filePath}"`);
    if (result.exitCode !== 0) {
      throw new Error(`Failed to read file: ${result.output}`);
    }
    return result.output;
  }

  // Write file in container
  async writeFile(containerId: string, filePath: string, content: string): Promise<void> {
    // Use base64 to handle special characters safely
    const base64Content = Buffer.from(content).toString('base64');
    const result = await this.execInContainer(
      containerId,
      `echo "${base64Content}" | base64 -d > "${filePath}"`
    );
    if (result.exitCode !== 0) {
      throw new Error(`Failed to write file: ${result.output}`);
    }
  }

  // Run tests inside the container
  async runTests(workspaceId: string, testCommand?: string): Promise<DockerExecResult> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    // Auto-detect test command if not provided
    const cmd = testCommand || this.getDefaultTestCommand(workspace.projectType);
    return this.execInContainer(workspace.containerId, cmd, workspace.workspaceDir);
  }

  private getDefaultTestCommand(projectType?: string): string {
    const commands: Record<string, string> = {
      node: 'npm test',
      python: 'pytest',
      go: 'go test ./...',
      rust: 'cargo test',
      generic: 'echo "No test command configured"',
    };
    return commands[projectType || 'generic'];
  }

  // Run build inside the container
  async runBuild(workspaceId: string, buildCommand?: string): Promise<DockerExecResult> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    const cmd = buildCommand || this.getDefaultBuildCommand(workspace.projectType);
    return this.execInContainer(workspace.containerId, cmd, workspace.workspaceDir);
  }

  private getDefaultBuildCommand(projectType?: string): string {
    const commands: Record<string, string> = {
      node: 'npm run build',
      python: 'python -m py_compile *.py',
      go: 'go build ./...',
      rust: 'cargo build',
      generic: 'echo "No build command configured"',
    };
    return commands[projectType || 'generic'];
  }

  // Start a stopped container
  async startWorkspace(workspaceId: string): Promise<void> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    await execAsync(`docker start ${workspace.containerId}`);
    workspace.status = 'running';
    await this.saveWorkspaces();
  }

  // Stop a running container
  async stopWorkspace(workspaceId: string): Promise<void> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    await execAsync(`docker stop ${workspace.containerId}`);
    workspace.status = 'stopped';
    await this.saveWorkspaces();
  }

  // Remove a workspace and its container
  async removeWorkspace(workspaceId: string): Promise<void> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) return;

    try {
      await execAsync(`docker rm -f ${workspace.containerId}`);
    } catch (e) {
      console.warn(`Failed to remove container: ${e}`);
    }

    this.workspaces.delete(workspaceId);
    if (this.activeWorkspaceId === workspaceId) {
      this.activeWorkspaceId = null;
    }
    await this.saveWorkspaces();
  }

  // Set active Docker workspace
  setActiveWorkspace(workspaceId: string): DockerWorkspace {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }
    this.activeWorkspaceId = workspaceId;
    return workspace;
  }

  getActiveWorkspace(): DockerWorkspace | null {
    if (!this.activeWorkspaceId) return null;
    return this.workspaces.get(this.activeWorkspaceId) || null;
  }

  getWorkspace(id: string): DockerWorkspace | null {
    return this.workspaces.get(id) || null;
  }

  listWorkspaces(): DockerWorkspace[] {
    return Array.from(this.workspaces.values());
  }

  // Get container logs
  async getLogs(workspaceId: string, tail?: number): Promise<string> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    const tailArg = tail ? `--tail ${tail}` : '';
    const { stdout } = await execAsync(`docker logs ${tailArg} ${workspace.containerId}`);
    return stdout;
  }

  // Get container resource usage
  async getStats(workspaceId: string): Promise<{ cpu: string; memory: string; network: string }> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    const { stdout } = await execAsync(
      `docker stats --no-stream --format "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}" ${workspace.containerId}`
    );
    const [cpu, memory, network] = stdout.trim().split('|');
    return { cpu, memory, network };
  }

  // Install additional packages in container
  async installPackage(workspaceId: string, packageName: string): Promise<DockerExecResult> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    const installCommands: Record<string, string> = {
      node: `npm install ${packageName}`,
      python: `pip install ${packageName}`,
      go: `go get ${packageName}`,
      rust: `cargo add ${packageName}`,
      generic: `apk add ${packageName}`,
    };

    const cmd = installCommands[workspace.projectType || 'generic'];
    return this.execInContainer(workspace.containerId, cmd, workspace.workspaceDir);
  }
}

// Singleton instance
let dockerManagerInstance: DockerManager | null = null;

export async function getDockerManager(): Promise<DockerManager> {
  if (!dockerManagerInstance) {
    dockerManagerInstance = new DockerManager();
    await dockerManagerInstance.initialize();
  }
  return dockerManagerInstance;
}
