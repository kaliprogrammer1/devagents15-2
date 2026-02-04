import * as fs from 'fs/promises';
import * as path from 'path';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface WorkspaceConfig {
  id: string;
  name: string;
  path: string;
  type: 'local' | 'git-clone' | 'git-worktree' | 'docker';
  gitRemote?: string;
  gitBranch?: string;
  createdAt: number;
  lastAccessed: number;
  isActive: boolean;
}

export interface FileVersion {
  id: string;
  filePath: string;
  content: string;
  timestamp: number;
  commitMessage?: string;
  hash: string;
}

export interface WorkspaceSnapshot {
  id: string;
  workspaceId: string;
  name: string;
  timestamp: number;
  gitCommit?: string;
  fileVersions: string[];
}

export interface DiffResult {
  filePath: string;
  hunks: DiffHunk[];
  additions: number;
  deletions: number;
}

export interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: string[];
}

const WORKSPACE_BASE = process.env.WORKSPACE_BASE || path.join(process.cwd(), '.workspaces');
const VERSION_STORE = path.join(WORKSPACE_BASE, '.versions');
const MAX_VERSIONS_PER_FILE = 50;

export class WorkspaceManager {
  private workspaces: Map<string, WorkspaceConfig> = new Map();
  private activeWorkspaceId: string | null = null;
  private fileVersions: Map<string, FileVersion[]> = new Map();

  async initialize(): Promise<void> {
    await fs.mkdir(WORKSPACE_BASE, { recursive: true });
    await fs.mkdir(VERSION_STORE, { recursive: true });
    await this.loadWorkspaces();
  }

  private async loadWorkspaces(): Promise<void> {
    try {
      const configPath = path.join(WORKSPACE_BASE, 'workspaces.json');
      const data = await fs.readFile(configPath, 'utf-8');
      const configs: WorkspaceConfig[] = JSON.parse(data);
      configs.forEach(w => this.workspaces.set(w.id, w));
    } catch {
      // No config file yet
    }
  }

  private async saveWorkspaces(): Promise<void> {
    const configPath = path.join(WORKSPACE_BASE, 'workspaces.json');
    const configs = Array.from(this.workspaces.values());
    await fs.writeFile(configPath, JSON.stringify(configs, null, 2));
  }

  // Mount a local project directory
  async mountLocalProject(projectPath: string, name?: string): Promise<WorkspaceConfig> {
    const resolvedPath = path.resolve(projectPath);
    
    // Verify path exists and is a directory
    const stat = await fs.stat(resolvedPath);
    if (!stat.isDirectory()) {
      throw new Error(`Path is not a directory: ${resolvedPath}`);
    }

    const id = `ws-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const workspace: WorkspaceConfig = {
      id,
      name: name || path.basename(resolvedPath),
      path: resolvedPath,
      type: 'local',
      createdAt: Date.now(),
      lastAccessed: Date.now(),
      isActive: false,
    };

    // Check if it's a git repository
    try {
      await execAsync('git rev-parse --git-dir', { cwd: resolvedPath });
      const { stdout: remote } = await execAsync('git remote get-url origin 2>/dev/null || echo ""', { cwd: resolvedPath });
      const { stdout: branch } = await execAsync('git branch --show-current', { cwd: resolvedPath });
      workspace.gitRemote = remote.trim() || undefined;
      workspace.gitBranch = branch.trim() || 'main';
    } catch {
      // Not a git repo
    }

    this.workspaces.set(id, workspace);
    await this.saveWorkspaces();
    return workspace;
  }

  // Clone a git repository into workspace
  async cloneRepository(gitUrl: string, name?: string, branch?: string): Promise<WorkspaceConfig> {
    const repoName = name || gitUrl.split('/').pop()?.replace('.git', '') || 'repo';
    const targetPath = path.join(WORKSPACE_BASE, `${repoName}-${Date.now()}`);
    
    const cloneCmd = branch 
      ? `git clone --branch ${branch} ${gitUrl} "${targetPath}"`
      : `git clone ${gitUrl} "${targetPath}"`;
    
    await execAsync(cloneCmd);

    const { stdout: currentBranch } = await execAsync('git branch --show-current', { cwd: targetPath });

    const id = `ws-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const workspace: WorkspaceConfig = {
      id,
      name: repoName,
      path: targetPath,
      type: 'git-clone',
      gitRemote: gitUrl,
      gitBranch: currentBranch.trim() || branch || 'main',
      createdAt: Date.now(),
      lastAccessed: Date.now(),
      isActive: false,
    };

    this.workspaces.set(id, workspace);
    await this.saveWorkspaces();
    return workspace;
  }

  // Create a git worktree for safe parallel editing
  async createWorktree(workspaceId: string, branchName: string, baseBranch?: string): Promise<WorkspaceConfig> {
    const sourceWorkspace = this.workspaces.get(workspaceId);
    if (!sourceWorkspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    const worktreePath = path.join(WORKSPACE_BASE, `${sourceWorkspace.name}-${branchName}-${Date.now()}`);
    
    // Create a new branch if baseBranch is provided, otherwise use existing branch
    if (baseBranch) {
      await execAsync(`git worktree add -b ${branchName} "${worktreePath}" ${baseBranch}`, { cwd: sourceWorkspace.path });
    } else {
      // Create worktree with new branch from current HEAD
      await execAsync(`git worktree add -b ${branchName} "${worktreePath}"`, { cwd: sourceWorkspace.path });
    }

    const id = `ws-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const workspace: WorkspaceConfig = {
      id,
      name: `${sourceWorkspace.name} (${branchName})`,
      path: worktreePath,
      type: 'git-worktree',
      gitRemote: sourceWorkspace.gitRemote,
      gitBranch: branchName,
      createdAt: Date.now(),
      lastAccessed: Date.now(),
      isActive: false,
    };

    this.workspaces.set(id, workspace);
    await this.saveWorkspaces();
    return workspace;
  }

  // Set active workspace
  async setActiveWorkspace(workspaceId: string): Promise<WorkspaceConfig> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceId}`);
    }

    // Deactivate previous
    if (this.activeWorkspaceId) {
      const prev = this.workspaces.get(this.activeWorkspaceId);
      if (prev) {
        prev.isActive = false;
      }
    }

    workspace.isActive = true;
    workspace.lastAccessed = Date.now();
    this.activeWorkspaceId = workspaceId;
    await this.saveWorkspaces();
    return workspace;
  }

  getActiveWorkspace(): WorkspaceConfig | null {
    if (!this.activeWorkspaceId) return null;
    return this.workspaces.get(this.activeWorkspaceId) || null;
  }

  getWorkspace(id: string): WorkspaceConfig | null {
    return this.workspaces.get(id) || null;
  }

  listWorkspaces(): WorkspaceConfig[] {
    return Array.from(this.workspaces.values()).sort((a, b) => b.lastAccessed - a.lastAccessed);
  }

  // File versioning
  async saveFileVersion(filePath: string, commitMessage?: string): Promise<FileVersion> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fullPath = path.join(workspace.path, filePath);
    const content = await fs.readFile(fullPath, 'utf-8');
    const hash = this.hashContent(content);

    const version: FileVersion = {
      id: `v-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      filePath,
      content,
      timestamp: Date.now(),
      commitMessage,
      hash,
    };

    // Store version
    const versionPath = path.join(VERSION_STORE, workspace.id, filePath.replace(/\//g, '_'));
    await fs.mkdir(path.dirname(versionPath), { recursive: true });
    
    let versions = this.fileVersions.get(fullPath) || [];
    versions.push(version);
    
    // Keep only last N versions
    if (versions.length > MAX_VERSIONS_PER_FILE) {
      versions = versions.slice(-MAX_VERSIONS_PER_FILE);
    }
    
    this.fileVersions.set(fullPath, versions);
    await fs.writeFile(`${versionPath}.json`, JSON.stringify(versions, null, 2));

    return version;
  }

  async getFileVersions(filePath: string): Promise<FileVersion[]> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) return [];

    const fullPath = path.join(workspace.path, filePath);
    
    // Check cache first
    if (this.fileVersions.has(fullPath)) {
      return this.fileVersions.get(fullPath) || [];
    }

    // Load from disk
    const versionPath = path.join(VERSION_STORE, workspace.id, filePath.replace(/\//g, '_'));
    try {
      const data = await fs.readFile(`${versionPath}.json`, 'utf-8');
      const versions = JSON.parse(data);
      this.fileVersions.set(fullPath, versions);
      return versions;
    } catch {
      return [];
    }
  }

  async rollbackFile(filePath: string, versionId: string): Promise<boolean> {
    const versions = await this.getFileVersions(filePath);
    const version = versions.find(v => v.id === versionId);
    if (!version) {
      throw new Error(`Version not found: ${versionId}`);
    }

    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    // Save current state before rollback
    await this.saveFileVersion(filePath, `Backup before rollback to ${versionId}`);

    // Write the old version
    const fullPath = path.join(workspace.path, filePath);
    await fs.writeFile(fullPath, version.content);
    return true;
  }

  // Git operations on active workspace
  async gitStatus(): Promise<{ staged: string[]; unstaged: string[]; untracked: string[] }> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const { stdout } = await execAsync('git status --porcelain', { cwd: workspace.path });
    const lines = stdout.trim().split('\n').filter(Boolean);
    
    const staged: string[] = [];
    const unstaged: string[] = [];
    const untracked: string[] = [];

    for (const line of lines) {
      const status = line.slice(0, 2);
      const file = line.slice(3);
      
      if (status[0] !== ' ' && status[0] !== '?') {
        staged.push(file);
      }
      if (status[1] !== ' ' && status[1] !== '?') {
        unstaged.push(file);
      }
      if (status === '??') {
        untracked.push(file);
      }
    }

    return { staged, unstaged, untracked };
  }

  async gitStage(files: string[]): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fileArgs = files.map(f => `"${f}"`).join(' ');
    await execAsync(`git add ${fileArgs}`, { cwd: workspace.path });
  }

  async gitUnstage(files: string[]): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fileArgs = files.map(f => `"${f}"`).join(' ');
    await execAsync(`git reset HEAD ${fileArgs}`, { cwd: workspace.path });
  }

  async gitCommit(message: string): Promise<string> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const { stdout } = await execAsync(`git commit -m "${message.replace(/"/g, '\\"')}"`, { cwd: workspace.path });
    const match = stdout.match(/\[[\w-]+ ([a-f0-9]+)\]/);
    return match ? match[1] : '';
  }

  async gitPush(remote: string = 'origin', branch?: string): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const branchArg = branch || workspace.gitBranch || 'HEAD';
    await execAsync(`git push ${remote} ${branchArg}`, { cwd: workspace.path });
  }

  async gitPull(remote: string = 'origin', branch?: string): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const branchArg = branch || workspace.gitBranch || '';
    await execAsync(`git pull ${remote} ${branchArg}`, { cwd: workspace.path });
  }

  async gitCreateBranch(branchName: string, checkout: boolean = true): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    if (checkout) {
      await execAsync(`git checkout -b ${branchName}`, { cwd: workspace.path });
      workspace.gitBranch = branchName;
      await this.saveWorkspaces();
    } else {
      await execAsync(`git branch ${branchName}`, { cwd: workspace.path });
    }
  }

  async gitCheckout(branchName: string): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    await execAsync(`git checkout ${branchName}`, { cwd: workspace.path });
    workspace.gitBranch = branchName;
    await this.saveWorkspaces();
  }

  async gitReset(mode: 'soft' | 'mixed' | 'hard' = 'mixed', ref: string = 'HEAD~1'): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    await execAsync(`git reset --${mode} ${ref}`, { cwd: workspace.path });
  }

  async gitDiff(file?: string): Promise<string> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fileArg = file ? `-- "${file}"` : '';
    const { stdout } = await execAsync(`git diff ${fileArg}`, { cwd: workspace.path });
    return stdout;
  }

  async gitLog(count: number = 10): Promise<Array<{ hash: string; message: string; author: string; date: string }>> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const format = '--format=%H|%s|%an|%ad';
    const { stdout } = await execAsync(`git log ${format} -n ${count}`, { cwd: workspace.path });
    
    return stdout.trim().split('\n').filter(Boolean).map(line => {
      const [hash, message, author, date] = line.split('|');
      return { hash, message, author, date };
    });
  }

  // File operations with versioning
  async readFile(filePath: string): Promise<string> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fullPath = path.join(workspace.path, filePath);
    return fs.readFile(fullPath, 'utf-8');
  }

  async writeFile(filePath: string, content: string, createVersion: boolean = true): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fullPath = path.join(workspace.path, filePath);

    // Save version before writing
    if (createVersion) {
      try {
        await this.saveFileVersion(filePath, `Auto-save before edit`);
      } catch {
        // File might not exist yet
      }
    }

    await fs.mkdir(path.dirname(fullPath), { recursive: true });
    await fs.writeFile(fullPath, content);
  }

  async deleteFile(filePath: string): Promise<void> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    // Save version before delete
    try {
      await this.saveFileVersion(filePath, `Backup before delete`);
    } catch {
      // File might not exist
    }

    const fullPath = path.join(workspace.path, filePath);
    await fs.rm(fullPath, { recursive: true });
  }

  async listFiles(dirPath: string = ''): Promise<Array<{ name: string; path: string; type: 'file' | 'dir'; size?: number }>> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const fullPath = path.join(workspace.path, dirPath);
    const entries = await fs.readdir(fullPath, { withFileTypes: true });

    const result = await Promise.all(entries.map(async entry => {
      const entryPath = path.join(dirPath, entry.name);
      const stat = entry.isFile() ? await fs.stat(path.join(fullPath, entry.name)) : null;
      
      return {
        name: entry.name,
        path: entryPath,
        type: entry.isDirectory() ? 'dir' as const : 'file' as const,
        size: stat?.size,
      };
    }));

    return result.sort((a, b) => {
      if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
  }

  // Execute command in workspace context
  async execInWorkspace(command: string, timeout: number = 30000): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: workspace.path,
        timeout,
        maxBuffer: 1024 * 1024 * 10,
        env: {
          ...process.env,
          PWD: workspace.path,
        },
      });
      return { stdout, stderr, exitCode: 0 };
    } catch (error: any) {
      return {
        stdout: error.stdout || '',
        stderr: error.stderr || error.message || '',
        exitCode: error.code || 1,
      };
    }
  }

  // Run tests in workspace
  async runTests(testCommand?: string): Promise<{ success: boolean; output: string; summary: { passed: number; failed: number; skipped: number } }> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    // Auto-detect test framework if no command provided
    const cmd = testCommand || await this.detectTestCommand();
    const result = await this.execInWorkspace(cmd, 120000);

    const output = result.stdout + result.stderr;
    const summary = this.parseTestOutput(output);

    return {
      success: result.exitCode === 0,
      output,
      summary,
    };
  }

  private async detectTestCommand(): Promise<string> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) return 'echo "No test command"';

    // Check package.json for test script
    try {
      const pkgPath = path.join(workspace.path, 'package.json');
      const pkg = JSON.parse(await fs.readFile(pkgPath, 'utf-8'));
      if (pkg.scripts?.test) {
        return 'npm test';
      }
    } catch {}

    // Check for common test runners
    const testFiles = ['pytest.ini', 'setup.py', 'pyproject.toml', 'Cargo.toml', 'go.mod'];
    for (const file of testFiles) {
      try {
        await fs.access(path.join(workspace.path, file));
        if (file.includes('py')) return 'pytest';
        if (file === 'Cargo.toml') return 'cargo test';
        if (file === 'go.mod') return 'go test ./...';
      } catch {}
    }

    return 'echo "No test framework detected"';
  }

  private parseTestOutput(output: string): { passed: number; failed: number; skipped: number } {
    let passed = 0, failed = 0, skipped = 0;

    // Jest/Vitest pattern
    const jestMatch = output.match(/(\d+) passed|(\d+) failed|(\d+) skipped/gi);
    if (jestMatch) {
      for (const m of jestMatch) {
        const num = parseInt(m);
        if (m.includes('passed')) passed = num;
        if (m.includes('failed')) failed = num;
        if (m.includes('skipped')) skipped = num;
      }
    }

    // pytest pattern
    const pytestMatch = output.match(/(\d+) passed|(\d+) failed|(\d+) skipped/gi);
    if (pytestMatch && passed === 0 && failed === 0) {
      for (const m of pytestMatch) {
        const num = parseInt(m);
        if (m.includes('passed')) passed = num;
        if (m.includes('failed')) failed = num;
        if (m.includes('skipped')) skipped = num;
      }
    }

    return { passed, failed, skipped };
  }

  // Run build in workspace
  async runBuild(buildCommand?: string): Promise<{ success: boolean; output: string; errors: string[] }> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) {
      throw new Error('No active workspace');
    }

    const cmd = buildCommand || await this.detectBuildCommand();
    const result = await this.execInWorkspace(cmd, 300000);

    const output = result.stdout + result.stderr;
    const errors = this.parseBuildErrors(output);

    return {
      success: result.exitCode === 0,
      output,
      errors,
    };
  }

  private async detectBuildCommand(): Promise<string> {
    const workspace = this.getActiveWorkspace();
    if (!workspace) return 'echo "No build command"';

    try {
      const pkgPath = path.join(workspace.path, 'package.json');
      const pkg = JSON.parse(await fs.readFile(pkgPath, 'utf-8'));
      if (pkg.scripts?.build) return 'npm run build';
    } catch {}

    const buildFiles: Record<string, string> = {
      'Makefile': 'make',
      'Cargo.toml': 'cargo build',
      'go.mod': 'go build ./...',
      'build.gradle': './gradlew build',
      'pom.xml': 'mvn package',
    };

    for (const [file, cmd] of Object.entries(buildFiles)) {
      try {
        await fs.access(path.join(workspace.path, file));
        return cmd;
      } catch {}
    }

    return 'echo "No build system detected"';
  }

  private parseBuildErrors(output: string): string[] {
    const errors: string[] = [];
    const lines = output.split('\n');

    for (const line of lines) {
      // TypeScript errors
      if (line.match(/error TS\d+:/)) {
        errors.push(line.trim());
      }
      // ESLint errors
      if (line.match(/\d+:\d+\s+error\s+/)) {
        errors.push(line.trim());
      }
      // Generic error patterns
      if (line.match(/^error(\[E\d+\])?:/i) || line.match(/:\d+:\d+: error:/)) {
        errors.push(line.trim());
      }
    }

    return errors;
  }

  // Remove workspace
  async removeWorkspace(workspaceId: string, deleteFiles: boolean = false): Promise<void> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) return;

    // Remove worktree if applicable
    if (workspace.type === 'git-worktree') {
      try {
        // Find the main repo
        const mainWorkspace = Array.from(this.workspaces.values()).find(
          w => w.type !== 'git-worktree' && w.gitRemote === workspace.gitRemote
        );
        if (mainWorkspace) {
          await execAsync(`git worktree remove "${workspace.path}" --force`, { cwd: mainWorkspace.path });
        }
      } catch (e) {
        console.error('Failed to remove worktree:', e);
      }
    }

    if (deleteFiles && (workspace.type === 'git-clone' || workspace.type === 'git-worktree')) {
      try {
        await fs.rm(workspace.path, { recursive: true, force: true });
      } catch (e) {
        console.error('Failed to delete workspace files:', e);
      }
    }

    this.workspaces.delete(workspaceId);
    if (this.activeWorkspaceId === workspaceId) {
      this.activeWorkspaceId = null;
    }
    await this.saveWorkspaces();
  }

  private hashContent(content: string): string {
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }
}

// Singleton instance
let workspaceManagerInstance: WorkspaceManager | null = null;

export async function getWorkspaceManager(): Promise<WorkspaceManager> {
  if (!workspaceManagerInstance) {
    workspaceManagerInstance = new WorkspaceManager();
    await workspaceManagerInstance.initialize();
  }
  return workspaceManagerInstance;
}
