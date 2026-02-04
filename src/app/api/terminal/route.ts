import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';
import { getWorkspaceManager } from '@/lib/workspaceManager';
import { getDockerManager, DockerWorkspace } from '@/lib/dockerManager';

const execAsync = promisify(exec);

const DEFAULT_WORKSPACE_DIR = process.cwd();
// In a real app, this would be a database or session-based storage
const sessionDirs: Record<string, string> = {};

// Get the effective workspace directory (mounted workspace or default)
async function getWorkspaceDir(): Promise<string> {
  try {
    const manager = await getWorkspaceManager();
    const activeWorkspace = manager.getActiveWorkspace();
    if (activeWorkspace) {
      return activeWorkspace.path;
    }
  } catch (e) {
    console.error('Failed to get workspace manager:', e);
  }
  return DEFAULT_WORKSPACE_DIR;
}

// Get Docker workspace if available
async function getDockerWorkspace(): Promise<DockerWorkspace | null> {
  try {
    const dockerManager = await getDockerManager();
    return dockerManager.getActiveWorkspace();
  } catch {
    return null;
  }
}

const ALLOWED_COMMANDS = [
  'ls', 'dir', 'pwd', 'cd', 'cat', 'head', 'tail', 'echo', 'mkdir', 'touch',
  'rm', 'cp', 'mv', 'grep', 'find', 'wc', 'sort', 'uniq', 'curl', 'wget',
  'node', 'npm', 'npx', 'python', 'python3', 'pip', 'pip3',
  'git', 'which', 'whoami', 'date', 'env', 'printenv', 'export',
  'tree', 'file', 'stat', 'chmod', 'df', 'du', 'tar', 'gzip', 'gunzip',
  'sed', 'awk', 'cut', 'tr', 'xargs', 'tee', 'diff',
  'bun', 'deno', 'pnpm', 'yarn',
];

const BLOCKED_COMMANDS = [
  'sudo', 'su', 'passwd', 'chown', 'useradd', 'userdel', 'groupadd',
  'shutdown', 'reboot', 'halt', 'poweroff', 'init',
  'mount', 'umount', 'fdisk', 'mkfs', 'dd',
  'iptables', 'firewall', 'systemctl', 'service',
  'kill', 'killall', 'pkill',
  'crontab', 'at',
];

function isCommandAllowed(command: string): { allowed: boolean; reason?: string } {
  const trimmed = command.trim();
  const firstWord = trimmed.split(/\s+/)[0].toLowerCase();
  
  for (const blocked of BLOCKED_COMMANDS) {
    if (firstWord === blocked || trimmed.includes(`; ${blocked}`) || trimmed.includes(`&& ${blocked}`) || trimmed.includes(`| ${blocked}`)) {
      return { allowed: false, reason: `Command '${blocked}' is not allowed for security reasons` };
    }
  }

  if (firstWord === 'rm') {
    if (trimmed.includes('-rf /') || trimmed.includes('-rf ~') || trimmed.includes('--no-preserve-root')) {
      return { allowed: false, reason: 'Dangerous rm command blocked' };
    }
  }

  return { allowed: true };
}

async function handleCd(args: string[], currentDir: string, workspaceDir: string): Promise<{ output: string; newDir: string }> {
  const target = args[0] || workspaceDir;
  
  let newPath: string;
  if (target === '~' || target === '$HOME') {
    newPath = workspaceDir;
  } else if (target === '-') {
    newPath = workspaceDir; // Simplified
  } else if (path.isAbsolute(target)) {
    newPath = target;
  } else {
    newPath = path.resolve(currentDir, target);
  }

  try {
    const stat = await fs.stat(newPath);
    if (!stat.isDirectory()) {
      return { output: `cd: ${target}: Not a directory`, newDir: currentDir };
    }
    return { output: '', newDir: newPath };
  } catch {
    return { output: `cd: ${target}: No such file or directory`, newDir: currentDir };
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { command, sessionId = 'default', host = 'localhost', useDocker = false } = body;

    if (!command || typeof command !== 'string') {
      return NextResponse.json({ error: 'Command is required' }, { status: 400 });
    }

    // Get the workspace directory (mounted workspace or default)
    const workspaceDir = await getWorkspaceDir();
    const currentDir = sessionDirs[sessionId] || workspaceDir;

    // Check for Docker execution
    const dockerWorkspace = await getDockerWorkspace();
    if (useDocker && dockerWorkspace) {
      // Execute command inside Docker container
      try {
        const dockerManager = await getDockerManager();
        const result = await dockerManager.execInContainer(dockerWorkspace.containerId, command, currentDir);
        return NextResponse.json({
          output: result.output,
          exitCode: result.exitCode,
          cwd: currentDir,
          docker: true,
          containerId: dockerWorkspace.containerId,
        });
      } catch (error: any) {
        return NextResponse.json({
          output: `Docker exec failed: ${error.message}`,
          exitCode: 1,
          cwd: currentDir,
          docker: true,
        });
      }
    }

    if (host !== 'localhost' && !dockerWorkspace) {
      // Simulation of remote SSH
      return NextResponse.json({
        output: `[${host}] Simulation: Executed "${command}" on remote host.`,
        exitCode: 0,
        cwd: '/root',
      });
    }

    const check = isCommandAllowed(command);
    if (!check.allowed) {
      return NextResponse.json({ output: check.reason || 'Command not allowed', exitCode: 1 });
    }

    const parts = command.trim().split(/\s+/);
    const cmd = parts[0];
    const args = parts.slice(1);

    if (cmd === 'cd') {
      const { output, newDir } = await handleCd(args, currentDir, workspaceDir);
      sessionDirs[sessionId] = newDir;
      return NextResponse.json({
        output: output || `Changed directory to ${newDir}`,
        exitCode: output ? 1 : 0,
        cwd: newDir,
      });
    }

    if (cmd === 'pwd') {
      return NextResponse.json({ output: currentDir, exitCode: 0, cwd: currentDir });
    }

    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: currentDir,
        timeout: 30000,
        maxBuffer: 1024 * 1024 * 10,
        env: {
          ...process.env,
          HOME: workspaceDir,
          PWD: currentDir,
          TERM: 'xterm-256color',
          FORCE_COLOR: '1',
        },
      });

      const output = (stdout + (stderr ? `\n${stderr}` : '')).trim();
      return NextResponse.json({
        output,
        exitCode: 0,
        cwd: currentDir,
      });
    } catch (error: any) {
      const output = (error.stdout || '') + (error.stderr || error.message || 'Command failed');
      return NextResponse.json({
        output: output.trim(),
        exitCode: error.code || 1,
        cwd: currentDir,
      });
    }
  } catch (error) {
    console.error('Terminal API error:', error);
    return NextResponse.json(
      { error: 'Failed to execute command', output: String(error) },
      { status: 500 }
    );
  }
}

export async function GET() {
  const workspaceDir = await getWorkspaceDir();
  const dockerWorkspace = await getDockerWorkspace();
  
  return NextResponse.json({
    workspace: workspaceDir,
    allowedCommands: ALLOWED_COMMANDS,
    docker: dockerWorkspace ? {
      active: true,
      containerId: dockerWorkspace.containerId,
      image: dockerWorkspace.image,
      mountPath: dockerWorkspace.mountPath,
    } : null,
  });
}
