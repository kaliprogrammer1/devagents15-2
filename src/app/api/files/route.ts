import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { getWorkspaceManager, WorkspaceConfig } from '@/lib/workspaceManager';
import { getDockerManager, DockerWorkspace } from '@/lib/dockerManager';

const DEFAULT_ROOT = process.cwd();

// Get the effective workspace root (mounted workspace or default)
async function getWorkspaceRoot(): Promise<string> {
  try {
    const manager = await getWorkspaceManager();
    const activeWorkspace = manager.getActiveWorkspace();
    if (activeWorkspace) {
      return activeWorkspace.path;
    }
  } catch (e) {
    console.error('Failed to get workspace manager:', e);
  }
  return DEFAULT_ROOT;
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

// Simple unified diff parser and applier
interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  changes: Array<{ type: 'context' | 'add' | 'remove'; content: string }>;
}

function parseUnifiedDiff(diffText: string): DiffHunk[] {
  const hunks: DiffHunk[] = [];
  const lines = diffText.split('\n');
  let currentHunk: DiffHunk | null = null;

  for (const line of lines) {
    // Match hunk header: @@ -1,3 +1,4 @@
    const hunkMatch = line.match(/^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@/);
    if (hunkMatch) {
      if (currentHunk) hunks.push(currentHunk);
      currentHunk = {
        oldStart: parseInt(hunkMatch[1], 10),
        oldLines: parseInt(hunkMatch[2] || '1', 10),
        newStart: parseInt(hunkMatch[3], 10),
        newLines: parseInt(hunkMatch[4] || '1', 10),
        changes: [],
      };
      continue;
    }

    if (!currentHunk) continue;

    if (line.startsWith('+') && !line.startsWith('+++')) {
      currentHunk.changes.push({ type: 'add', content: line.slice(1) });
    } else if (line.startsWith('-') && !line.startsWith('---')) {
      currentHunk.changes.push({ type: 'remove', content: line.slice(1) });
    } else if (line.startsWith(' ') || line === '') {
      currentHunk.changes.push({ type: 'context', content: line.startsWith(' ') ? line.slice(1) : line });
    }
  }

  if (currentHunk) hunks.push(currentHunk);
  return hunks;
}

function applyDiff(originalContent: string, hunks: DiffHunk[]): { success: boolean; content: string; error?: string } {
  const lines = originalContent.split('\n');
  
  // Apply hunks in reverse order to preserve line numbers
  const sortedHunks = [...hunks].sort((a, b) => b.oldStart - a.oldStart);
  
  for (const hunk of sortedHunks) {
    const startIndex = hunk.oldStart - 1;
    
    // Verify context lines match
    let oldLineIndex = startIndex;
    let contextMatches = true;
    
    for (const change of hunk.changes) {
      if (change.type === 'context' || change.type === 'remove') {
        if (lines[oldLineIndex] !== change.content) {
          // Allow fuzzy matching for whitespace differences
          if (lines[oldLineIndex]?.trim() !== change.content.trim()) {
            contextMatches = false;
            break;
          }
        }
        oldLineIndex++;
      }
    }
    
    if (!contextMatches) {
      return {
        success: false,
        content: originalContent,
        error: `Context mismatch at line ${hunk.oldStart}. The file may have been modified.`,
      };
    }
    
    // Apply the hunk
    const newLines: string[] = [];
    for (const change of hunk.changes) {
      if (change.type === 'context' || change.type === 'add') {
        newLines.push(change.content);
      }
    }
    
    // Calculate how many lines to remove
    const removeCount = hunk.changes.filter(c => c.type === 'context' || c.type === 'remove').length;
    
    // Replace the old lines with new lines
    lines.splice(startIndex, removeCount, ...newLines);
  }
  
  return { success: true, content: lines.join('\n') };
}

// Generate unified diff between two strings
function generateDiff(oldContent: string, newContent: string, filePath: string): string {
  const oldLines = oldContent.split('\n');
  const newLines = newContent.split('\n');
  
  // Simple LCS-based diff (for small files)
  const diff: string[] = [];
  diff.push(`--- a/${filePath}`);
  diff.push(`+++ b/${filePath}`);
  
  // Find differences using a simple algorithm
  let oldIdx = 0;
  let newIdx = 0;
  let hunkOldStart = 1;
  let hunkNewStart = 1;
  let hunkChanges: string[] = [];
  let inHunk = false;
  
  while (oldIdx < oldLines.length || newIdx < newLines.length) {
    if (oldIdx < oldLines.length && newIdx < newLines.length && oldLines[oldIdx] === newLines[newIdx]) {
      if (inHunk) {
        hunkChanges.push(` ${oldLines[oldIdx]}`);
      }
      oldIdx++;
      newIdx++;
    } else {
      if (!inHunk) {
        inHunk = true;
        hunkOldStart = oldIdx + 1;
        hunkNewStart = newIdx + 1;
        // Add context before
        const contextStart = Math.max(0, oldIdx - 3);
        for (let i = contextStart; i < oldIdx; i++) {
          hunkChanges.push(` ${oldLines[i]}`);
        }
        hunkOldStart = contextStart + 1;
        hunkNewStart = contextStart + 1;
      }
      
      // Look ahead to find next match
      let foundMatch = false;
      for (let lookAhead = 1; lookAhead <= 5 && !foundMatch; lookAhead++) {
        if (oldIdx + lookAhead < oldLines.length && newIdx < newLines.length && 
            oldLines[oldIdx + lookAhead] === newLines[newIdx]) {
          // Lines were removed
          for (let i = 0; i < lookAhead; i++) {
            hunkChanges.push(`-${oldLines[oldIdx + i]}`);
          }
          oldIdx += lookAhead;
          foundMatch = true;
        } else if (newIdx + lookAhead < newLines.length && oldIdx < oldLines.length && 
                   newLines[newIdx + lookAhead] === oldLines[oldIdx]) {
          // Lines were added
          for (let i = 0; i < lookAhead; i++) {
            hunkChanges.push(`+${newLines[newIdx + i]}`);
          }
          newIdx += lookAhead;
          foundMatch = true;
        }
      }
      
      if (!foundMatch) {
        // Changed line
        if (oldIdx < oldLines.length) {
          hunkChanges.push(`-${oldLines[oldIdx]}`);
          oldIdx++;
        }
        if (newIdx < newLines.length) {
          hunkChanges.push(`+${newLines[newIdx]}`);
          newIdx++;
        }
      }
    }
    
    // Flush hunk if we have enough trailing context
    if (inHunk) {
      const trailingContext = hunkChanges.slice(-3).filter(l => l.startsWith(' ')).length;
      if (trailingContext >= 3 || (oldIdx >= oldLines.length && newIdx >= newLines.length)) {
        const oldCount = hunkChanges.filter(l => l.startsWith(' ') || l.startsWith('-')).length;
        const newCount = hunkChanges.filter(l => l.startsWith(' ') || l.startsWith('+')).length;
        diff.push(`@@ -${hunkOldStart},${oldCount} +${hunkNewStart},${newCount} @@`);
        diff.push(...hunkChanges);
        hunkChanges = [];
        inHunk = false;
      }
    }
  }
  
  return diff.join('\n');
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, path: targetPath, content, name, type, diff: diffContent, newName, useDocker = false } = body;

    // Get the workspace root (mounted workspace or default)
    const ROOT = await getWorkspaceRoot();
    const dockerWorkspace = await getDockerWorkspace();

    const fullPath = path.join(ROOT, targetPath || '');
    
    // Security: ensure path is within workspace
    const normalizedFull = path.normalize(fullPath);
    const normalizedRoot = path.normalize(ROOT);
    if (!normalizedFull.startsWith(normalizedRoot)) {
      return NextResponse.json({ error: 'Access denied: path traversal detected' }, { status: 403 });
    }

    // Get workspace manager for versioning
    let workspaceManager;
    try {
      workspaceManager = await getWorkspaceManager();
    } catch {
      workspaceManager = null;
    }

    switch (action) {
      case 'list': {
        // For Docker, execute ls command in container
        if (useDocker && dockerWorkspace) {
          try {
            const dockerManager = await getDockerManager();
            const result = await dockerManager.execInContainer(
              dockerWorkspace.containerId,
              `ls -la "${targetPath || '/'}"`,
              '/'
            );
            // Parse ls output (simplified)
            const files = result.output.split('\n')
              .filter(line => line.trim() && !line.startsWith('total'))
              .map(line => {
                const parts = line.split(/\s+/);
                const name = parts[parts.length - 1];
                const isDir = line.startsWith('d');
                return {
                  id: path.join(targetPath || '', name),
                  name,
                  type: isDir ? 'folder' : 'file',
                };
              })
              .filter(f => f.name !== '.' && f.name !== '..');
            return NextResponse.json({ files, docker: true });
          } catch (error: any) {
            return NextResponse.json({ error: error.message }, { status: 500 });
          }
        }

        const entries = await fs.readdir(fullPath, { withFileTypes: true });
        const files = await Promise.all(entries.map(async entry => {
          const entryPath = path.join(fullPath, entry.name);
          let size;
          try {
            const stat = await fs.stat(entryPath);
            size = stat.size;
          } catch {}
          
          return {
            id: path.join(targetPath || '', entry.name),
            name: entry.name,
            type: entry.isDirectory() ? 'folder' : 'file',
            size,
          };
        }));
        return NextResponse.json({ files, workspace: ROOT });
      }

      case 'read': {
        // For Docker, read file from container
        if (useDocker && dockerWorkspace) {
          try {
            const dockerManager = await getDockerManager();
            const result = await dockerManager.execInContainer(
              dockerWorkspace.containerId,
              `cat "${targetPath}"`,
              '/'
            );
            return NextResponse.json({ content: result.output, docker: true });
          } catch (error: any) {
            return NextResponse.json({ error: error.message }, { status: 500 });
          }
        }

        const fileContent = await fs.readFile(fullPath, 'utf-8');
        return NextResponse.json({ content: fileContent, workspace: ROOT });
      }

      case 'write': {
        // Save version before writing (rollback support)
        if (workspaceManager) {
          try {
            const activeWorkspace = workspaceManager.getActiveWorkspace();
            if (activeWorkspace) {
              await workspaceManager.saveFileVersion(targetPath, 'Auto-backup before write');
            }
          } catch {
            // File might not exist yet
          }
        }

        // For Docker, write file in container
        if (useDocker && dockerWorkspace) {
          try {
            const dockerManager = await getDockerManager();
            // Escape content for shell
            const escapedContent = (content || '').replace(/'/g, "'\\''");
            const result = await dockerManager.execInContainer(
              dockerWorkspace.containerId,
              `echo '${escapedContent}' > "${targetPath}"`,
              '/'
            );
            return NextResponse.json({ success: result.exitCode === 0, docker: true });
          } catch (error: any) {
            return NextResponse.json({ error: error.message }, { status: 500 });
          }
        }

        // Ensure parent directory exists
        await fs.mkdir(path.dirname(fullPath), { recursive: true });
        await fs.writeFile(fullPath, content || '');
        return NextResponse.json({ success: true, workspace: ROOT });
      }

      case 'patch': {
        // Apply a unified diff patch to a file
        if (!diffContent) {
          return NextResponse.json({ error: 'diff content is required for patch action' }, { status: 400 });
        }

        // Read current content
        let currentContent: string;
        try {
          currentContent = await fs.readFile(fullPath, 'utf-8');
        } catch {
          return NextResponse.json({ error: 'File not found' }, { status: 404 });
        }

        // Save version before patching (rollback support)
        if (workspaceManager) {
          try {
            const activeWorkspace = workspaceManager.getActiveWorkspace();
            if (activeWorkspace) {
              await workspaceManager.saveFileVersion(targetPath, 'Auto-backup before patch');
            }
          } catch {}
        }

        // Parse and apply diff
        const hunks = parseUnifiedDiff(diffContent);
        if (hunks.length === 0) {
          return NextResponse.json({ error: 'No valid hunks found in diff' }, { status: 400 });
        }

        const result = applyDiff(currentContent, hunks);
        if (!result.success) {
          return NextResponse.json({ error: result.error, success: false }, { status: 409 });
        }

        // Write patched content
        await fs.writeFile(fullPath, result.content);
        return NextResponse.json({ 
          success: true, 
          hunksApplied: hunks.length,
          workspace: ROOT,
        });
      }

      case 'diff': {
        // Generate diff between current content and provided new content
        if (content === undefined) {
          return NextResponse.json({ error: 'content is required for diff action' }, { status: 400 });
        }

        let currentContent: string;
        try {
          currentContent = await fs.readFile(fullPath, 'utf-8');
        } catch {
          currentContent = '';
        }

        const diffOutput = generateDiff(currentContent, content, targetPath);
        return NextResponse.json({ diff: diffOutput, workspace: ROOT });
      }

      case 'create': {
        // For Docker, create file/folder in container
        if (useDocker && dockerWorkspace) {
          try {
            const dockerManager = await getDockerManager();
            const newPath = path.join(targetPath || '', name);
            const cmd = type === 'folder' 
              ? `mkdir -p "${newPath}"`
              : `touch "${newPath}"`;
            const result = await dockerManager.execInContainer(
              dockerWorkspace.containerId,
              cmd,
              '/'
            );
            return NextResponse.json({ success: result.exitCode === 0, docker: true });
          } catch (error: any) {
            return NextResponse.json({ error: error.message }, { status: 500 });
          }
        }

        if (type === 'folder') {
          await fs.mkdir(path.join(fullPath, name), { recursive: true });
        } else {
          await fs.mkdir(fullPath, { recursive: true });
          await fs.writeFile(path.join(fullPath, name), content || '');
        }
        return NextResponse.json({ success: true, workspace: ROOT });
      }

      case 'delete': {
        // Save version before delete (rollback support)
        if (workspaceManager) {
          try {
            const activeWorkspace = workspaceManager.getActiveWorkspace();
            if (activeWorkspace) {
              await workspaceManager.saveFileVersion(targetPath, 'Auto-backup before delete');
            }
          } catch {}
        }

        // For Docker, delete file/folder in container
        if (useDocker && dockerWorkspace) {
          try {
            const dockerManager = await getDockerManager();
            const result = await dockerManager.execInContainer(
              dockerWorkspace.containerId,
              `rm -rf "${targetPath}"`,
              '/'
            );
            return NextResponse.json({ success: result.exitCode === 0, docker: true });
          } catch (error: any) {
            return NextResponse.json({ error: error.message }, { status: 500 });
          }
        }

        const stats = await fs.stat(fullPath);
        if (stats.isDirectory()) {
          await fs.rm(fullPath, { recursive: true });
        } else {
          await fs.unlink(fullPath);
        }
        return NextResponse.json({ success: true, workspace: ROOT });
      }

      case 'rename': {
        if (!newName) {
          return NextResponse.json({ error: 'newName is required for rename action' }, { status: 400 });
        }

        // For Docker, rename in container
        if (useDocker && dockerWorkspace) {
          try {
            const dockerManager = await getDockerManager();
            const newPath = path.join(path.dirname(targetPath || ''), newName);
            const result = await dockerManager.execInContainer(
              dockerWorkspace.containerId,
              `mv "${targetPath}" "${newPath}"`,
              '/'
            );
            return NextResponse.json({ success: result.exitCode === 0, docker: true });
          } catch (error: any) {
            return NextResponse.json({ error: error.message }, { status: 500 });
          }
        }

        const newPath = path.join(path.dirname(fullPath), newName);
        await fs.rename(fullPath, newPath);
        return NextResponse.json({ success: true, workspace: ROOT });
      }

      case 'versions': {
        // Get file version history
        if (!workspaceManager) {
          return NextResponse.json({ versions: [] });
        }

        try {
          const versions = await workspaceManager.getFileVersions(targetPath);
          return NextResponse.json({ 
            versions: versions.map(v => ({
              id: v.id,
              timestamp: v.timestamp,
              commitMessage: v.commitMessage,
              hash: v.hash,
            })),
            workspace: ROOT,
          });
        } catch {
          return NextResponse.json({ versions: [] });
        }
      }

      case 'rollback': {
        // Rollback to a specific version
        const { versionId } = body;
        if (!versionId) {
          return NextResponse.json({ error: 'versionId is required for rollback' }, { status: 400 });
        }

        if (!workspaceManager) {
          return NextResponse.json({ error: 'Workspace manager not available' }, { status: 500 });
        }

        try {
          await workspaceManager.rollbackFile(targetPath, versionId);
          return NextResponse.json({ success: true, workspace: ROOT });
        } catch (error: any) {
          return NextResponse.json({ error: error.message }, { status: 500 });
        }
      }

      case 'workspace-info': {
        // Get current workspace information
        const activeWorkspace = workspaceManager?.getActiveWorkspace();
        const allWorkspaces = workspaceManager?.listWorkspaces() || [];
        
        return NextResponse.json({
          current: activeWorkspace ? {
            id: activeWorkspace.id,
            name: activeWorkspace.name,
            path: activeWorkspace.path,
            type: activeWorkspace.type,
            gitBranch: activeWorkspace.gitBranch,
            gitRemote: activeWorkspace.gitRemote,
          } : null,
          workspaces: allWorkspaces.map(w => ({
            id: w.id,
            name: w.name,
            path: w.path,
            type: w.type,
            isActive: w.isActive,
          })),
          docker: dockerWorkspace ? {
            active: true,
            containerId: dockerWorkspace.containerId,
            image: dockerWorkspace.image,
          } : null,
        });
      }

      default:
        return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Files API error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function GET() {
  // Return workspace info and capabilities
  const ROOT = await getWorkspaceRoot();
  const dockerWorkspace = await getDockerWorkspace();
  
  let workspaceManager;
  try {
    workspaceManager = await getWorkspaceManager();
  } catch {
    workspaceManager = null;
  }

  const activeWorkspace = workspaceManager?.getActiveWorkspace();

  return NextResponse.json({
    workspace: ROOT,
    activeWorkspace: activeWorkspace ? {
      id: activeWorkspace.id,
      name: activeWorkspace.name,
      type: activeWorkspace.type,
      gitBranch: activeWorkspace.gitBranch,
    } : null,
    capabilities: [
      'list',
      'read', 
      'write',
      'create',
      'delete',
      'rename',
      'patch',      // Diff-based editing
      'diff',       // Generate diffs
      'versions',   // File version history
      'rollback',   // Rollback to previous version
    ],
    docker: dockerWorkspace ? {
      active: true,
      containerId: dockerWorkspace.containerId,
      image: dockerWorkspace.image,
    } : null,
  });
}
