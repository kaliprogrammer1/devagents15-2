import { NextRequest, NextResponse } from 'next/server';
import { getWorkspaceManager, type WorkspaceConfig } from '@/lib/workspaceManager';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, ...params } = body;
    const manager = await getWorkspaceManager();

    switch (action) {
      case 'mount': {
        const { path, name } = params;
        if (!path) {
          return NextResponse.json({ error: 'Path is required' }, { status: 400 });
        }
        const workspace = await manager.mountLocalProject(path, name);
        return NextResponse.json({ workspace });
      }

      case 'clone': {
        const { gitUrl, name, branch } = params;
        if (!gitUrl) {
          return NextResponse.json({ error: 'Git URL is required' }, { status: 400 });
        }
        const workspace = await manager.cloneRepository(gitUrl, name, branch);
        return NextResponse.json({ workspace });
      }

      case 'worktree': {
        const { workspaceId, branchName, baseBranch } = params;
        if (!workspaceId || !branchName) {
          return NextResponse.json({ error: 'workspaceId and branchName are required' }, { status: 400 });
        }
        const workspace = await manager.createWorktree(workspaceId, branchName, baseBranch);
        return NextResponse.json({ workspace });
      }

      case 'activate': {
        const { workspaceId } = params;
        if (!workspaceId) {
          return NextResponse.json({ error: 'workspaceId is required' }, { status: 400 });
        }
        const workspace = await manager.setActiveWorkspace(workspaceId);
        return NextResponse.json({ workspace });
      }

      case 'remove': {
        const { workspaceId, deleteFiles } = params;
        if (!workspaceId) {
          return NextResponse.json({ error: 'workspaceId is required' }, { status: 400 });
        }
        await manager.removeWorkspace(workspaceId, deleteFiles);
        return NextResponse.json({ success: true });
      }

      // File operations with versioning
      case 'read': {
        const { filePath } = params;
        if (!filePath) {
          return NextResponse.json({ error: 'filePath is required' }, { status: 400 });
        }
        const content = await manager.readFile(filePath);
        return NextResponse.json({ content });
      }

      case 'write': {
        const { filePath, content, createVersion = true } = params;
        if (!filePath || content === undefined) {
          return NextResponse.json({ error: 'filePath and content are required' }, { status: 400 });
        }
        await manager.writeFile(filePath, content, createVersion);
        return NextResponse.json({ success: true });
      }

      case 'delete': {
        const { filePath } = params;
        if (!filePath) {
          return NextResponse.json({ error: 'filePath is required' }, { status: 400 });
        }
        await manager.deleteFile(filePath);
        return NextResponse.json({ success: true });
      }

      case 'list': {
        const { dirPath } = params;
        const files = await manager.listFiles(dirPath || '');
        return NextResponse.json({ files });
      }

      // Version history
      case 'versions': {
        const { filePath } = params;
        if (!filePath) {
          return NextResponse.json({ error: 'filePath is required' }, { status: 400 });
        }
        const versions = await manager.getFileVersions(filePath);
        return NextResponse.json({ versions });
      }

      case 'rollback': {
        const { filePath, versionId } = params;
        if (!filePath || !versionId) {
          return NextResponse.json({ error: 'filePath and versionId are required' }, { status: 400 });
        }
        await manager.rollbackFile(filePath, versionId);
        return NextResponse.json({ success: true });
      }

      // Git operations
      case 'git-status': {
        const status = await manager.gitStatus();
        return NextResponse.json(status);
      }

      case 'git-stage': {
        const { files } = params;
        if (!files || !Array.isArray(files)) {
          return NextResponse.json({ error: 'files array is required' }, { status: 400 });
        }
        await manager.gitStage(files);
        return NextResponse.json({ success: true });
      }

      case 'git-unstage': {
        const { files } = params;
        if (!files || !Array.isArray(files)) {
          return NextResponse.json({ error: 'files array is required' }, { status: 400 });
        }
        await manager.gitUnstage(files);
        return NextResponse.json({ success: true });
      }

      case 'git-commit': {
        const { message } = params;
        if (!message) {
          return NextResponse.json({ error: 'message is required' }, { status: 400 });
        }
        const commitHash = await manager.gitCommit(message);
        return NextResponse.json({ commitHash });
      }

      case 'git-push': {
        const { remote, branch } = params;
        await manager.gitPush(remote, branch);
        return NextResponse.json({ success: true });
      }

      case 'git-pull': {
        const { remote, branch } = params;
        await manager.gitPull(remote, branch);
        return NextResponse.json({ success: true });
      }

      case 'git-branch': {
        const { branchName, checkout = true } = params;
        if (!branchName) {
          return NextResponse.json({ error: 'branchName is required' }, { status: 400 });
        }
        await manager.gitCreateBranch(branchName, checkout);
        return NextResponse.json({ success: true });
      }

      case 'git-checkout': {
        const { branchName } = params;
        if (!branchName) {
          return NextResponse.json({ error: 'branchName is required' }, { status: 400 });
        }
        await manager.gitCheckout(branchName);
        return NextResponse.json({ success: true });
      }

      case 'git-reset': {
        const { mode = 'mixed', ref = 'HEAD~1' } = params;
        await manager.gitReset(mode, ref);
        return NextResponse.json({ success: true });
      }

      case 'git-diff': {
        const { file } = params;
        const diff = await manager.gitDiff(file);
        return NextResponse.json({ diff });
      }

      case 'git-log': {
        const { count = 10 } = params;
        const commits = await manager.gitLog(count);
        return NextResponse.json({ commits });
      }

      // Execute in workspace
      case 'exec': {
        const { command, timeout } = params;
        if (!command) {
          return NextResponse.json({ error: 'command is required' }, { status: 400 });
        }
        const result = await manager.execInWorkspace(command, timeout);
        return NextResponse.json(result);
      }

      // Run tests
      case 'test': {
        const { testCommand } = params;
        const result = await manager.runTests(testCommand);
        return NextResponse.json(result);
      }

      // Run build
      case 'build': {
        const { buildCommand } = params;
        const result = await manager.runBuild(buildCommand);
        return NextResponse.json(result);
      }

      default:
        return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Workspace API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const manager = await getWorkspaceManager();
    const workspaces = manager.listWorkspaces();
    const activeWorkspace = manager.getActiveWorkspace();

    return NextResponse.json({
      workspaces,
      activeWorkspace,
    });
  } catch (error: any) {
    console.error('Workspace API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
