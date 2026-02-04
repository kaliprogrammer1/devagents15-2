import { NextRequest, NextResponse } from 'next/server';
import { GitHubIntegration, getGitHubConnection, disconnectGitHub } from '@/lib/github';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, userId, ...params } = body;
    
    if (!userId) {
      return NextResponse.json({ error: 'userId is required' }, { status: 400 });
    }
    
    const github = new GitHubIntegration(userId);
    
    switch (action) {
      case 'connect': {
        const { token } = params;
        if (!token) {
          return NextResponse.json({ error: 'GitHub token is required' }, { status: 400 });
        }
        const success = await github.initialize(token);
        if (success) {
          const user = await github.getAuthenticatedUser();
          return NextResponse.json({ success: true, user });
        }
        return NextResponse.json({ error: 'Failed to connect to GitHub' }, { status: 401 });
      }
      
      case 'disconnect': {
        const success = await disconnectGitHub(userId);
        return NextResponse.json({ success });
      }
      
      case 'status': {
        const connection = await getGitHubConnection(userId);
        return NextResponse.json({ 
          connected: !!connection, 
          username: connection?.github_username 
        });
      }
      
      case 'repos': {
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const repos = await github.listRepositories();
        return NextResponse.json({ repos });
      }
      
      case 'repo': {
        const { owner, repo } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const repository = await github.getRepository(owner, repo);
        return NextResponse.json({ repo: repository });
      }
      
      case 'branches': {
        const { owner, repo } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const branches = await github.listBranches(owner, repo);
        return NextResponse.json({ branches });
      }
      
      case 'prs': {
        const { owner, repo, state = 'open' } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const prs = await github.listPullRequests(owner, repo, state);
        return NextResponse.json({ prs });
      }
      
      case 'pr': {
        const { owner, repo, pull_number } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const pr = await github.getPullRequest(owner, repo, pull_number);
        return NextResponse.json({ pr });
      }
      
      case 'create_pr': {
        const { owner, repo, title, head, base, body: prBody } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const pr = await github.createPullRequest(owner, repo, title, head, base, prBody);
        return NextResponse.json({ pr });
      }
      
      case 'merge_pr': {
        const { owner, repo, pull_number, commit_title, merge_method = 'squash' } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const success = await github.mergePullRequest(owner, repo, pull_number, commit_title, merge_method);
        return NextResponse.json({ success });
      }
      
      case 'file': {
        const { owner, repo, path, ref } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const content = await github.getFileContent(owner, repo, path, ref);
        return NextResponse.json({ content });
      }
      
      case 'files': {
        const { owner, repo, path = '', ref } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const files = await github.listFiles(owner, repo, path, ref);
        return NextResponse.json({ files });
      }
      
      case 'create_branch': {
        const { owner, repo, branchName, fromRef } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const success = await github.createBranch(owner, repo, branchName, fromRef);
        return NextResponse.json({ success });
      }
      
      case 'create_file':
      case 'update_file': {
        const { owner, repo, path, content, message, branch } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const success = await github.createOrUpdateFile(owner, repo, path, content, message, branch);
        return NextResponse.json({ success });
      }
      
      case 'commits': {
        const { owner, repo, sha, per_page = 30 } = params;
        await github.initialize();
        if (!github.isConnected()) {
          return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
        }
        const commits = await github.getCommits(owner, repo, sha, per_page);
        return NextResponse.json({ commits });
      }
      
        case 'comment': {
          const { owner, repo, issue_number, body: commentBody } = params;
          await github.initialize();
          if (!github.isConnected()) {
            return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
          }
          const success = await github.addComment(owner, repo, issue_number, commentBody);
          return NextResponse.json({ success });
        }

        case 'workflow_runs': {
          const { owner, repo } = params;
          await github.initialize();
          if (!github.isConnected()) {
            return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
          }
          const runs = await github.listWorkflowRuns(owner, repo);
          return NextResponse.json({ runs });
        }

        case 'workflow_run': {
          const { owner, repo, run_id } = params;
          await github.initialize();
          if (!github.isConnected()) {
            return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
          }
          const run = await github.getWorkflowRun(owner, repo, run_id);
          return NextResponse.json({ run });
        }

        case 'workflow_run_jobs': {
          const { owner, repo, run_id } = params;
          await github.initialize();
          if (!github.isConnected()) {
            return NextResponse.json({ error: 'Not connected to GitHub' }, { status: 401 });
          }
          const jobs = await github.getWorkflowRunJobs(owner, repo, run_id);
          return NextResponse.json({ jobs });
        }

      
      default:
        return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
    }
  } catch (error) {
    console.error('GitHub API error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
