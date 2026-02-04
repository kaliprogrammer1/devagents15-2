import { Octokit } from 'octokit';
import { supabaseAdmin, type GitHubConnection, type PRWorkflow } from './supabase';

export interface GitHubRepo {
  id: number;
  name: string;
  full_name: string;
  description: string | null;
  html_url: string;
  private: boolean;
  default_branch: string;
  language: string | null;
  stargazers_count: number;
  open_issues_count: number;
}

export interface GitHubPR {
  id: number;
  number: number;
  title: string;
  body: string | null;
  state: string;
  html_url: string;
  head: { ref: string; sha: string };
  base: { ref: string };
  user: { login: string } | null;
  created_at: string;
  updated_at: string;
  merged_at: string | null;
  mergeable: boolean | null;
  additions: number;
  deletions: number;
  changed_files: number;
}

export interface GitHubBranch {
  name: string;
  commit: { sha: string };
  protected: boolean;
}

export interface GitHubCommit {
  sha: string;
  message: string;
  author: { name: string; email: string; date: string } | null;
  html_url: string;
}

export interface GitHubFile {
  name: string;
  path: string;
  type: 'file' | 'dir';
  size?: number;
  sha: string;
  content?: string;
}

export class GitHubIntegration {
  private octokit: Octokit | null = null;
  private userId: string;
  
  constructor(userId: string) {
    this.userId = userId;
  }
  
  async initialize(token?: string): Promise<boolean> {
    if (token) {
      this.octokit = new Octokit({ auth: token, userAgent: 'Agent-3D-Room/1.0' });
      const user = await this.getAuthenticatedUser();
      if (user) {
        await this.saveConnection(token, user.login);
        return true;
      }
      return false;
    }
    
    const { data: connection } = await supabaseAdmin
      .from('github_connections')
      .select('*')
      .eq('user_id', this.userId)
      .single();
    
    if (connection?.access_token) {
      this.octokit = new Octokit({ auth: connection.access_token, userAgent: 'Agent-3D-Room/1.0' });
      return true;
    }
    
    return false;
  }
  
  private async saveConnection(token: string, username: string): Promise<void> {
    await supabaseAdmin.from('github_connections').upsert({
      user_id: this.userId,
      access_token: token,
      github_username: username,
      connected_repos: [],
      updated_at: new Date().toISOString()
    }, { onConflict: 'user_id' });
  }
  
  async getAuthenticatedUser(): Promise<{ login: string; id: number; name: string | null } | null> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.users.getAuthenticated();
      return { login: data.login, id: data.id, name: data.name };
    } catch (error) {
      console.error('GitHub auth error:', error);
      return null;
    }
  }
  
  async listRepositories(): Promise<GitHubRepo[]> {
    if (!this.octokit) return [];
    try {
      const repos = await this.octokit.paginate(this.octokit.rest.repos.listForAuthenticatedUser, {
        per_page: 100,
        sort: 'updated'
      });
      return repos.map(r => ({
        id: r.id,
        name: r.name,
        full_name: r.full_name,
        description: r.description,
        html_url: r.html_url,
        private: r.private,
        default_branch: r.default_branch,
        language: r.language,
        stargazers_count: r.stargazers_count,
        open_issues_count: r.open_issues_count
      }));
    } catch (error) {
      console.error('Error listing repos:', error);
      return [];
    }
  }
  
  async getRepository(owner: string, repo: string): Promise<GitHubRepo | null> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.repos.get({ owner, repo });
      return {
        id: data.id,
        name: data.name,
        full_name: data.full_name,
        description: data.description,
        html_url: data.html_url,
        private: data.private,
        default_branch: data.default_branch,
        language: data.language,
        stargazers_count: data.stargazers_count,
        open_issues_count: data.open_issues_count
      };
    } catch (error) {
      console.error('Error getting repo:', error);
      return null;
    }
  }
  
  async listBranches(owner: string, repo: string): Promise<GitHubBranch[]> {
    if (!this.octokit) return [];
    try {
      const { data } = await this.octokit.rest.repos.listBranches({ owner, repo, per_page: 100 });
      return data.map(b => ({
        name: b.name,
        commit: { sha: b.commit.sha },
        protected: b.protected
      }));
    } catch (error) {
      console.error('Error listing branches:', error);
      return [];
    }
  }
  
  async listPullRequests(owner: string, repo: string, state: 'open' | 'closed' | 'all' = 'open'): Promise<GitHubPR[]> {
    if (!this.octokit) return [];
    try {
      const { data } = await this.octokit.rest.pulls.list({
        owner, repo, state, sort: 'updated', direction: 'desc', per_page: 50
      });
      return data.map(pr => ({
        id: pr.id,
        number: pr.number,
        title: pr.title,
        body: pr.body,
        state: pr.state,
        html_url: pr.html_url,
        head: { ref: pr.head.ref, sha: pr.head.sha },
        base: { ref: pr.base.ref },
        user: pr.user ? { login: pr.user.login } : null,
        created_at: pr.created_at,
        updated_at: pr.updated_at,
        merged_at: pr.merged_at,
        mergeable: null,
        additions: 0,
        deletions: 0,
        changed_files: 0
      }));
    } catch (error) {
      console.error('Error listing PRs:', error);
      return [];
    }
  }
  
  async getPullRequest(owner: string, repo: string, pull_number: number): Promise<GitHubPR | null> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.pulls.get({ owner, repo, pull_number });
      return {
        id: data.id,
        number: data.number,
        title: data.title,
        body: data.body,
        state: data.state,
        html_url: data.html_url,
        head: { ref: data.head.ref, sha: data.head.sha },
        base: { ref: data.base.ref },
        user: data.user ? { login: data.user.login } : null,
        created_at: data.created_at,
        updated_at: data.updated_at,
        merged_at: data.merged_at,
        mergeable: data.mergeable,
        additions: data.additions,
        deletions: data.deletions,
        changed_files: data.changed_files
      };
    } catch (error) {
      console.error('Error getting PR:', error);
      return null;
    }
  }

  async listWorkflowRuns(owner: string, repo: string): Promise<any[]> {
    if (!this.octokit) return [];
    try {
      const { data } = await this.octokit.rest.actions.listWorkflowRunsForRepo({
        owner,
        repo,
        per_page: 10
      });
      return data.workflow_runs;
    } catch (error) {
      console.error('Error listing workflow runs:', error);
      return [];
    }
  }

  async getWorkflowRun(owner: string, repo: string, run_id: number): Promise<any> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.actions.getWorkflowRun({
        owner,
        repo,
        run_id
      });
      return data;
    } catch (error) {
      console.error('Error getting workflow run:', error);
      return null;
    }
  }

  async getWorkflowRunLogs(owner: string, repo: string, run_id: number): Promise<string | null> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.actions.downloadWorkflowRunLogs({
        owner,
        repo,
        run_id
      });
      // In a real environment, this is a binary zip. 
      // For this agent, we'll try to get the jobs and their steps instead for more actionable info.
      const jobs = await this.getWorkflowRunJobs(owner, repo, run_id);
      let logSummary = `Workflow Run ${run_id} Jobs:\n`;
      for (const job of jobs) {
        logSummary += `Job: ${job.name} (Status: ${job.status}, Conclusion: ${job.conclusion})\n`;
        if (job.steps) {
          for (const step of job.steps) {
            logSummary += `  - Step: ${step.name} (Status: ${step.status}, Conclusion: ${step.conclusion})\n`;
          }
        }
      }
      return logSummary;
    } catch (error) {
      console.error('Error getting logs:', error);
      return null;
    }
  }

  async monitorAndFixBuilds(owner: string, repo: string): Promise<{ fixed: boolean; message: string }> {
    if (!this.octokit) return { fixed: false, message: 'Not connected to GitHub' };
    
    try {
      const runs = await this.listWorkflowRuns(owner, repo);
      const failingRuns = runs.filter(r => r.status === 'completed' && r.conclusion === 'failure');
      
      if (failingRuns.length === 0) {
        return { fixed: false, message: 'No failing builds found' };
      }
      
      const latestFailing = failingRuns[0];
      const logs = await this.getWorkflowRunLogs(owner, repo, latestFailing.id);
      
      // Here we would normally trigger the agent's reasoning engine to analyze the logs
      // For now, we return the failure info so the agent can act on it
      return { 
        fixed: false, 
        message: `Found failing build ${latestFailing.id} in ${repo}. Analysis needed.\nLogs summary:\n${logs}`
      };
    } catch (error) {
      return { fixed: false, message: `Error monitoring builds: ${error}` };
    }
  }

  async getWorkflowRunJobs(owner: string, repo: string, run_id: number): Promise<any[]> {
    if (!this.octokit) return [];
    try {
      const { data } = await this.octokit.rest.actions.listJobsForWorkflowRun({
        owner,
        repo,
        run_id
      });
      return data.jobs;
    } catch (error) {
      console.error('Error listing jobs:', error);
      return [];
    }
  }
  
  async createPullRequest(
    owner: string,
    repo: string,
    title: string,
    head: string,
    base: string,
    body?: string
  ): Promise<GitHubPR | null> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.pulls.create({ owner, repo, title, head, base, body });
      
      await supabaseAdmin.from('pr_workflows').insert({
        user_id: this.userId,
        github_repo: `${owner}/${repo}`,
        pr_number: data.number,
        pr_title: title,
        pr_body: body,
        status: 'open',
        branch_name: head,
        commits: []
      });
      
      return {
        id: data.id,
        number: data.number,
        title: data.title,
        body: data.body,
        state: data.state,
        html_url: data.html_url,
        head: { ref: data.head.ref, sha: data.head.sha },
        base: { ref: data.base.ref },
        user: data.user ? { login: data.user.login } : null,
        created_at: data.created_at,
        updated_at: data.updated_at,
        merged_at: data.merged_at,
        mergeable: data.mergeable,
        additions: data.additions,
        deletions: data.deletions,
        changed_files: data.changed_files
      };
    } catch (error) {
      console.error('Error creating PR:', error);
      return null;
    }
  }
  
  async mergePullRequest(
    owner: string,
    repo: string,
    pull_number: number,
    commit_title?: string,
    merge_method: 'merge' | 'squash' | 'rebase' = 'squash'
  ): Promise<boolean> {
    if (!this.octokit) return false;
    try {
      await this.octokit.rest.pulls.merge({ owner, repo, pull_number, commit_title, merge_method });
      
      await supabaseAdmin
        .from('pr_workflows')
        .update({ status: 'merged', updated_at: new Date().toISOString() })
        .eq('github_repo', `${owner}/${repo}`)
        .eq('pr_number', pull_number);
      
      return true;
    } catch (error) {
      console.error('Error merging PR:', error);
      return false;
    }
  }
  
  async getFileContent(owner: string, repo: string, path: string, ref?: string): Promise<string | null> {
    if (!this.octokit) return null;
    try {
      const { data } = await this.octokit.rest.repos.getContent({ owner, repo, path, ref });
      if ('content' in data && data.type === 'file') {
        return Buffer.from(data.content, 'base64').toString('utf-8');
      }
      return null;
    } catch (error) {
      console.error('Error getting file:', error);
      return null;
    }
  }
  
  async listFiles(owner: string, repo: string, path: string = '', ref?: string): Promise<GitHubFile[]> {
    if (!this.octokit) return [];
    try {
      const { data } = await this.octokit.rest.repos.getContent({ owner, repo, path, ref });
      if (Array.isArray(data)) {
        return data.map(f => ({
          name: f.name,
          path: f.path,
          type: f.type as 'file' | 'dir',
          size: f.size,
          sha: f.sha
        }));
      }
      return [];
    } catch (error) {
      console.error('Error listing files:', error);
      return [];
    }
  }
  
  async createBranch(owner: string, repo: string, branchName: string, fromRef?: string): Promise<boolean> {
    if (!this.octokit) return false;
    try {
      const repository = await this.getRepository(owner, repo);
      if (!repository) return false;
      
      const baseBranch = fromRef || repository.default_branch;
      const { data: refData } = await this.octokit.rest.git.getRef({
        owner, repo, ref: `heads/${baseBranch}`
      });
      
      await this.octokit.rest.git.createRef({
        owner, repo, ref: `refs/heads/${branchName}`, sha: refData.object.sha
      });
      
      return true;
    } catch (error) {
      console.error('Error creating branch:', error);
      return false;
    }
  }
  
  async createOrUpdateFile(
    owner: string,
    repo: string,
    path: string,
    content: string,
    message: string,
    branch: string
  ): Promise<boolean> {
    if (!this.octokit) return false;
    try {
      let sha: string | undefined;
      try {
        const { data } = await this.octokit.rest.repos.getContent({ owner, repo, path, ref: branch });
        if ('sha' in data) sha = data.sha;
      } catch { /* File doesn't exist */ }
      
      await this.octokit.rest.repos.createOrUpdateFileContents({
        owner, repo, path,
        message,
        content: Buffer.from(content).toString('base64'),
        branch,
        sha
      });
      
      return true;
    } catch (error) {
      console.error('Error creating/updating file:', error);
      return false;
    }
  }
  
  async getCommits(owner: string, repo: string, sha?: string, per_page: number = 30): Promise<GitHubCommit[]> {
    if (!this.octokit) return [];
    try {
      const { data } = await this.octokit.rest.repos.listCommits({ owner, repo, sha, per_page });
      return data.map(c => ({
        sha: c.sha,
        message: c.commit.message,
        author: c.commit.author ? {
          name: c.commit.author.name || '',
          email: c.commit.author.email || '',
          date: c.commit.author.date || ''
        } : null,
        html_url: c.html_url
      }));
    } catch (error) {
      console.error('Error getting commits:', error);
      return [];
    }
  }
  
  async addComment(owner: string, repo: string, issue_number: number, body: string): Promise<boolean> {
    if (!this.octokit) return false;
    try {
      await this.octokit.rest.issues.createComment({ owner, repo, issue_number, body });
      return true;
    } catch (error) {
      console.error('Error adding comment:', error);
      return false;
    }
  }
  
  async requestReview(owner: string, repo: string, pull_number: number, reviewers: string[]): Promise<boolean> {
    if (!this.octokit) return false;
    try {
      await this.octokit.rest.pulls.requestReviewers({ owner, repo, pull_number, reviewers });
      return true;
    } catch (error) {
      console.error('Error requesting review:', error);
      return false;
    }
  }
  
  isConnected(): boolean {
    return this.octokit !== null;
  }
}

export async function getGitHubConnection(userId: string): Promise<GitHubConnection | null> {
  const { data } = await supabaseAdmin
    .from('github_connections')
    .select('*')
    .eq('user_id', userId)
    .single();
  return data;
}

export async function disconnectGitHub(userId: string): Promise<boolean> {
  const { error } = await supabaseAdmin
    .from('github_connections')
    .delete()
    .eq('user_id', userId);
  return !error;
}
