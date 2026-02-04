import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);

export interface AgentSkill {
  id?: string;
  skill_name: string;
  skill_type: string;
  description: string;
  proficiency_level: number;
  times_used: number;
  usage_count: number;
  last_used_at?: string;
  compressed_data: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  success_rate: number;
}

export interface UserMemory {
  id?: string;
  user_id: string;
  memory_type: string;
  content_compressed: string;
  context?: Record<string, unknown>;
  importance_score: number;
  embedding_1536?: number[];
}

export interface UniversalMemory {
  id?: string;
  memory_type: string;
  content_compressed: string;
  embedding_vector?: string;
  embedding_1536?: number[];
  importance_score: number;
  access_count: number;
}

export interface KnowledgeNode {
  id?: string;
  name: string;
  node_type: string;
  content_compressed: string;
  metadata?: Record<string, unknown>;
  embedding_1536?: number[];
}

export interface GitHubConnection {
  id?: string;
  user_id: string;
  access_token: string;
  github_username?: string;
  connected_repos: string[];
}

export interface CodeExecution {
  id?: string;
  user_id: string;
  language: string;
  code: string;
  output?: string;
  exit_code?: number;
  execution_time_ms?: number;
  success: boolean;
}

export interface PRWorkflow {
  id?: string;
  user_id: string;
  github_repo: string;
  pr_number?: number;
  pr_title?: string;
  pr_body?: string;
  status: string;
  branch_name?: string;
  commits: unknown[];
}

export interface TestResult {
  id?: string;
  user_id: string;
  execution_id?: string;
  test_framework?: string;
  total_tests: number;
  passed: number;
  failed: number;
  skipped: number;
  coverage_percent?: number;
  output?: string;
}
