// Persona-specific application configurations and skills
import { RoomConfig } from './rooms';

export type PersonaType = 'architect' | 'operator' | 'creator' | 'analyst' | 'hacker';

export interface PersonaApp {
  id: string;
  name: string;
  icon: string;
  component: string;
  description: string;
  defaultOpen?: boolean;
}

export interface PersonaSkillSet {
  primarySkills: string[];
  tools: string[];
  specializations: string[];
}

export interface PersonaConfig {
  id: PersonaType;
  name: string;
  title: string;
  description: string;
  apps: PersonaApp[];
  skills: PersonaSkillSet;
  databaseTables: string[];
  defaultBrowserUrls: string[];
  terminalPrompt: string;
  codeLanguages: string[];
  aiPromptStyle: string;
}

export const PERSONA_CONFIGS: Record<PersonaType, PersonaConfig> = {
  architect: {
    id: 'architect',
    name: 'The Architect',
    title: 'System Design Engineer',
    description: 'Designs scalable systems, defines architecture patterns, and ensures structural integrity.',
    apps: [
      { id: 'systemDesigner', name: 'System Designer', icon: 'Box', component: 'SystemDesignerApp', description: 'Visual system architecture tool' },
      { id: 'umlEditor', name: 'UML Editor', icon: 'GitBranch', component: 'UMLEditorApp', description: 'Create UML diagrams' },
      { id: 'apiDesigner', name: 'API Designer', icon: 'Webhook', component: 'APIDesignerApp', description: 'Design RESTful and GraphQL APIs' },
      { id: 'terminal', name: 'Terminal', icon: 'Terminal', component: 'RealTerminal', description: 'System commands', defaultOpen: true },
      { id: 'editor', name: 'Code Editor', icon: 'FileText', component: 'CodeEditor', description: 'Edit code files' },
      { id: 'browser', name: 'Documentation', icon: 'Globe', component: 'RealBrowser', description: 'Browse documentation' },
    ],
    skills: {
      primarySkills: ['System Design', 'Architecture Patterns', 'API Design', 'Database Modeling', 'Microservices'],
      tools: ['Kubernetes', 'Docker', 'Terraform', 'AWS/GCP', 'PostgreSQL'],
      specializations: ['Distributed Systems', 'Event-Driven Architecture', 'Domain-Driven Design'],
    },
    databaseTables: ['architect_designs', 'architect_patterns', 'architect_api_specs'],
    defaultBrowserUrls: ['https://aws.amazon.com/architecture/', 'https://martinfowler.com/'],
    terminalPrompt: 'architect@system:~$',
    codeLanguages: ['typescript', 'go', 'rust', 'yaml', 'terraform'],
    aiPromptStyle: 'Precise, formal, and structured. Focus on scalability, maintainability, and best practices.',
  },

  operator: {
    id: 'operator',
    name: 'The Operator',
    title: 'DevOps & SRE Specialist',
    description: 'Manages infrastructure, monitors systems, and ensures operational excellence.',
    apps: [
      { id: 'monitoring', name: 'Monitoring Hub', icon: 'Activity', component: 'MonitoringApp', description: 'System metrics and alerts', defaultOpen: true },
      { id: 'deployManager', name: 'Deploy Manager', icon: 'Rocket', component: 'DeployManagerApp', description: 'CI/CD pipeline management' },
      { id: 'logViewer', name: 'Log Analyzer', icon: 'ScrollText', component: 'LogViewerApp', description: 'Centralized log analysis' },
      { id: 'terminal', name: 'Terminal', icon: 'Terminal', component: 'RealTerminal', description: 'Execute operations' },
      { id: 'editor', name: 'Config Editor', icon: 'FileText', component: 'CodeEditor', description: 'Edit configs' },
      { id: 'browser', name: 'Cloud Console', icon: 'Globe', component: 'RealBrowser', description: 'Cloud dashboards' },
    ],
    skills: {
      primarySkills: ['CI/CD', 'Infrastructure as Code', 'Monitoring', 'Incident Response', 'Container Orchestration'],
      tools: ['GitHub Actions', 'Jenkins', 'Prometheus', 'Grafana', 'Docker', 'Kubernetes'],
      specializations: ['SRE', 'Platform Engineering', 'Chaos Engineering'],
    },
    databaseTables: ['operator_deployments', 'operator_incidents', 'operator_metrics'],
    defaultBrowserUrls: ['https://grafana.com/', 'https://prometheus.io/'],
    terminalPrompt: 'operator@prod:~#',
    codeLanguages: ['bash', 'yaml', 'python', 'dockerfile', 'hcl'],
    aiPromptStyle: 'Concise, action-oriented, and urgent. Use operational terminology and focus on uptime.',
  },

  creator: {
    id: 'creator',
    name: 'The Creator',
    title: 'Full-Stack Developer & Designer',
    description: 'Builds user interfaces, crafts experiences, and brings ideas to life.',
    apps: [
      { id: 'designStudio', name: 'Design Studio', icon: 'Palette', component: 'DesignStudioApp', description: 'UI/UX design workspace', defaultOpen: true },
      { id: 'componentLib', name: 'Component Library', icon: 'Layers', component: 'ComponentLibraryApp', description: 'Reusable UI components' },
      { id: 'colorPicker', name: 'Color Lab', icon: 'Pipette', component: 'ColorLabApp', description: 'Color palette generator' },
      { id: 'editor', name: 'Code Editor', icon: 'FileText', component: 'CodeEditor', description: 'Write code' },
      { id: 'browser', name: 'Inspiration', icon: 'Globe', component: 'RealBrowser', description: 'Design inspiration' },
      { id: 'terminal', name: 'Dev Server', icon: 'Terminal', component: 'RealTerminal', description: 'Run dev commands' },
    ],
    skills: {
      primarySkills: ['React/Next.js', 'CSS/Tailwind', 'UI Design', 'Animation', 'Responsive Design'],
      tools: ['Figma', 'Framer Motion', 'Three.js', 'Storybook', 'Chromatic'],
      specializations: ['Design Systems', 'Motion Design', 'Accessibility'],
    },
    databaseTables: ['creator_designs', 'creator_components', 'creator_palettes'],
    defaultBrowserUrls: ['https://dribbble.com/', 'https://tailwindcss.com/'],
    terminalPrompt: 'creator@studio:~$',
    codeLanguages: ['typescript', 'javascript', 'css', 'html', 'jsx'],
    aiPromptStyle: 'Creative, metaphorical, and slightly informal. Encourage experimentation and visual thinking.',
  },

  analyst: {
    id: 'analyst',
    name: 'The Analyst',
    title: 'Data Scientist & ML Engineer',
    description: 'Analyzes data, builds models, and extracts insights from complex datasets.',
    apps: [
      { id: 'dataDashboard', name: 'Data Dashboard', icon: 'BarChart3', component: 'DataDashboardApp', description: 'Interactive data visualization', defaultOpen: true },
      { id: 'notebook', name: 'Jupyter Notebook', icon: 'FileCode', component: 'NotebookApp', description: 'Python notebooks' },
      { id: 'sqlEditor', name: 'SQL Editor', icon: 'Database', component: 'SQLEditorApp', description: 'Query databases' },
      { id: 'modelTrainer', name: 'ML Studio', icon: 'Brain', component: 'MLStudioApp', description: 'Train and evaluate models' },
      { id: 'editor', name: 'Code Editor', icon: 'FileText', component: 'CodeEditor', description: 'Edit scripts' },
      { id: 'browser', name: 'Research', icon: 'Globe', component: 'RealBrowser', description: 'Research papers' },
    ],
    skills: {
      primarySkills: ['Python', 'SQL', 'Statistics', 'Machine Learning', 'Data Visualization'],
      tools: ['Pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'Jupyter', 'Tableau'],
      specializations: ['Deep Learning', 'NLP', 'Computer Vision', 'Time Series'],
    },
    databaseTables: ['analyst_datasets', 'analyst_models', 'analyst_experiments'],
    defaultBrowserUrls: ['https://arxiv.org/', 'https://kaggle.com/'],
    terminalPrompt: 'analyst@lab:~$',
    codeLanguages: ['python', 'sql', 'r', 'julia'],
    aiPromptStyle: 'Fact-based, detailed, and analytical. Use data points, statistics, and logical reasoning.',
  },

  hacker: {
    id: 'hacker',
    name: 'The Hacker',
    title: 'Security Researcher & Penetration Tester',
    description: 'Finds vulnerabilities, tests security, and protects systems from threats.',
    apps: [
      { id: 'securityTerminal', name: 'Security Terminal', icon: 'Terminal', component: 'SecurityTerminalApp', description: 'Security-focused terminal', defaultOpen: true },
      { id: 'vulnScanner', name: 'Vuln Scanner', icon: 'Shield', component: 'VulnScannerApp', description: 'Scan for vulnerabilities' },
      { id: 'networkMap', name: 'Network Map', icon: 'Network', component: 'NetworkMapApp', description: 'Network topology visualization' },
      { id: 'packetAnalyzer', name: 'Packet Analyzer', icon: 'Radio', component: 'PacketAnalyzerApp', description: 'Analyze network traffic' },
      { id: 'editor', name: 'Exploit Editor', icon: 'FileText', component: 'CodeEditor', description: 'Write scripts' },
      { id: 'browser', name: 'Dark Web Intel', icon: 'Globe', component: 'RealBrowser', description: 'Security research' },
    ],
    skills: {
      primarySkills: ['Penetration Testing', 'Reverse Engineering', 'Cryptography', 'Network Security', 'Malware Analysis'],
      tools: ['Metasploit', 'Burp Suite', 'Wireshark', 'Nmap', 'IDA Pro', 'Ghidra'],
      specializations: ['Web Security', 'Binary Exploitation', 'Red Teaming'],
    },
    databaseTables: ['hacker_findings', 'hacker_exploits', 'hacker_targets'],
    defaultBrowserUrls: ['https://hackerone.com/', 'https://owasp.org/'],
    terminalPrompt: 'root@kali:~#',
    codeLanguages: ['python', 'c', 'assembly', 'bash', 'javascript'],
    aiPromptStyle: 'Terse, technical, and uses slang or terminal metaphors. Focus on security implications.',
  },
};

export function getPersonaConfig(personaId: string): PersonaConfig {
  return PERSONA_CONFIGS[personaId as PersonaType] || PERSONA_CONFIGS.architect;
}

export function getPersonaFromRoom(room: RoomConfig): PersonaConfig {
  return PERSONA_CONFIGS[room.id as PersonaType] || PERSONA_CONFIGS.architect;
}

export function getPersonaApps(personaId: string): PersonaApp[] {
  const config = getPersonaConfig(personaId);
  return config.apps;
}

export function getPersonaSkills(personaId: string): PersonaSkillSet {
  const config = getPersonaConfig(personaId);
  return config.skills;
}
