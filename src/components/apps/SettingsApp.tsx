"use client";

import { useState, useEffect } from 'react';
import { useSettingsStore, Settings } from '@/lib/settingsStore';
import { Settings as SettingsIcon, User, Palette, Cpu, FolderOpen, Link, Info, ChevronRight, Moon, Sun, Monitor, Volume2, VolumeX, Bell, BellOff, Check, Zap, Coffee, Clock, Save, RotateCcw } from 'lucide-react';

type SettingsTab = 'general' | 'appearance' | 'agent' | 'workspace' | 'integrations' | 'about';

const TABS: { id: SettingsTab; label: string; icon: React.ReactNode }[] = [
  { id: 'general', label: 'General', icon: <SettingsIcon size={16} /> },
  { id: 'appearance', label: 'Appearance', icon: <Palette size={16} /> },
  { id: 'agent', label: 'Agent', icon: <Cpu size={16} /> },
  { id: 'workspace', label: 'Workspace', icon: <FolderOpen size={16} /> },
  { id: 'integrations', label: 'Integrations', icon: <Link size={16} /> },
  { id: 'about', label: 'About', icon: <Info size={16} /> },
];

const THEMES = [
  { id: 'dark', label: 'Dark', icon: <Moon size={16} />, colors: { bg: '#0d1117', accent: '#58a6ff' } },
  { id: 'light', label: 'Light', icon: <Sun size={16} />, colors: { bg: '#ffffff', accent: '#0969da' } },
  { id: 'system', label: 'System', icon: <Monitor size={16} />, colors: { bg: '#1e1e2e', accent: '#89b4fa' } },
  { id: 'midnight', label: 'Midnight', icon: <Moon size={16} />, colors: { bg: '#0a0a0f', accent: '#a855f7' } },
  { id: 'forest', label: 'Forest', icon: <Moon size={16} />, colors: { bg: '#0d1912', accent: '#22c55e' } },
];

const FONTS = [
  { id: 'jetbrains', label: 'JetBrains Mono', family: "'JetBrains Mono', monospace" },
  { id: 'fira', label: 'Fira Code', family: "'Fira Code', monospace" },
  { id: 'cascadia', label: 'Cascadia Code', family: "'Cascadia Code', monospace" },
  { id: 'source', label: 'Source Code Pro', family: "'Source Code Pro', monospace" },
  { id: 'inter', label: 'Inter', family: "'Inter', sans-serif" },
];

const PERSONALITIES = [
  { id: 'professional', label: 'Professional', description: 'Formal, concise communication' },
  { id: 'friendly', label: 'Friendly', description: 'Warm, casual interactions' },
  { id: 'mentor', label: 'Mentor', description: 'Educational, detailed explanations' },
  { id: 'minimal', label: 'Minimal', description: 'Brief, to-the-point responses' },
];

export default function SettingsApp() {
  const { settings, updateSettings, resetSettings } = useSettingsStore();
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');
  const [saved, setSaved] = useState(false);
  const [localSettings, setLocalSettings] = useState<Settings>(settings);

  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleSave = () => {
    updateSettings(localSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    resetSettings();
    setLocalSettings(useSettingsStore.getState().settings);
  };

  const updateLocal = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'general':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white mb-6">General Settings</h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
                <div>
                  <div className="text-white font-medium">Agent Name</div>
                  <div className="text-sm text-[#8b949e]">Display name for the AI agent</div>
                </div>
                <input
                  type="text"
                  value={localSettings.agentName}
                  onChange={(e) => updateLocal('agentName', e.target.value)}
                  className="bg-[#21262d] border border-[#30363d] rounded-md px-3 py-1.5 text-sm text-white w-40 outline-none focus:border-[#58a6ff]"
                />
              </div>

              <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
                <div>
                  <div className="text-white font-medium">Language</div>
                  <div className="text-sm text-[#8b949e]">Interface language</div>
                </div>
                <select
                  value={localSettings.language}
                  onChange={(e) => updateLocal('language', e.target.value)}
                  className="bg-[#21262d] border border-[#30363d] rounded-md px-3 py-1.5 text-sm text-white outline-none focus:border-[#58a6ff]"
                >
                  <option value="en">English</option>
                  <option value="es">Español</option>
                  <option value="fr">Français</option>
                  <option value="de">Deutsch</option>
                  <option value="ja">日本語</option>
                </select>
              </div>

              <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
                <div>
                  <div className="text-white font-medium">Time Format</div>
                  <div className="text-sm text-[#8b949e]">12-hour or 24-hour clock</div>
                </div>
                <select
                  value={localSettings.timeFormat}
                  onChange={(e) => updateLocal('timeFormat', e.target.value as '12h' | '24h')}
                  className="bg-[#21262d] border border-[#30363d] rounded-md px-3 py-1.5 text-sm text-white outline-none focus:border-[#58a6ff]"
                >
                  <option value="12h">12-hour</option>
                  <option value="24h">24-hour</option>
                </select>
              </div>

              <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
                <div>
                  <div className="text-white font-medium">Sound Effects</div>
                  <div className="text-sm text-[#8b949e]">Play sounds for actions</div>
                </div>
                <button
                  onClick={() => updateLocal('soundEnabled', !localSettings.soundEnabled)}
                  className={`p-2 rounded-md transition-colors ${localSettings.soundEnabled ? 'bg-[#238636] text-white' : 'bg-[#21262d] text-[#8b949e]'}`}
                >
                  {localSettings.soundEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
                </button>
              </div>

              <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
                <div>
                  <div className="text-white font-medium">Notifications</div>
                  <div className="text-sm text-[#8b949e]">Show desktop notifications</div>
                </div>
                <button
                  onClick={() => updateLocal('notificationsEnabled', !localSettings.notificationsEnabled)}
                  className={`p-2 rounded-md transition-colors ${localSettings.notificationsEnabled ? 'bg-[#238636] text-white' : 'bg-[#21262d] text-[#8b949e]'}`}
                >
                  {localSettings.notificationsEnabled ? <Bell size={18} /> : <BellOff size={18} />}
                </button>
              </div>
            </div>
          </div>
        );

      case 'appearance':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white mb-6">Appearance</h2>
            
            <div>
              <div className="text-white font-medium mb-3">Theme</div>
              <div className="grid grid-cols-5 gap-2">
                {THEMES.map(theme => (
                  <button
                    key={theme.id}
                    onClick={() => updateLocal('theme', theme.id)}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      localSettings.theme === theme.id
                        ? 'border-[#58a6ff] bg-[#58a6ff]/10'
                        : 'border-[#30363d] hover:border-[#8b949e]'
                    }`}
                  >
                    <div 
                      className="w-full h-8 rounded mb-2 flex items-center justify-center"
                      style={{ backgroundColor: theme.colors.bg, border: `1px solid ${theme.colors.accent}` }}
                    >
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: theme.colors.accent }} />
                    </div>
                    <div className="text-xs text-center text-white">{theme.label}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t border-[#30363d] pt-6">
              <div className="text-white font-medium mb-3">Font Family</div>
              <div className="space-y-2">
                {FONTS.map(font => (
                  <button
                    key={font.id}
                    onClick={() => updateLocal('fontFamily', font.id)}
                    className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all ${
                      localSettings.fontFamily === font.id
                        ? 'border-[#58a6ff] bg-[#58a6ff]/10'
                        : 'border-[#30363d] hover:border-[#8b949e]'
                    }`}
                  >
                    <span style={{ fontFamily: font.family }} className="text-white">{font.label}</span>
                    {localSettings.fontFamily === font.id && <Check size={16} className="text-[#58a6ff]" />}
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t border-[#30363d] pt-6">
              <div className="flex items-center justify-between mb-3">
                <div className="text-white font-medium">Font Size</div>
                <span className="text-[#8b949e]">{localSettings.fontSize}px</span>
              </div>
              <input
                type="range"
                min="10"
                max="18"
                value={localSettings.fontSize}
                onChange={(e) => updateLocal('fontSize', parseInt(e.target.value))}
                className="w-full accent-[#58a6ff]"
              />
              <div className="flex justify-between text-xs text-[#8b949e] mt-1">
                <span>Small</span>
                <span>Large</span>
              </div>
            </div>

            <div className="border-t border-[#30363d] pt-6">
              <div className="flex items-center justify-between py-3">
                <div>
                  <div className="text-white font-medium">Animations</div>
                  <div className="text-sm text-[#8b949e]">Enable smooth transitions</div>
                </div>
                <button
                  onClick={() => updateLocal('animationsEnabled', !localSettings.animationsEnabled)}
                  className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.animationsEnabled ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.animationsEnabled ? 'left-7' : 'left-1'}`} />
                </button>
              </div>
            </div>
          </div>
        );

      case 'agent':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white mb-6">Agent Settings</h2>
            
            <div>
              <div className="text-white font-medium mb-3">Personality</div>
              <div className="grid grid-cols-2 gap-2">
                {PERSONALITIES.map(p => (
                  <button
                    key={p.id}
                    onClick={() => updateLocal('agentPersonality', p.id)}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      localSettings.agentPersonality === p.id
                        ? 'border-[#58a6ff] bg-[#58a6ff]/10'
                        : 'border-[#30363d] hover:border-[#8b949e]'
                    }`}
                  >
                    <div className="text-white font-medium">{p.label}</div>
                    <div className="text-xs text-[#8b949e]">{p.description}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t border-[#30363d] pt-6">
              <div className="flex items-center justify-between mb-3">
                <div className="text-white font-medium">Work Speed</div>
                <span className="text-[#8b949e] capitalize">{localSettings.agentSpeed}</span>
              </div>
              <div className="flex gap-2">
                {(['slow', 'normal', 'fast'] as const).map(speed => (
                  <button
                    key={speed}
                    onClick={() => updateLocal('agentSpeed', speed)}
                    className={`flex-1 py-2 rounded-lg border transition-all capitalize ${
                      localSettings.agentSpeed === speed
                        ? 'border-[#58a6ff] bg-[#58a6ff]/10 text-white'
                        : 'border-[#30363d] text-[#8b949e] hover:border-[#8b949e]'
                    }`}
                  >
                    {speed}
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t border-[#30363d] pt-6 space-y-4">
              <div className="flex items-center justify-between py-3">
                <div>
                  <div className="text-white font-medium">Auto Start</div>
                  <div className="text-sm text-[#8b949e]">Begin working automatically</div>
                </div>
                <button
                  onClick={() => updateLocal('autoStartEnabled', !localSettings.autoStartEnabled)}
                  className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.autoStartEnabled ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.autoStartEnabled ? 'left-7' : 'left-1'}`} />
                </button>
              </div>

              <div className="flex items-center justify-between py-3">
                <div>
                  <div className="text-white font-medium">Break Reminders</div>
                  <div className="text-sm text-[#8b949e]">Agent takes periodic breaks</div>
                </div>
                <button
                  onClick={() => updateLocal('breakRemindersEnabled', !localSettings.breakRemindersEnabled)}
                  className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.breakRemindersEnabled ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.breakRemindersEnabled ? 'left-7' : 'left-1'}`} />
                </button>
              </div>

              <div className="flex items-center justify-between py-3">
                <div>
                  <div className="text-white font-medium">Verbose Logging</div>
                  <div className="text-sm text-[#8b949e]">Show detailed action logs</div>
                </div>
                <button
                  onClick={() => updateLocal('verboseLogging', !localSettings.verboseLogging)}
                  className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.verboseLogging ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.verboseLogging ? 'left-7' : 'left-1'}`} />
                </button>
              </div>
            </div>
          </div>
        );

      case 'workspace':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white mb-6">Workspace Settings</h2>
            
            <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
              <div>
                <div className="text-white font-medium">Project Name</div>
                <div className="text-sm text-[#8b949e]">Current workspace name</div>
              </div>
              <input
                type="text"
                value={localSettings.workspaceName}
                onChange={(e) => updateLocal('workspaceName', e.target.value)}
                className="bg-[#21262d] border border-[#30363d] rounded-md px-3 py-1.5 text-sm text-white w-48 outline-none focus:border-[#58a6ff]"
              />
            </div>

            <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
              <div>
                <div className="text-white font-medium">Default Branch</div>
                <div className="text-sm text-[#8b949e]">Git default branch</div>
              </div>
              <input
                type="text"
                value={localSettings.defaultBranch}
                onChange={(e) => updateLocal('defaultBranch', e.target.value)}
                className="bg-[#21262d] border border-[#30363d] rounded-md px-3 py-1.5 text-sm text-white w-32 outline-none focus:border-[#58a6ff]"
              />
            </div>

            <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
              <div>
                <div className="text-white font-medium">Tab Size</div>
                <div className="text-sm text-[#8b949e]">Editor indentation</div>
              </div>
              <select
                value={localSettings.tabSize}
                onChange={(e) => updateLocal('tabSize', parseInt(e.target.value))}
                className="bg-[#21262d] border border-[#30363d] rounded-md px-3 py-1.5 text-sm text-white outline-none"
              >
                <option value={2}>2 spaces</option>
                <option value={4}>4 spaces</option>
                <option value={8}>8 spaces</option>
              </select>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
              <div>
                <div className="text-white font-medium">Auto Save</div>
                <div className="text-sm text-[#8b949e]">Automatically save files</div>
              </div>
              <button
                onClick={() => updateLocal('autoSaveEnabled', !localSettings.autoSaveEnabled)}
                className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.autoSaveEnabled ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
              >
                <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.autoSaveEnabled ? 'left-7' : 'left-1'}`} />
              </button>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
              <div>
                <div className="text-white font-medium">Format on Save</div>
                <div className="text-sm text-[#8b949e]">Auto-format code when saving</div>
              </div>
              <button
                onClick={() => updateLocal('formatOnSave', !localSettings.formatOnSave)}
                className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.formatOnSave ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
              >
                <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.formatOnSave ? 'left-7' : 'left-1'}`} />
              </button>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-[#30363d]">
              <div>
                <div className="text-white font-medium">Show Minimap</div>
                <div className="text-sm text-[#8b949e]">Code preview sidebar</div>
              </div>
              <button
                onClick={() => updateLocal('showMinimap', !localSettings.showMinimap)}
                className={`w-12 h-6 rounded-full transition-colors relative ${localSettings.showMinimap ? 'bg-[#238636]' : 'bg-[#21262d]'}`}
              >
                <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localSettings.showMinimap ? 'left-7' : 'left-1'}`} />
              </button>
            </div>
          </div>
        );

      case 'integrations':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white mb-6">Integrations</h2>
            
            {[
              { name: 'GitHub', key: 'githubToken' as const, connected: !!localSettings.githubToken, icon: '🐙' },
              { name: 'Vercel', key: 'vercelToken' as const, connected: !!localSettings.vercelToken, icon: '▲' },
              { name: 'OpenAI', key: 'openaiKey' as const, connected: !!localSettings.openaiKey, icon: '🤖' },
              { name: 'Slack', key: 'slackWebhook' as const, connected: !!localSettings.slackWebhook, icon: '💬' },
            ].map(integration => (
              <div key={integration.name} className="p-4 bg-[#161b22] rounded-lg border border-[#30363d]">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{integration.icon}</span>
                    <div>
                      <div className="text-white font-medium">{integration.name}</div>
                      <div className="text-xs text-[#8b949e]">
                        {integration.connected ? 'Connected' : 'Not connected'}
                      </div>
                    </div>
                  </div>
                  <div className={`w-3 h-3 rounded-full ${integration.connected ? 'bg-[#238636]' : 'bg-[#8b949e]'}`} />
                </div>
                <input
                  type="password"
                  value={localSettings[integration.key] || ''}
                  onChange={(e) => updateLocal(integration.key, e.target.value)}
                  placeholder={`Enter ${integration.name} API key...`}
                  className="w-full bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm text-white outline-none focus:border-[#58a6ff]"
                />
              </div>
            ))}
          </div>
        );

      case 'about':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-white mb-6">About DevAgent</h2>
            
            <div className="text-center py-8">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-[#58a6ff] to-[#a855f7] rounded-2xl flex items-center justify-center text-3xl">
                🤖
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">DevAgent</h3>
              <p className="text-[#8b949e]">Your AI-powered development assistant</p>
            </div>

            <div className="space-y-3">
              {[
                { label: 'Version', value: '2.0.0' },
                { label: 'Build', value: '2025.01.21' },
                { label: 'AI Model', value: 'Llama 3.3 70B' },
                { label: 'Runtime', value: 'Next.js 15' },
                { label: 'License', value: 'MIT' },
              ].map(item => (
                <div key={item.label} className="flex items-center justify-between py-2 border-b border-[#30363d]">
                  <span className="text-[#8b949e]">{item.label}</span>
                  <span className="text-white">{item.value}</span>
                </div>
              ))}
            </div>

            <div className="pt-4 text-center text-[#8b949e] text-sm">
              <p>Built with ❤️ by the DevAgent team</p>
              <p className="mt-2">© 2025 DevAgent. All rights reserved.</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="h-full bg-[#0d1117] text-[#c9d1d9] flex">
      <div className="w-48 bg-[#161b22] border-r border-[#30363d] p-3 flex flex-col">
        <div className="space-y-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 w-full p-2 rounded-md text-sm transition-colors ${
                activeTab === tab.id
                  ? 'bg-[#21262d] text-white'
                  : 'text-[#8b949e] hover:bg-[#21262d] hover:text-white'
              }`}
            >
              {tab.icon}
              {tab.label}
              {activeTab === tab.id && <ChevronRight size={14} className="ml-auto" />}
            </button>
          ))}
        </div>
        
        <div className="mt-auto pt-4 border-t border-[#30363d] space-y-2">
          <button
            onClick={handleSave}
            className="flex items-center justify-center gap-2 w-full p-2 rounded-md text-sm bg-[#238636] hover:bg-[#2ea043] text-white transition-colors"
          >
            {saved ? <Check size={16} /> : <Save size={16} />}
            {saved ? 'Saved!' : 'Save'}
          </button>
          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 w-full p-2 rounded-md text-sm bg-[#21262d] hover:bg-[#30363d] text-[#8b949e] transition-colors"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        {renderContent()}
      </div>
    </div>
  );
}
