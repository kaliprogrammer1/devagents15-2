"use client";

import { useState } from 'react';
import { useWorkspaceStore, Task } from '@/lib/workspaceStore';
import { Plus, MoreHorizontal, Clock, Tag, User, AlertCircle, CheckCircle2, Circle, ArrowUpCircle, Flame } from 'lucide-react';

const COLUMNS: { id: Task['status']; title: string; color: string }[] = [
  { id: 'backlog', title: 'Backlog', color: '#8b949e' },
  { id: 'todo', title: 'To Do', color: '#58a6ff' },
  { id: 'in_progress', title: 'In Progress', color: '#d29922' },
  { id: 'review', title: 'Review', color: '#a371f7' },
  { id: 'done', title: 'Done', color: '#7ee787' },
];

const PRIORITY_CONFIG = {
  low: { icon: Circle, color: '#8b949e', label: 'Low' },
  medium: { icon: ArrowUpCircle, color: '#58a6ff', label: 'Medium' },
  high: { icon: AlertCircle, color: '#d29922', label: 'High' },
  urgent: { icon: Flame, color: '#f85149', label: 'Urgent' },
};

export default function TaskBoardApp() {
  const { tasks, activeTaskId, addTask, updateTask, moveTask, setActiveTask } = useWorkspaceStore();
  
  const [showNewTask, setShowNewTask] = useState(false);
  const [newTaskColumn, setNewTaskColumn] = useState<Task['status']>('todo');
  const [newTaskTitle, setNewTaskTitle] = useState('');
  const [newTaskDescription, setNewTaskDescription] = useState('');
  const [newTaskPriority, setNewTaskPriority] = useState<Task['priority']>('medium');
  const [draggedTask, setDraggedTask] = useState<string | null>(null);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);

  const handleAddTask = () => {
    if (newTaskTitle.trim()) {
      addTask({
        title: newTaskTitle.trim(),
        description: newTaskDescription.trim(),
        status: newTaskColumn,
        priority: newTaskPriority,
        assignee: 'DevAgent',
        storyPoints: 2,
        labels: [],
      });
      setNewTaskTitle('');
      setNewTaskDescription('');
      setShowNewTask(false);
    }
  };

  const handleDragStart = (taskId: string) => {
    setDraggedTask(taskId);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (status: Task['status']) => {
    if (draggedTask) {
      moveTask(draggedTask, status);
      setDraggedTask(null);
    }
  };

  const formatTimeAgo = (timestamp: number) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    return `${Math.floor(seconds / 86400)}d`;
  };

  return (
    <div className="h-full bg-[#0d1117] text-[#c9d1d9] flex flex-col">
      <div className="h-12 bg-[#161b22] border-b border-[#30363d] flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <span className="font-semibold">Task Board</span>
          <span className="text-xs text-[#8b949e] bg-[#21262d] px-2 py-0.5 rounded">
            {tasks.length} tasks
          </span>
        </div>
        <button 
          onClick={() => { setShowNewTask(true); setNewTaskColumn('todo'); }}
          className="flex items-center gap-1.5 text-xs bg-[#238636] hover:bg-[#2ea043] px-3 py-1.5 rounded-md"
        >
          <Plus size={14} />
          New Task
        </button>
      </div>

      <div className="flex-1 flex overflow-x-auto p-4 gap-4">
        {COLUMNS.map((column) => {
          const columnTasks = tasks.filter(t => t.status === column.id);
          
          return (
            <div 
              key={column.id}
              className="w-72 flex-shrink-0 flex flex-col"
              onDragOver={handleDragOver}
              onDrop={() => handleDrop(column.id)}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: column.color }} />
                  <span className="text-sm font-medium">{column.title}</span>
                  <span className="text-xs text-[#8b949e] bg-[#21262d] px-1.5 rounded">
                    {columnTasks.length}
                  </span>
                </div>
                <button 
                  onClick={() => { setShowNewTask(true); setNewTaskColumn(column.id); }}
                  className="p-1 hover:bg-[#21262d] rounded"
                >
                  <Plus size={14} />
                </button>
              </div>

              <div className="flex-1 space-y-2 overflow-y-auto">
                {columnTasks.map((task) => {
                  const PriorityIcon = PRIORITY_CONFIG[task.priority].icon;
                  
                  return (
                    <div
                      key={task.id}
                      draggable
                      onDragStart={() => handleDragStart(task.id)}
                      onClick={() => setSelectedTask(task)}
                      className={`bg-[#161b22] border rounded-lg p-3 cursor-pointer hover:border-[#30363d] transition-all ${
                        draggedTask === task.id ? 'opacity-50' : ''
                      } ${
                        activeTaskId === task.id ? 'border-[#58a6ff]' : 'border-[#21262d]'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <span className="text-sm font-medium leading-tight">{task.title}</span>
                        <button className="p-0.5 hover:bg-[#21262d] rounded opacity-0 group-hover:opacity-100">
                          <MoreHorizontal size={14} />
                        </button>
                      </div>
                      
                      {task.description && (
                        <p className="text-xs text-[#8b949e] mb-2 line-clamp-2">{task.description}</p>
                      )}
                      
                      <div className="flex flex-wrap gap-1 mb-2">
                        {task.labels.map((label, i) => (
                          <span 
                            key={i}
                            className="text-[10px] px-1.5 py-0.5 rounded-full bg-[#388bfd33] text-[#58a6ff]"
                          >
                            {label}
                          </span>
                        ))}
                      </div>
                      
                      <div className="flex items-center justify-between text-xs text-[#8b949e]">
                        <div className="flex items-center gap-2">
                          <PriorityIcon 
                            size={12} 
                            style={{ color: PRIORITY_CONFIG[task.priority].color }} 
                          />
                          <span className="bg-[#21262d] px-1.5 py-0.5 rounded">{task.storyPoints}SP</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock size={10} />
                          <span>{formatTimeAgo(task.updatedAt)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
                
                {columnTasks.length === 0 && (
                  <div className="text-center text-[#8b949e] text-xs py-8 border border-dashed border-[#30363d] rounded-lg">
                    No tasks
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {showNewTask && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg w-[400px] p-4">
            <h3 className="text-lg font-semibold mb-4">Create New Task</h3>
            
            <div className="space-y-3">
              <input
                type="text"
                value={newTaskTitle}
                onChange={(e) => setNewTaskTitle(e.target.value)}
                placeholder="Task title..."
                className="w-full bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none focus:border-[#58a6ff]"
                autoFocus
              />
              
              <textarea
                value={newTaskDescription}
                onChange={(e) => setNewTaskDescription(e.target.value)}
                placeholder="Description (optional)..."
                className="w-full h-20 bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none focus:border-[#58a6ff] resize-none"
              />
              
              <div className="flex gap-2">
                <select
                  value={newTaskColumn}
                  onChange={(e) => setNewTaskColumn(e.target.value as Task['status'])}
                  className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none"
                >
                  {COLUMNS.map(col => (
                    <option key={col.id} value={col.id}>{col.title}</option>
                  ))}
                </select>
                
                <select
                  value={newTaskPriority}
                  onChange={(e) => setNewTaskPriority(e.target.value as Task['priority'])}
                  className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none"
                >
                  {Object.entries(PRIORITY_CONFIG).map(([key, config]) => (
                    <option key={key} value={key}>{config.label}</option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="flex justify-end gap-2 mt-4">
              <button 
                onClick={() => setShowNewTask(false)}
                className="px-4 py-2 text-sm bg-[#21262d] hover:bg-[#30363d] rounded-md"
              >
                Cancel
              </button>
              <button 
                onClick={handleAddTask}
                disabled={!newTaskTitle.trim()}
                className="px-4 py-2 text-sm bg-[#238636] hover:bg-[#2ea043] disabled:opacity-50 rounded-md"
              >
                Create Task
              </button>
            </div>
          </div>
        </div>
      )}

      {selectedTask && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedTask(null)}>
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg w-[500px] max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="p-4 border-b border-[#30363d]">
              <div className="flex items-center gap-2 text-xs text-[#8b949e] mb-2">
                <span className="bg-[#21262d] px-2 py-0.5 rounded">{selectedTask.id}</span>
                <span>•</span>
                <span>{COLUMNS.find(c => c.id === selectedTask.status)?.title}</span>
              </div>
              <h2 className="text-xl font-semibold">{selectedTask.title}</h2>
            </div>
            
            <div className="p-4 space-y-4">
              <div>
                <label className="text-xs text-[#8b949e] uppercase mb-1 block">Description</label>
                <p className="text-sm">{selectedTask.description || 'No description'}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-[#8b949e] uppercase mb-1 block">Priority</label>
                  <div className="flex items-center gap-2">
                    {(() => {
                      const PriorityIcon = PRIORITY_CONFIG[selectedTask.priority].icon;
                      return (
                        <>
                          <PriorityIcon size={14} style={{ color: PRIORITY_CONFIG[selectedTask.priority].color }} />
                          <span className="text-sm">{PRIORITY_CONFIG[selectedTask.priority].label}</span>
                        </>
                      );
                    })()}
                  </div>
                </div>
                
                <div>
                  <label className="text-xs text-[#8b949e] uppercase mb-1 block">Assignee</label>
                  <div className="flex items-center gap-2">
                    <User size={14} />
                    <span className="text-sm">{selectedTask.assignee}</span>
                  </div>
                </div>
                
                <div>
                  <label className="text-xs text-[#8b949e] uppercase mb-1 block">Story Points</label>
                  <span className="text-sm">{selectedTask.storyPoints} SP</span>
                </div>
                
                <div>
                  <label className="text-xs text-[#8b949e] uppercase mb-1 block">Updated</label>
                  <span className="text-sm">{new Date(selectedTask.updatedAt).toLocaleDateString()}</span>
                </div>
              </div>
              
              {selectedTask.labels.length > 0 && (
                <div>
                  <label className="text-xs text-[#8b949e] uppercase mb-1 block">Labels</label>
                  <div className="flex flex-wrap gap-1">
                    {selectedTask.labels.map((label, i) => (
                      <span key={i} className="text-xs px-2 py-1 rounded-full bg-[#388bfd33] text-[#58a6ff]">
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            <div className="p-4 border-t border-[#30363d] flex justify-between">
              <select
                value={selectedTask.status}
                onChange={(e) => {
                  moveTask(selectedTask.id, e.target.value as Task['status']);
                  setSelectedTask({ ...selectedTask, status: e.target.value as Task['status'] });
                }}
                className="bg-[#21262d] border border-[#30363d] rounded px-3 py-1.5 text-sm"
              >
                {COLUMNS.map(col => (
                  <option key={col.id} value={col.id}>{col.title}</option>
                ))}
              </select>
              
              <button 
                onClick={() => setSelectedTask(null)}
                className="px-4 py-1.5 text-sm bg-[#21262d] hover:bg-[#30363d] rounded"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
