"use client";

import { useState, useEffect } from 'react';
import { Plus, Trash2, CheckCircle, Circle, Calendar, Tag, AlertCircle } from 'lucide-react';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

interface Todo {
  id: string;
  title: string;
  completed: boolean;
  created_at: string;
}

export default function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [newTodo, setNewTodo] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTodos();
      const subscription = supabase
        .channel('todos_changes')
        .on('postgres_changes' as any, { event: '*', table: 'todos' }, () => {
          fetchTodos();
        })
        .subscribe();


    return () => {
      supabase.removeChannel(subscription);
    };
  }, []);

  async function fetchTodos() {
    const { data, error } = await supabase
      .from('todos')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) console.error('Error fetching todos:', error);
    else setTodos(data || []);
    setLoading(loading && false);
  }

  async function addTodo(e: React.FormEvent) {
    e.preventDefault();
    if (!newTodo.trim()) return;

    const { error } = await supabase
      .from('todos')
      .insert([{ title: newTodo.trim(), completed: false }]);

    if (error) console.error('Error adding todo:', error);
    else setNewTodo('');
  }

  async function toggleTodo(id: string, completed: boolean) {
    const { error } = await supabase
      .from('todos')
      .update({ completed: !completed })
      .eq('id', id);

    if (error) console.error('Error updating todo:', error);
  }

  async function deleteTodo(id: string) {
    const { error } = await supabase
      .from('todos')
      .delete()
      .eq('id', id);

    if (error) console.error('Error deleting todo:', error);
  }

  return (
    <div className="h-full bg-[#f8fafc] text-slate-800 flex flex-col font-sans">
      <div className="p-6 border-b bg-white/50 backdrop-blur-md sticky top-0 z-10">
        <h1 className="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2">
          <CheckCircle className="text-blue-500" size={24} />
          Tasks
        </h1>
        <form onSubmit={addTodo} className="relative">
          <input
            type="text"
            value={newTodo}
            onChange={(e) => setNewTodo(e.target.value)}
            placeholder="Add a new task..."
            className="w-full bg-slate-100 border-none rounded-xl px-4 py-3 pl-11 text-sm focus:ring-2 focus:ring-blue-500/20 transition-all"
          />
          <Plus className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
        </form>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-3">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mb-2" />
            <p className="text-sm">Loading tasks...</p>
          </div>
        ) : todos.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-400 opacity-60">
            <Calendar size={48} className="mb-4" />
            <p>No tasks yet. Stay productive!</p>
          </div>
        ) : (
          todos.map((todo) => (
            <div
              key={todo.id}
              className="group flex items-center justify-between p-4 bg-white rounded-xl border border-slate-100 shadow-sm hover:shadow-md hover:border-blue-100 transition-all"
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => toggleTodo(todo.id, todo.completed)}
                  className={`transition-colors ${todo.completed ? 'text-blue-500' : 'text-slate-300 hover:text-blue-400'}`}
                >
                  {todo.completed ? <CheckCircle size={22} /> : <Circle size={22} />}
                </button>
                <span className={`text-sm font-medium transition-all ${todo.completed ? 'text-slate-400 line-through' : 'text-slate-700'}`}>
                  {todo.title}
                </span>
              </div>
              <button
                onClick={() => deleteTodo(todo.id)}
                className="opacity-0 group-hover:opacity-100 p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 transition-all rounded-lg"
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))
        )}
      </div>

      <div className="p-4 bg-slate-50 border-t flex items-center justify-between text-[10px] text-slate-400 font-bold uppercase tracking-wider">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <Tag size={10} /> {todos.length} Total
          </span>
          <span className="flex items-center gap-1">
            <CheckCircle size={10} /> {todos.filter(t => t.completed).length} Done
          </span>
        </div>
        <div className="flex items-center gap-1 text-blue-500">
          <AlertCircle size={10} /> Live Sync Active
        </div>
      </div>
    </div>
  );
}
