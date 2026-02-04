"use client";

import { useState, useRef, useEffect } from 'react';
import { Hash, Send, Plus, Search, Settings, AtSign, Smile, Paperclip, MoreHorizontal, ChevronDown, Users } from 'lucide-react';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

interface Message {
  id: string;
  channel: string;
  sender: string;
  content: string;
  timestamp: string;
  is_agent: boolean;
}

export default function TeamChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeChannel, setActiveChannel] = useState('general');
  const [messageInput, setMessageInput] = useState('');
  const [loading, setLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchMessages();
      const subscription = supabase
        .channel('messages_changes')
        .on('postgres_changes' as any, { event: 'INSERT', table: 'messages' }, (payload: any) => {
          const newMessage = payload.new as Message;

        if (newMessage.channel === activeChannel) {
          setMessages(prev => [...prev, newMessage]);
        }
      })
      .subscribe();

    return () => {
      supabase.removeChannel(subscription);
    };
  }, [activeChannel]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function fetchMessages() {
    setLoading(true);
    const { data, error } = await supabase
      .from('messages')
      .select('*')
      .eq('channel', activeChannel)
      .order('timestamp', { ascending: true });

    if (error) console.error('Error fetching messages:', error);
    else setMessages(data || []);
    setLoading(false);
  }

  async function handleSendMessage() {
    if (!messageInput.trim()) return;

    const { error } = await supabase
      .from('messages')
      .insert([{
        channel: activeChannel,
        sender: 'You',
        content: messageInput.trim(),
        is_agent: false
      }]);

    if (error) console.error('Error sending message:', error);
    else setMessageInput('');
  }

  const channels = ['general', 'dev', 'standup', 'random'];

  return (
    <div className="h-full bg-[#1a1d21] text-white flex font-sans">
      <div className="w-60 bg-[#19171d] border-r border-[#35363a] flex flex-col hidden md:flex">
        <div className="h-12 px-4 flex items-center justify-between border-b border-[#35363a]">
          <div className="flex items-center gap-2">
            <span className="font-bold">Workspace</span>
            <ChevronDown size={14} />
          </div>
          <button className="p-1 hover:bg-white/10 rounded">
            <Settings size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-2">
          <div className="px-3 mb-2">
            <div className="flex items-center justify-between text-[#ababad] text-xs font-semibold mb-1">
              <span>Channels</span>
              <button className="hover:bg-white/10 rounded p-0.5">
                <Plus size={14} />
              </button>
            </div>
            
            {channels.map((channel) => (
              <button
                key={channel}
                onClick={() => setActiveChannel(channel)}
                className={`w-full flex items-center gap-2 px-2 py-1 rounded text-sm ${
                  activeChannel === channel
                    ? 'bg-[#1164a3] text-white'
                    : 'text-[#ababad] hover:bg-white/5'
                }`}
              >
                <Hash size={14} />
                <span className="flex-1 text-left">{channel}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <div className="h-12 px-4 flex items-center justify-between border-b border-[#35363a]">
          <div className="flex items-center gap-2">
            <Hash size={18} className="text-[#ababad]" />
            <span className="font-bold">{activeChannel}</span>
          </div>
          <div className="flex items-center gap-2">
            <Users size={18} className="text-[#ababad]" />
            <Search size={18} className="text-[#ababad]" />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-2">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
            </div>
          ) : messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-[#ababad]">
              <Hash size={48} className="mb-2" />
              <span className="text-lg">Welcome to #{activeChannel}</span>
              <span className="text-sm">No messages yet.</span>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, i) => (
                <div key={msg.id} className="group flex items-start gap-3">
                  <div className={`w-9 h-9 rounded flex-shrink-0 flex items-center justify-center font-bold ${
                    msg.is_agent ? 'bg-blue-600' : 'bg-purple-600'
                  }`}>
                    {msg.sender[0]}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2">
                      <span className="font-bold">{msg.sender}</span>
                      <span className="text-xs text-[#ababad]">
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <div className="text-sm whitespace-pre-wrap break-words text-slate-200">{msg.content}</div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="px-4 pb-4">
          <div className="bg-[#222529] rounded-lg border border-[#35363a] overflow-hidden">
            <div className="flex items-center gap-1 px-2 py-1 border-b border-[#35363a]">
              <button className="p-1.5 hover:bg-white/10 rounded"><Plus size={16} /></button>
              <button className="p-1.5 hover:bg-white/10 rounded text-[#ababad]"><Paperclip size={16} /></button>
              <button className="p-1.5 hover:bg-white/10 rounded text-[#ababad]"><Smile size={16} /></button>
              <button className="p-1.5 hover:bg-white/10 rounded text-[#ababad]"><AtSign size={16} /></button>
            </div>
            
            <div className="flex items-center px-3 py-2">
              <input
                type="text"
                value={messageInput}
                onChange={(e) => setMessageInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder={`Message #${activeChannel}`}
                className="flex-1 bg-transparent outline-none text-sm placeholder-[#ababad]"
              />
              <button 
                onClick={handleSendMessage}
                disabled={!messageInput.trim()}
                className={`p-1.5 rounded ${messageInput.trim() ? 'bg-[#007a5a] text-white' : 'text-[#ababad]'}`}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
