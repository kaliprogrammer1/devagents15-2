"use client";

import { useState, useEffect } from 'react';
import { useWorkspaceStore, CalendarEvent } from '@/lib/workspaceStore';
import { ChevronLeft, ChevronRight, Plus, Clock, Calendar as CalendarIcon, Users, Video, Focus, Coffee, GitPullRequest } from 'lucide-react';

const EVENT_COLORS: Record<CalendarEvent['type'], { bg: string; border: string; icon: React.ReactNode }> = {
  meeting: { bg: 'bg-blue-500/20', border: 'border-blue-500', icon: <Video size={12} /> },
  focus: { bg: 'bg-purple-500/20', border: 'border-purple-500', icon: <Focus size={12} /> },
  break: { bg: 'bg-green-500/20', border: 'border-green-500', icon: <Coffee size={12} /> },
  standup: { bg: 'bg-yellow-500/20', border: 'border-yellow-500', icon: <Users size={12} /> },
  review: { bg: 'bg-orange-500/20', border: 'border-orange-500', icon: <GitPullRequest size={12} /> },
};

const HOURS = Array.from({ length: 13 }, (_, i) => i + 7);

export default function CalendarApp() {
  const { calendarEvents, addCalendarEvent, removeCalendarEvent } = useWorkspaceStore();
  
  const [currentDate, setCurrentDate] = useState(new Date());
  const [view, setView] = useState<'day' | 'week'>('day');
  const [showNewEvent, setShowNewEvent] = useState(false);
  const [newEventTitle, setNewEventTitle] = useState('');
  const [newEventType, setNewEventType] = useState<CalendarEvent['type']>('meeting');
  const [newEventStart, setNewEventStart] = useState('09:00');
  const [newEventEnd, setNewEventEnd] = useState('10:00');
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  const getEventsForDay = (date: Date) => {
    return calendarEvents.filter(event => {
      const eventDate = new Date(event.start);
      return eventDate.toDateString() === date.toDateString();
    }).sort((a, b) => a.start - b.start);
  };

  const todayEvents = getEventsForDay(currentDate);

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getEventPosition = (event: CalendarEvent) => {
    const startDate = new Date(event.start);
    const endDate = new Date(event.end);
    const startHour = startDate.getHours() + startDate.getMinutes() / 60;
    const endHour = endDate.getHours() + endDate.getMinutes() / 60;
    const top = (startHour - 7) * 60;
    const height = (endHour - startHour) * 60;
    return { top, height };
  };

  const getCurrentTimePosition = () => {
    const hour = currentTime.getHours() + currentTime.getMinutes() / 60;
    return (hour - 7) * 60;
  };

  const handleAddEvent = () => {
    if (newEventTitle.trim()) {
      const [startH, startM] = newEventStart.split(':').map(Number);
      const [endH, endM] = newEventEnd.split(':').map(Number);
      
      const startDate = new Date(currentDate);
      startDate.setHours(startH, startM, 0, 0);
      
      const endDate = new Date(currentDate);
      endDate.setHours(endH, endM, 0, 0);
      
      addCalendarEvent({
        title: newEventTitle.trim(),
        start: startDate.getTime(),
        end: endDate.getTime(),
        type: newEventType,
      });
      
      setNewEventTitle('');
      setShowNewEvent(false);
    }
  };

  const navigateDate = (direction: number) => {
    const newDate = new Date(currentDate);
    newDate.setDate(newDate.getDate() + direction);
    setCurrentDate(newDate);
  };

  const isToday = currentDate.toDateString() === new Date().toDateString();

  return (
    <div className="h-full bg-[#0d1117] text-[#c9d1d9] flex flex-col">
      <div className="h-14 bg-[#161b22] border-b border-[#30363d] flex items-center justify-between px-4">
        <div className="flex items-center gap-4">
          <CalendarIcon size={20} className="text-[#58a6ff]" />
          <div className="flex items-center gap-2">
            <button 
              onClick={() => navigateDate(-1)}
              className="p-1 hover:bg-[#21262d] rounded"
            >
              <ChevronLeft size={18} />
            </button>
            <button 
              onClick={() => setCurrentDate(new Date())}
              className={`px-3 py-1 text-sm rounded ${isToday ? 'bg-[#238636]' : 'bg-[#21262d] hover:bg-[#30363d]'}`}
            >
              Today
            </button>
            <button 
              onClick={() => navigateDate(1)}
              className="p-1 hover:bg-[#21262d] rounded"
            >
              <ChevronRight size={18} />
            </button>
          </div>
          <span className="text-lg font-semibold">
            {currentDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="flex bg-[#21262d] rounded-md overflow-hidden">
            <button 
              onClick={() => setView('day')}
              className={`px-3 py-1.5 text-sm ${view === 'day' ? 'bg-[#30363d]' : 'hover:bg-[#30363d]'}`}
            >
              Day
            </button>
            <button 
              onClick={() => setView('week')}
              className={`px-3 py-1.5 text-sm ${view === 'week' ? 'bg-[#30363d]' : 'hover:bg-[#30363d]'}`}
            >
              Week
            </button>
          </div>
          <button 
            onClick={() => setShowNewEvent(true)}
            className="flex items-center gap-1.5 text-sm bg-[#238636] hover:bg-[#2ea043] px-3 py-1.5 rounded-md"
          >
            <Plus size={14} />
            Add Event
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="w-64 bg-[#161b22] border-r border-[#30363d] p-4 overflow-y-auto">
          <h3 className="text-sm font-semibold mb-3">Today's Schedule</h3>
          
          {todayEvents.length === 0 ? (
            <div className="text-center text-[#8b949e] py-4 text-sm">
              No events scheduled
            </div>
          ) : (
            <div className="space-y-2">
              {todayEvents.map((event) => {
                const config = EVENT_COLORS[event.type];
                return (
                  <div 
                    key={event.id}
                    className={`p-2 rounded-md border-l-2 ${config.bg} ${config.border}`}
                  >
                    <div className="flex items-center gap-1.5 text-xs text-[#8b949e] mb-1">
                      {config.icon}
                      <span>{formatTime(event.start)} - {formatTime(event.end)}</span>
                    </div>
                    <div className="text-sm font-medium">{event.title}</div>
                  </div>
                );
              })}
            </div>
          )}

          <div className="mt-6">
            <h3 className="text-sm font-semibold mb-3">Quick Actions</h3>
            <div className="space-y-1">
              <button 
                onClick={() => {
                  setNewEventTitle('Daily Standup');
                  setNewEventType('standup');
                  setNewEventStart('09:00');
                  setNewEventEnd('09:15');
                  setShowNewEvent(true);
                }}
                className="w-full text-left text-sm p-2 hover:bg-[#21262d] rounded flex items-center gap-2"
              >
                <Users size={14} className="text-yellow-400" />
                Add Standup
              </button>
              <button 
                onClick={() => {
                  setNewEventTitle('Focus Time');
                  setNewEventType('focus');
                  setNewEventStart('10:00');
                  setNewEventEnd('12:00');
                  setShowNewEvent(true);
                }}
                className="w-full text-left text-sm p-2 hover:bg-[#21262d] rounded flex items-center gap-2"
              >
                <Focus size={14} className="text-purple-400" />
                Add Focus Block
              </button>
              <button 
                onClick={() => {
                  setNewEventTitle('Code Review');
                  setNewEventType('review');
                  setNewEventStart('14:00');
                  setNewEventEnd('15:00');
                  setShowNewEvent(true);
                }}
                className="w-full text-left text-sm p-2 hover:bg-[#21262d] rounded flex items-center gap-2"
              >
                <GitPullRequest size={14} className="text-orange-400" />
                Add Review
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="relative" style={{ height: `${HOURS.length * 60}px` }}>
            {HOURS.map((hour) => (
              <div 
                key={hour} 
                className="absolute w-full flex border-t border-[#21262d]"
                style={{ top: `${(hour - 7) * 60}px`, height: '60px' }}
              >
                <div className="w-16 text-xs text-[#8b949e] pr-2 text-right -mt-2">
                  {hour === 12 ? '12 PM' : hour > 12 ? `${hour - 12} PM` : `${hour} AM`}
                </div>
                <div className="flex-1 border-l border-[#21262d]" />
              </div>
            ))}

            {isToday && currentTime.getHours() >= 7 && currentTime.getHours() < 20 && (
              <div 
                className="absolute left-16 right-0 border-t-2 border-red-500 z-20"
                style={{ top: `${getCurrentTimePosition()}px` }}
              >
                <div className="absolute -left-1 -top-1.5 w-3 h-3 bg-red-500 rounded-full" />
              </div>
            )}

            <div className="absolute left-16 right-0 top-0 bottom-0">
              {todayEvents.map((event) => {
                const { top, height } = getEventPosition(event);
                const config = EVENT_COLORS[event.type];
                
                return (
                  <div
                    key={event.id}
                    className={`absolute left-1 right-1 rounded-md border-l-4 ${config.bg} ${config.border} p-2 overflow-hidden cursor-pointer hover:opacity-80 transition-opacity`}
                    style={{ top: `${top}px`, height: `${Math.max(height, 24)}px` }}
                    onClick={() => removeCalendarEvent(event.id)}
                  >
                    <div className="flex items-center gap-1 text-xs mb-0.5">
                      {config.icon}
                      <span className="text-[#8b949e]">
                        {formatTime(event.start)} - {formatTime(event.end)}
                      </span>
                    </div>
                    {height >= 40 && (
                      <div className="text-sm font-medium truncate">{event.title}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {showNewEvent && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg w-[400px] p-4">
            <h3 className="text-lg font-semibold mb-4">Add Event</h3>
            
            <div className="space-y-3">
              <input
                type="text"
                value={newEventTitle}
                onChange={(e) => setNewEventTitle(e.target.value)}
                placeholder="Event title..."
                className="w-full bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none focus:border-[#58a6ff]"
                autoFocus
              />
              
              <select
                value={newEventType}
                onChange={(e) => setNewEventType(e.target.value as CalendarEvent['type'])}
                className="w-full bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none"
              >
                <option value="meeting">Meeting</option>
                <option value="focus">Focus Time</option>
                <option value="break">Break</option>
                <option value="standup">Standup</option>
                <option value="review">Code Review</option>
              </select>
              
              <div className="flex gap-2">
                <div className="flex-1">
                  <label className="text-xs text-[#8b949e] block mb-1">Start</label>
                  <input
                    type="time"
                    value={newEventStart}
                    onChange={(e) => setNewEventStart(e.target.value)}
                    className="w-full bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-xs text-[#8b949e] block mb-1">End</label>
                  <input
                    type="time"
                    value={newEventEnd}
                    onChange={(e) => setNewEventEnd(e.target.value)}
                    className="w-full bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2 text-sm outline-none"
                  />
                </div>
              </div>
            </div>
            
            <div className="flex justify-end gap-2 mt-4">
              <button 
                onClick={() => setShowNewEvent(false)}
                className="px-4 py-2 text-sm bg-[#21262d] hover:bg-[#30363d] rounded-md"
              >
                Cancel
              </button>
              <button 
                onClick={handleAddEvent}
                disabled={!newEventTitle.trim()}
                className="px-4 py-2 text-sm bg-[#238636] hover:bg-[#2ea043] disabled:opacity-50 rounded-md"
              >
                Add Event
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
