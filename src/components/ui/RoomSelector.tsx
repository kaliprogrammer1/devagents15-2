"use client";

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ROOMS } from '@/lib/rooms';
import { ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

interface RoomSelectorProps {
  currentRoomId: string;
  onRoomChange: (roomId: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

function RoomThumbnail({
  room,
  isActive,
  onClick,
  onSelect,
}: {
  room: typeof ROOMS[0];
  isActive: boolean;
  onClick: () => void;
  onSelect: () => void;
}) {
  const handleClick = () => {
    onClick();
    onSelect();
  };
  
  return (
    <button
      type="button"
      onClick={handleClick}
      className="relative flex flex-col items-center gap-2 group cursor-pointer bg-transparent border-none outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50 focus-visible:ring-offset-2 rounded-lg"
      aria-label={`Switch to ${room.name} workspace`}
    >
      <div className="relative w-[72px] h-[72px]">
        {/* Segmented ring indicator */}
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 100 100"
        >
          <circle
            cx="50"
            cy="50"
            r="46"
            fill="none"
            strokeWidth="4"
            strokeLinecap="round"
            className={`transition-colors duration-300 ${
              isActive
                ? 'stroke-cyan-400'
                : 'stroke-white/20 group-hover:stroke-white/40'
            }`}
          />
        </svg>

        {/* Avatar container with gradient */}
        <div className="absolute inset-[5px] rounded-full bg-black/30 p-[2px]">
          <div
            className="w-full h-full rounded-full overflow-hidden transition-transform duration-300 group-hover:scale-105"
            style={{
              background: `linear-gradient(135deg, ${room.colors.accent}, ${room.colors.wall})`,
            }}
          >
            {/* Overlay icon or visual */}
            <div className="w-full h-full flex items-center justify-center">
                <Sparkles
                  size={24}
                  className={`transition-all duration-300 ${
                    isActive ? 'text-white' : 'text-slate-200 group-hover:text-white'
                  }`}
                />
            </div>
          </div>
        </div>

        {/* Active indicator dot */}
        {isActive && (
          <motion.div
            layoutId="activeIndicator"
            className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-cyan-400 rounded-full shadow-lg shadow-cyan-400/50"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
          />
        )}
      </div>

        <span
          className={`text-xs truncate max-w-[80px] transition-colors duration-200 ${
            isActive ? 'text-cyan-600 font-bold' : 'text-slate-500 group-hover:text-slate-700'
          }`}
        >
          {room.name.replace('The ', '')}
        </span>
    </button>
  );
}

export default function RoomSelector({
  currentRoomId,
  onRoomChange,
  isOpen,
  onToggle,
}: RoomSelectorProps) {
  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50">
      <AnimatePresence>
        {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              transition={{ duration: 0.3, ease: [0.32, 0.72, 0, 1] }}
              className="bg-white/95 backdrop-blur-2xl border border-white/20 rounded-2xl p-4 shadow-2xl"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4 px-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-600 rounded-full animate-pulse" />
                  <span className="text-slate-500 text-xs font-bold uppercase tracking-wider">
                    AI Workspaces
                  </span>
                </div>
                <button
                  onClick={onToggle}
                  className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-400 hover:text-slate-600 transition-colors"
                >
                  <ChevronUp size={14} />
                </button>
              </div>

              {/* Room thumbnails - horizontal scroll */}
                <div className="flex gap-4 overflow-x-auto py-2 px-1 [&::-webkit-scrollbar]:hidden md:[&::-webkit-scrollbar]:block md:[&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-slate-200 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-slate-300">
                  {ROOMS.map((room) => (
                    <RoomThumbnail
                      key={room.id}
                      room={room}
                      isActive={currentRoomId === room.id}
                      onClick={() => onRoomChange(room.id)}
                      onSelect={onToggle}
                    />
                  ))}
                </div>

              {/* Current room info */}
              <div className="mt-3 pt-3 border-t border-slate-100 px-2">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-900 text-sm font-bold">
                      {ROOMS.find((r) => r.id === currentRoomId)?.name}
                    </p>
                    <p className="text-slate-500 text-[10px] italic font-medium">
                      {ROOMS.find((r) => r.id === currentRoomId)?.persona}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                    <span className="text-[10px] text-emerald-600 font-bold uppercase tracking-wider">
                      Active
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
        )}
      </AnimatePresence>

      {/* Collapsed toggle button */}
      {!isOpen && (
        <motion.button
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          onClick={onToggle}
          className="flex items-center gap-3 px-4 py-2.5 bg-white/80 backdrop-blur-2xl border border-white/20 rounded-full text-slate-800 hover:text-slate-900 hover:bg-white/90 transition-all shadow-lg"
        >
          <div
            className="w-8 h-8 rounded-full"
            style={{
              background: `linear-gradient(135deg, ${
                ROOMS.find((r) => r.id === currentRoomId)?.colors.accent || '#3b82f6'
              }, ${ROOMS.find((r) => r.id === currentRoomId)?.colors.wall || '#f3f4f6'})`,
            }}
          />
          <span className="text-sm font-medium text-slate-800">
            {ROOMS.find((r) => r.id === currentRoomId)?.name.replace('The ', '')}
          </span>
          <ChevronDown size={16} className="text-slate-600" />
        </motion.button>
      )}
    </div>
  );
}
