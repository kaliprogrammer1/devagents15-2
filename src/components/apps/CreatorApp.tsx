"use client";

import { useState, useRef, useEffect } from 'react';
import { Palette, Brush, Square, Circle, Type, Image, Layers, Undo, Redo, Download, Trash2, Sun, Sparkles, Wand2 } from 'lucide-react';

interface CanvasElement {
  id: string;
  type: 'rect' | 'circle' | 'text' | 'freehand';
  x: number;
  y: number;
  width?: number;
  height?: number;
  radius?: number;
  text?: string;
  color: string;
  points?: { x: number; y: number }[];
}

const colorPalette = [
  '#f43f5e', '#ec4899', '#a855f7', '#8b5cf6', '#6366f1',
  '#3b82f6', '#0ea5e9', '#14b8a6', '#22c55e', '#84cc16',
  '#eab308', '#f97316', '#1e293b', '#64748b', '#f1f5f9',
];

export default function CreatorApp() {
  const [elements, setElements] = useState<CanvasElement[]>([]);
  const [selectedTool, setSelectedTool] = useState<'select' | 'rect' | 'circle' | 'text' | 'freehand'>('select');
  const [selectedColor, setSelectedColor] = useState('#6366f1');
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPath, setCurrentPath] = useState<{ x: number; y: number }[]>([]);
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [history, setHistory] = useState<CanvasElement[][]>([[]]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const canvasRef = useRef<HTMLDivElement>(null);

  const saveToHistory = (newElements: CanvasElement[]) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newElements);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setElements(history[historyIndex - 1]);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setElements(history[historyIndex + 1]);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (selectedTool === 'rect') {
      const newElement: CanvasElement = {
        id: Date.now().toString(),
        type: 'rect',
        x: x - 50,
        y: y - 30,
        width: 100,
        height: 60,
        color: selectedColor,
      };
      const newElements = [...elements, newElement];
      setElements(newElements);
      saveToHistory(newElements);
    } else if (selectedTool === 'circle') {
      const newElement: CanvasElement = {
        id: Date.now().toString(),
        type: 'circle',
        x,
        y,
        radius: 40,
        color: selectedColor,
      };
      const newElements = [...elements, newElement];
      setElements(newElements);
      saveToHistory(newElements);
    } else if (selectedTool === 'text') {
      const text = prompt('Enter text:');
      if (text) {
        const newElement: CanvasElement = {
          id: Date.now().toString(),
          type: 'text',
          x,
          y,
          text,
          color: selectedColor,
        };
        const newElements = [...elements, newElement];
        setElements(newElements);
        saveToHistory(newElements);
      }
    } else if (selectedTool === 'select') {
      setSelectedElement(null);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (selectedTool === 'freehand' && canvasRef.current) {
      setIsDrawing(true);
      const rect = canvasRef.current.getBoundingClientRect();
      setCurrentPath([{ x: e.clientX - rect.left, y: e.clientY - rect.top }]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDrawing && selectedTool === 'freehand' && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      setCurrentPath(prev => [...prev, { x: e.clientX - rect.left, y: e.clientY - rect.top }]);
    }
  };

  const handleMouseUp = () => {
    if (isDrawing && currentPath.length > 1) {
      const newElement: CanvasElement = {
        id: Date.now().toString(),
        type: 'freehand',
        x: 0,
        y: 0,
        points: currentPath,
        color: selectedColor,
      };
      const newElements = [...elements, newElement];
      setElements(newElements);
      saveToHistory(newElements);
    }
    setIsDrawing(false);
    setCurrentPath([]);
  };

  const deleteSelected = () => {
    if (selectedElement) {
      const newElements = elements.filter(e => e.id !== selectedElement);
      setElements(newElements);
      saveToHistory(newElements);
      setSelectedElement(null);
    }
  };

  const generateIdea = () => {
    const ideas = [
      'Try adding a gradient background',
      'How about some geometric patterns?',
      'Consider adding depth with shadows',
      'Mix complementary colors for contrast',
      'Add some organic, flowing shapes',
      'Try the golden ratio for proportions',
    ];
    alert(ideas[Math.floor(Math.random() * ideas.length)]);
  };

  return (
    <div className="h-full bg-[#fdf6e3] text-slate-800 flex flex-col font-sans text-xs">
      {/* Header */}
      <div className="p-3 border-b border-amber-200 bg-white/50 backdrop-blur-sm flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Palette size={18} className="text-orange-500" />
            <span className="font-bold text-slate-900 text-sm">Creative Canvas</span>
          </div>
          <div className="h-4 w-px bg-amber-200" />
          <div className="flex items-center gap-1">
            <button onClick={undo} disabled={historyIndex === 0} className="p-1.5 hover:bg-amber-100 rounded transition-colors disabled:opacity-30">
              <Undo size={14} />
            </button>
            <button onClick={redo} disabled={historyIndex === history.length - 1} className="p-1.5 hover:bg-amber-100 rounded transition-colors disabled:opacity-30">
              <Redo size={14} />
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={generateIdea} className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg text-[10px] font-bold hover:opacity-90 transition-opacity">
            <Sparkles size={12} />
            Inspire Me
          </button>
          <button className="p-1.5 hover:bg-amber-100 rounded transition-colors">
            <Download size={14} />
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Tools Sidebar */}
        <div className="w-16 border-r border-amber-200 bg-white/30 p-2 flex flex-col items-center gap-2">
          {[
            { tool: 'select' as const, icon: Wand2, label: 'Select' },
            { tool: 'rect' as const, icon: Square, label: 'Rectangle' },
            { tool: 'circle' as const, icon: Circle, label: 'Circle' },
            { tool: 'freehand' as const, icon: Brush, label: 'Draw' },
            { tool: 'text' as const, icon: Type, label: 'Text' },
          ].map(({ tool, icon: Icon, label }) => (
            <button
              key={tool}
              onClick={() => setSelectedTool(tool)}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${
                selectedTool === tool 
                  ? 'bg-orange-500 text-white shadow-lg shadow-orange-500/30' 
                  : 'hover:bg-amber-100 text-slate-600'
              }`}
              title={label}
            >
              <Icon size={18} />
            </button>
          ))}
          
          <div className="my-2 w-8 h-px bg-amber-200" />
          
          <button 
            onClick={deleteSelected}
            disabled={!selectedElement}
            className="w-10 h-10 rounded-xl flex items-center justify-center hover:bg-rose-100 text-rose-500 transition-all disabled:opacity-30"
            title="Delete"
          >
            <Trash2 size={18} />
          </button>
        </div>

        {/* Canvas Area */}
        <div className="flex-1 p-4 overflow-hidden">
          <div 
            ref={canvasRef}
            className="w-full h-full bg-white rounded-2xl shadow-inner border border-amber-100 relative overflow-hidden cursor-crosshair"
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Grid Pattern */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#f1f5f9_1px,transparent_1px),linear-gradient(to_bottom,#f1f5f9_1px,transparent_1px)] bg-[size:20px_20px] opacity-50" />
            
            {/* Rendered Elements */}
            <svg className="absolute inset-0 w-full h-full">
              {elements.map(el => {
                if (el.type === 'rect') {
                  return (
                    <rect
                      key={el.id}
                      x={el.x}
                      y={el.y}
                      width={el.width}
                      height={el.height}
                      fill={el.color}
                      rx="8"
                      className={`cursor-pointer transition-all ${selectedElement === el.id ? 'stroke-2 stroke-orange-500' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setSelectedElement(el.id); }}
                    />
                  );
                }
                if (el.type === 'circle') {
                  return (
                    <circle
                      key={el.id}
                      cx={el.x}
                      cy={el.y}
                      r={el.radius}
                      fill={el.color}
                      className={`cursor-pointer transition-all ${selectedElement === el.id ? 'stroke-2 stroke-orange-500' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setSelectedElement(el.id); }}
                    />
                  );
                }
                if (el.type === 'text') {
                  return (
                    <text
                      key={el.id}
                      x={el.x}
                      y={el.y}
                      fill={el.color}
                      className={`text-lg font-bold cursor-pointer ${selectedElement === el.id ? 'underline' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setSelectedElement(el.id); }}
                    >
                      {el.text}
                    </text>
                  );
                }
                if (el.type === 'freehand' && el.points) {
                  const pathData = el.points.reduce((acc, p, i) => 
                    i === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`, ''
                  );
                  return (
                    <path
                      key={el.id}
                      d={pathData}
                      stroke={el.color}
                      strokeWidth="3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      fill="none"
                      className={`cursor-pointer ${selectedElement === el.id ? 'opacity-70' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setSelectedElement(el.id); }}
                    />
                  );
                }
                return null;
              })}
              
              {/* Current Drawing Path */}
              {isDrawing && currentPath.length > 1 && (
                <path
                  d={currentPath.reduce((acc, p, i) => 
                    i === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`, ''
                  )}
                  stroke={selectedColor}
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                  opacity="0.7"
                />
              )}
            </svg>

            {/* Empty State */}
            {elements.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                <div className="text-center">
                  <Sun size={48} className="mx-auto mb-4 text-amber-300" />
                  <p className="text-lg font-medium">Start creating!</p>
                  <p className="text-sm mt-1">Select a tool and click to add shapes</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Color Palette Sidebar */}
        <div className="w-20 border-l border-amber-200 bg-white/30 p-3 flex flex-col gap-3">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest text-center">Colors</div>
          <div className="grid grid-cols-3 gap-1.5">
            {colorPalette.map(color => (
              <button
                key={color}
                onClick={() => setSelectedColor(color)}
                className={`w-5 h-5 rounded-md transition-all ${
                  selectedColor === color ? 'ring-2 ring-offset-2 ring-orange-500 scale-110' : 'hover:scale-110'
                }`}
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
          
          <div className="mt-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest text-center">Current</div>
          <div 
            className="w-full h-8 rounded-lg border-2 border-white shadow-inner"
            style={{ backgroundColor: selectedColor }}
          />
          
          <div className="mt-auto">
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest text-center mb-2">Layers</div>
            <div className="flex items-center justify-center gap-1 text-slate-400">
              <Layers size={14} />
              <span className="text-[10px]">{elements.length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="p-2 px-4 border-t border-amber-200 bg-white/50 text-[9px] text-slate-500 flex justify-between items-center">
        <span>CREATIVE_MODE // TOOL: {selectedTool.toUpperCase()} // ELEMENTS: {elements.length}</span>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-orange-500" />
          <span className="text-orange-600">IMAGINATION UNLEASHED</span>
        </div>
      </div>
    </div>
  );
}
