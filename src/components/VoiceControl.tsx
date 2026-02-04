"use client";

import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Camera, ScreenShare, Loader2, X } from 'lucide-react';

interface VoiceControlProps {
  onTranscript: (text: string) => void;
  onVisualInput: (type: 'screen' | 'camera') => void;
  isProcessing: boolean;
}

export default function VoiceControl({ onTranscript, onVisualInput, isProcessing }: VoiceControlProps) {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'speechRecognition' in window)) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).speechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event: any) => {
        const current = event.resultIndex;
        const transcriptText = event.results[current][0].transcript;
        setTranscript(transcriptText);
        
        if (event.results[current].isFinal) {
          onTranscript(transcriptText);
          setIsListening(false);
        }
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [onTranscript]);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      setTranscript('');
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  return (
    <div className="flex items-center gap-2 p-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-full shadow-lg transition-all hover:bg-white/15">
      <button
        onClick={toggleListening}
        className={`p-2 rounded-full transition-all ${
          isListening ? 'bg-red-500 text-white animate-pulse' : 'text-white/70 hover:text-white'
        }`}
        title={isListening ? 'Stop Listening' : 'Voice Command'}
      >
        {isListening ? <Mic size={20} /> : <MicOff size={20} />}
      </button>

      {isListening && (
        <div className="px-3 py-1 text-sm text-white/90 max-w-[200px] truncate animate-in fade-in slide-in-from-left-2">
          {transcript || 'Listening...'}
        </div>
      )}

      {!isListening && (
        <div className="flex items-center gap-1">
          <button
            onClick={() => onVisualInput('screen')}
            className="p-2 text-white/70 hover:text-white transition-all rounded-full hover:bg-white/10"
            title="Show Screen"
          >
            <ScreenShare size={18} />
          </button>
          <button
            onClick={() => onVisualInput('camera')}
            className="p-2 text-white/70 hover:text-white transition-all rounded-full hover:bg-white/10"
            title="Show via Camera"
          >
            <Camera size={18} />
          </button>
        </div>
      )}

      {isProcessing && (
        <div className="p-2 text-cyan-400">
          <Loader2 size={18} className="animate-spin" />
        </div>
      )}
    </div>
  );
}
