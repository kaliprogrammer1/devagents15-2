"use client";

import { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import Agent3D from './Agent3D';
import { RoomConfig } from '@/lib/rooms';

interface AgentState {
  position: { x: number; y: number; z: number };
  rotation: number;
  currentAction: string;
  currentTask: string | null;
  thoughts: { type: string; content: string; timestamp: number }[];
  isProcessing: boolean;
  isTyping?: boolean;
  isScrolling?: boolean;
}

interface RoomProps {
  agentState: AgentState;
  onAgentPositionChange: (pos: { x: number; y: number; z: number }) => void;
  wireframe: boolean;
  showAgentView: boolean;
  focusOnAgent?: boolean;
  roomConfig: RoomConfig;
}

function DeskSetup({ position, wireframe, screenOn, color, layout }: { position: [number, number, number]; wireframe: boolean; screenOn?: boolean; color: string; layout: string }) {
  const monitorColor = "#1a1a1a";
  const isUltrawide = layout === 'minimal';
  const isMulti = layout === 'cyberpunk' || layout === 'analytical' || layout === 'hacker';
  
  return (
    <group position={position}>
      {/* Table top */}
      <mesh position={[0, 0.75, 0]} castShadow receiveShadow>
        <boxGeometry args={[isUltrawide ? 2.2 : 1.6, 0.05, 0.8]} />
        <meshStandardMaterial color={color} roughness={0.4} metalness={0.1} wireframe={wireframe} />
      </mesh>
      
      {/* Legs */}
      {[[-0.7, 0.3], [0.7, 0.3], [-0.7, -0.3], [0.7, -0.3]].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.375, z]} castShadow receiveShadow>
          <boxGeometry args={[0.05, 0.75, 0.05]} />
          <meshStandardMaterial color="#333" roughness={0.3} metalness={0.6} wireframe={wireframe} />
        </mesh>
      ))}
      
      {/* Monitors */}
      <group position={[0, 0.78, -0.2]}>
        {/* Main Monitor */}
        <mesh position={[0, 0.30, 0]} castShadow receiveShadow>
          <boxGeometry args={[isUltrawide ? 1.4 : 0.8, isUltrawide ? 0.45 : 0.5, 0.03]} />
          <meshStandardMaterial color={monitorColor} roughness={0.2} metalness={0.6} wireframe={wireframe} />
        </mesh>
        <mesh position={[0, 0.30, 0.016]} castShadow receiveShadow>
          <boxGeometry args={[isUltrawide ? 1.35 : 0.75, isUltrawide ? 0.40 : 0.45, 0.01]} />
          <meshStandardMaterial 
            color={screenOn ? "#1a1a2e" : "#0a0a0a"} 
            roughness={0.1} 
            metalness={0.3} 
            emissive={screenOn ? "#3b82f6" : "#000"} 
            emissiveIntensity={screenOn ? 0.2 : 0} 
            wireframe={wireframe} 
          />
        </mesh>

        {isMulti && (
          <>
            {/* Left Monitor */}
            <group position={[-0.65, 0.25, 0.1]} rotation={[0, 0.3, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.55, 0.40, 0.02]} />
                <meshStandardMaterial color={monitorColor} />
              </mesh>
              <mesh position={[0, 0, 0.011]}>
                <boxGeometry args={[0.50, 0.35, 0.01]} />
                <meshStandardMaterial color={screenOn ? "#064e3b" : "#000"} emissive={screenOn ? "#10b981" : "#000"} emissiveIntensity={0.15} />
              </mesh>
            </group>
            {/* Right Monitor */}
            <group position={[0.65, 0.25, 0.1]} rotation={[0, -0.3, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.55, 0.40, 0.02]} />
                <meshStandardMaterial color={monitorColor} />
              </mesh>
              <mesh position={[0, 0, 0.011]}>
                <boxGeometry args={[0.50, 0.35, 0.01]} />
                <meshStandardMaterial color={screenOn ? "#4c1d95" : "#000"} emissive={screenOn ? "#8b5cf6" : "#000"} emissiveIntensity={0.15} />
              </mesh>
            </group>
          </>
        )}
        
        {/* Monitor Stand */}
        <mesh position={[0, 0.03, 0.05]} castShadow receiveShadow>
          <boxGeometry args={[0.1, 0.06, 0.15]} />
          <meshStandardMaterial color={monitorColor} roughness={0.2} metalness={0.6} wireframe={wireframe} />
        </mesh>
        <mesh position={[0, 0, 0.08]} castShadow receiveShadow>
          <boxGeometry args={[0.25, 0.015, 0.18]} />
          <meshStandardMaterial color={monitorColor} roughness={0.2} metalness={0.6} wireframe={wireframe} />
        </mesh>
      </group>
      
      {/* Keyboard */}
      <mesh position={[0, 0.78, 0.15]} castShadow receiveShadow>
        <boxGeometry args={[0.45, 0.02, 0.15]} />
        <meshStandardMaterial color="#2a2a2a" roughness={0.4} metalness={0.3} wireframe={wireframe} />
      </mesh>
      
      {/* Mouse */}
      <mesh position={[0.35, 0.78, 0.18]} castShadow receiveShadow>
        <boxGeometry args={[0.06, 0.025, 0.10]} />
        <meshStandardMaterial color="#2a2a2a" roughness={0.4} metalness={0.3} wireframe={wireframe} />
      </mesh>
    </group>
  );
}

// Animated Gaming Chair with rotation support
function AnimatedChair({ 
  position, 
  wireframe, 
  color, 
  accent,
  chairRotation = 0,
  isVisible = true
}: { 
  position: [number, number, number]; 
  wireframe: boolean; 
  color: string; 
  accent: string;
  chairRotation?: number;
  isVisible?: boolean;
}) {
  const chairRef = useRef<THREE.Group>(null);
  
  useFrame((_, delta) => {
    if (chairRef.current) {
      chairRef.current.rotation.y = THREE.MathUtils.lerp(
        chairRef.current.rotation.y, 
        chairRotation, 
        delta * 4
      );
    }
  });

  if (!isVisible) return null;

  return (
    <group ref={chairRef} position={position}>
      {/* Base with wheels */}
      <mesh position={[0, 0.05, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.28, 0.30, 0.03, 24]} />
        <meshStandardMaterial color="#222" roughness={0.3} metalness={0.7} wireframe={wireframe} />
      </mesh>
      
      {/* Wheels */}
      {[0, 1, 2, 3, 4].map((i) => {
        const angle = (i / 5) * Math.PI * 2;
        return (
          <mesh key={i} position={[Math.sin(angle) * 0.25, 0.03, Math.cos(angle) * 0.25]} castShadow>
            <sphereGeometry args={[0.04, 8, 8]} />
            <meshStandardMaterial color="#111" roughness={0.5} metalness={0.5} />
          </mesh>
        );
      })}
      
      {/* Hydraulic cylinder */}
      <mesh position={[0, 0.22, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.04, 0.04, 0.34, 12]} />
        <meshStandardMaterial color="#444" roughness={0.3} metalness={0.8} wireframe={wireframe} />
      </mesh>
      
      {/* Seat */}
      <mesh position={[0, 0.45, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.50, 0.10, 0.50]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.2} wireframe={wireframe} />
      </mesh>
      
      {/* Back rest */}
      <mesh position={[0, 0.80, -0.22]} castShadow receiveShadow>
        <boxGeometry args={[0.48, 0.65, 0.08]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.2} wireframe={wireframe} />
      </mesh>
      
      {/* Head rest */}
      <mesh position={[0, 1.18, -0.22]} castShadow receiveShadow>
        <boxGeometry args={[0.38, 0.14, 0.10]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.2} wireframe={wireframe} />
      </mesh>
      
      {/* Arm rests */}
      <mesh position={[-0.28, 0.58, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.06, 0.18, 0.35]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.2} wireframe={wireframe} />
      </mesh>
      <mesh position={[0.28, 0.58, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.06, 0.18, 0.35]} />
        <meshStandardMaterial color={color} roughness={0.6} metalness={0.2} wireframe={wireframe} />
      </mesh>
      
      {/* Accent stripe */}
      <mesh position={[0, 0.80, -0.27]} castShadow receiveShadow>
        <boxGeometry args={[0.03, 0.55, 0.01]} />
        <meshStandardMaterial color={accent} roughness={0.3} metalness={0.5} wireframe={wireframe} />
      </mesh>
    </group>
  );
}

function ServerRack({ position, wireframe, accentColor = "#06b6d4" }: { position: [number, number, number]; wireframe: boolean; accentColor?: string }) {
  return (
    <group position={position}>
      <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.6, 1.2, 0.5]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.2} metalness={0.8} wireframe={wireframe} />
      </mesh>
      {[0.2, 0.4, 0.6, 0.8, 1.0].map((y, i) => (
        <group key={i} position={[0, y - 0.6, 0.251]}>
          <mesh castShadow>
            <boxGeometry args={[0.5, 0.15, 0.01]} />
            <meshStandardMaterial color="#2a2a2a" />
          </mesh>
          <pointLight position={[0.2, 0, 0.02]} intensity={0.15} color={i % 2 === 0 ? accentColor : "#8b5cf6"} distance={0.6} />
        </group>
      ))}
    </group>
  );
}

// Architect room props - Blueprint/Schematic panels
function BlueprintPanel({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      {/* Frame */}
      <mesh castShadow>
        <boxGeometry args={[1.2, 0.8, 0.02]} />
        <meshStandardMaterial color="#1e293b" roughness={0.3} metalness={0.5} wireframe={wireframe} />
      </mesh>
      {/* Blueprint surface */}
      <mesh position={[0, 0, 0.011]}>
        <planeGeometry args={[1.1, 0.7]} />
        <meshStandardMaterial color="#1e3a5f" emissive="#3b82f6" emissiveIntensity={0.1} />
      </mesh>
      {/* Grid lines */}
      {[-0.4, -0.2, 0, 0.2, 0.4].map((x, i) => (
        <mesh key={`v${i}`} position={[x, 0, 0.012]}>
          <boxGeometry args={[0.005, 0.65, 0.001]} />
          <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.3} transparent opacity={0.5} />
        </mesh>
      ))}
      {[-0.25, 0, 0.25].map((y, i) => (
        <mesh key={`h${i}`} position={[0, y, 0.012]}>
          <boxGeometry args={[1.05, 0.005, 0.001]} />
          <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.3} transparent opacity={0.5} />
        </mesh>
      ))}
    </group>
  );
}

// Architect room - 3D model display
function ArchitectureModel({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      {/* Base platform */}
      <mesh position={[0, 0, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.3, 0.35, 0.05, 32]} />
        <meshStandardMaterial color="#334155" roughness={0.3} metalness={0.6} wireframe={wireframe} />
      </mesh>
      {/* Building model */}
      <group position={[0, 0.15, 0]}>
        <mesh castShadow>
          <boxGeometry args={[0.2, 0.25, 0.2]} />
          <meshStandardMaterial color="#60a5fa" transparent opacity={0.6} emissive="#3b82f6" emissiveIntensity={0.2} />
        </mesh>
        <mesh position={[0.08, 0.18, 0]}>
          <boxGeometry args={[0.08, 0.15, 0.08]} />
          <meshStandardMaterial color="#93c5fd" transparent opacity={0.5} emissive="#60a5fa" emissiveIntensity={0.2} />
        </mesh>
      </group>
      {/* Hologram ring */}
      <mesh position={[0, 0.15, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.25, 0.01, 8, 32]} />
        <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.5} transparent opacity={0.4} />
      </mesh>
    </group>
  );
}

// Operator room - Status dashboard
function StatusDashboard({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      <mesh castShadow>
        <boxGeometry args={[0.8, 0.5, 0.03]} />
        <meshStandardMaterial color="#1e1e2e" roughness={0.2} metalness={0.6} wireframe={wireframe} />
      </mesh>
      <mesh position={[0, 0, 0.016]}>
        <planeGeometry args={[0.75, 0.45]} />
        <meshStandardMaterial color="#0a0a1a" emissive="#8b5cf6" emissiveIntensity={0.15} />
      </mesh>
      {/* Status bars */}
      {[0.15, 0.05, -0.05, -0.15].map((y, i) => (
        <mesh key={i} position={[-0.1, y, 0.017]}>
          <boxGeometry args={[0.4 * (0.5 + Math.random() * 0.5), 0.04, 0.001]} />
          <meshStandardMaterial 
            color={i === 0 ? "#22c55e" : i === 1 ? "#3b82f6" : i === 2 ? "#f59e0b" : "#ef4444"} 
            emissive={i === 0 ? "#22c55e" : i === 1 ? "#3b82f6" : i === 2 ? "#f59e0b" : "#ef4444"} 
            emissiveIntensity={0.4}
          />
        </mesh>
      ))}
      <pointLight position={[0, 0, 0.1]} intensity={0.3} color="#a855f7" distance={1} />
    </group>
  );
}

// Operator room - Alert panel
function AlertPanel({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      <mesh castShadow>
        <boxGeometry args={[0.4, 0.6, 0.02]} />
        <meshStandardMaterial color="#1a1a2e" roughness={0.3} metalness={0.5} wireframe={wireframe} />
      </mesh>
      {/* Alert indicators */}
      {[0.2, 0.1, 0, -0.1, -0.2].map((y, i) => (
        <group key={i} position={[-0.12, y, 0.011]}>
          <mesh>
            <circleGeometry args={[0.03, 16]} />
            <meshStandardMaterial 
              color={i < 2 ? "#22c55e" : i === 2 ? "#f59e0b" : "#ef4444"} 
              emissive={i < 2 ? "#22c55e" : i === 2 ? "#f59e0b" : "#ef4444"}
              emissiveIntensity={i >= 3 ? 0.8 : 0.4}
            />
          </mesh>
          <mesh position={[0.1, 0, 0]}>
            <boxGeometry args={[0.12, 0.02, 0.001]} />
            <meshStandardMaterial color="#4a5568" />
          </mesh>
        </group>
      ))}
    </group>
  );
}

// Creator room - Art easel
function ArtEasel({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      {/* Legs */}
      <mesh position={[-0.15, 0.5, 0.1]} rotation={[0.1, 0, -0.15]} castShadow>
        <boxGeometry args={[0.03, 1.0, 0.03]} />
        <meshStandardMaterial color="#8b5a2b" roughness={0.7} wireframe={wireframe} />
      </mesh>
      <mesh position={[0.15, 0.5, 0.1]} rotation={[0.1, 0, 0.15]} castShadow>
        <boxGeometry args={[0.03, 1.0, 0.03]} />
        <meshStandardMaterial color="#8b5a2b" roughness={0.7} wireframe={wireframe} />
      </mesh>
      <mesh position={[0, 0.5, -0.15]} rotation={[-0.2, 0, 0]} castShadow>
        <boxGeometry args={[0.03, 1.0, 0.03]} />
        <meshStandardMaterial color="#8b5a2b" roughness={0.7} wireframe={wireframe} />
      </mesh>
      {/* Canvas */}
      <mesh position={[0, 0.8, 0.05]} rotation={[0.1, 0, 0]} castShadow>
        <boxGeometry args={[0.5, 0.6, 0.02]} />
        <meshStandardMaterial color="#f5f5dc" roughness={0.9} wireframe={wireframe} />
      </mesh>
      {/* Color splashes */}
      <mesh position={[-0.1, 0.85, 0.062]}>
        <circleGeometry args={[0.08, 16]} />
        <meshStandardMaterial color="#f97316" emissive="#f97316" emissiveIntensity={0.2} />
      </mesh>
      <mesh position={[0.1, 0.75, 0.062]}>
        <circleGeometry args={[0.06, 16]} />
        <meshStandardMaterial color="#ec4899" emissive="#ec4899" emissiveIntensity={0.2} />
      </mesh>
    </group>
  );
}

// Creator room - Plant
function Plant({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      {/* Pot */}
      <mesh position={[0, 0.1, 0]} castShadow>
        <cylinderGeometry args={[0.1, 0.08, 0.2, 16]} />
        <meshStandardMaterial color="#d4a373" roughness={0.8} wireframe={wireframe} />
      </mesh>
      {/* Soil */}
      <mesh position={[0, 0.2, 0]}>
        <cylinderGeometry args={[0.09, 0.09, 0.02, 16]} />
        <meshStandardMaterial color="#3d2914" roughness={1} />
      </mesh>
      {/* Leaves */}
      {[0, 0.8, 1.6, 2.4, 3.2].map((angle, i) => (
        <mesh key={i} position={[Math.sin(angle) * 0.05, 0.35 + i * 0.05, Math.cos(angle) * 0.05]} rotation={[0.3, angle, 0.2]}>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial color="#22c55e" roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}

// Creator room - Color palette
function ColorPalette({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  const colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899"];
  return (
    <group position={position}>
      <mesh castShadow>
        <boxGeometry args={[0.4, 0.08, 0.25]} />
        <meshStandardMaterial color="#1e293b" roughness={0.3} metalness={0.5} wireframe={wireframe} />
      </mesh>
      {colors.map((color, i) => (
        <mesh key={i} position={[-0.15 + i * 0.05, 0.041, 0]}>
          <cylinderGeometry args={[0.02, 0.02, 0.01, 16]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
        </mesh>
      ))}
    </group>
  );
}

// Analyst room - Data chart display
function DataChart({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      <mesh castShadow>
        <boxGeometry args={[0.9, 0.6, 0.02]} />
        <meshStandardMaterial color="#0f172a" roughness={0.2} metalness={0.6} wireframe={wireframe} />
      </mesh>
      <mesh position={[0, 0, 0.011]}>
        <planeGeometry args={[0.85, 0.55]} />
        <meshStandardMaterial color="#020617" emissive="#0891b2" emissiveIntensity={0.1} />
      </mesh>
      {/* Bar chart */}
      {[-0.3, -0.15, 0, 0.15, 0.3].map((x, i) => {
        const height = 0.1 + Math.random() * 0.3;
        return (
          <mesh key={i} position={[x, -0.2 + height/2, 0.012]}>
            <boxGeometry args={[0.08, height, 0.005]} />
            <meshStandardMaterial color="#0891b2" emissive="#06b6d4" emissiveIntensity={0.4} />
          </mesh>
        );
      })}
      {/* Line trend */}
      <mesh position={[0, 0.1, 0.012]} rotation={[0, 0, 0.2]}>
        <boxGeometry args={[0.7, 0.01, 0.001]} />
        <meshStandardMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.6} />
      </mesh>
      <pointLight position={[0, 0, 0.15]} intensity={0.2} color="#0891b2" distance={0.8} />
    </group>
  );
}

// Analyst room - Data sphere
function DataSphere({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      <mesh position={[0, 0.3, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#0891b2" transparent opacity={0.3} emissive="#06b6d4" emissiveIntensity={0.3} wireframe />
      </mesh>
      <mesh position={[0, 0.3, 0]}>
        <sphereGeometry args={[0.15, 8, 8]} />
        <meshStandardMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.5} />
      </mesh>
      {/* Orbit rings */}
      <mesh position={[0, 0.3, 0]} rotation={[Math.PI / 3, 0, 0]}>
        <torusGeometry args={[0.25, 0.008, 8, 32]} />
        <meshStandardMaterial color="#06b6d4" emissive="#06b6d4" emissiveIntensity={0.4} />
      </mesh>
      <mesh position={[0, 0.3, 0]} rotation={[Math.PI / 3, Math.PI / 2, 0]}>
        <torusGeometry args={[0.25, 0.008, 8, 32]} />
        <meshStandardMaterial color="#0891b2" emissive="#0891b2" emissiveIntensity={0.4} />
      </mesh>
      <pointLight position={[0, 0.3, 0]} intensity={0.4} color="#06b6d4" distance={1} />
    </group>
  );
}

// Hacker room - Matrix rain panel
function MatrixPanel({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      <mesh castShadow>
        <boxGeometry args={[0.6, 0.8, 0.02]} />
        <meshStandardMaterial color="#0a0a0a" roughness={0.1} metalness={0.8} wireframe={wireframe} />
      </mesh>
      <mesh position={[0, 0, 0.011]}>
        <planeGeometry args={[0.55, 0.75]} />
        <meshStandardMaterial color="#001a00" emissive="#22c55e" emissiveIntensity={0.15} />
      </mesh>
      {/* Matrix columns */}
      {[-0.2, -0.1, 0, 0.1, 0.2].map((x, i) => (
        <group key={i}>
          {[0.3, 0.15, 0, -0.15, -0.3].map((y, j) => (
            <mesh key={j} position={[x, y, 0.012]} >
              <boxGeometry args={[0.04, 0.08, 0.001]} />
              <meshStandardMaterial 
                color="#22c55e" 
                emissive="#22c55e" 
                emissiveIntensity={0.3 + (j * 0.15)} 
                transparent 
                opacity={0.3 + (j * 0.15)} 
              />
            </mesh>
          ))}
        </group>
      ))}
      <pointLight position={[0, 0, 0.1]} intensity={0.4} color="#22c55e" distance={1} />
    </group>
  );
}

// Hacker room - Network visualization
function NetworkNode({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      {/* Central node */}
      <mesh position={[0, 0.25, 0]}>
        <octahedronGeometry args={[0.1, 0]} />
        <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.6} />
      </mesh>
      {/* Connection nodes */}
      {[0, 1.2, 2.4, 3.6, 4.8].map((angle, i) => {
        const x = Math.sin(angle) * 0.25;
        const z = Math.cos(angle) * 0.25;
        return (
          <group key={i}>
            <mesh position={[x, 0.25, z]}>
              <sphereGeometry args={[0.03, 8, 8]} />
              <meshStandardMaterial color="#16a34a" emissive="#22c55e" emissiveIntensity={0.4} />
            </mesh>
            {/* Connection line */}
            <mesh position={[x/2, 0.25, z/2]} rotation={[0, -angle, Math.PI/2]}>
              <cylinderGeometry args={[0.005, 0.005, 0.2, 4]} />
              <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.3} transparent opacity={0.6} />
            </mesh>
          </group>
        );
      })}
      <pointLight position={[0, 0.25, 0]} intensity={0.3} color="#22c55e" distance={0.8} />
    </group>
  );
}

// Hacker room - Cable mess
function CableMess({ position, wireframe }: { position: [number, number, number]; wireframe: boolean }) {
  return (
    <group position={position}>
      {[0, 0.5, 1, 1.5, 2].map((i) => (
        <mesh key={i} position={[Math.sin(i * 2) * 0.1, 0.02, Math.cos(i * 2) * 0.1]} rotation={[0, i, 0]}>
          <torusGeometry args={[0.15 + i * 0.03, 0.015, 8, 16, Math.PI * 1.5]} />
          <meshStandardMaterial 
            color={i % 2 === 0 ? "#1a1a1a" : "#22c55e"} 
            emissive={i % 2 === 0 ? "#000" : "#22c55e"}
            emissiveIntensity={i % 2 === 0 ? 0 : 0.2}
            roughness={0.6} 
          />
        </mesh>
      ))}
    </group>
  );
}

function DynamicFloor({ config, wireframe }: { config: RoomConfig; wireframe: boolean }) {
  const { shape, floorSize } = config.environment.geometry;
  const [w, h] = floorSize;

  if (shape === 'L') {
    return (
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <mesh position={[w/4, h/4, 0]} receiveShadow>
          <planeGeometry args={[w/2, h/2]} />
          <meshStandardMaterial color={config.colors.floor} roughness={0.8} wireframe={wireframe} />
        </mesh>
        <mesh position={[-w/4, -h/4, 0]} receiveShadow>
          <planeGeometry args={[w/2, h/2]} />
          <meshStandardMaterial color={config.colors.floor} roughness={0.8} wireframe={wireframe} />
        </mesh>
        <mesh position={[-w/4, h/4, 0]} receiveShadow>
          <planeGeometry args={[w/2, h/2]} />
          <meshStandardMaterial color={config.colors.floor} roughness={0.8} wireframe={wireframe} />
        </mesh>
      </group>
    );
  }

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[w, h]} />
      <meshStandardMaterial color={config.colors.floor} roughness={0.8} metalness={0.1} wireframe={wireframe} />
    </mesh>
  );
}

function DynamicWalls({ config, wireframe }: { config: RoomConfig; wireframe: boolean }) {
  const { shape, floorSize, wallHeight } = config.environment.geometry;
  const [w, h] = floorSize;

  return (
    <group>
      {/* Back Wall */}
      <mesh position={[0, wallHeight/2, -h/2]} receiveShadow>
        <boxGeometry args={[w, wallHeight, 0.1]} />
        <meshStandardMaterial color={config.colors.wall} roughness={0.9} wireframe={wireframe} />
      </mesh>
      {/* Right Wall */}
      <mesh position={[w/2, wallHeight/2, 0]} receiveShadow>
        <boxGeometry args={[0.1, wallHeight, h]} />
        <meshStandardMaterial color={config.colors.wall} roughness={0.9} wireframe={wireframe} />
      </mesh>
      {/* Left Wall */}
      {shape === 'alcove' && (
        <mesh position={[-w/2, wallHeight/2, 0]} receiveShadow>
          <boxGeometry args={[0.1, wallHeight, h]} />
          <meshStandardMaterial color={config.colors.wall} roughness={0.9} wireframe={wireframe} />
        </mesh>
      )}
      
      {/* Window */}
      <group position={[w/2 - 0.05, wallHeight/2, 0]} rotation={[0, -Math.PI / 2, 0]}>
        <mesh>
          <boxGeometry args={[2, 1.5, 0.05]} />
          <meshStandardMaterial color="#334155" />
        </mesh>
        <mesh position={[0, 0, 0.03]}>
          <planeGeometry args={[1.8, 1.3]} />
          <meshStandardMaterial color={config.lighting.window.color} emissive={config.lighting.window.color} emissiveIntensity={0.5} transparent opacity={0.6} />
        </mesh>
      </group>
    </group>
  );
}

function CameraController({ config, enabled }: { config: RoomConfig; enabled: boolean }) {
  const { camera } = useThree();
  const targetPos = useRef(new THREE.Vector3(...config.environment.camera.position));
  const targetLook = useRef(new THREE.Vector3(...config.environment.camera.target));
  
  const currentPos = useRef(new THREE.Vector3(...config.environment.camera.position));
  const currentLook = useRef(new THREE.Vector3(...config.environment.camera.target));

  useEffect(() => {
    targetPos.current.set(...config.environment.camera.position);
    targetLook.current.set(...config.environment.camera.target);
  }, [config.id]);

  useFrame((_, delta) => {
    if (!enabled) return;
    
    currentPos.current.lerp(targetPos.current, delta * 2);
    currentLook.current.lerp(targetLook.current, delta * 2);
    
    camera.position.copy(currentPos.current);
    camera.lookAt(currentLook.current);
    camera.fov = THREE.MathUtils.lerp(camera.fov, config.environment.camera.fov, delta * 2);
    camera.updateProjectionMatrix();
  });

  return null;
}

// Sitting animation phases
type SitPhase = 'idle' | 'approaching' | 'rotating_chair' | 'sitting_down' | 'rotating_to_desk' | 'seated' | 'using_computer';

function RoomScene({ agentState, onAgentPositionChange, wireframe, showAgentView, focusOnAgent, roomConfig }: RoomProps) {
  const targetPosition = useRef(agentState.position);
  const currentPosition = useRef({ ...agentState.position });
  const currentRotation = useRef(0);
  const [sitPhase, setSitPhase] = useState<SitPhase>('idle');
  const [chairRotation, setChairRotation] = useState(Math.PI); // Initially facing away from desk
  const phaseStartTime = useRef(0);
  
  const h = roomConfig.environment.geometry.floorSize[1];
  const chairPosition: [number, number, number] = [0, 0, -h/2 + 1.6];
  const deskPosition: [number, number, number] = [0, 0, -h/2 + 0.8];

  // Handle action changes
  useEffect(() => {
    if (agentState.currentAction.startsWith('WALK_TO:')) {
      const coords = agentState.currentAction.replace('WALK_TO:', '').split(',');
      if (coords.length >= 2) {
        targetPosition.current = {
          x: parseFloat(coords[0]),
          y: 0,
          z: parseFloat(coords[1])
        };
        setSitPhase('idle');
      }
    } else if (agentState.currentAction === 'SIT_COMPUTER') {
      // Start sitting sequence - first approach the chair
      targetPosition.current = { x: 0.6, y: 0, z: chairPosition[2] + 0.3 };
      setSitPhase('approaching');
      phaseStartTime.current = Date.now();
    } else if (agentState.currentAction === 'USE_COMPUTER') {
      // If already seated, go to using computer
      if (sitPhase === 'seated' || sitPhase === 'using_computer') {
        setSitPhase('using_computer');
      } else {
        // Start full sequence
        targetPosition.current = { x: 0.6, y: 0, z: chairPosition[2] + 0.3 };
        setSitPhase('approaching');
        phaseStartTime.current = Date.now();
      }
    } else if (agentState.currentAction === 'STAND_UP') {
      setSitPhase('idle');
      setChairRotation(Math.PI);
      targetPosition.current = { x: 0.6, y: 0, z: chairPosition[2] + 0.5 };
    }
  }, [agentState.currentAction, h]);

  useFrame((_, delta) => {
    const baseSpeed = 1.5;
    const speed = baseSpeed * roomConfig.behavior.speedMultiplier;
    const elapsed = Date.now() - phaseStartTime.current;
    
    // Handle sitting animation phases
    if (sitPhase === 'approaching') {
      const dx = targetPosition.current.x - currentPosition.current.x;
      const dz = targetPosition.current.z - currentPosition.current.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      
      if (dist > 0.1) {
        currentPosition.current.x += (dx / dist) * speed * delta;
        currentPosition.current.z += (dz / dist) * speed * delta;
        currentRotation.current = Math.atan2(dx, dz);
      } else {
        // Arrived at chair side, start rotating chair
        setSitPhase('rotating_chair');
        phaseStartTime.current = Date.now();
        currentRotation.current = Math.PI / 2; // Face the chair
      }
    } else if (sitPhase === 'rotating_chair') {
      // Agent reaches to rotate the chair (1 second)
      if (elapsed < 1000) {
        // Rotate chair from facing away (PI) to facing agent (PI/2)
        const progress = elapsed / 1000;
        setChairRotation(Math.PI - progress * (Math.PI / 2));
      } else {
        setSitPhase('sitting_down');
        phaseStartTime.current = Date.now();
        // Move to chair position
        targetPosition.current = { x: 0, y: 0, z: chairPosition[2] };
      }
    } else if (sitPhase === 'sitting_down') {
      // Move to sit on chair (0.8 seconds)
      const dx = targetPosition.current.x - currentPosition.current.x;
      const dz = targetPosition.current.z - currentPosition.current.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      
      if (dist > 0.05) {
        currentPosition.current.x += (dx / dist) * speed * 1.5 * delta;
        currentPosition.current.z += (dz / dist) * speed * 1.5 * delta;
        // Turn to sit (back to chair)
        currentRotation.current = THREE.MathUtils.lerp(currentRotation.current, -Math.PI / 2, delta * 5);
      } else if (elapsed > 800) {
        setSitPhase('rotating_to_desk');
        phaseStartTime.current = Date.now();
      }
    } else if (sitPhase === 'rotating_to_desk') {
      // Rotate chair to face desk (0.6 seconds)
      if (elapsed < 600) {
        const progress = elapsed / 600;
        // Rotate from PI/2 (facing right) to PI (facing desk/back wall)
        setChairRotation(Math.PI / 2 + progress * (Math.PI / 2));
        currentRotation.current = THREE.MathUtils.lerp(currentRotation.current, Math.PI, delta * 8);
      } else {
        setSitPhase('seated');
        setChairRotation(Math.PI);
        currentRotation.current = Math.PI;
        // If action is USE_COMPUTER, go to that phase
        if (agentState.currentAction === 'USE_COMPUTER') {
          setSitPhase('using_computer');
        }
      }
    } else if (sitPhase === 'idle') {
      // Normal walking behavior
      const dx = targetPosition.current.x - currentPosition.current.x;
      const dz = targetPosition.current.z - currentPosition.current.z;
      const dist = Math.sqrt(dx * dx + dz * dz);

      if (dist > 0.03) {
        currentPosition.current.x += (dx / dist) * speed * delta;
        currentPosition.current.z += (dz / dist) * speed * delta;
        currentRotation.current = Math.atan2(dx, dz);
      }
    }
    
    onAgentPositionChange({ ...currentPosition.current });
  });

  const isUsingComputer = sitPhase === 'using_computer';
  const isSitting = sitPhase === 'seated' || sitPhase === 'using_computer' || sitPhase === 'rotating_to_desk';
  const isRotatingChair = sitPhase === 'rotating_chair';
  const showChair = sitPhase !== 'seated' && sitPhase !== 'using_computer' && sitPhase !== 'rotating_to_desk' && sitPhase !== 'sitting_down';

  // Determine agent action for animation
  let agentAction = agentState.currentAction;
  if (sitPhase === 'approaching') agentAction = 'WALK_TO:0,0';
  if (sitPhase === 'rotating_chair') agentAction = 'ROTATE_CHAIR';
  if (sitPhase === 'sitting_down') agentAction = 'SITTING_DOWN';
  if (sitPhase === 'rotating_to_desk') agentAction = 'SEATED';
  if (sitPhase === 'seated') agentAction = 'SEATED';
  if (sitPhase === 'using_computer') agentAction = 'USE_COMPUTER';

  return (
    <>
      <CameraController config={roomConfig} enabled={focusOnAgent && !showAgentView} />
      
      <ambientLight intensity={roomConfig.lighting.ambient} />
      <directionalLight
        position={roomConfig.lighting.directional.position}
        intensity={roomConfig.lighting.directional.intensity}
        color={roomConfig.lighting.directional.color}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      
      {roomConfig.lighting.secondary && (
        <pointLight
          position={roomConfig.lighting.secondary.position}
          intensity={roomConfig.lighting.secondary.intensity}
          color={roomConfig.lighting.secondary.color}
        />
      )}

      <DynamicFloor config={roomConfig} wireframe={wireframe} />
      <DynamicWalls config={roomConfig} wireframe={wireframe} />

      <DeskSetup 
        position={deskPosition} 
        wireframe={wireframe} 
        screenOn={isUsingComputer} 
        color={roomConfig.colors.desk} 
        layout={roomConfig.environment.layout}
      />
      
      <AnimatedChair 
        position={chairPosition}
        wireframe={wireframe}
        color={roomConfig.colors.chair}
        accent={roomConfig.colors.accent}
        chairRotation={chairRotation}
        isVisible={showChair}
      />

        {/* Props based on room layout/persona */}
        {/* Architect room - minimal layout */}
        {roomConfig.environment.layout === 'minimal' && (
          <group>
            <BlueprintPanel position={[3.5, 1.8, -2.5]} wireframe={wireframe} />
            <ArchitectureModel position={[2.5, 0.78, -2]} wireframe={wireframe} />
          </group>
        )}

        {/* Operator room - cyberpunk layout */}
        {roomConfig.environment.layout === 'cyberpunk' && (
          <group>
            <group position={[2, 0, -2]}>
              <ServerRack position={[0, 0, 0]} wireframe={wireframe} accentColor="#a855f7" />
              <ServerRack position={[0.7, 0, 0]} wireframe={wireframe} accentColor="#22d3ee" />
            </group>
            <StatusDashboard position={[-2, 1.5, -2.4]} wireframe={wireframe} />
            <AlertPanel position={[2.4, 1.5, -1]} wireframe={wireframe} />
          </group>
        )}

        {/* Creator room - creative layout */}
        {roomConfig.environment.layout === 'creative' && (
          <group>
            <ArtEasel position={[2.5, 0, -1]} wireframe={wireframe} />
            <Plant position={[-2, 0, 1]} wireframe={wireframe} />
            <Plant position={[3, 0, 0.5]} wireframe={wireframe} />
            <ColorPalette position={[0.8, 0.78, -2.2]} wireframe={wireframe} />
          </group>
        )}

        {/* Analyst room - analytical layout */}
        {roomConfig.environment.layout === 'analytical' && (
          <group>
            <DataChart position={[-2, 1.5, -2.4]} wireframe={wireframe} />
            <DataChart position={[2, 1.8, -2.4]} wireframe={wireframe} />
            <DataSphere position={[3.5, 0.78, -1]} wireframe={wireframe} />
          </group>
        )}

        {/* Hacker room - hacker layout */}
        {roomConfig.environment.layout === 'hacker' && (
          <group>
            <MatrixPanel position={[-1.5, 1.2, -1.9]} wireframe={wireframe} />
            <NetworkNode position={[1.2, 0, 0.5]} wireframe={wireframe} />
            <ServerRack position={[-1.5, 0, 1]} wireframe={wireframe} accentColor="#22c55e" />
            <CableMess position={[0.5, 0, 0.8]} wireframe={wireframe} />
          </group>
        )}

      <Agent3D
        position={[currentPosition.current.x, currentPosition.current.y, currentPosition.current.z]}
        action={agentAction}
        wireframe={wireframe}
        isTyping={agentState.isTyping}
        isScrolling={agentState.isScrolling}
        targetRotation={currentRotation.current}
        outfit={roomConfig.agent.outfit}
        sitPhase={sitPhase}
      />
    </>
  );
}

export default function Room3D({
  agentState,
  onAgentPositionChange,
  wireframe,
  showAgentView,
  focusOnAgent = false,
  roomConfig
}: RoomProps) {
  if (!roomConfig) return null;
  
  return (
    <Canvas shadows dpr={[1, 2]} gl={{ antialias: true }}>
      {!focusOnAgent && (
        <>
          <PerspectiveCamera makeDefault position={roomConfig.environment.camera.position} fov={roomConfig.environment.camera.fov} />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={1}
            maxDistance={15}
            maxPolarAngle={Math.PI / 2.1}
            target={roomConfig.environment.camera.target}
          />
        </>
      )}
      {focusOnAgent && (
        <PerspectiveCamera makeDefault position={roomConfig.environment.camera.position} fov={roomConfig.environment.camera.fov} />
      )}
      <RoomScene
        agentState={agentState}
        onAgentPositionChange={onAgentPositionChange}
        wireframe={wireframe}
        showAgentView={showAgentView}
        focusOnAgent={focusOnAgent}
        roomConfig={roomConfig}
      />
    </Canvas>
  );
}
