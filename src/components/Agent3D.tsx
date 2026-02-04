"use client";

import AIAgentCharacter from "./AIAgentCharacter";

interface Agent3DProps {
  position: [number, number, number];
  action: string;
  wireframe: boolean;
  isTyping?: boolean;
  isScrolling?: boolean;
  targetRotation?: number;
  outfit?: {
    skin: string;
    hair: string;
    hoodie: string;
    hoodieDark: string;
    pants: string;
    shoes: string;
  };
  sitPhase?: string;
}

export default function Agent3D({
  position,
  action,
  wireframe,
  isTyping,
  isScrolling,
  targetRotation = 0,
  outfit,
  sitPhase = 'idle',
}: Agent3DProps) {
  // No offset needed - character is built to stand on y=0
  const adjustedPosition: [number, number, number] = [
    position[0],
    position[1],
    position[2]
  ];

  return (
    <AIAgentCharacter
      position={adjustedPosition}
      action={action}
      wireframe={wireframe}
      isTyping={isTyping}
      isScrolling={isScrolling}
      rotation={targetRotation}
      outfit={outfit}
      sitPhase={sitPhase}
    />
  );
}
