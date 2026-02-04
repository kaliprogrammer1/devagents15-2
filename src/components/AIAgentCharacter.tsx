"use client";

import React, { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface AIAgentCharacterProps {
  action: string;
  isTyping?: boolean;
  isScrolling?: boolean;
  wireframe?: boolean;
  position?: [number, number, number];
  rotation?: number;
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

export default function AIAgentCharacter({
  action,
  isTyping,
  wireframe = false,
  position = [0, 0, 0],
  rotation = 0,
  outfit,
  sitPhase = 'idle',
}: AIAgentCharacterProps) {
  const group = useRef<THREE.Group>(null);
  const timeRef = useRef(0);

  const colors = {
    skin: outfit?.skin || "#f3ccb0",
    hair: outfit?.hair || "#2c1810",
    hoodie: outfit?.hoodie || "#4f46e5",
    hoodieDark: outfit?.hoodieDark || "#3730a3",
    pants: outfit?.pants || "#1e293b",
    shoes: outfit?.shoes || "#111111",
  };

  const isWalking = action.startsWith("WALK_TO");
  const isRotatingChair = sitPhase === 'rotating_chair';
  const isSittingDown = sitPhase === 'sitting_down';
  const isSeated = sitPhase === 'seated' || sitPhase === 'using_computer' || sitPhase === 'rotating_to_desk';
  const isUsingComputer = sitPhase === 'using_computer';

  useFrame((state, delta) => {
    timeRef.current = state.clock.getElapsedTime();
    if (group.current) {
      group.current.rotation.y = THREE.MathUtils.lerp(group.current.rotation.y, rotation, delta * 5);
    }
  });

  const t = timeRef.current || 0;
  const walkCycle = isWalking ? Math.sin(t * 8) : 0;
  const breathe = Math.sin(t * 2) * 0.01;

  // Animation states based on sitPhase
  let pelvisY = 0.95;
  let thighRotationL = 0;
  let thighRotationR = 0;
  let calfRotationL = 0;
  let calfRotationR = 0;
  let armRotationL = 0;
  let armRotationR = 0;
  let forearmRotationL = -0.2;
  let forearmRotationR = -0.2;
  let torsoLean = 0;

  if (isWalking) {
    thighRotationL = walkCycle * 0.5;
    thighRotationR = -walkCycle * 0.5;
    calfRotationL = Math.max(0, -walkCycle * 0.4);
    calfRotationR = Math.max(0, walkCycle * 0.4);
    armRotationL = -walkCycle * 0.4;
    armRotationR = walkCycle * 0.4;
  } else if (isRotatingChair) {
    // Reaching forward to grab chair
    armRotationL = -0.8;
    armRotationR = -0.8;
    forearmRotationL = -1.2;
    forearmRotationR = -1.2;
    torsoLean = 0.2;
  } else if (isSittingDown) {
    // Transitioning to sit
    pelvisY = 0.55;
    thighRotationL = -1.2;
    thighRotationR = -1.2;
    calfRotationL = 1.0;
    calfRotationR = 1.0;
    armRotationL = -0.3;
    armRotationR = -0.3;
  } else if (isSeated) {
    // Fully seated
    pelvisY = 0.50;
    thighRotationL = -Math.PI / 2;
    thighRotationR = -Math.PI / 2;
    calfRotationL = Math.PI / 2;
    calfRotationR = Math.PI / 2;
    
    if (isUsingComputer) {
      // Typing animation
      const typeSpeed = isTyping ? 15 : 0;
      armRotationL = -0.7;
      armRotationR = -0.7;
      forearmRotationL = -1.0 + (isTyping ? Math.sin(t * typeSpeed) * 0.15 : 0);
      forearmRotationR = -1.0 + (isTyping ? Math.sin(t * typeSpeed + 1.5) * 0.15 : 0);
    } else {
      armRotationL = -0.3;
      armRotationR = -0.3;
      forearmRotationL = -0.5;
      forearmRotationR = -0.5;
    }
  }

  return (
    <group ref={group} position={position}>
      {/* PELVIS - Root of body hierarchy */}
      <group position={[0, pelvisY, 0]} rotation={[torsoLean, 0, 0]}>
        <mesh castShadow>
          <boxGeometry args={[0.35, 0.15, 0.2]} />
          <meshStandardMaterial color={colors.pants} wireframe={wireframe} />
        </mesh>

        {/* TORSO */}
        <group position={[0, 0.35, 0]}>
          <mesh castShadow scale={[1, 1 + breathe, 1]}>
            <boxGeometry args={[0.42, 0.55, 0.22]} />
            <meshStandardMaterial color={colors.hoodie} wireframe={wireframe} />
          </mesh>
          
          {/* Hood detail */}
          <mesh position={[0, 0.2, -0.08]} castShadow>
            <boxGeometry args={[0.30, 0.18, 0.12]} />
            <meshStandardMaterial color={colors.hoodieDark} wireframe={wireframe} />
          </mesh>

          {/* NECK */}
          <group position={[0, 0.32, 0]}>
            <mesh castShadow>
              <cylinderGeometry args={[0.06, 0.07, 0.1, 8]} />
              <meshStandardMaterial color={colors.skin} wireframe={wireframe} />
            </mesh>

            {/* HEAD */}
            <group position={[0, 0.18, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.20, 0.25, 0.22]} />
                <meshStandardMaterial color={colors.skin} wireframe={wireframe} />
              </mesh>
              {/* Hair */}
              <mesh position={[0, 0.08, -0.02]} castShadow>
                <boxGeometry args={[0.21, 0.12, 0.23]} />
                <meshStandardMaterial color={colors.hair} wireframe={wireframe} />
              </mesh>
              {/* Face features */}
              <mesh position={[-0.05, 0.02, 0.11]}>
                <sphereGeometry args={[0.02, 8, 8]} />
                <meshStandardMaterial color="#1a1a1a" />
              </mesh>
              <mesh position={[0.05, 0.02, 0.11]}>
                <sphereGeometry args={[0.02, 8, 8]} />
                <meshStandardMaterial color="#1a1a1a" />
              </mesh>
              {/* Nose */}
              <mesh position={[0, -0.02, 0.12]}>
                <boxGeometry args={[0.03, 0.04, 0.02]} />
                <meshStandardMaterial color={colors.skin} wireframe={wireframe} />
              </mesh>
            </group>
          </group>

          {/* LEFT ARM */}
          <group position={[-0.26, 0.22, 0]} rotation={[armRotationL, 0, 0.1]}>
            {/* Upper arm */}
            <mesh position={[0, -0.14, 0]} castShadow>
              <boxGeometry args={[0.10, 0.28, 0.10]} />
              <meshStandardMaterial color={colors.hoodie} wireframe={wireframe} />
            </mesh>
            {/* Forearm */}
            <group position={[0, -0.28, 0]} rotation={[forearmRotationL, 0, 0]}>
              <mesh position={[0, -0.12, 0]} castShadow>
                <boxGeometry args={[0.08, 0.24, 0.08]} />
                <meshStandardMaterial color={colors.hoodie} wireframe={wireframe} />
              </mesh>
              {/* Hand */}
              <mesh position={[0, -0.28, 0]} castShadow>
                <boxGeometry args={[0.07, 0.10, 0.04]} />
                <meshStandardMaterial color={colors.skin} wireframe={wireframe} />
              </mesh>
            </group>
          </group>

          {/* RIGHT ARM */}
          <group position={[0.26, 0.22, 0]} rotation={[armRotationR, 0, -0.1]}>
            {/* Upper arm */}
            <mesh position={[0, -0.14, 0]} castShadow>
              <boxGeometry args={[0.10, 0.28, 0.10]} />
              <meshStandardMaterial color={colors.hoodie} wireframe={wireframe} />
            </mesh>
            {/* Forearm */}
            <group position={[0, -0.28, 0]} rotation={[forearmRotationR, 0, 0]}>
              <mesh position={[0, -0.12, 0]} castShadow>
                <boxGeometry args={[0.08, 0.24, 0.08]} />
                <meshStandardMaterial color={colors.hoodie} wireframe={wireframe} />
              </mesh>
              {/* Hand */}
              <mesh position={[0, -0.28, 0]} castShadow>
                <boxGeometry args={[0.07, 0.10, 0.04]} />
                <meshStandardMaterial color={colors.skin} wireframe={wireframe} />
              </mesh>
            </group>
          </group>
        </group>

        {/* LEFT LEG */}
        <group position={[-0.09, -0.08, 0]} rotation={[thighRotationL, 0, 0]}>
          {/* Thigh */}
          <mesh position={[0, -0.21, 0]} castShadow>
            <boxGeometry args={[0.13, 0.42, 0.13]} />
            <meshStandardMaterial color={colors.pants} wireframe={wireframe} />
          </mesh>
          {/* Calf */}
          <group position={[0, -0.42, 0]} rotation={[calfRotationL, 0, 0]}>
            <mesh position={[0, -0.19, 0]} castShadow>
              <boxGeometry args={[0.11, 0.38, 0.11]} />
              <meshStandardMaterial color={colors.pants} wireframe={wireframe} />
            </mesh>
            {/* Foot */}
            <mesh position={[0, -0.40, 0.04]} castShadow>
              <boxGeometry args={[0.11, 0.07, 0.22]} />
              <meshStandardMaterial color={colors.shoes} wireframe={wireframe} />
            </mesh>
          </group>
        </group>

        {/* RIGHT LEG */}
        <group position={[0.09, -0.08, 0]} rotation={[thighRotationR, 0, 0]}>
          {/* Thigh */}
          <mesh position={[0, -0.21, 0]} castShadow>
            <boxGeometry args={[0.13, 0.42, 0.13]} />
            <meshStandardMaterial color={colors.pants} wireframe={wireframe} />
          </mesh>
          {/* Calf */}
          <group position={[0, -0.42, 0]} rotation={[calfRotationR, 0, 0]}>
            <mesh position={[0, -0.19, 0]} castShadow>
              <boxGeometry args={[0.11, 0.38, 0.11]} />
              <meshStandardMaterial color={colors.pants} wireframe={wireframe} />
            </mesh>
            {/* Foot */}
            <mesh position={[0, -0.40, 0.04]} castShadow>
              <boxGeometry args={[0.11, 0.07, 0.22]} />
              <meshStandardMaterial color={colors.shoes} wireframe={wireframe} />
            </mesh>
          </group>
        </group>
      </group>
    </group>
  );
}
