import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Float, Html, Line } from "@react-three/drei";
import { useState, useRef, useMemo } from "react";
import * as THREE from "three";
import Particles from "./Particles";

function Node({ position, color, label, size = 0.6 }) {
  const [hovered, setHover] = useState(false);
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle pulse effect
      const t = state.clock.getElapsedTime();
      meshRef.current.scale.setScalar(1 + Math.sin(t * 3) * 0.05);
    }
  });

  return (
    <mesh
      position={position}
      ref={meshRef}
      onPointerOver={() => setHover(true)}
      onPointerOut={() => setHover(false)}
    >
      <sphereGeometry args={[size, 32, 32]} />
      <meshStandardMaterial 
        color={color} 
        emissive={color} 
        emissiveIntensity={hovered ? 1.5 : 0.8}
        roughness={0.2}
        metalness={0.8}
      />
      {hovered && (
        <Html position={[0, size + 0.5, 0]} center>
          <div className="bg-black/80 backdrop-blur-md px-3 py-1 rounded border border-white/20 text-white font-mono text-sm whitespace-nowrap shadow-[0_0_15px_rgba(255,255,255,0.2)]">
            {label}
          </div>
        </Html>
      )}
    </mesh>
  );
}

function DataPacket({ start, end, color, speed = 1, delay = 0 }) {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    // Wrap around from 0 to 1 with an offset
    const progress = ((t * speed + delay) % 2); 
    
    if (progress > 1) {
      meshRef.current.visible = false;
    } else {
      meshRef.current.visible = true;
      const x = start[0] + (end[0] - start[0]) * progress;
      const y = start[1] + (end[1] - start[1]) * progress;
      const z = start[2] + (end[2] - start[2]) * progress;
      meshRef.current.position.set(x, y, z);
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.1, 16, 16]} />
      <meshBasicMaterial color={color} />
      <pointLight color={color} intensity={1} distance={2} />
    </mesh>
  );
}

function ConnectionLine({ start, end, color }) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end]);

  return (
    <Line
      points={points}
      color={color}
      lineWidth={1}
      transparent
      opacity={0.3}
    />
  );
}

function SceneContent() {
  const posEnv = [-3.5, 0, 0];
  const posOracle = [0, 2.5, 0];
  const posAgent = [3.5, 0, 0];

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1.5} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      <Particles />

      <group>
        {/* Connection Lines */}
        <ConnectionLine start={posEnv} end={posOracle} color="#8b5cf6" />
        <ConnectionLine start={posOracle} end={posAgent} color="#06b6d4" />
        <ConnectionLine start={posEnv} end={posAgent} color="#475569" />

        {/* Data/Goods Packets Flowing */}
        {/* Env -> Oracle (Disruption signals) */}
        <DataPacket start={posEnv} end={posOracle} color="#c084fc" speed={0.8} />
        <DataPacket start={posEnv} end={posOracle} color="#c084fc" speed={0.8} delay={0.5} />
        
        {/* Oracle -> Agent (Risk Scores) */}
        <DataPacket start={posOracle} end={posAgent} color="#22d3ee" speed={1.2} />
        <DataPacket start={posOracle} end={posAgent} color="#22d3ee" speed={1.2} delay={0.5} />

        {/* Env -> Agent (Supply Chain State / Goods) */}
        <DataPacket start={posEnv} end={posAgent} color="#94a3b8" speed={0.5} />
        <DataPacket start={posEnv} end={posAgent} color="#94a3b8" speed={0.5} delay={0.33} />
        <DataPacket start={posEnv} end={posAgent} color="#94a3b8" speed={0.5} delay={0.66} />

        {/* Nodes */}
        <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
          <Node position={posEnv} color="#8b5cf6" label="Supply Chain Environment" />
        </Float>

        <Float speed={3} rotationIntensity={1} floatIntensity={1.5}>
          <Node position={posOracle} color="#06b6d4" label="Oracle News Module (LLM)" />
        </Float>

        <Float speed={2.5} rotationIntensity={0.5} floatIntensity={1}>
          <Node position={posAgent} color="#f43f5e" label="PPO Augmented Agent" size={0.8} />
        </Float>
      </group>

      {/* Orbit controls limited to prevent getting lost */}
      <OrbitControls 
        enableZoom={false} 
        enablePan={false}
        minPolarAngle={Math.PI / 3} 
        maxPolarAngle={Math.PI / 1.5}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </>
  );
}

export default function Scene3D() {
  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 45 }} dpr={[1, 2]}>
      <SceneContent />
    </Canvas>
  );
}
