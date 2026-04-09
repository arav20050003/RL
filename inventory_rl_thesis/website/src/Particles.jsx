import { useRef } from "react";
import { useFrame } from "@react-three/fiber";

export default function Particles() {
  const ref = useRef();

  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += 0.0005;
      ref.current.rotation.x += 0.0002;
    }
  });

  // Create 1000 random particles spread in a wide radius
  const particleCount = 1000;
  const positions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 20;
  }

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>

      <pointsMaterial size={0.03} color="#ffffff" transparent opacity={0.6} sizeAttenuation={true} />
    </points>
  );
}
