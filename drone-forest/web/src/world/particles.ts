/**
 * Drifting motes / falling leaves. One Points object whose vertex shader wraps every particle
 * into a box centred on the drone, so the layer follows the flight with zero CPU work per
 * frame. Additive, depth-tested, no depth write: they glow softly in the sun through bloom.
 */
import * as THREE from 'three';
import { markScenery } from './layers';
import type { Rng } from '../core/types';

const COUNT = 900;
const BOX = new THREE.Vector3(90, 40, 120);

export interface Particles {
  readonly points: THREE.Points;
  update(time: number, anchor: THREE.Vector3): void;
  dispose(): void;
}

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform vec3 uAnchor;
  uniform vec3 uBox;
  uniform float uPixelRatio;
  attribute float aSize;
  attribute float aPhase;
  varying float vAlpha;
  void main() {
    vec3 drift = vec3(
      sin(uTime * 0.35 + aPhase) * 2.0,
      -uTime * (0.45 + 0.35 * fract(aPhase * 0.618)),
      cos(uTime * 0.27 + aPhase * 1.7) * 2.0
    );
    vec3 rel = mod(position + drift - uAnchor + uBox * 0.5, uBox) - uBox * 0.5;
    vec3 world = rel + uAnchor;
    vec4 mv = modelViewMatrix * vec4(world, 1.0);
    float dist = -mv.z;
    gl_PointSize = aSize * uPixelRatio * clamp(220.0 / max(dist, 1.0), 0.6, 18.0);
    vAlpha = smoothstep(0.0, 6.0, dist) * (1.0 - smoothstep(45.0, 80.0, dist));
    gl_Position = projectionMatrix * mv;
  }
`;

const fragmentShader = /* glsl */ `
  varying float vAlpha;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float soft = 1.0 - smoothstep(0.02, 0.25, r2);
    gl_FragColor = vec4(1.0, 0.86, 0.55, 1.0) * soft * vAlpha * 0.35;
  }
`;

export function createParticles(scene: THREE.Scene, rng: Rng): Particles {
  const positions = new Float32Array(COUNT * 3);
  const sizes = new Float32Array(COUNT);
  const phases = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i++) {
    positions[i * 3] = rng.range(0, BOX.x);
    positions[i * 3 + 1] = rng.range(0, BOX.y);
    positions[i * 3 + 2] = rng.range(0, BOX.z);
    sizes[i] = rng.range(0.3, 1.0);
    phases[i] = rng.range(0, Math.PI * 2);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uAnchor: { value: new THREE.Vector3() },
      uBox: { value: BOX.clone() },
      uPixelRatio: { value: 1 },
    },
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  markScenery(points);
  scene.add(points);

  return {
    points,
    update(time, anchor) {
      material.uniforms['uTime']!.value = time;
      (material.uniforms['uAnchor']!.value as THREE.Vector3).copy(anchor);
      material.uniforms['uPixelRatio']!.value = Math.min(window.devicePixelRatio, 2);
    },
    dispose() {
      scene.remove(points);
      geometry.dispose();
      material.dispose();
    },
  };
}
